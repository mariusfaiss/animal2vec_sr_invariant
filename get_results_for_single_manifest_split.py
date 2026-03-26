import os
import torch
import h5py
import argparse
import contextlib
import numpy as np
from tqdm import tqdm
from itertools import groupby
import sys
from pathlib import Path

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

from torch.utils.data import DataLoader
from fairseq import checkpoint_utils
from nn.audio_tasks import FileAudioLabelDataset  # for registering everything


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--normalize",
        default=True,
        type=bool,
        help="Should we normalize the input to zero mean and unit variance",
    )
    parser.add_argument(
        "--export_embeddings",
        default=False,
        type=bool,
        help="Should we export the embeddings (This will produce a very large file)",
    )
    parser.add_argument(
        "--export_predictions",
        default=True,
        type=bool,
        help="Should we export the predictions",
    )
    parser.add_argument(
        "--use_softmax",
        default=False,
        type=bool,
        help="If set to True, then we use a softmax as the final activation."
             "Otherwise, we use a sigmoid.",
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join("/Volumes/EAS001598_4tb/SR_MeerKAT_XCbirds_10s/linear_eval/checkpoints/TE_pretrain_small_temporal_mix_samplerate_checkpoint30.pt"),
        type=str,
        help="Path to pretrained model. "
             "This should point to a *.pt file created by pytorch."
    )
    parser.add_argument(
        "--device", default="cuda",
        type=str, choices=["cuda", "cpu"],
        help="The device you want to use. "
             "We will fall back to cpu if no cuda device is available."
    )
    parser.add_argument(
        "--batch_size", default=1,
        type=int,
        help="Every file that is being split into smaller segments and fed to the model"
             "in batches. --batch-size gives this batch size and --segment-length gives"
             "the segment length in sec. Default values are 10s segment length and "
             "12 segments batch size."
             "Adjust if you have more - or less. Keep in mind that first reducing --batch-size"
             "is recommend as reducing segment-length reduces the timespan the transformer"
             "can use for contextualizing."
    )
    parser.add_argument(
        "--average_start_k_layers", default=0,
        type=int,
        help="The transformer layer from which we start averaging"
    )
    parser.add_argument(
        "--average_end_k_layers", default=12,
        type=int,
        help="The transformer layer with which we end averaging"
    )
    parser.add_argument(
        "--out_path", default="/Volumes/EAS001598_4tb/SR_MeerKAT_XCbirds_10s/linear_eval",
        type=str,
        help="The path to where embeddings.h5 file should be written to."
             "The folder does not need to exists. The script can take care of that. "
             "Default is the current workdir."
    )
    parser.add_argument(
        "--conv_feature_layers",
        default='[(512, 10, 5)] + [(512, 3, 3)] + [(512, 3, 2)] * 4 + [(512, 3, 1)] + [(512, 2, 1)] * 2',
        type=str,
        help="string describing convolutional feature extraction layers in form of a python list that contains "
             "[(dim, kernel_size, stride), ...]"
    )
    parser.add_argument(
        "--unique_labels",
        default="['beep', 'synch', 'sn', 'cc', 'ld', 'oth', 'mo', 'al', 'soc', 'agg', 'eating', 'Parus_major', "
                "'Turdus_merula', 'Fringilla_coelebs', 'Phylloscopus_collybita', 'Erithacus_rubecula', "
                "'Turdus_philomelos', 'Sylvia_atricapilla', 'Phylloscopus_trochilus', 'Loxia_curvirostra', "
                "'Troglodytes_troglodytes', 'Cyanistes_caeruleus']",
        type=str,
        help="A string list that, when evaluated, holds the names for the used classes."
    )
    parser.add_argument(
        "--manifest_path", default="/home/jupyter-mfaiss/Datasets/SR_MeerKAT_XCbirds_10s/manifests",
        type=str,
        help="The path to the manifest file folder."
    )
    parser.add_argument(
        "--split", default="valid_0",
        type=str,
        help="The split manifest file to load."
    )
    parser.add_argument(
        "--method",
        default="avg",
        type=str,
        choices=["avg", "max", "canny"],
        help="Which method to use for fusing the predictions into time bins."
             "avg: Average pooling, then thresholding."
             "max: Max pooling, then thresholding."
             "canny: Canny edge detector",
    )
    parser.add_argument(
        "--sigma_s",
        default=0.1,
        type=float,
        help="Size of Gaussian (std dev) in seconds for the canny method. "
             "Filter width in seconds for avg and max methods.",
    )
    parser.add_argument(
        "--metric_threshold",
        default=0.5,
        type=float,
        help="Threshold for the filtered predictions. Only used when --method={max, avg}",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.,
        type=float,
        help="The minimum IoU that is needed for a prediction to be counted as "
             "overlapping with focal. 0 means that even a single frame overlap is enough, "
             "while 1 means that only a perfect overlap is counted as overlap.",
    )
    parser.add_argument(
        "--maxfilt_s",
        default=0.1,
        type=float,
        help="Time to smooth with maxixmum filter. Only used when --method=canny",
    )
    parser.add_argument(
        "--max_duration_s",
        default=0.05,
        type=float,
        help="Detections are never longer than max duration (s). Only used when --method=canny",
    )
    parser.add_argument(
        "--sample_rate",
        default=48000,
        type=int,
        help="The sample rate in Hz for the input data",
    )
    parser.add_argument(
        "--min_label_size",
        default=3032,
        type=int,
        help="if set, we only load files from the manifest whose label file bytesize "
             "is larger then this value. Hint: Empty label files have Bytesize 3032."
             "Only used when with_labels is True",
    )
    parser.add_argument(
        "--lowP",
        default=0.125,
        type=float,
        help="Low threshold. Detections are based on Canny edge detector, "
             "but the detector sometimes misses the minima on some transients"
             "if they are not sharp enough.  lowP is used for pruning detections."
             "Detections whose Gaussian-smoothed signal is beneath lowP are discarded."
             "Only used when --method=canny",
    )
    parser.add_argument(
        "--event_detection",
        default=False,
        type=bool,
        help="Detects events on a per-frame basis, exports full embeddings,"
             "If False, does simple sample-level classification and pools the "
             "embeddings along the time axis.",
    )
    parser.add_argument(
        "--pooling_method",
        default="max",
        type=str,
        choices=["mean", "max"],
        help="When not doing event detection, sets how embeddings are pooled along the time axis. Can be either max-"
             "pooling or mean_pooling",
    )
    parser.add_argument(
        "--return_labels",
        default="True",
        type=bool,
        help="return labels or not",
    )
    return parser


def get_intervalls(data, shift=0):
    # Group the array in segments with itertools groupby function
    grouped = (list(g) for _, g in groupby(enumerate(data), lambda t: t[1]))
    # Only add the interval if it is with values larger than 0
    return [(g[0][0] + shift, min([len(data) - 1, g[-1][0] + shift])) for g in grouped if g[0][1] == 1]


def main(args):
    model_path = args.model_path
    assert os.path.isfile(model_path)

    # Check if out path exists
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    method_dict = {
        "sigma_s": args.sigma_s,
        "metric_threshold": args.metric_threshold,
        "maxfilt_s": args.maxfilt_s,
        "max_duration_s": args.max_duration_s,
        "lowP": args.lowP,
        "iou_threshold": args.iou_threshold,
    }

    # load the model
    device = "cuda" if torch.cuda.is_available() and args.device.lower() == "cuda" else "cpu"
    print("Loading the model and placing on {} ... ".format(device), end="")
    models, cfg = checkpoint_utils.load_model_ensemble([model_path])
    model = models[0].to(device)  # place on appropriate device
    print("done")

    # The labels on which the model was trained on
    # print("\n args.unique_labels", args.unique_labels)
    print("Loading the data ... ", end="")
    dataset = FileAudioLabelDataset(
        manifest_path=os.path.join(args.manifest_path, args.split + ".tsv"),
        sample_rate=args.sample_rate,
        pad=True,
        min_sample_size=1612,
        normalize=args.normalize,
        shuffle=False,
        return_labels=args.return_labels,
        unique_labels=eval(args.unique_labels),
        use_focal_loss=cfg.criterion.use_focal_loss,
        min_label_size=args.min_label_size,
        conv_feature_layers=args.conv_feature_layers,
        segmentation_metrics=False,
        do_focal_prediction=False
    )

    num_cpus = os.cpu_count()
    num_workers = num_cpus

    print(f"Using {num_workers} worker processes for data loading...")

    # args.export_embeddings = False
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=dataset.collater,
                            num_workers=num_workers,
                            pin_memory=True
                            )
    print("done")

    # file name will "embeddings_model-name_checkpoint-name_avg-start_avg-end_split-name_foldername.h5"
    folder_name = model_path[len(model_path[:model_path.find("/checkpoints/")]) -
                             model_path[
                             :model_path.find("/checkpoints/")][::-1].find("/"):model_path.find("/checkpoints/")]
    file_args = (args.pooling_method, os.path.basename(model_path),
                 args.average_start_k_layers, args.average_end_k_layers,
                 args.split, folder_name)
    if args.export_embeddings:
        emb_out_filename = "embeddings_{}_{}_{}_{}_{}_{}.h5".format(*file_args)
        out_path = os.path.join(args.out_path, emb_out_filename)
        if os.path.isfile(out_path):
            os.remove(out_path)
        embedding_context = h5py.File(out_path, "w")
    else:
        embedding_context = contextlib.nullcontext()

    if args.export_predictions:
        pred_out_filename = "predictions_{}_{}_{}_{}_{}_{}.h5".format(*file_args)
        out_path = os.path.join(args.out_path, pred_out_filename)
        if os.path.isfile(out_path):
            os.remove(out_path)
        prediction_context = h5py.File(out_path, "w")
    else:
        prediction_context = contextlib.nullcontext()

    if args.export_predictions or args.export_embeddings:
        with torch.inference_mode(), embedding_context as f_h5, prediction_context as f_pred_h5:
            model.eval()
            for samples in tqdm(dataloader, desc="Starting inference", file=sys.stdout):
                net_input = samples["net_input"]
                source = net_input["source"].squeeze()
                sample_rates = samples["sample_rate"]
                padding_mask = net_input["padding_mask"]
                targets = samples["target"]

                res = model.extract_features(source=source.to(device), sample_rates=sample_rates,
                                             padding_mask=padding_mask, target=targets)
                targets = res["target"]
                res_padding_mask = res["padding_mask"]
                # IMPLEMENT EVENT DETECTION DISABLED HERE

                if args.event_detection:
                    if cfg.criterion.use_focal_loss:
                        seg_target_idx = [[get_intervalls(x) for x in y.T] for y in targets]
                    else:
                        seg_target_idx = [get_intervalls(y) for y in targets]
                else:
                    # Assumes targets shape: [Batch, Classes]
                    # We simply identify which class indices are active (value == 1)
                    seg_target_idx = [torch.nonzero(y).flatten().tolist() for y in targets]

                samples["seg_target_idx"] = seg_target_idx
                samples["source_size"] = source.size(-1)

                if args.export_predictions:
                    if "linear_eval_projection" in res:
                        if args.use_softmax:
                            probs = torch.softmax(res["linear_eval_projection"].float(), -1)
                        else:
                            probs = torch.sigmoid(res["linear_eval_projection"].float())
                    else:
                        if args.use_softmax:
                            probs = torch.softmax(res["encoder_out"].float(), -1)
                        else:
                            probs = torch.sigmoid(res["encoder_out"].float())
                    if not args.use_softmax and args.event_detection:
                        pr, ta, ios, sp, me = model.get_segmented_probs_and_targets(
                            samples,
                            torch.tensor(probs), method_dict,
                            method=args.method
                        )

                    bs = probs.size(0)
                    likelihoods_export = probs.detach().cpu().numpy()
                    targets_export = targets.detach().cpu().numpy()
                    if args.event_detection:
                        time_dim = probs.size(1)
                        if not args.use_softmax:
                            segmented_likelihoods_export = pr.detach().cpu().numpy()
                            segmented_targets_export = ta.detach().cpu().numpy()
                        else:
                            segmented_likelihoods_export = likelihoods_export.copy()
                            segmented_targets_export = targets_export.copy()
                        segmented_likelihoods_export = segmented_likelihoods_export.reshape(bs, time_dim, -1)
                        segmented_targets_export = segmented_targets_export.reshape(bs, time_dim, -1)

                        like_tar_shapes = likelihoods_export.shape == targets_export.shape
                        seg_like_seg_tar_shapes = segmented_likelihoods_export.shape == segmented_targets_export.shape
                        seg_like_like_shapes = segmented_likelihoods_export.shape == likelihoods_export.shape
                        seg_tar_tar_shapes = segmented_targets_export.shape == targets_export.shape
                        _assert(
                            like_tar_shapes and seg_like_seg_tar_shapes and seg_like_like_shapes and seg_tar_tar_shapes,
                            "Predictions and Target do not share the first two dimensions.\n"
                            "Model Name: {},"
                            "Predictions Shape: {}, Target Shape {},"
                            "Segmented Predictions Shape: {}, Segmented Target Shape {}".format(
                                model.__class__.__name__,
                                likelihoods_export.shape, targets_export.shape,
                                segmented_likelihoods_export.shape, segmented_targets_export.shape
                            )
                        )
                    else:
                        # Assertions for File Level (Just Batch and Classes must match)
                        _assert(
                            likelihoods_export.shape == targets_export.shape,
                            "Predictions and Target do not share dimensions (File Level Mode).\n"
                            "Shapes: Pred {}, Tar {}".format(likelihoods_export.shape, targets_export.shape)
                        )

                    # We iterate over the batch. If event_detection is on, we also grab the segmented arrays.
                    for enu in range(bs):
                        # Data for this sample
                        like = likelihoods_export[enu]
                        tar = targets_export[enu]

                        # Create group with unix index
                        index = samples["id"][enu]
                        grp_pred = f_pred_h5.create_group("{:06.0f}".format(index))

                        # Filename handling
                        fn_pred = dataset.fnames[index]
                        fn_pred = fn_pred if isinstance(dataset.fnames, list) else fn_pred.as_py()
                        fn_pred = dataset.text_compressor.decompress(fn_pred)
                        
                        # get sample rate
                        sr_pred = sample_rates[enu].item()

                        grp_pred.create_dataset("fname", data=fn_pred)
                        grp_pred.create_dataset("samplerate", data=sr_pred)
                        grp_pred.create_dataset("likelihood", data=like, dtype=np.float32)
                        grp_pred.create_dataset("target", data=tar, dtype=np.float32)

                        # Only save segmented data if we are in event detection mode and not using softmax
                        if args.event_detection and not args.use_softmax:
                            seg_like = segmented_likelihoods_export[enu]
                            seg_tar = segmented_targets_export[enu]
                            grp_pred.create_dataset("segmented_likelihood", data=seg_like, dtype=np.float32)
                            grp_pred.create_dataset("segmented_target", data=seg_tar, dtype=np.float32)

                if args.export_embeddings:
                    layer_results = res["layer_results"]
                    min_layer = args.average_start_k_layers
                    max_layer = args.average_end_k_layers

                    finetuned = False
                    if hasattr(model, "w2v_encoder"):  # is finetuned or not
                        finetuned = True

                    d2v = getattr(model, "is_d2v_multi", False) or "data2vec" in model.__class__.__name__.lower()
                    if finetuned:
                        if hasattr(model.w2v_encoder, "is_d2v_multi"):  # finetuned data2vec model
                            d2v = getattr(model.w2v_encoder, "is_d2v_multi", False)

                    if d2v:
                        target_layer_results = [l for l in layer_results[min_layer:max_layer]]
                    else:
                        target_layer_results = [l[0] for l in layer_results[min_layer:max_layer]]
                    if not d2v:  # transpose if w2v
                        target_layer_results = [tl.transpose(0, 1) for tl in target_layer_results]  # HTBC -> HBTC
                    # Average over transformer layers
                    avg_emb = (sum(target_layer_results) / len(target_layer_results)).float()  # BTC

                    if not args.event_detection:

                        if args.pooling_method == "max":
                            if res_padding_mask is not None and res_padding_mask.any():
                                expanded_padding_mask = res_padding_mask.unsqueeze(-1).expand_as(avg_emb)
                                avg_emb.masked_fill_(expanded_padding_mask, -torch.inf)

                            avg_emb, _ = torch.max(avg_emb, dim=1)  # Output shape is BC

                        if args.pooling_method == "mean":
                            if res_padding_mask is not None and res_padding_mask.any():
                                mask = ~res_padding_mask.unsqueeze(-1)
                                masked_embeddings = avg_emb * mask.float()
                                summed_embeddings = torch.sum(masked_embeddings, dim=1)
                                valid_token_count = mask.sum(dim=1).float()
                                valid_token_count = torch.clamp(valid_token_count, min=1e-9)
                                avg_emb = summed_embeddings / valid_token_count
                            else:
                                avg_emb = torch.sum(avg_emb, dim=1) / len(avg_emb[1].float())

                        targets, _ = torch.max(targets.type(torch.int64), dim=1)

                    embeddings_export = avg_emb.detach().cpu().numpy()
                    targets_export = targets.detach().cpu().numpy()

                    if args.event_detection:
                        _assert(
                            embeddings_export.shape[:2] == targets_export.shape[:2],
                            "Embeddings and Target do not share the first two dimensions.\n"
                            "Model Name: {}, d2v detected: {}\n"
                            "Embeddings Shape: {}, Target Shape {}".format(
                                model.__class__.__name__, d2v,
                                embeddings_export.shape, targets_export.shape)
                        )

                    for enu, (emb, tar) in enumerate(zip(embeddings_export, targets_export)):  # Iterate over Batch
                        # Create group with unix index
                        index = samples["id"][enu]
                        grp = f_h5.create_group("{:06.0f}".format(index))
                        # This group holds information about the filename, the target, and the embedding
                        fn = dataset.fnames[index]
                        fn = fn if isinstance(dataset.fnames, list) else fn.as_py()
                        fn = dataset.text_compressor.decompress(fn)
                        grp.create_dataset("fname", data=fn)
                        grp.create_dataset("embedding", data=emb, dtype=np.float32)  # BTC
                        grp.create_dataset("target", data=tar, dtype=np.float32)  # BTC


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    main(args)
