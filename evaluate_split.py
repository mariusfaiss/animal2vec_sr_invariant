import os
import sys
import h5py
import json
import numpy as np
import pandas as pd
import argparse
import warnings
from sklearn.metrics import classification_report, average_precision_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# ==========================================
# CONFIGURATION
# ==========================================
CLASS_NAMES = ['beep', 'synch', 'sn', 'cc', 'ld', 'oth', 'mo', 'al', 'soc', 'agg', 'eating', 'Parus_major',
               'Turdus_merula', 'Fringilla_coelebs', 'Phylloscopus_collybita', 'Erithacus_rubecula',
               'Turdus_philomelos', 'Sylvia_atricapilla', 'Phylloscopus_trochilus', 'Loxia_curvirostra',
               'Troglodytes_troglodytes', 'Cyanistes_caeruleus']

THRESHOLD = 0.175


def find_h5_file(cwd):
    files = [f for f in os.listdir(cwd) if f.endswith('.h5')]
    if len(files) == 0:
        print(f"Error: No .h5 files found in {cwd}")
        sys.exit(1)
    elif len(files) > 1:
        print(f"Error: Multiple .h5 files found in {cwd}. Specify one with --file.")
        sys.exit(1)
    print(f"Auto-detected prediction file: {files[0]}")
    return os.path.join(cwd, files[0])


def load_data(h5_path):
    print(f"Loading data from: {h5_path}")
    y_true = []
    y_score = []
    sample_rates = []
    filenames = []

    if not os.path.exists(h5_path):
        print(f"Error: File not found at {h5_path}")
        sys.exit(1)

    try:
        with h5py.File(h5_path, 'r') as f:
            keys = sorted(list(f.keys()))
            for key in keys:
                lik = f[key]['likelihood'][()]
                tar = f[key]['target'][()]

                # get filename
                fname_data = f[key]['fname'][()]
                if isinstance(fname_data, bytes):
                    fname = fname_data.decode('utf-8')
                else:
                    fname = str(fname_data)

                # get sample rate
                if "samplerate" in f[key]:
                    sr = f[key]['samplerate'][()]
                    if hasattr(sr, 'item'):
                        sr = sr.item()
                else:
                    sr = -1

                if lik.ndim == 2:
                    lik = np.max(lik, axis=0)
                    tar = np.max(tar, axis=0)

                y_score.append(lik)
                y_true.append(tar)
                sample_rates.append(sr)
                filenames.append(fname)

    except OSError:
        print("Error: Could not open file.")
        sys.exit(1)

    return np.array(y_true), np.array(y_score), np.array(sample_rates), np.array(filenames)


def generate_full_report(y_true, y_score, class_names=None):
    num_classes = y_true.shape[1]

    # calculate support manually
    support_per_class = np.sum(y_true, axis=0)
    active_indices = np.where(support_per_class > 0)[0]

    if not class_names or len(class_names) != num_classes:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    y_pred_binary = (y_score > THRESHOLD).astype(int)

    report_dict = classification_report(
        y_true, y_pred_binary,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(report_dict).transpose()

    # calculate AP
    per_class_ap = average_precision_score(y_true, y_score, average=None)

    # mask inactive classes for mAP calculation
    per_class_ap_masked = per_class_ap.copy()
    for i in range(num_classes):
        if i not in active_indices:
            per_class_ap_masked[i] = np.nan

    if len(active_indices) > 0:
        mean_ap = np.nanmean(per_class_ap_masked)
    else:
        mean_ap = 0.0

    # recalculate macro averages
    if len(active_indices) > 0:
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_true, y_pred_binary,
            average='macro',
            labels=active_indices,
            zero_division=0
        )
        df.loc['macro avg', 'precision'] = p_macro
        df.loc['macro avg', 'recall'] = r_macro
        df.loc['macro avg', 'f1-score'] = f_macro

    df['AP'] = np.nan
    for idx, name in enumerate(class_names):
        if name in df.index:
            val = per_class_ap_masked[idx]
            df.loc[name, 'AP'] = val
            if idx not in active_indices:
                df.loc[name, 'precision'] = np.nan
                df.loc[name, 'recall'] = np.nan
                df.loc[name, 'f1-score'] = np.nan

    if 'micro avg' in df.index:
        df.loc['micro avg', 'AP'] = average_precision_score(y_true, y_score, average='micro')

    if 'macro avg' in df.index:
        df.loc['macro avg', 'AP'] = mean_ap
        df = df.rename(index={'macro avg': 'macro avg (mAP)'})

    if 'weighted avg' in df.index:
        df.loc['weighted avg', 'AP'] = average_precision_score(y_true, y_score, average='weighted')

    if 'samples avg' in df.index:
        try:
            df.loc['samples avg', 'AP'] = average_precision_score(y_true, y_score, average='samples')
        except:
            df.loc['samples avg', 'AP'] = 0.0

    cols = ['precision', 'recall', 'f1-score', 'AP', 'support']
    df = df[cols]

    return df, mean_ap


def log_metrics_to_tensorboard(writer, df_results, step=1, group_name="Overall"):
    print(f"Logging group '{group_name}' to TensorBoard...")

    if 'macro avg (mAP)' in df_results.index:
        map_val = df_results.loc['macro avg (mAP)', 'AP']
        writer.add_scalar(f'mAP/{group_name}', map_val, step)

    if 'micro avg' in df_results.index:
        f1_micro = df_results.loc['micro avg', 'f1-score']
        writer.add_scalar(f'F1_Micro/{group_name}', f1_micro, step)

    try:
        md_table = df_results.fillna('').to_markdown(floatfmt=".4f")
        writer.add_text(f'Evaluation/Report_{group_name}', md_table, step)
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Predictions")
    parser.add_argument('--dir', type=str, default="",
                        help="Directory with .h5 file")
    parser.add_argument('--file', type=str, default=None, help="Specific filename")
    parser.add_argument('--step', type=int, default=1, help="TensorBoard Step")
    parser.add_argument('--sr_map', type=str, default="/home/jupyter-mfaiss/Datasets/48_MeerKAT_XCbirds_10s/original_SRs_48_MeerKAT_XCbirds_10s.json",
                        help="Path to sr_mapping.json to override sample rates")

    args = parser.parse_args()

    # logging setup
    log_file = os.path.join(args.dir, "evaluation_results.txt")
    sys.stdout = Logger(log_file)
    print(f"Saving console output to: {log_file}")

    if args.file:
        file_path = os.path.join(args.dir, args.file)
    else:
        file_path = find_h5_file(args.dir)

    # load Data
    y_true_all, y_score_all, sr_all, filenames_all = load_data(file_path)

    # use sample rate map if provided
    if args.sr_map:
        print(f"\nLoading Sample Rate Map from: {args.sr_map}")
        try:
            with open(args.sr_map, 'r') as f:
                sr_map_dict = json.load(f)

            print("Mapping original sample rates to current predictions...")
            mapped_count = 0
            missing_count = 0

            # create new array for mapped rates
            sr_mapped = np.full_like(sr_all, -1)

            for i, fname in enumerate(filenames_all):
                if fname in sr_map_dict:
                    sr_mapped[i] = sr_map_dict[fname]
                    mapped_count += 1
                else:
                    # check if it was converted to .wav
                    fname_wav = fname[:-4] + ".wav"
                    if fname_wav in sr_map_dict:
                        sr_mapped[i] = sr_map_dict[fname_wav]
                        mapped_count += 1
                    else:
                        # check if original was .WAV
                        fname_WAV = fname[:-4] + ".WAV"
                        if fname_WAV in sr_map_dict:
                            sr_mapped[i] = sr_map_dict[fname_WAV]
                            mapped_count += 1
                        else:
                            # leave as -1 sentinel
                            missing_count += 1
                            print(f"missing: {fname}")

            print(f"Mapping Complete: {mapped_count} mapped, {missing_count} missing (set to -1).")

            # replace the array used for evaluation
            sr_all = sr_mapped

        except Exception as e:
            print(f"Error loading SR Map: {e}")
            sys.exit(1)

    if len(y_true_all) > 0:
        writer = SummaryWriter(log_dir=args.dir)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)

        evaluation_groups = [
            ("Overall", 0, 22),
            ("Meerkats", 0, 11),
            ("Birds", 11, 22)
        ]

        unique_srs = np.unique(sr_all)
        # exclude -1 (files that were not found in the map)
        unique_srs = unique_srs[unique_srs != -1]

        for group_name, start_idx, end_idx in evaluation_groups:

            print("\n" + "#" * 60)
            print(f"  ANALYSIS GROUP: {group_name.upper()}")
            print("#" * 60)

            # slice data
            y_true_group = y_true_all[:, start_idx:end_idx]
            y_score_group = y_score_all[:, start_idx:end_idx]
            class_names_group = CLASS_NAMES[start_idx:end_idx]

            # first evaluate all sample rates
            if np.sum(y_true_group) > 0:
                print("\n" + "=" * 40)
                print(f"Dataset: {group_name} (All Sample Rates)")
                print("=" * 40)

                df_group, mAP_group = generate_full_report(y_true_group, y_score_group, class_names_group)

                print(df_group.fillna(''))
                print(f"\n{group_name} (All) mAP: {mAP_group:.4f}")
                log_metrics_to_tensorboard(writer, df_group, step=args.step, group_name=f"{group_name}/All_Files")
            else:
                print(f"Skipping {group_name} (All): No positive labels.")

            # then evaluate per sample rate
            for sr in unique_srs:
                mask = (sr_all == sr)

                y_true_sr = y_true_group[mask]
                y_score_sr = y_score_group[mask]

                if len(y_true_sr) == 0: continue
                if np.sum(y_true_sr) == 0: continue

                print("\n" + "-" * 40)
                print(f"Dataset: {group_name} | Sample Rate: {sr} Hz")
                print("-" * 40)
                print(f"Files found: {len(y_true_sr)}")

                df_sr, mAP_sr = generate_full_report(y_true_sr, y_score_sr, class_names_group)

                print(df_sr.fillna(''))
                print(f"{group_name} (SR {sr}) mAP: {mAP_sr:.4f}")

                log_metrics_to_tensorboard(writer, df_sr, step=args.step, group_name=f"{group_name}/SR_{sr}")

        writer.close()
        print("\n" + "=" * 60)
        print(f"TensorBoard logs written to: {args.dir}")
        print("=" * 60)

    else:
        print("No data found.")
