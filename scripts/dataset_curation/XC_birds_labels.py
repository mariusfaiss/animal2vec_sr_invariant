import os
import glob
import h5py
import ntpath
import soundfile as sf
import pandas as pd
import ast

#############################################################################
# script for creating labels for each file

audio_folder = "/Volumes/EAS001598_4tb/xc_birds/10s_splits"
label_folder = "/Volumes/EAS001598_4tb/xc_birds/lbl"

metadata = pd.read_csv("/Volumes/EAS001598_4tb/xc_birds/metadata_files/XC_filtered_bird_recordings.csv")
metadata_indexed = metadata.set_index('observation_id')

unique_labels = {"beep": 0, "synch": 1, "sn": 2, "cc": 3, "ld": 4, "oth": 5, "mo": 6, "al": 7, "soc": 8, "agg": 9,
                 "eating": 10,

                 "Parus_major": 11, "Turdus_merula": 12, "Fringilla_coelebs": 13, "Phylloscopus_collybita": 14,
                 "Erithacus_rubecula": 15, "Turdus_philomelos": 16, "Sylvia_atricapilla": 17,
                 "Phylloscopus_trochilus": 18, "Loxia_curvirostra": 19, "Troglodytes_troglodytes": 20,
                 "Cyanistes_caeruleus": 21}


#######################################################################################################################

audio_files = len(glob.glob(f"{audio_folder}/*"))
counter = 0

for filename in glob.glob(os.path.join(f"{audio_folder}/*")):
    counter += 1
    basename = ntpath.basename(filename)

    # output filename
    f_name_lbl = basename.split(".")[0] + ".h5"

    if not os.path.isfile(os.path.join(label_folder, f_name_lbl)):
        print(f"{counter}/{audio_files}: {f_name_lbl}")

        # get the labels
        observation = basename.split("_")[2].split(".")[0]
        labels_string = metadata_indexed.loc[observation, 'species_labels']
        labels_list = [species.strip() for species in labels_string.split(',')]
        foc_label = basename.split("XC")[0][:-1]

        # Get the number of items in your labels_list
        num_labels = len(labels_list)

        # read waveform
        try:
            waveform, samplerate = sf.read(filename)
        except sf.LibsndfileError as e:
            print("{} is corrupt and raised a LibsndfileError. We skip it.".format(basename), flush=True)
            continue
        if len(waveform) == 0:
            print("{} is corrupt and has a length of zero. We skip it.".format(basename), flush=True)
            continue

        # calculate time stamps
        start_frame_lbl = [0] * num_labels
        start_time_lbl = [0.0] * num_labels
        end_frame_lbl = [len(waveform)] * num_labels
        end_time_lbl = [len(waveform)/samplerate] * num_labels

        # prepare label metadata
        lbl_cat = []
        foc = []
        for label in labels_list:
            lbl_cat.append(unique_labels[label])
            if label == foc_label:
                foc.append(1)
            else:
                foc.append(0)

        labels_list = [species.encode('utf-8') for species in labels_list]

        # write hdf5 file
        with h5py.File(os.path.join(label_folder, f_name_lbl), mode="w") as f:
            f.create_dataset(name="start_time_lbl", data=start_time_lbl)
            f.create_dataset(name="start_frame_lbl", data=start_frame_lbl)
            f.create_dataset(name="end_time_lbl", data=end_time_lbl)
            f.create_dataset(name="end_frame_lbl", data=end_frame_lbl)
            f.create_dataset(name="lbl", data=labels_list)
            f.create_dataset(name="lbl_cat", data=lbl_cat)
            f.create_dataset(name="foc", data=foc)

    else:
        print(f"{counter}/{audio_files}")
