"""
Label parser for the meerkat data
"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import timedelta, datetime

base_path = ["/Volumes/Extreme_SSD/XC_birds/converted"]

metadata_file = pd.read_csv("/Volumes/Extreme_SSD/XC_birds/metadata_files/XC_filtered_bird_recordings.csv")

supported_file_types = ["WAV", "AIFF", "AIFC", "FLAC", "OGG", "MP3", "MAT"]
files = []
for d in base_path:
    for ft in supported_file_types:
        regex_ft = "".join(["[{}{}]".format(x.lower(), x.upper()) for x in ft])
        files += list(Path(d).rglob('*.{}'.format(regex_ft)))

# Remove duplicates
files = list(set(files))

labels = []
for file in files:
    filename = os.path.basename(file)
    label = filename.split("_")[0] + "_" + filename.split("_")[1]
    labels.append(label)

unique, counts = np.unique(labels, return_counts=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warnings.filterwarnings('ignore')
    err_files = 0
    durations = {}
    for ff in files:
        try:
            metadata = sf.info(ff)
            durations[os.path.basename(ff)] = metadata.duration
        except Exception as e:
            err_files += 1
            durations[os.path.basename(ff)] = np.nan

df = pd.DataFrame.from_dict(durations, orient="index", columns=["Duration"]).reset_index(names="AudioFile")
df["Name"] = labels
df = df.dropna(how="any")
# mix_length = 5  # no file shorter than 5 seconds will be considered (this still covers 94.4% of all the data)
max_length = 250  # no file longer than 250 seconds will be considered (this still covers 97.6% of all the data)
# ultimately we still have 92.1 % of the data -> 568h
df = df.where(df["Duration"] < max_length).dropna(how="any").reset_index(drop=True)
# df = df.where(df["Duration"] > mix_length).dropna(how="any").reset_index(drop=True)
df["Focal"] = "focal"
df["StartRelative"] = timedelta(seconds=0)
df["EndRelative"] = [timedelta(seconds=x) for x in df["Duration"].tolist()]

out_path = "/Volumes/Extreme_SSD/XC_birds/metadata_files"
df.to_pickle(os.path.join(out_path, "XC_foreground_labels_{}.P".format(datetime.today().strftime('%Y-%m-%d'))))
