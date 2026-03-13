from urllib import request
import os
import pandas as pd
import glob
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

convert_gbif = False

folder = f"/Users/mfaiss/PycharmProjects/self-supervised-animal-vocalizations/GBIF"

if convert_gbif:
    # import multimedia file with links and create data frame with links and GBIF IDs for sound recordings
    multimedia = pd.read_csv(f"{folder}/GBIF_0021944-250525065834625/multimedia.txt", sep="\t")
    Sounds = multimedia.loc[multimedia["type"] == "Sound"]
    links = Sounds[["gbifID", "identifier", "description"]]
    links = links.dropna()
    links.to_csv(f"{folder}/processed/links.csv", sep="\t", index=False)

    # import occurrence file with species identifiers and create data frame
    occurrence_file = pd.read_csv(f"{folder}/GBIF_0021944-250525065834625/occurrence.txt", sep="\t", on_bad_lines='skip')

    # remove empty columns
    non_null_columns = [col for col in occurrence_file.columns if occurrence_file.loc[:, col].notna().any()]
    occurrences = occurrence_file[non_null_columns]
    sIDs = pd.DataFrame(occurrences)
    sIDs = sIDs[["gbifID", "species", "catalogNumber"]]
    sIDs = sIDs.dropna()

    species_counts = sIDs['species'].value_counts()
    top_11_species = species_counts.head(11)
    # Get the list of names of the top 11 species
    top_species_names = top_11_species.index.tolist()

    # Filter the sIDs DataFrame
    # sIDs['species'].isin(top_species_names) creates a boolean Series:
    # True if the species in that row is one of the top 11, False otherwise.
    sIDs = sIDs[sIDs['species'].isin(top_species_names)]
    print(len(sIDs))

    sIDs.to_csv(f"{folder}/processed/species_identifiers.csv", sep="\t", index=False)
else:
    links = pd.read_csv(f"{folder}/processed/links.csv", sep="\t")
    sIDs = pd.read_csv(f"{folder}/processed/species_identifiers.csv", sep="\t")

print(f"Total observations: {len(sIDs)}")

# create dictionary for each GBIF ID and species name
species_dict = dict([(i, x) for i, x in zip(sIDs.gbifID, sIDs.species)])
catalog_dict = dict([(i, x) for i, x in zip(sIDs.gbifID, sIDs.catalogNumber)])

# get number of wav files
n_recordings = 0
allowed_formats = ["wav", "mp3"]
file_types = defaultdict(int)
for index, row in links.iterrows():
    URL = row["identifier"]
    file_type = URL.rsplit(".", 1)[-1].rsplit("?", 1)[0].lower()        # get the file type

    if row["gbifID"] in species_dict.keys() and file_type in allowed_formats:
        n_recordings += 1
        file_types[file_type] += 1

print(f"Filtered recordings: {n_recordings}")

downloaded = glob.glob(f"{folder}*.wav")        # create list of already downloaded files
counter = len(downloaded)                       # init downloaded file counter
