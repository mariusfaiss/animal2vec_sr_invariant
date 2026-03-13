# This script downloads the metadata for a given set of species from xeno-canto
# Then it can be used to filter the data and calculate statistics

import requests
import json
import pandas as pd
from collections import defaultdict
import math
import glob
import os
import ntpath
import time
import sys
import requests
import ast

pd.set_option("display.max_columns", None)

folder = "/Volumes/Extreme_SSD/XC_birds/metadata_files"

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1
download_list = False           # downloads a list of all species currently on XC

# which species to download
common_species = ["Parus major", "Turdus merula", "Fringilla coelebs", "Phylloscopus collybita",
                  "Erithacus rubecula", "Turdus philomelos", "Sylvia atricapilla", "Phylloscopus trochilus",
                  "Loxia curvirostra", "Troglodytes troglodytes", "Cyanistes caeruleus"]

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2
process_list = True             # create stats about downloaded list

remove_no_derivatives = True    # remove files with no derivatives licenses

remove_long = False             # remove long files
max_seconds = 120               # how long can the files be at max?

remove_sr = True                # remove low sample rate files
min_sr = 16000

allowed_formats = ["wav", "mp3"]

# # # # # # # # # # # # # # # # # # # # # # # # # # #
# 3
download_files = False

# # # # # # # # # # # # # # # # # # # # # # # # # # #


def hours_of_audio(file_time):
    # for converting time formats

    if isinstance(file_time, str):
        time_format = file_time.count(":")

        if time_format == 2:
            h, m, s = file_time.split(':')
            audio_hours = int(h) * 3600 + int(m) * 60 + int(s)

        if time_format == 1:
            m, s = file_time.split(':')
            audio_hours = int(m) * 60 + int(s)

    elif isinstance(file_time, int):
        m, s = divmod(file_time, 60)
        h, m = divmod(m, 60)
        audio_hours = f'{h:d}:{m:02d}:{s:02d}'

    else:
        raise ValueError("file time is not valid")

    return audio_hours


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# this downloads the metadata from xc
if download_list:

    # specify metadata columns of interest and create d
    columns = ["id", "gen", "sp", "smp", "length", "also", "lic", "rec", "url", "file", "file-name", "lat", "lng",
               "time", "date", "loc", "alt"]
    recordings_df = pd.DataFrame(columns=columns)

    for species in common_species:
        print(f"Species: {species}")
        url = f"https://xeno-canto.org/api/2/recordings?query=sp:\"{species}\""

        search = requests.get(url)                      # get data from url
        xc_text = json.loads(search.text)               # convert to json

        num_pages = xc_text["numPages"]                 # get number of pages
        num_recordings = xc_text["numRecordings"]       # get number of recordings

        # go through all pages
        page = 1
        while page <= num_pages:
            print(f"Page {page} of {num_pages}")
            response = requests.get(f"{url}&page={page}")                   # request with page specification
            xc_text = json.loads(response.text)                             # convert to json

            # get only recording dictionaries
            recordings = xc_text.pop("recordings")
            recordings = {item['id']: item for item in recordings}

            # create dataframe with relevant data and concatenate of main dataframe
            dataframe = pd.DataFrame.from_dict(recordings, orient="index")
            dataframe = dataframe[columns]
            recordings_df = pd.concat([recordings_df, dataframe])

            page += 1   # update page counter

    # save for further processing
    recordings_df.to_csv(f"{folder}/XC_common_11_birds.csv", index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# remove recordings based on certain criteria set at the top of the script. Also calculate stats
if process_list:

    # initializing dictionaries to track dataset tatistics
    number_of_recordings = defaultdict(int)
    length_dict = defaultdict(int)
    time_dict = defaultdict(str)
    sr_dict = defaultdict(int)
    longer_files = defaultdict(int)
    with_background_species = defaultdict(int)
    n_no_derivatives = 0
    series_ID_dict = defaultdict(int)
    file_type_dict = defaultdict(int)

    # separator for the species label list
    separator = ", "

    # read downloaded metadata
    All_recordings = pd.read_csv(f"{folder}/XC_common_11_birds.csv")

    # initialize main dataframe
    recordings_df = pd.DataFrame(columns=["file_name", "species_labels", "observation_id", "main_species_name",
                                          "license", "contributor", "observation", "file", "sample_rate", "format",
                                          "series_ID", "og_file_name"])

    # iterate over all rows in the metadata
    total_rows = len(All_recordings)
    for index, row in All_recordings.iterrows():
        print(f"{index}/{total_rows}")

        # read variables from metadata
        species_name = row["gen"] + "_" + row["sp"]                         # species name without space
        file_type = row["file-name"].split(".")[-1]
        file_length = row["length"]

        ########################################################################################
        # FILE REMOVAL
        ########################################################################################

        # remove no derivatives in case we want to process and publish the files
        if remove_no_derivatives:
            if "nd" in row["lic"]:
                n_no_derivatives += 1
                continue

        # remove empty file_types because they might be weird files
        if file_type.lower() not in allowed_formats:
            continue

        # remove nan sample rates because they might be weird files
        sr = row["smp"]
        if math.isnan(sr):
            continue

        # remove files with low sample rates
        if remove_sr:
            if sr < min_sr:
                continue

        # remove files longer than the max length
        length = hours_of_audio(file_length)

        if remove_long:
            if length > max_seconds:
                continue

        # check multiples from serial recordings and remove
        # these recordings likely originate from longer recordings or were made close in time at the same location,
        # likely containing the same individuals vocalizing
        location_ID = str(row["lat"]) + "_" + str(row["lng"])
        if location_ID == "nan_nan":
            location_ID = row["rec"]
        series_ID = species_name + "_" + location_ID + "_" + row["date"] + "_" + row["time"][:2]

        if series_ID not in series_ID_dict:
            series_ID_dict[series_ID] += 1
        else:
            series_ID_dict[series_ID] += 1
            continue

        ########################################################################################
        # FILE REMOVAL
        ########################################################################################

        # get more metadata
        file_name = species_name + "_XC" + str(row["id"]) + "." + file_type
        background_species = ast.literal_eval(row["also"])
        observation_url = "https:" + row["url"]
        file_link = row["file"]
        license_link = "https:" + row["lic"]

        # make the species labels
        species_labels = [bird.replace(' ', '_') for bird in background_species if bird in common_species]
        species_labels.insert(0, species_name)
        species_labels = list(dict.fromkeys(species_labels))    # remove duplicate species

        # updating dicts for stats
        file_type_dict["*." + file_type] += 1
        sr_dict[str(sr).split(".")[0]] += 1

        if not remove_long and length > max_seconds:
            longer_files[species_name] += 1
            length_dict[species_name] += max_seconds    # simulate trimming long files, if they are not removed
        else:
            length_dict[species_name] += length

        if species_labels != [species_name]:
            with_background_species[species_name] += 1

        # add to species dict
        number_of_recordings[species_name] += 1

        # add to dataframe
        recordings_df.loc[index, "file_name"] = file_name
        recordings_df.loc[index, "species_labels"] = separator.join(species_labels)
        recordings_df.loc[index, "main_species_name"] = species_name
        recordings_df.loc[index, "observation_id"] = "XC" + observation_url.split("/")[-1]
        recordings_df.loc[index, "contributor"] = row["rec"]
        recordings_df.at[index, "observation"] = observation_url           # link to observation
        recordings_df.at[index, "file"] = file_link                # link to file
        recordings_df.at[index, "license"] = license_link
        recordings_df.loc[index, "series_ID"] = series_ID
        recordings_df.loc[index, "sample_rate"] = sr
        recordings_df.loc[index, "format"] = file_type
        recordings_df.loc[index, "og_file_name"] = row["file-name"]

    # save to csv
    recordings_df.to_csv(f"{folder}/XC_filtered_bird_recordings.csv", index=False)

    # print all sample rates and number of files
    sr_dict = dict(sorted(sr_dict.items(), key=lambda item: item[1], reverse=True))
    for sample_rate, number in sr_dict.items():
        print(f"{sample_rate}: {number}")

    # track serial recordings
    series_ID_dict = {k: v for k, v in series_ID_dict.items() if v > 1}
    series_ID_dict = dict(sorted(series_ID_dict.items(), key=lambda item: item[1]))
    number_of_replicates = sum(series_ID_dict.values()) - len(series_ID_dict)
    # print(f"Number of series replicates: {number_of_replicates}")

    # calculate times from seconds
    for key in length_dict:
        length = length_dict[key]
        time = hours_of_audio(length)
        time_dict[key] += time

    # make dataframes and sort
    numbers_df = pd.DataFrame.from_dict(number_of_recordings, orient="index", columns=["n_recordings"])
    numbers_df = numbers_df.T
    length_df = pd.DataFrame.from_dict(length_dict, orient="index", columns=["length"])
    length_df = length_df.T
    time_df = pd.DataFrame.from_dict(time_dict, orient="index", columns=["time"])
    time_df = time_df.T

    # make main dataframe
    dataframe = pd.concat([numbers_df, length_df, time_df])
    dataframe = dataframe.T
    dataframe = dataframe.reset_index()
    dataframe.rename(columns={"index": "species_name"}, inplace=True)

    # add long files and background species counts
    dataframe[f"{max_seconds}min_files"] = 0
    dataframe["bg_species"] = 0

    for index, row in dataframe.iterrows():
        species = row["species_name"]

        if species in longer_files.keys():
            dataframe.at[index, f"{max_seconds}min_files"] = longer_files[species]

        if species in with_background_species.keys():
            dataframe.at[index, "bg_species"] = with_background_species[species]

    dataframe = dataframe.sort_values(["n_recordings", "length"], ascending=False)

    n_files = dataframe["n_recordings"].sum()
    long_files = dataframe[f"{max_seconds}min_files"].sum()
    total_length = hours_of_audio(dataframe["length"].sum())[:-3]

    print(f"{len(dataframe)} species, {n_files:,} files, {long_files:,} longer than "
          f"{max_seconds} sec, {total_length}h long")

    dataframe.to_csv(f"{folder}/XC_bird_species_summary.csv", index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# download the recordings
if download_files:

    to_download = pd.read_csv(f"{folder}/XC_filtered_bird_recordings.csv")
    download_folder = "/Volumes/Extreme_SSD/XC_birds/download"
    n_recordings = len(to_download)
    downloaded = []

    for filename in glob.glob(os.path.join(f"{download_folder}/*")):
        file = ntpath.basename(filename)
        downloaded.append(file)

    counter = len(downloaded)
    print(f"Downloaded {counter} files already")

    for index, row in to_download.iterrows():
        file_name = row["species_name"] + "_XC" + row["observation"].split("/")[-1] + "." + row["file_name"].split(".")[-1]
        final_filepath = f"{download_folder}/{file_name}"

        if file_name not in downloaded:
            counter += 1
            URL = row["file"]
            temp_filepath = f"{download_folder}/temp_{file_name}"
            print(f"{counter}/{n_recordings}    {file_name}")

            try:
                # Use stream=True to avoid loading the whole file into memory at once
                with requests.get(URL, stream=True, timeout=10) as r:
                    # Check if the request was successful
                    r.raise_for_status()

                    # Open the temporary file for writing in binary mode
                    with open(temp_filepath, 'wb') as f:
                        # Download the file in chunks
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                # If the download was successful, rename the temporary file
                os.rename(temp_filepath, final_filepath)

            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")

            except KeyboardInterrupt:
                print("\nDownload interrupted by user.")
                # The finally block will handle cleanup
                sys.exit(1)  # Exit the script

            finally:
                # This block will run NO MATTER WHAT.
                # If the temporary file still exists, it means the download
                # was not completed successfully, so we clean it up.
                if os.path.exists(temp_filepath):
                    print(f"Cleaning up partial file: {temp_filepath}")
                    os.remove(temp_filepath)

            # request.urlretrieve(URL, full_file_name)
            time.sleep(1)

            # download until x files
            # if counter >= 30000:
            #     break

    print()
