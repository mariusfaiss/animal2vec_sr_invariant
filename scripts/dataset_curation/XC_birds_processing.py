import os
import glob
import ntpath
import librosa
from pydub import AudioSegment
from pydub.utils import mediainfo
from pydub.exceptions import CouldntDecodeError
from collections import defaultdict
from collections import Counter
import hashlib
import pickle
import numpy as np
import shutil
import sys

audio_folder = "/Volumes/Extreme_SSD/XC_birds/download"
# audio_folder = "/Volumes/Extreme_SSD/XC_birds/conversion_test"

#############################################################################
checksum = False    # DONE
calculate = False   # calculate checksums else read in pickle file?

#############################################################################
# downsampling and trimming all files
downsampling = False

max_sr = 48000
max_len = 120
default_bitrate = "320k"
downsampled_folder = "/Volumes/Extreme_SSD/XC_birds/converted"

#############################################################################
# splitting into 10s second chunks
splitting = True

split_folder = downsampled_folder
out_folder = "/Volumes/Extreme_SSD/XC_birds/10s_splits"
chunk_length = 10  # Desired chunk length in seconds
min_length = 3  # Minimum length for a chunk to be saved (in seconds)

#######################################################################################################################
# script to check for duplicate audio files in multiple datasets
if checksum:
    if calculate:
        downloaded = []
        for filename in glob.glob(os.path.join(f"{audio_folder}/*")):
            file = ntpath.basename(filename)
            downloaded.append(file)

        total_files = len(downloaded)


        def sha256_checksum(fname):     # f.read(4096) was used originally
            hash_func = hashlib.sha256()
            # 1. Load the audio with librosa
            #    sr=None:       Preserves the original sample rate.
            #    mono=False:    Preserves all channels.
            #    dtype=np.float32: Ensures data is 32-bit float, matching torchaudio.
            waveform, sr = librosa.load(fname, sr=None, mono=False, dtype=np.float32)

            # 2. Librosa returns a 1D array for mono audio. We must reshape it to
            #    (1, n_samples) to match torchaudio's 2D output for all files.
            if waveform.ndim == 1:
                waveform = waveform.reshape(1, -1)

            hash_func.update(waveform.tobytes())
            return hash_func.hexdigest()


        def add_values_in_dict(sample_dict, key, value):
            if key not in sample_dict:
                sample_dict[key] = list()
            sample_dict[key].append(value)
            return sample_dict


        list_of_sha256_checksums = defaultdict(int)
        list_of_files = defaultdict(str)
        counter = 0

        for filename in glob.glob(os.path.join(f"{audio_folder}/*")):
            counter += 1
            # file = ntpath.basename(filename)
            file = filename
            print(f"{counter}/{total_files}    {file}")
            sha256 = sha256_checksum(filename)
            list_of_sha256_checksums[sha256] += 1
            list_of_files = add_values_in_dict(list_of_files, sha256, file)

        with open('/Volumes/Extreme_SSD/XC_birds/checksums/list_of_sha256_checksums.pkl', 'wb') as fp:
            pickle.dump(list_of_sha256_checksums, fp)

        with open('/Volumes/Extreme_SSD/XC_birds/checksums/list_of_files.pkl', 'wb') as fp:
            pickle.dump(list_of_files, fp)

    with open('/Volumes/Extreme_SSD/XC_birds/checksums/list_of_sha256_checksums.pkl', 'rb') as fp:
        list_of_sha256_checksums = pickle.load(fp)

    with open('/Volumes/Extreme_SSD/XC_birds/checksums/list_of_files.pkl', 'rb') as fp:
        list_of_files = pickle.load(fp)

    summary = Counter(list_of_sha256_checksums.values())
    print(summary)

    test_counter = 0
    for index in list_of_sha256_checksums:
        if list_of_sha256_checksums[index] > 1:
            test_counter += 1
            # print(f"{test_counter}: {list_of_files[index]}")


    def all_same(items):
        return all(x == items[0] for x in items)


    removed = []
    total_removed = 0
    for key, value in list_of_sha256_checksums.items():
        species_IDs = []                                        # init species IDs for entry
        counter = 1
        if value > 1:
            # print(f"{list_of_files[key]}")
            for file in list_of_files[key]:
                file = ntpath.basename(file)
                genus = file.split("_")[0]
                species = file.split("_")[1]
                species_ID = genus + species
                species_IDs.append(species_ID)

            if not all_same(species_IDs):                       # remove all files from different species
                for file in list_of_files[key]:
                    shutil.copy(file, "/Volumes/Extreme_SSD/XC_birds/checksums/")
                    os.remove(f"{file}")
                    total_removed += 1
                    removed.append(file)
                    # print(f"Remove confusion {file}\n")

            if all_same(species_IDs):                           # remove all files from
                for file in list_of_files[key]:
                    if counter > 1:
                        shutil.copy(file, "/Volumes/Extreme_SSD/XC_birds/checksums/")
                        os.remove(f"{file}")
                        total_removed += 1
                        removed.append(file)
                        # print(f"Remove duplicate {counter} {file}\n")
                    counter += 1

    print(f"Total removed: {total_removed}")

    species_dict = defaultdict(int)

    for file in removed:
        file = ntpath.basename(file)
        genus = file.split("_")[0]
        species = file.split("_")[1]
        species_ID = genus + " " + species
        species_dict[species_ID] += 1

    print(species_dict)

#######################################################################################################################
if downsampling:

    # --- Script Start ---
    print("Starting linear audio processing script using pydub...")

    # Create the output directory if it doesn't exist to prevent errors
    if not os.path.exists(downsampled_folder):
        os.makedirs(downsampled_folder)
        print(f"Created output directory: {downsampled_folder}")

    # Get a list of all files in the input directory
    # We wrap this in a try-except block in case the input directory doesn't exist
    try:
        files_to_process = os.listdir(audio_folder)
    except FileNotFoundError:
        print(f"ERROR: Input directory '{audio_folder}' not found.")
        print("Please create it and place your audio files inside.")
        exit()  # Stop the script if there's nothing to process

    processed_files = os.listdir(downsampled_folder)
    n_processed_files = len(processed_files)

    # Loop through each filename found in the input directory
    for filename in files_to_process:
        # Construct the full path for the input file and the desired output file
        input_path = os.path.join(audio_folder, filename)
        output_path = os.path.join(downsampled_folder, filename)

        # skip already processed files
        if filename in processed_files:
            continue

        # Check if the item is a file and if it has a supported extension
        if os.path.isfile(input_path) and filename.lower().endswith(('.wav', '.mp3')):
            n_processed_files += 1
            print(f"\n--- Processing: {n_processed_files}/{len(files_to_process)} {filename} ---")

            # Use a try-except block to handle corrupted files or other errors gracefully
            try:
                # --- Step 1a: Get Original File Info (Metadata and Bitrate) ---
                # This logic now applies to both WAV and MP3
                info = mediainfo(input_path)
                tags_to_preserve = info.get('TAG', {})

                # Remove the 'title' tag from the metadata dictionary before saving.
                # This prevents it from overriding the filename in file explorers like macOS Finder.
                # We use .pop() with a default value of None to avoid errors if the tag doesn't exist.
                # We check for both common casings ('title' and 'TITLE') to be robust.
                if tags_to_preserve:
                    title_removed = tags_to_preserve.pop('title', None)
                    title_removed_upper = tags_to_preserve.pop('TITLE', None)

                # This is specific to MP3 files
                bitrate_to_use = default_bitrate
                is_mp3 = filename.lower().endswith('.mp3')
                if is_mp3:
                    # Try to get the exact bitrate from the media info
                    original_bitrate_str = info.get('bit_rate')
                    if original_bitrate_str:
                        # Convert '192000' into '192k' for pydub's export function
                        original_bitrate_k = int(int(original_bitrate_str) / 1000)
                        bitrate_to_use = f"{original_bitrate_k}k"
                        print(f"Original bitrate detected: {bitrate_to_use}. This will be used for the new file.")

                # Step 1b: Load the audio file.
                # pydub automatically detects the format from the file content/extension.
                audio = AudioSegment.from_file(input_path)

                # Step 2: Process the sample rate.
                current_samplerate = audio.frame_rate
                if current_samplerate < 16000:
                    print(f"sample rate too low!: {current_samplerate}")
                if current_samplerate > max_sr:
                    # If the sample rate is too high, downsample it to the target rate.
                    audio = audio.set_frame_rate(max_sr)
                    print(f"Downsampled from {current_samplerate} Hz to {max_sr} Hz.")

                # Step 3: Convert to mono if the file is stereo.
                # The .channels attribute tells us the number of audio channels.
                if audio.channels > 1:
                    # If there is more than 1 channel, convert it to mono.
                    audio = audio.set_channels(1)

                # Step 4: Process the duration.
                # pydub works with time in milliseconds, so convert seconds to ms.
                max_duration_ms = max_len * 1000
                duration_ms = len(audio)  # len() on an AudioSegment returns its duration in ms.

                if duration_ms > max_duration_ms:
                    # The file is longer than 2 minutes, so we need to trim it.
                    # Calculate the middle point of the audio.
                    middle_point_ms = duration_ms / 2

                    # Calculate where our 2-minute slice should start and end.
                    start_trim_ms = middle_point_ms - (max_duration_ms / 2)
                    end_trim_ms = middle_point_ms + (max_duration_ms / 2)

                    # Slice the audio. pydub uses Python's list slicing syntax.
                    audio = audio[start_trim_ms:end_trim_ms]
                    print(f"Trimmed file from {duration_ms / 1000:.2f}s to {len(audio) / 1000:.2f}s (from the middle).")

                # Step 5: Save the processed audio file.
                # We determine the format from the original filename's extension.
                file_format = filename.split('.')[-1].lower()

                temp_output_path = output_path + ".part"

                try:
                    # The 'tags' parameter works for both formats.
                    # The 'bitrate' parameter is only used for MP3s.
                    if file_format == 'mp3':
                        print(f"Saving MP3 with bitrate {bitrate_to_use} and original metadata...")
                        audio.export(
                            temp_output_path,
                            format="mp3",
                            bitrate=bitrate_to_use,
                            tags=tags_to_preserve
                        )
                    else:  # For WAV files
                        print("Saving WAV file with original metadata...")
                        audio.export(
                            temp_output_path,
                            format="wav",
                            tags=tags_to_preserve
                        )
                    # If the download was successful, rename the temporary file
                    os.rename(temp_output_path, output_path)

                except KeyboardInterrupt:
                    print("\nDownload interrupted by user.")
                    # The finally block will handle cleanup
                    sys.exit(1)  # Exit the script

                finally:
                    # This block will run NO MATTER WHAT.
                    # If the temporary file still exists, it means the download
                    # was not completed successfully, so we clean it up.
                    if os.path.exists(temp_output_path):
                        print(f"Cleaning up partial file: {temp_output_path}")
                        os.remove(temp_output_path)

            except CouldntDecodeError:
                print(f"ERROR: Could not decode {filename}. It might be corrupted or an unsupported format.")
            except Exception as e:
                # Catch any other unexpected errors during processing.
                print(f"An unexpected error occurred while processing {filename}: {e}")
        else:
            # This skips any subdirectories or non-audio files in the input folder.
            if os.path.isfile(input_path):
                print(f"\n--- Skipping: {filename} (not a .wav or .mp3 file) ---")

    print("\n\nProcessing complete!")

#######################################################################################################################
if splitting:

    # Convert chunk length to milliseconds (pydub works with ms)
    chunk_length_ms = chunk_length * 1000

    # Create the output directory if it doesn't exist
    os.makedirs(out_folder, exist_ok=True)

    # Get a list of all files in the input folder
    files_in_folder = os.listdir(split_folder)
    files_out_folder = os.listdir(out_folder)
    unique_files_out_folder = [file.split("_")[2] for file in files_out_folder]
    unique_files_out_folder = list(dict.fromkeys(unique_files_out_folder))  # remove duplicate species
    total_files = len(files_in_folder)
    n_split = len(unique_files_out_folder)

    # Loop through each file in the input folder
    for filename in files_in_folder:
        # Process only .mp3 and .wav files
        if filename.lower().endswith(('.mp3', '.wav')):

            # check if file has been chunked already
            check_chunk = filename.split(".")[0] + "_00_10." + filename.split(".")[1]
            if filename in files_out_folder or check_chunk in files_out_folder:
                continue
            if filename[0] == ".":
                continue

            n_split += 1
            print(f"Processing {n_split}/{total_files} {filename}")

            # Construct full file path
            input_path = os.path.join(split_folder, filename)

            try:
                # --- Step 1a: Get Original File Info (Metadata and Bitrate) ---
                # This logic now applies to both WAV and MP3
                info = mediainfo(input_path)
                tags_to_preserve = info.get('TAG', {})

                # This is specific to MP3 files
                bitrate_to_use = default_bitrate
                is_mp3 = filename.lower().endswith('.mp3')
                if is_mp3:
                    # Try to get the exact bitrate from the media info
                    original_bitrate_str = info.get('bit_rate')
                    if original_bitrate_str:
                        # Convert '192000' into '192k' for pydub's export function
                        original_bitrate_k = int(int(original_bitrate_str) / 1000)
                        bitrate_to_use = f"{original_bitrate_k}k"

                # Load the audio file
                audio = AudioSegment.from_file(input_path)
                duration_ms = len(audio)

                # --- CASE 1: File is shorter than or equal to the chunk length ---
                if duration_ms <= chunk_length_ms:
                    print(f"'{filename}' is shorter than {chunk_length}s. Copying as is.")

                    # Construct the destination path
                    output_path = os.path.join(out_folder, filename)
                    temp_output_path = output_path + ".part"

                    try:
                        # Copy the file using shutil.copy2 to preserve metadata
                        shutil.copy2(input_path, temp_output_path)

                        # If the download was successful, rename the temporary file
                        os.rename(temp_output_path, output_path)
                    except KeyboardInterrupt:
                        print("\nDownload interrupted by user.")
                        # The finally-block will handle cleanup
                        sys.exit(1)  # Exit the script

                    finally:
                        # This block will run NO MATTER WHAT.
                        # If the temporary file still exists, it means the download
                        # was not completed successfully, so we clean it up.
                        if os.path.exists(temp_output_path):
                            print(f"Cleaning up partial file: {temp_output_path}")
                            os.remove(temp_output_path)

                # --- CASE 2: File is longer than the chunk length ---
                else:
                    # Get file name without extension and the extension itself
                    base_name, extension = os.path.splitext(filename)

                    # tracking chunks so they can be removed if the process is interrupted
                    file_list = []

                    try:
                        # Iterate through the audio and create chunks
                        for i in range(0, duration_ms, chunk_length_ms):
                            start_ms = i
                            end_ms = i + chunk_length_ms

                            # Ensure the last chunk does not go past the audio length
                            if end_ms > duration_ms:
                                end_ms = duration_ms

                            # Extract the chunk
                            chunk = audio[start_ms:end_ms]

                            # NEW: Check if the created chunk is too short before saving
                            if len(chunk) < min_length*1000:
                                continue

                            # Create the new filename
                            start_sec = start_ms // 1000
                            end_sec = end_ms // 1000
                            chunk_filename = f"{base_name}_{start_sec:02d}_{end_sec:02d}{extension}"

                            # Construct the full output path for the chunk
                            chunk_output_path = os.path.join(out_folder, chunk_filename)
                            file_list.append(chunk_output_path)
                            temp_output_path = chunk_output_path + ".part"
                            file_list.append(temp_output_path)

                            print(f"  - Creating chunk: {chunk_filename}")

                            try:
                                # The 'tags' parameter works for both formats.
                                # The 'bitrate' parameter is only used for MP3s.
                                if extension == '.mp3':
                                    chunk.export(
                                        temp_output_path,
                                        format="mp3",
                                        bitrate=bitrate_to_use,
                                        tags=tags_to_preserve
                                    )
                                else:  # For WAV files
                                    chunk.export(
                                        temp_output_path,
                                        format="wav",
                                        tags=tags_to_preserve
                                    )
                                # If the download was successful, rename the temporary file
                                os.rename(temp_output_path, chunk_output_path)

                            except KeyboardInterrupt:
                                print("\nDownload interrupted by user.")
                                # The finally-block will handle cleanup
                                sys.exit(1)  # Exit the script

                            finally:
                                # This block will run NO MATTER WHAT.
                                # If the temporary file still exists, it means the download
                                # was not completed successfully, so we clean it up.
                                if os.path.exists(temp_output_path):
                                    print(f"Cleaning up partial file: {temp_output_path}")
                                    os.remove(temp_output_path)

                        # clear file list if all chunks have been successfully written, so they are not removed
                        file_list = []

                    except KeyboardInterrupt:
                        print("\nDownload interrupted by user.")
                        # The finally-block will handle cleanup
                        sys.exit(1)  # Exit the script

                    finally:
                        # This block will run NO MATTER WHAT.
                        # If the temporary file still exists, it means the download
                        # was not completed successfully, so we clean it up.
                        for file_name in file_list:

                            if os.path.exists(file_name):
                                print(f"Cleaning up incomplete splits: {file_name}")
                                os.remove(file_name)

            except Exception as e:
                print(f"Could not process file '{filename}'. Reason: {e}")

    print("\nProcessing complete.")
