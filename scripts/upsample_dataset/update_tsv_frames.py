import argparse
import csv
from pathlib import Path
import sys
import librosa
from tqdm import tqdm

# this script updates the number of audio samples in the manifest files after upsampling the original files

# the sample rate that the dataset was upsampled to
TARGET_SAMPLE_RATE = 48000


def get_frame_count(audio_path: Path) -> int:
    """
    Calculates the number of audio frames in a file at the target sample rate.

    This works by getting the audio's duration in seconds and multiplying
    by the target sample rate. This is more robust than re-sampling in memory.

    Args:
        audio_path: The full path to the audio file.

    Returns:
        The total number of frames the audio would have at TARGET_SAMPLE_RATE.
    """
    duration_seconds = librosa.get_duration(path=audio_path)
    # round() is used to avoid floating point inaccuracies before converting to int
    return int(round(duration_seconds * TARGET_SAMPLE_RATE))


def process_tsv_file(tsv_path: Path, out_dir_path: Path):
    """
    Processes a single TSV config file, updating the sample counts, base path, and extensions.
    Writes the updated TSV to out_dir_path.

    Args:
        tsv_path: The path to the original .tsv file to process.
        out_dir_path: The directory where the updated .tsv file should be saved.
    """
    print(f"\n--- Processing file: {tsv_path.name} ---")

    # define paths for the output file and a safe temporary file
    final_output_path = out_dir_path / tsv_path.name
    temp_output_path = out_dir_path / f"{tsv_path.name}.tmp"

    try:
        with open(tsv_path, 'r', encoding='utf-8') as infile:
            # read all lines to get a total for the progress bar
            lines = infile.readlines()
            if not lines:
                print(f"Warning: File {tsv_path.name} is empty. Skipping.")
                return

            # the first line is the base directory for audio files
            base_audio_dir = Path(lines[0].strip())
            print(f"Original base audio directory: {base_audio_dir}")

            # check if the original base directory exists to read original files
            if not base_audio_dir.is_dir():
                print(f"Error: Base audio directory '{base_audio_dir}' does not exist. Cannot process this file.")
                print("Please check the first line of your tsv file.")
                return

            # modify the base directory name to reflect the new sample rate (e.g., SR_ -> 48_)
            target_sr_prefix = f"{TARGET_SAMPLE_RATE // 1000}_"
            new_base_dir_name = base_audio_dir.name.replace("SR_", target_sr_prefix)
            new_base_audio_dir = base_audio_dir.parent / new_base_dir_name

            # the rest of the lines are the data rows
            data_rows = lines[1:]

            with open(temp_output_path, 'w', encoding='utf-8', newline='') as outfile:
                # write the updated header line to the new file (.as_posix() forces forward slashes)
                outfile.write(f"{new_base_audio_dir.as_posix()}\n")
                writer = csv.writer(outfile, delimiter='\t')

                pbar = tqdm(data_rows, desc="Updating samples", unit="files")
                for line in pbar:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        relative_path, old_samples_str = line.split('\t')
                    except ValueError:
                        print(f"Warning: Skipping malformed line: {line}")
                        continue

                    # reference the original audio path to get the exact duration
                    full_audio_path = base_audio_dir / relative_path

                    # force .wav extension for the upsampled dataset manifest
                    new_relative_path = Path(relative_path).with_suffix('.wav').as_posix()

                    try:
                        if not full_audio_path.exists():
                            raise FileNotFoundError

                        # get the new frame count based on duration and target rate
                        new_samples = get_frame_count(full_audio_path)
                        writer.writerow([new_relative_path, new_samples])

                    except FileNotFoundError:
                        print(f"\nWarning: Audio file not found: {full_audio_path}")
                        print("         Writing old sample count back to the config.")
                        writer.writerow([new_relative_path, old_samples_str])
                    except Exception as e:
                        print(f"\nWarning: Could not process audio file: {full_audio_path}")
                        print(f"         Error: {e}. Writing old sample count back.")
                        writer.writerow([new_relative_path, old_samples_str])

        # if the temp file was written successfully, rename it to the final output file
        temp_output_path.rename(final_output_path)
        print(f"Successfully created updated manifest at: {final_output_path}")

    except FileNotFoundError:
        print(f"Error: Config file not found: {tsv_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {tsv_path.name}: {e}")
        # clean up the temporary file if an error occurred
        if temp_output_path.exists():
            temp_output_path.unlink()


def main():
    """
    Main function to parse arguments and find all .tsv files to process.
    """
    parser = argparse.ArgumentParser(
        description="Update sample counts in TSV config files for audio resampled to 48kHz."
    )
    parser.add_argument(
        "config_dir",
        nargs='?',
        default="/home/jupyter-mfaiss/Datasets/SR_MeerKAT_XCbirds_10s",
        type=str,
        help="The directory containing the original .tsv config files to update."
    )
    parser.add_argument(
        "out_dir",
        nargs='?',
        default="/home/jupyter-mfaiss/Datasets/48_MeerKAT_XCbirds_10s",
        type=str,
        help="The manifests directory of the upsampled dataset."
    )
    args = parser.parse_args()

    config_dir_path = Path(args.config_dir)
    out_dir_path = Path(args.out_dir)

    if not config_dir_path.is_dir():
        print(f"Error: The input config directory does not exist: {config_dir_path}")
        sys.exit(1)

    # make sure output directory exists, create parents if needed
    out_dir_path.mkdir(parents=True, exist_ok=True)

    tsv_files = sorted(list(config_dir_path.glob("*.tsv")))

    if not tsv_files:
        print(f"No .tsv files found in directory: {config_dir_path}")
        sys.exit(0)

    print(f"Found {len(tsv_files)} .tsv files to process.")
    for tsv_file in tsv_files:
        process_tsv_file(tsv_file, out_dir_path)

    print("\nAll files processed. Script finished.")


if __name__ == "__main__":
    main()