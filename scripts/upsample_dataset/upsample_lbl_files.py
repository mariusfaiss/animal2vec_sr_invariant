import h5py
import numpy as np
from pathlib import Path
from pydub.utils import mediainfo
from tqdm import tqdm
import shutil
from typing import Union
import argparse

# this script updates the hdf5 label files by "upsampling" the start and end frame values for the labels
# this is likely unnecessary for the sample-based classification task, but was done for due diligence

# CONFIGURATION
TARGET_SAMPLE_RATE = 48000
DATASETS_TO_MODIFY = ["start_frame_lbl", "end_frame_lbl"]
AUDIO_EXTENSIONS = [".wav", ".mp3", ".MP3", ".WAV", ".Mp3"]


def get_audio_sample_rate_from_ffprobe(audio_file_path: Path) -> Union[int, None]:
    """Extracts the sample rate from an audio file using ffprobe (via pydub)."""
    try:
        info = mediainfo(str(audio_file_path))
        return int(info['sample_rate'])
    except Exception as e:
        tqdm.write(f"[Warning] Could not get sample rate for {audio_file_path} using ffprobe. Error: {e}")
        return None


def find_corresponding_audio(h5_path: Path, base_lbl_dir: Path, base_audio_dir: Path) -> Union[Path, None]:
    """Finds the audio file corresponding to an H5 file by checking multiple extensions."""
    relative_path = h5_path.relative_to(base_lbl_dir)
    audio_stem_path = base_audio_dir / relative_path.with_suffix('')

    for extension in AUDIO_EXTENSIONS:
        potential_path = audio_stem_path.with_suffix(extension)
        if potential_path.exists():
            return potential_path
    return None


def process_h5_file(h5_path: Path, base_lbl_dir: Path, output_dir: Path, base_audio_dir: Path) -> bool:
    """
    Processes a single HDF5 file.
    Returns True on success, False on failure.
    """

    # determine corresponding audio file path
    relative_path = h5_path.relative_to(base_lbl_dir)
    output_h5_path = output_dir / relative_path

    if output_h5_path.exists():
        return True  # skip already processed files

    audio_path = find_corresponding_audio(h5_path, base_lbl_dir, base_audio_dir)
    if not audio_path:
        tqdm.write(f"[Warning] No corresponding audio file found for {h5_path}")
        return False

    # get sample rate and calculate ratio
    sample_rate = None

    try:
        sr_str = audio_path.parent.name
        if sr_str.lower().endswith('hz'):
            sample_rate = int(sr_str[:-2])
    except (ValueError, IndexError):
        sample_rate = None  # ensure it's None if parsing fails

        # if parsing from path failed, fall back to the slow ffprobe method
    if sample_rate is None:
        sample_rate = get_audio_sample_rate_from_ffprobe(audio_path)

    if not sample_rate:  # handles both None and 0
        tqdm.write(f"[Warning] Skipping {h5_path} due to invalid sample rate.")
        return False

    ratio = TARGET_SAMPLE_RATE / sample_rate

    # create output path and ensure parent directory exists
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    # read, modify, and write to the new HDF5 file
    try:
        # first, copy the entire file to the destination.
        shutil.copy2(h5_path, output_h5_path)

        # open the new file to modify it.
        with h5py.File(output_h5_path, 'r+') as f:
            for dset_name in DATASETS_TO_MODIFY:
                if dset_name not in f:
                    tqdm.write(f"[Info] Dataset '{dset_name}' not in {h5_path}, skipping.")
                    continue

                source_dset = f[dset_name]
                original_data = source_dset[...]
                original_dtype = source_dset.dtype

                calculated_data = original_data * ratio

                if np.issubdtype(original_dtype, np.floating):
                    modified_data = calculated_data.astype(original_dtype)
                else:
                    modified_data = np.round(calculated_data).astype(original_dtype)

                # delete the old dataset and create a new one with the same name.
                del f[dset_name]
                f.create_dataset(dset_name, data=modified_data)

        return True

    except Exception as e:
        tqdm.write(f"[ERROR] Failed to process file {h5_path}. Error: {e}")
        # clean up partially created file on error
        if output_h5_path.exists():
            output_h5_path.unlink()
        return False


def main():
    """Main function to find and process all HDF5 files."""

    parser = argparse.ArgumentParser(
        description="Update the start and end audio sample values of labels for upsampled audio data."
    )
    parser.add_argument(
        "base_folder",
        nargs='?',
        default="/home/jupyter-mfaiss/Datasets/SR_MeerKAT_XCbirds_10s",
        type=str,
        help="The base directory of the original dataset containing the wav and lbl subdirectories."
    )
    parser.add_argument(
        "out_folder",
        nargs='?',
        default="/home/jupyter-mfaiss/Datasets/48_MeerKAT_XCbirds_10s",
        type=str,
        help="The base directory of the upsampled dataset."
    )
    args = parser.parse_args()

    base_lbl_dir = Path(args.base_folder) / "lbl"
    base_audio_dir = Path(args.base_folder) / "wav"
    output_dir = Path(args.out_folder) / "lbl"

    if not base_lbl_dir.is_dir() or not base_audio_dir.is_dir():
        print(f"Error: Source directories not found.")
        print(f"HDF5 dir (exists: {base_lbl_dir.is_dir()}): {base_lbl_dir}")
        print(f"Audio dir (exists: {base_audio_dir.is_dir()}): {base_audio_dir}")
        return

    print(f"HDF5 source:      {base_lbl_dir.resolve()}")
    print(f"Audio source:     {base_audio_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print("-" * 35)

    h5_files = list(base_lbl_dir.rglob("*.h5"))
    if not h5_files:
        print("No .h5 files found to process. Exiting.")
        return

    success_count = 0
    for h5_file in tqdm(h5_files, desc="Processing files", unit="file"):
        if process_h5_file(h5_file, base_lbl_dir, output_dir, base_audio_dir):
            success_count += 1

    print("\n--- Processing Complete ---")
    total_files = len(h5_files)
    failed_count = total_files - success_count
    print(f"Successfully processed: {success_count}/{total_files} files.")
    if failed_count > 0:
        print(f"Failed to process:      {failed_count}/{total_files} files. (Check warnings above)")
    print(f"Modified files are saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
