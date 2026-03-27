import os
import subprocess
import argparse
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
from tqdm import tqdm
import torch
import torchaudio.functional as F
import soundfile as sf

TARGET_SAMPLE_RATE = "48000"
SUPPORTED_EXTENSIONS = {".mp3", ".wav"}


def find_audio_files(input_dir: str) -> List[str]:
    """Finds all supported audio files in the input directory and subdirectories."""
    audio_files = []
    print(f"Searching for audio files in '{input_dir}'...")
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                audio_files.append(os.path.join(root, file))
    print(f"Found {len(audio_files)} audio files.")
    return audio_files


def check_ffmpeg():
    """Checks if FFmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: FFmpeg not found.")
        print("Please install FFmpeg and ensure it's in your system's PATH.")
        return False


def process_file(task: Tuple[str, str]) -> Tuple[str, bool, str]:
    input_path, output_path = task

    # force output to be a .wav file
    output_path = os.path.splitext(output_path)[0] + ".wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # load the audio EXACTLY as animal2vecs dataloader does
        wav, curr_sample_rate = sf.read(input_path, dtype="float32")
        feats = torch.from_numpy(wav).float()

        # soundfile returns audio as [frames, channels] (or 1D for mono)
        # PyTorch's resample function expects [channels, frames]
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)  # Convert to [1, frames]
        else:
            feats = feats.t()  # Convert to [channels, frames]

        # resample if necessary using PyTorch's native resampler
        if curr_sample_rate != TARGET_SAMPLE_RATE:
            feats = F.resample(
                feats,
                orig_freq=curr_sample_rate,
                new_freq=int(TARGET_SAMPLE_RATE)
            )

        # convert back to numpy for soundfile to write
        # reshape back to [frames, channels] or 1D mono
        feats = feats.t().numpy() if feats.shape[0] > 1 else feats.squeeze(0).numpy()

        # save as uncompressed WAV
        # we use subtype="FLOAT" to preserve the exact float32 values.
        # this prevents 16-bit integer quantization differences!
        sf.write(
            file=output_path,
            data=feats,
            samplerate=int(TARGET_SAMPLE_RATE),
            subtype="FLOAT"
        )

        return (input_path, True, "Success")

    except Exception as e:
        return (input_path, False, f"Soundfile/Resampling error: {str(e)}")

def main():
    """Main function to orchestrate the resampling process."""
    parser = argparse.ArgumentParser(
        description=f"Resample all MP3 and WAV files in a directory to {int(TARGET_SAMPLE_RATE) / 1000}kHz."
    )
    parser.add_argument("--input_dir", help="The root directory containing audio files.",
                        default="/home/jupyter-mfaiss/Datasets/SR_MeerKAT_XCbirds_10s/wav")
    parser.add_argument("--output_dir", help="The directory where resampled files will be saved.",
                        default="/home/jupyter-mfaiss/Datasets/48_MeerKAT_XCbirds_10s/wav")
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=cpu_count(),
        help=f"Number of parallel processes to use (default: all available cores, {cpu_count()})"
    )

    args = parser.parse_args()

    if not check_ffmpeg():
        return

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    if args.input_dir == args.output_dir:
        print("Error: Input and output directories cannot be the same.")
        return

    # find all files to process
    all_files = find_audio_files(args.input_dir)
    if not all_files:
        print("No audio files to process. Exiting.")
        return

    # create the list of tasks (input_path, output_path)
    tasks = []
    for input_path in all_files:
        relative_path = os.path.relpath(input_path, args.input_dir)
        output_path = os.path.join(args.output_dir, relative_path)
        tasks.append((input_path, output_path))

    # use a multiprocessing Pool to process files in parallel
    print(f"\nStarting resampling with {args.processes} processes...")
    failed_files = []

    with Pool(processes=args.processes) as pool:
        # tqdm shows a progress bar
        with tqdm(total=len(tasks), desc="Resampling Files") as pbar:
            for result in pool.imap_unordered(process_file, tasks):
                input_file, success, message = result
                if not success:
                    failed_files.append((input_file, message))
                pbar.update(1)

    # report results
    print("\n--------------------")
    print("      SUMMARY       ")
    print("--------------------")
    print(f"Total files processed: {len(tasks)}")
    print(f"Successfully converted: {len(tasks) - len(failed_files)}")
    print(f"Failed conversions: {len(failed_files)}")

    if failed_files:
        print("\nThe following files failed to convert:")
        for file, reason in failed_files:
            print(f" - {file}\n   Reason: {reason}")

    print("\nResampling complete.")
    print(f"Resampled files are located in: '{os.path.abspath(args.output_dir)}'")


if __name__ == "__main__":
    main()
