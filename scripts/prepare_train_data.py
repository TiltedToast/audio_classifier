import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import typer
from torch import Tensor
from tqdm.auto import tqdm
from typer import Argument, Option


class SaveMode(str, Enum):
    image = "image"
    tensor = "tensor"


@dataclass
class Args:
    input_file: Path
    output_dir: Path
    num_workers: int
    save_modes: list[SaveMode]
    target_length: int
    sample_rate: int
    bit_depth: int
    n_fft: int
    hop_length: int
    n_mels: int
    f_min: int
    f_max: int


def save_spectrogram_plot(
    spectrogram: Tensor,
    output_path: Path,
    sample_rate: int,
    waveform: Tensor,
    n_mels: int,
):
    """Save a mel spectrogram plot.

    Args:
        spectrogram: Tensor of shape (Mels, Time)
        output_path: Path to save the plot to.
        sample_rate (int): The sample rate of the audio.
        waveform (Tensor): Tensor of shape (channels, time).
        n_mels (int): Number of mel bands.
    """

    plt.figure(figsize=(10, 4))
    plt.imshow(
        spectrogram.squeeze(),
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, waveform.size(1) / sample_rate, 0, n_mels],
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Frequency")
    plt.savefig(output_path.as_posix(), dpi=300, bbox_inches="tight")
    plt.close()


def trim_silence(
    waveform: Tensor,
    sample_rate: int,
    threshold: float,
    min_duration: float,
):
    """
    Trim trailing silence based on energy threshold.

    Args:
        waveform: Tensor of shape (channels, time).
        sample_rate: Sampling rate of the waveform.
        threshold: Energy threshold below which is considered silence.
        min_duration: Minimum duration of non-silence in seconds.

    Returns:
        Trimmed waveform.
    """

    energy = waveform.pow(2).mean(dim=0)
    non_silent_indices = (energy > threshold).nonzero(as_tuple=True)[0]

    if len(non_silent_indices) == 0:
        return waveform[:, :0]

    start_index = non_silent_indices[0]
    end_index = non_silent_indices[-1]

    min_samples = int(min_duration * sample_rate)
    end_index = max(end_index, start_index + min_samples - 1)

    return waveform[:, start_index : end_index + 1]


def postprocess_spectrogram(mel_spect: Tensor, args: Args, mode: SaveMode):
    """Post-process the mel spectrogram."""
    mel_spect_db = T.AmplitudeToDB()(mel_spect)

    # Average channels if 2+ channels
    mel_spect_db = mel_spect_db.mean(dim=0)

    # Pad/truncate spectrogram if necessary
    if mode == SaveMode.tensor:
        if mel_spect_db.size(1) < args.target_length:
            pad_amount = args.target_length - mel_spect_db.size(1)
            mel_spect_db = F.pad(mel_spect_db, (0, pad_amount))
        elif mel_spect_db.size(1) > args.target_length:
            mel_spect_db = mel_spect_db[:, : args.target_length]

    return mel_spect_db


def preprocess_waveform(
    waveform: Tensor,
    original_sample_rate: int,
    args: Args,
):
    waveform -= waveform.mean()

    if original_sample_rate != args.sample_rate:
        resampler = T.Resample(orig_freq=original_sample_rate, new_freq=args.sample_rate)
        waveform = resampler(waveform)

    waveform = trim_silence(
        waveform,
        args.sample_rate,
        threshold=1e-4,
        min_duration=0.2,
    )

    return quantize_waveform(waveform, args.bit_depth)


def save_spectrogram(
    spectrogram: Tensor,
    waveform: Tensor,
    args: Args,
    mode: SaveMode,
    output_path: Path,
):
    """Save the spectrogram to the specified output path."""

    if mode == SaveMode.image:
        save_spectrogram_plot(
            spectrogram,
            output_path,
            args.sample_rate,
            waveform,
            args.n_mels,
        )
    elif mode == SaveMode.tensor:
        torch.save(spectrogram, output_path.as_posix())
    else:
        raise ValueError(f"Unknown save mode {mode}")


def convert_to_spectrogram(
    output_path: Path,
    mode: SaveMode,
    args: Args,
):
    input_path = args.input_file.absolute()
    output_path = output_path.absolute()

    if output_path.exists():
        return

    waveform, original_sample_rate = torchaudio.load(input_path.as_posix())

    waveform = preprocess_waveform(waveform, original_sample_rate, args)

    mel_spect = T.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
    )(waveform)

    mel_spect_db = postprocess_spectrogram(mel_spect, args, mode)

    save_spectrogram(mel_spect_db, waveform, args, mode, output_path)


def quantize_waveform(waveform: Tensor, bit_depth: int):
    """Quantize the waveform to the specified bit depth."""
    max_val = float(2 ** (bit_depth - 1) - 1)
    waveform = waveform.clamp_(-1.0, 1.0)
    waveform = torch.round_(waveform * max_val) / max_val
    return waveform


def worker(args: Args):
    mapping = {
        SaveMode.image: ".png",
        SaveMode.tensor: ".pt",
    }

    output_paths: list[tuple[Path, SaveMode]] = []

    for mode in args.save_modes:
        output_paths.append(
            (
                args.output_dir
                / args.input_file.with_suffix(
                    args.input_file.suffix + mapping[mode],
                ).relative_to("/"),
                mode,
            )
        )

    for output_path, mode in output_paths:
        if output_path.exists():
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        convert_to_spectrogram(output_path, mode, args)


def valid_file(file: Path):
    try:
        torchaudio.info(file.as_posix())
        return True
    except Exception:
        return False


def main(
    input_dir: Path = Argument(..., help="Input directory to recursively search for audio files"),
    output_dir: Path = Option(Path("./data/train"), help="Output directory"),
    num_workers: int = Option(mp.cpu_count(), help="Number of files to process in parallel"),
    save_modes: list[SaveMode] = Option(
        [SaveMode.image],
        "--save-mode",
        help="Mode(s) to save the spectrograms",
        case_sensitive=False,
    ),
    target_length: int = Option(
        8192,
        help="Target length of the spectrogram (only relevant for tensor mode)",
    ),
    sample_rate: int = Option(16000, help="Sample rate of the audio (resampling if necessary)"),
    bit_depth: int = Option(16, help="Bit depth of the audio (quantization if necessary)"),
    n_fft: int = Option(2048, help="FFT size"),
    hop_length: int = Option(512, help="Hop length"),
    n_mels: int = Option(128, help="Number of mel bands"),
    f_min: int = Option(0, help="Minimum frequency"),
    f_max: int = Option(8000, help="Maximum frequency"),
):
    if not input_dir.is_dir():
        raise typer.BadParameter(f"Input directory {input_dir} does not exist")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("**/*"))
    with mp.Pool(num_workers) as pool:
        files = [file for file, is_valid in zip(files, pool.map(valid_file, files)) if is_valid]

    args_list = [
        Args(
            file,
            output_dir,
            num_workers,
            save_modes,
            target_length,
            sample_rate,
            bit_depth,
            n_fft,
            hop_length,
            n_mels,
            f_min,
            f_max,
        )
        for file in files
    ]

    with mp.Pool(num_workers) as pool, tqdm(
        total=len(args_list),
        desc="Processing files",
    ) as pbar:
        for _ in pool.imap_unordered(worker, args_list):
            pbar.update()


if __name__ == "__main__":
    typer.run(main)
