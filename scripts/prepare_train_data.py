import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
import typer
from tqdm.auto import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from enum import Enum
from typer import Option, Argument
import numpy as np

f_max = 8000


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


def save_spectrogram_plot(
    spectrogram: torch.Tensor,
    output_path: Path,
    sample_rate: int,
    waveform: torch.Tensor,
):
    # take first channel if stereo
    spec = spectrogram[0].numpy() if spectrogram.dim() > 2 else spectrogram.numpy()

    def mel_to_freq(mel_values):
        return 700 * (10 ** (mel_values / 2595) - 1)

    # Compute Mel frequencies
    mel_bins = spec.shape[1]
    mel_values = np.linspace(0, 2595 * np.log10(1 + f_max / 700), mel_bins)
    mel_frequencies = mel_to_freq(mel_values)

    plt.figure(figsize=(10, 4))
    plt.imshow(
        spec.squeeze(),
        aspect="auto",
        origin="lower",
        extent=[
            0,
            waveform.size(1) / sample_rate,
            mel_frequencies[0],
            mel_frequencies[-1],
        ],
        cmap="magma",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.savefig(output_path.as_posix(), dpi=300, bbox_inches="tight")
    plt.close()


def convert_to_spectrogram(
    input_path: Path,
    output_path: Path,
    mode: SaveMode,
    target_length: int,
):
    input_path = input_path.absolute()
    output_path = output_path.absolute()

    if output_path.exists():
        return

    waveform, sample_rate = torchaudio.load(input_path.as_posix())

    mel_spect = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=80,
        f_min=20,
        f_max=f_max,
    )(waveform)

    mel_spect_db = T.AmplitudeToDB()(mel_spect)

    if mode == SaveMode.image:
        save_spectrogram_plot(mel_spect_db, output_path, sample_rate, waveform)
    elif mode == SaveMode.tensor:
        torch.save(mel_spect_db, output_path.as_posix())
    else:
        raise ValueError(f"Unknown save mode {mode}")


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
        convert_to_spectrogram(
            args.input_file,
            output_path,
            mode,
            args.target_length,
        )


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
    target_length: int = Option(8192, help="Target length of the spectrogram"),
):
    if not input_dir.is_dir():
        raise typer.BadParameter(f"Input directory {input_dir} does not exist")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for f in input_dir.glob("**/*"):
        try:
            torchaudio.info(f.as_posix())
            files.append(f.resolve())
        except Exception:
            continue

    args_list = [
        Args(
            file,
            output_dir,
            num_workers,
            save_modes,
            target_length,
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
