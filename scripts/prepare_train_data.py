import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
import typer
from tqdm.auto import tqdm
import torch.nn.functional as F

@dataclass
class Args:
    input_file: Path
    output_dir: Path
    num_workers: int


def convert_to_spectrogram(input_path: Path, output_path: Path):
    input_path = input_path.absolute()
    output_path = output_path.absolute()

    if output_path.exists():
        return

    waveform, sample_rate = torchaudio.load(input_path.as_posix())

    spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=80,
        f_min=20,
        f_max=8000,
    )(waveform)

    spectrogram = torch.log(spectrogram + 1e-9)

    # TODO: Standardise spectrogram size for classification tasks
    # target_length = 1024
    # if spectrogram.size(-1) != target_length:
    #     spectrogram = F.interpolate(
    #         spectrogram.unsqueeze(0),
    #         size=(80, target_length),
    #         mode="bilinear",
    #     ).squeeze(0)

    torch.save(spectrogram, output_path.as_posix())


def worker(args: Args):
    output_path = args.output_dir / args.input_file.with_suffix(
        args.input_file.suffix + ".pt",
    ).relative_to("/")

    if output_path.exists():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_to_spectrogram(args.input_file, output_path)


def main(
    input_dir: Path,
    output_dir: Path = Path("./data/train"),
    num_workers: int = mp.cpu_count(),
):
    if not input_dir.is_dir():
        raise typer.BadParameter(f"Input directory {input_dir} does not exist")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.glob("**/*") if f.suffix in [".wav", ".mp3", ".flac"]]

    args_list = [Args(file, output_dir, num_workers) for file in files]

    with mp.Pool(num_workers) as pool, tqdm(
        total=len(args_list),
        desc="Processing files",
    ) as pbar:
        for _ in pool.imap_unordered(worker, args_list):
            pbar.update()


if __name__ == "__main__":
    typer.run(main)
