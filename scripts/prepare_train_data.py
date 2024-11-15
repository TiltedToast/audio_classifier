import typer
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass
import torchaudio
import torchaudio.transforms as T
import torch


@dataclass
class Args:
    input_file: Path
    output_dir: Path
    num_workers: int


spectrogram_transform = T.Spectrogram(
    n_fft = 1024,
    win_length = None,
    hop_length = 512,
)


def convert_to_spectrogram(input_path: Path, output_path: Path):
    input_path = input_path.absolute()
    output_path = output_path.absolute()

    if output_path.exists():
        return

    waveform, _ = torchaudio.load(input_path.as_posix())

    spectrogram = spectrogram_transform(waveform)

    torch.save(spectrogram, output_path.as_posix())


def worker(args: Args):
    output_path = args.output_dir / args.input_file.with_suffix(
        args.input_file.suffix + ".pt",
    ).relative_to("/")

    if output_path.exists():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {args.input_file} into {output_path}")
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

    with mp.Pool(num_workers) as pool:
        pool.map(worker, [Args(file, output_dir, num_workers) for file in files])


if __name__ == "__main__":
    typer.run(main)
