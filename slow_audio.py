"""Time-stretch WAV(s) slower without pitch shift.

Engines (least to most echo on typical TTS):
  - pedalboard   — Rubber Band (default). Usually cleanest for voice.
  - wsola        — audiotsm overlap-add; can sound grainy or doubled.
  - ffmpeg-rubberband — system FFmpeg + rubberband filter (quality varies by build).

Batch mode: numbered 0001.wav … in --src folder.
Single-file mode: --input one.wav [--output out.wav]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def trim_trailing_quiet(
    y: np.ndarray,
    sr: int,
    db_below_peak: float,
    pad_ms: float = 30.0,
) -> np.ndarray:
    """Shorten file by trimming very quiet tail (stretched noise floor can sound like reverb)."""
    peak = float(np.max(np.abs(y)))
    if peak < 1e-8:
        return y
    thresh = peak * (10.0 ** (-db_below_peak / 20.0))
    loud = np.where(np.any(np.abs(y) > thresh, axis=1))[0]
    if loud.size == 0:
        return y
    end = min(y.shape[0], loud[-1] + 1 + int(sr * pad_ms / 1000.0))
    return y[:end]


def stretch_pedalboard(y_ch_first: np.ndarray, sr: int, stretch_factor: float) -> np.ndarray:
    """Rubber Band via pedalboard. y shape (channels, samples) float32."""
    import pedalboard as pb

    x = np.asarray(y_ch_first, dtype=np.float32)
    return pb.time_stretch(
        x,
        float(sr),
        stretch_factor=float(stretch_factor),
        pitch_shift_in_semitones=0.0,
        high_quality=True,
        transient_mode="crisp",
        transient_detector="compound",
        retain_phase_continuity=True,
        use_long_fft_window=False,
        use_time_domain_smoothing=False,
        preserve_formants=True,
    )


def stretch_wsola(y_ch_first: np.ndarray, channels: int, speed: float) -> np.ndarray:
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter

    x = np.asarray(y_ch_first, dtype=np.float64)
    reader = ArrayReader(x)
    writer = ArrayWriter(channels)
    wsola(channels, speed=float(speed)).run(reader, writer)
    return np.asarray(writer.data, dtype=np.float32)


def stretch_ffmpeg_rubberband(src: Path, dst: Path, tempo: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not on PATH (install FFmpeg for this engine).")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-af",
        f"rubberband=tempo={tempo}",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def default_slow_output_path(root: Path, input_path: Path, rate: float) -> Path:
    """e.g. Lafufu Raw Voice.wav -> <root>/Lafufu Raw Voice_slow_r0p75.wav"""
    tag = str(rate).replace(".", "p")
    return root / f"{input_path.stem}_slow_r{tag}.wav"


def process_one(
    src: Path,
    dst: Path,
    *,
    rate: float,
    engine: str,
    trim_tail_db: float,
) -> tuple[float, float]:
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)
    if trim_tail_db and trim_tail_db > 0:
        y = trim_trailing_quiet(y, int(sr), float(trim_tail_db))

    dst.parent.mkdir(parents=True, exist_ok=True)

    if engine == "ffmpeg-rubberband":
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tpath = Path(tmp.name)
        try:
            sf.write(str(tpath), y, int(sr), subtype="PCM_16")
            stretch_ffmpeg_rubberband(tpath, dst, float(rate))
        finally:
            tpath.unlink(missing_ok=True)
        y_b4, sr0 = sf.read(str(src), dtype="float32", always_2d=False)
        y_af, sr1 = sf.read(str(dst), dtype="float32", always_2d=False)
        dur_in = (y_b4.shape[0] if y_b4.ndim > 1 else len(y_b4)) / float(sr0)
        dur_out = (y_af.shape[0] if y_af.ndim > 1 else len(y_af)) / float(sr1)
        return dur_in, dur_out

    if engine == "pedalboard":
        y_in = y.T.copy()
        y_out = stretch_pedalboard(y_in, int(sr), float(rate))
        y_write = y_out.T if y_out.shape[0] > 1 else y_out[0]
        sf.write(str(dst), y_write, int(sr), subtype="PCM_16")
        dur_in = y.shape[0] / float(sr)
        dur_out = (y_write.shape[0] if y_write.ndim > 1 else len(y_write)) / float(sr)
        return dur_in, dur_out

    y_in = y.T.copy()
    ch = y_in.shape[0]
    y_out = stretch_wsola(y_in, ch, float(rate))
    y_write = y_out.T if ch > 1 else y_out[0]
    sf.write(str(dst), y_write, int(sr), subtype="PCM_16")
    dur_in = y.shape[0] / float(sr)
    dur_out = (y_write.shape[0] if y_write.ndim > 1 else len(y_write)) / float(sr)
    return dur_in, dur_out


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Slow WAV(s) without changing pitch (Rubber Band default).")
    ap.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Single input WAV (use with --output or default name in project root). Ignores batch --src/--count.",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Single output WAV path. Default with -i: <project_root>/<stem>_slow_r<rate>.wav",
    )
    ap.add_argument("--src", type=Path, default=root / "out_training", help="Input folder (batch mode)")
    ap.add_argument(
        "--dst",
        type=Path,
        default=root / "out_training(slowed)",
        help="Output folder (batch mode only)",
    )
    ap.add_argument("--count", type=int, default=5, help="Batch: how many files from the start (0001, 0002, ...)")
    ap.add_argument(
        "--rate",
        type=float,
        default=0.95,
        help="Playback speed factor (<1 = slower). 0.75 = slower; closer to 1 = less artifacts.",
    )
    ap.add_argument(
        "--engine",
        choices=("pedalboard", "wsola", "ffmpeg-rubberband"),
        default="pedalboard",
        help="pedalboard = Rubber Band (recommended). wsola = lighter dep, more grain. ffmpeg = external binary.",
    )
    ap.add_argument(
        "--trim-tail-db",
        type=float,
        default=0.0,
        metavar="DB",
        help="If >0, trim tail quieter than this many dB below peak (e.g. 45). Reduces stretched noise tail.",
    )
    args = ap.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if not (0.5 <= args.rate <= 1.5):
        raise SystemExit("--rate keep between 0.5 and 1.5 for slowdown/speedup around 1.0.")

    if args.engine == "pedalboard":
        try:
            import pedalboard  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "Engine 'pedalboard' needs: pip install pedalboard\n"
                "Or use --engine wsola (audiotsm, more artifacts)."
            ) from e

    # --- Single file ---
    if args.input is not None:
        src_path = Path(args.input).resolve()
        if not src_path.is_file():
            raise SystemExit(f"Not a file: {src_path}")
        out_path = Path(args.output).resolve() if args.output is not None else default_slow_output_path(root, src_path, args.rate)
        print(f"Input:  {src_path}")
        print(f"Output: {out_path}")
        dur_in, dur_out = process_one(
            src_path,
            out_path,
            rate=float(args.rate),
            engine=args.engine,
            trim_tail_db=float(args.trim_tail_db),
        )
        print(f"{src_path.name}: {dur_in:.2f}s -> {dur_out:.2f}s (rate={args.rate}, {args.engine})")
        print("Done.")
        return

    # --- Batch folder ---
    args.dst.mkdir(parents=True, exist_ok=True)
    print(f"Input dir:  {args.src}")
    print(f"Output dir: {args.dst}")

    for n in range(1, args.count + 1):
        name = f"{n:04d}.wav"
        src = args.src / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing: {src}")
        dst = args.dst / name
        dur_in, dur_out = process_one(
            src,
            dst,
            rate=float(args.rate),
            engine=args.engine,
            trim_tail_db=float(args.trim_tail_db),
        )
        print(f"{name}: {dur_in:.2f}s -> {dur_out:.2f}s (rate={args.rate}, {args.engine}) -> {dst}")

    print("Done.")


if __name__ == "__main__":
    main()
