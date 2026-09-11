from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from navmem3d.schemas import CaptureSequence


def make_video(
    sequence_path: str | Path,
    output_path: str | Path,
    fps: float = 10.0,
    max_duration_s: float = 30.0,
    max_size_mb: float = 100.0,
) -> dict[str, object]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to create a video")
    if fps <= 0 or max_duration_s <= 0 or max_size_mb <= 0:
        raise ValueError("fps, max_duration_s and max_size_mb must be positive")

    source = Path(sequence_path).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sequence = CaptureSequence.load(source)
    max_frames = max(1, int(fps * max_duration_s))
    frames = sequence.frames[:max_frames]
    concat_path = output.with_suffix(output.suffix + ".ffconcat")
    duration = 1.0 / fps
    lines = ["ffconcat version 1.0"]
    for frame in frames:
        image = (source.parent / frame.rgb_uri).resolve()
        if not image.exists():
            raise FileNotFoundError(image)
        lines.append(f"file '{_ffconcat_quote(str(image))}'")
        lines.append(f"duration {duration:.9f}")
    lines.append(f"file '{_ffconcat_quote(str((source.parent / frames[-1].rgb_uri).resolve()))}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(concat_path), "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    size_mb = output.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise RuntimeError(
            f"video is {size_mb:.2f} MB, exceeding the {max_size_mb:.2f} MB limit"
        )
    return {
        "output": str(output),
        "frame_count": len(frames),
        "duration_s": len(frames) / fps,
        "size_mb": round(size_mb, 3),
    }


def _ffconcat_quote(value: str) -> str:
    return value.replace("'", "'\\''")

