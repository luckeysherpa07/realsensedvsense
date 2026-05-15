from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "dataset_processed" / "check_mailbox_split" / "check_mailbox_day_rgb.mp4"
DEFAULT_AUDIO = PROJECT_ROOT / "dataset_processed" / "check_mailbox_split" / "check_mailbox_day.m4a"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset_processed" / "check_mailbox_split" / "check_mailbox_day_rgb_with_audio.mp4"


def run_command(cmd: list[str], dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(printable)
    if dry_run:
        return None
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def probe_duration(path: Path, ffprobe_bin: str) -> float:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    duration = payload.get("format", {}).get("duration")
    if duration is None:
        raise ValueError(f"Could not read duration from: {path}")
    return float(duration)


def verify_output(output: Path, ffprobe_bin: str) -> None:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name",
        "-of",
        "json",
        str(output),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_video or not has_audio:
        raise RuntimeError(f"Verification failed for {output}: missing video or audio stream")


def mux(video: Path, audio: Path, output: Path, ffmpeg_bin: str, dry_run: bool) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ]
    run_command(cmd, dry_run=dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mux .m4a audio into an .mp4 video using ffmpeg.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Input MP4 video path.")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Input M4A audio path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output MP4 path.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting output if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running ffmpeg.")
    parser.add_argument("--no-verify", action="store_true", help="Skip ffprobe stream verification.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name/path.")
    parser.add_argument("--ffprobe-bin", default="ffprobe", help="ffprobe executable name/path.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    video = args.video.resolve()
    audio = args.audio.resolve()
    output = args.output.resolve()

    if not video.exists():
        print(f"Error: video file not found: {video}", file=sys.stderr)
        return 1
    if not audio.exists():
        print(f"Error: audio file not found: {audio}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.overwrite and not args.dry_run:
        print(f"Error: output already exists: {output}", file=sys.stderr)
        print("Use --overwrite to replace it.", file=sys.stderr)
        return 1

    if output.exists() and args.dry_run and not args.overwrite:
        print(f"Notice: output already exists and would be overwritten: {output}")

    try:
        video_duration = probe_duration(video, args.ffprobe_bin)
        audio_duration = probe_duration(audio, args.ffprobe_bin)
        print(f"Video duration: {video_duration:.3f}s")
        print(f"Audio duration: {audio_duration:.3f}s")

        mux(video, audio, output, ffmpeg_bin=args.ffmpeg_bin, dry_run=args.dry_run)

        if not args.dry_run and not args.no_verify:
            verify_output(output, args.ffprobe_bin)
            out_duration = probe_duration(output, args.ffprobe_bin)
            print(f"Output duration: {out_duration:.3f}s")
            print("Verification passed: output has both video and audio streams.")
    except subprocess.CalledProcessError as exc:
        print(exc.stderr or str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Dry-run completed. Planned output: {output}")
    else:
        print(f"Done: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

