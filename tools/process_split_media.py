from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


VIDEO_SUFFIXES = ("rgb", "depth", "ir", "event")

# Edit these values directly if you prefer not to use command line arguments.
USE_INLINE_CONFIG = True
DATASET_ROOT = Path("../dataset")
OUTPUT_ROOT = Path("dataset_processed")
VIDEO_MODE = "reencode"  # "copy" is faster but only keyframe-accurate.
DRY_RUN = True
INLINE_JOBS = [
    {
        "split_dir": "wash_hands_split",
        "prefix": "wash_hands_day",
        "video_start": "00:00:38.000",
        "video_end": "",
        "audio_start": "00:00:05.000",
        "audio_end": "00:02:09.000",
    },
    {
        "split_dir": "wash_hands_split",
        "prefix": "wash_hands_night2",
        "video_start": "00:00:42.000",
        "video_end": "",
        "audio_start": "00:00:01.000",
        "audio_end": "00:02:20.000",
    },
]


@dataclass
class Job:
    split_dir: str
    prefix: str
    video_start: Optional[float] = None
    video_end: Optional[float] = None
    audio_start: Optional[float] = None
    audio_end: Optional[float] = None
    audio_offset: Optional[float] = None

    @property
    def duration(self) -> float:
        if self.video_start is None or self.video_end is None:
            raise ValueError("job duration is not available until start/end are resolved")
        return self.video_end - self.video_start


def parse_timecode(value: str) -> float:
    value = value.strip()
    if not value:
        raise ValueError("empty time value")
    if ":" not in value:
        return float(value)
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid timecode: {value}")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_optional_timecode(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return parse_timecode(text)


def format_timecode(value: float) -> str:
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value - hours * 3600 - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def run_ffmpeg(args: list[str], dry_run: bool) -> None:
    cmd = ["ffmpeg", "-y", *args]
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def load_jobs(manifest_path: Path) -> list[Job]:
    jobs: list[Job] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"split_dir", "prefix", "trim_start", "trim_end", "audio_offset"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if not row["split_dir"].strip():
                continue
            job = Job(
                split_dir=row["split_dir"].strip(),
                prefix=row["prefix"].strip(),
                video_start=parse_optional_timecode(row["trim_start"]),
                video_end=parse_optional_timecode(row["trim_end"]),
                audio_offset=parse_optional_timecode(row["audio_offset"]),
            )
            if (
                job.video_start is not None
                and job.video_end is not None
                and job.video_end <= job.video_start
            ):
                raise ValueError(
                    f"{job.split_dir}/{job.prefix}: trim_end must be greater than trim_start"
                )
            jobs.append(job)
    return jobs


def load_inline_jobs() -> list[Job]:
    jobs: list[Job] = []
    for row in INLINE_JOBS:
        job = Job(
            split_dir=row["split_dir"].strip(),
            prefix=row["prefix"].strip(),
            video_start=parse_optional_timecode(row.get("video_start")),
            video_end=parse_optional_timecode(row.get("video_end")),
            audio_start=parse_optional_timecode(row.get("audio_start")),
            audio_end=parse_optional_timecode(row.get("audio_end")),
            audio_offset=parse_optional_timecode(row.get("audio_offset")),
        )
        if (
            job.video_start is not None
            and job.video_end is not None
            and job.video_end <= job.video_start
        ):
            raise ValueError(
                f"{job.split_dir}/{job.prefix}: video_end must be greater than video_start"
            )
        if (job.audio_start is None) != (job.audio_end is None):
            raise ValueError(
                f"{job.split_dir}/{job.prefix}: audio_start and audio_end must be provided together"
            )
        if job.audio_start is not None and job.audio_end is not None and job.audio_end <= job.audio_start:
            raise ValueError(
                f"{job.split_dir}/{job.prefix}: audio_end must be greater than audio_start"
            )
        jobs.append(job)
    return jobs


def trim_video(
    input_path: Path,
    output_path: Path,
    trim_start: float,
    trim_end: float,
    mode: str,
    dry_run: bool,
) -> None:
    if mode == "copy":
        args = [
            "-ss",
            format_timecode(trim_start),
            "-to",
            format_timecode(trim_end),
            "-i",
            str(input_path),
            "-c",
            "copy",
            str(output_path),
        ]
    else:
        args = [
            "-i",
            str(input_path),
            "-ss",
            format_timecode(trim_start),
            "-to",
            format_timecode(trim_end),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-an",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    run_ffmpeg(args, dry_run=dry_run)


def probe_duration(input_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(input_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    duration = payload.get("format", {}).get("duration")
    if duration is None:
        raise ValueError(f"could not probe duration for {input_path}")
    return float(duration)


def trim_audio(
    input_path: Path,
    output_path: Path,
    trim_start: float,
    duration: float,
    audio_offset: Optional[float],
    audio_start: Optional[float],
    audio_end: Optional[float],
    dry_run: bool,
) -> None:
    if audio_start is not None and audio_end is not None:
        args = [
            "-ss",
            format_timecode(audio_start),
            "-to",
            format_timecode(audio_end),
            "-i",
            str(input_path),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
        run_ffmpeg(args, dry_run=dry_run)
        return

    source_start = trim_start + (audio_offset or 0.0)
    leading_silence = 0.0
    if source_start < 0:
        leading_silence = -source_start
        source_start = 0.0

    remaining = max(duration - leading_silence, 0.0)
    delay_ms = round(leading_silence * 1000)
    filters: list[str] = []
    if delay_ms > 0:
        filters.append(f"adelay={delay_ms}:all=1")
    filters.append("apad")
    filters.append(f"atrim=duration={duration:.3f}")
    audio_filter = ",".join(filters)

    if remaining <= 0:
        args = [
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            format_timecode(duration),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
        run_ffmpeg(args, dry_run=dry_run)
        return

    args = [
        "-ss",
        format_timecode(source_start),
        "-t",
        format_timecode(remaining),
        "-i",
        str(input_path),
        "-af",
        audio_filter,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]
    run_ffmpeg(args, dry_run=dry_run)


def process_job(
    dataset_root: Path,
    output_root: Path,
    job: Job,
    video_mode: str,
    dry_run: bool,
) -> None:
    input_dir = dataset_root / job.split_dir
    output_dir = output_root / job.split_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"missing split directory: {input_dir}")

    print(f"\nProcessing {job.split_dir} / {job.prefix}")
    reference_video = input_dir / f"{job.prefix}_rgb.mp4"
    if not reference_video.exists():
        for suffix in VIDEO_SUFFIXES:
            candidate = input_dir / f"{job.prefix}_{suffix}.mp4"
            if candidate.exists():
                reference_video = candidate
                break
    if not reference_video.exists():
        raise FileNotFoundError(f"missing reference video for {job.split_dir}/{job.prefix}")

    video_start = job.video_start if job.video_start is not None else 0.0
    video_end = job.video_end if job.video_end is not None else probe_duration(reference_video)
    if video_end <= video_start:
        raise ValueError(f"{job.split_dir}/{job.prefix}: resolved video_end must be greater than video_start")

    for suffix in VIDEO_SUFFIXES:
        input_path = input_dir / f"{job.prefix}_{suffix}.mp4"
        if not input_path.exists():
            print(f"  skip missing video: {input_path.name}")
            continue
        output_path = output_dir / input_path.name
        trim_video(
            input_path=input_path,
            output_path=output_path,
            trim_start=video_start,
            trim_end=video_end,
            mode=video_mode,
            dry_run=dry_run,
        )

    audio_input = input_dir / f"{job.prefix}.m4a"
    if audio_input.exists():
        audio_output = output_dir / audio_input.name
        audio_start = job.audio_start if job.audio_start is not None else 0.0
        audio_end = job.audio_end if job.audio_end is not None else probe_duration(audio_input)
        if job.audio_start is None and job.audio_end is None and job.audio_offset is not None:
            audio_start = video_start + job.audio_offset
            if audio_start < 0:
                audio_start = 0.0
            audio_end = audio_start + (video_end - video_start)
        if audio_end <= audio_start:
            raise ValueError(f"{job.split_dir}/{job.prefix}: resolved audio_end must be greater than audio_start")
        trim_audio(
            input_path=audio_input,
            output_path=audio_output,
            trim_start=video_start,
            duration=video_end - video_start,
            audio_offset=job.audio_offset,
            audio_start=audio_start,
            audio_end=audio_end,
            dry_run=dry_run,
        )
    else:
        print(f"  skip missing audio: {audio_input.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trim grouped media inside dataset/*_split and align external .m4a audio "
            "with a per-group offset."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root directory that contains the *_split folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset_processed"),
        help="Directory where trimmed outputs will be written.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV manifest with split_dir,prefix,trim_start,trim_end,audio_offset columns.",
    )
    parser.add_argument(
        "--video-mode",
        choices=("copy", "reencode"),
        default="reencode",
        help="copy is faster but keyframe-limited; reencode is frame-accurate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ffmpeg commands without executing them.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if USE_INLINE_CONFIG:
            jobs = load_inline_jobs()
            dataset_root = DATASET_ROOT
            output_root = OUTPUT_ROOT
            video_mode = VIDEO_MODE
            dry_run = DRY_RUN
        else:
            if args.manifest is None:
                raise ValueError("--manifest is required when USE_INLINE_CONFIG is False")
            jobs = load_jobs(args.manifest)
            dataset_root = args.dataset_root
            output_root = args.output_root
            video_mode = args.video_mode
            dry_run = args.dry_run
        for job in jobs:
            process_job(
                dataset_root=dataset_root,
                output_root=output_root,
                job=job,
                video_mode=video_mode,
                dry_run=dry_run,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nFinished. Output root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
