"""Extract full-length mono 16 kHz WAV tracks from AVE MP4 files."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = ROOT / "data" / "raw" / "AVE" / "ave" / "video"
AUDIO_DIR = ROOT / "data" / "raw" / "AVE" / "ave" / "audio"
REPORT_PATH = ROOT / "data" / "raw" / "AVE" / "ave_extras" / "audio_extraction_report.json"

def find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install ffmpeg or imageio-ffmpeg to extract audio") from exc

def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not VIDEO_DIR.is_dir():
        raise FileNotFoundError(f"AVE videos not found: {VIDEO_DIR}")
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No MP4 files in {VIDEO_DIR}")
    ffmpeg = find_ffmpeg()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    skipped = 0
    for index, video in enumerate(videos, start=1):
        output = AUDIO_DIR / f"{video.stem}.wav"
        if output.is_file() and output.stat().st_size > 44:
            skipped += 1
        else:
            temp = AUDIO_DIR / f"{video.stem}.partial.wav"
            command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                       "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                       "-c:a", "pcm_s16le", "-f", "wav", str(temp)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode == 0 and temp.is_file() and temp.stat().st_size > 44:
                temp.replace(output)
            else:
                temp.unlink(missing_ok=True)
                failures.append({"video_id": video.stem, "return_code": result.returncode,
                                 "error": result.stderr.strip() or "No audio output produced"})
        if index % 100 == 0 or index == len(videos):
            print(f"Processed {index}/{len(videos)}; skipped {skipped}; failures {len(failures)}")
    report = {"video_count": len(videos), "audio_count": len(list(AUDIO_DIR.glob("*.wav")),),
              "sample_rate_hz": 16000, "channels": 1,
              "encoding": "PCM signed 16-bit little-endian WAV",
              "skipped_existing_count": skipped, "failure_count": len(failures), "failures": failures}
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {REPORT_PATH}")
    if failures:
        raise SystemExit(f"Failed on {len(failures)} of {len(videos)} videos")

if __name__ == "__main__":
    main()




