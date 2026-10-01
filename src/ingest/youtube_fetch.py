"""Fetch a YouTube video's auto-captions and convert them into the same
transcript JSON format the rest of the pipeline consumes.

Output shape (mirrors lenny_transcripts.py):
{
    "video_id": "dQw4w9WgXcQ",
    "text": "full concatenated text ...",
    "words": [{"word": "token", "start_ms": 0, "end_ms": 120}, ...],
    "cues": [{"speaker": "Unknown", "start_ms": 0, "end_ms": 2100, "text": "..."}, ...]
}

Usage:
    from src.ingest.youtube_fetch import fetch_transcript
    path = fetch_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

from ..config import TRANSCRIPT_DIR

WORDS_PER_SEC_FALLBACK = 2.5


def extract_video_id(url: str) -> str:
    """Extract the 11-char YouTube video ID from any YouTube URL."""
    m = re.search(r"(?:youtu\.be/|[?&]v=)([\w-]{11})", url)
    if not m:
        raise ValueError(f"Could not extract video ID from URL: {url!r}")
    return m.group(1)


def _seg_to_cues(segments: list[dict]) -> list[dict]:
    """Convert youtube-transcript-api segments into cues with ms timestamps."""
    cues = []
    for i, seg in enumerate(segments):
        start_ms = int(seg["start"] * 1000)
        # end_ms: use next segment's start, or estimate from duration
        if i + 1 < len(segments):
            end_ms = int(segments[i + 1]["start"] * 1000)
        else:
            end_ms = start_ms + int(seg.get("duration", 3.0) * 1000)
        cues.append({
            "speaker": "Unknown",
            "start_ms": start_ms,
            "end_ms": max(end_ms, start_ms + 1),
            "text": seg["text"].strip(),
        })
    return cues


def _interpolate_words(cues: list[dict]) -> list[dict]:
    """Spread each cue's words evenly across [start_ms, end_ms)."""
    words: list[dict] = []
    for c in cues:
        toks = c["text"].split()
        if not toks:
            continue
        start, end = c["start_ms"], c["end_ms"]
        span = max(end - start, 1)
        step = span / len(toks)
        for j, tok in enumerate(toks):
            w_start = int(start + j * step)
            w_end = int(start + (j + 1) * step)
            words.append({"word": tok, "start_ms": w_start, "end_ms": max(w_end, w_start + 1)})
    return words


def fetch_transcript(url: str, force: bool = False) -> Path:
    """Fetch captions for a YouTube URL and save to data/transcripts/<video_id>.json.

    Returns the path to the saved transcript file.
    Skips download if the file already exists unless force=True.
    """
    video_id = extract_video_id(url)
    out = TRANSCRIPT_DIR / f"{video_id}.json"

    if out.exists() and not force:
        print(f"[youtube_fetch] {video_id} already exists -> {out}")
        return out

    print(f"[youtube_fetch] fetching captions for {video_id} ...")
    # v1.x API: fetch() is now an instance method
    fetched = YouTubeTranscriptApi().fetch(video_id)
    transcript_list = [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]

    cues = _seg_to_cues(transcript_list)
    if not cues:
        raise ValueError(f"No captions found for video {video_id}. The video may have captions disabled.")

    words = _interpolate_words(cues)
    full_text = " ".join(w["word"] for w in words).strip()

    out.write_text(json.dumps({
        "video_id": video_id,
        "text": full_text,
        "words": words,
        "cues": cues,
    }, ensure_ascii=False))

    print(f"[youtube_fetch] {video_id} -> {len(cues)} cues, {len(words)} words saved to {out}")
    return out
