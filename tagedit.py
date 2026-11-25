#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mutagen>=1.47.0",
#     "pillow>=10.0.0",
#     "wcwidth>=0.2.13",
# ]
# ///
"""
tagedit — A fast, terminal-native audio tag editor for MP3, M4A, Opus, Ogg, and FLAC.

Run in one line with uv:
    uv run tagedit.py [directory_or_files...]
"""

from __future__ import annotations

import argparse
import base64
import curses
import io
import locale
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mutagen
from mutagen.flac import FLAC, Picture
from mutagen.id3 import (
    COMM,
    ID3,
    ID3NoHeaderError,
    TALB,
    TCMP,
    TCOM,
    TCON,
    TDRC,
    TIT2,
    TPE1,
    TPE2,
    TPOS,
    TRCK,
    TXXX,
)
from mutagen.mp4 import MP4, MP4FreeForm
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis
from PIL import Image
from wcwidth import wcwidth, wcswidth

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".opus", ".ogg", ".flac"}

# Standard tags always shown in the right-hand table (even when blank).
# Tuple: (key, display_label, color_group)
STANDARD_TAGS: List[Tuple[str, str, str]] = [
    ("title", "Title", "core"),
    ("artist", "Artist", "core"),
    ("albumartist", "Album Artist", "core"),
    ("album", "Album", "core"),
    ("tracknumber", "Track", "num"),
    ("tracktotal", "Track Total", "num"),
    ("discnumber", "Disc", "num"),
    ("disctotal", "Disc Total", "num"),
    ("date", "Date", "num"),
    ("genre", "Genre", "meta"),
    ("composer", "Composer", "meta"),
    ("compilation", "Compilation", "meta"),
    ("comment", "Comment", "meta"),
]

STANDARD_TAG_KEYS = {k for k, _, _ in STANDARD_TAGS}
STANDARD_TAG_LABELS = {k: label for k, label, _ in STANDARD_TAGS}
STANDARD_TAG_GROUPS = {k: group for k, _, group in STANDARD_TAGS}

# MP4 atom mappings
MP4_TEXT_ATOMS: Dict[str, str] = {
    "title": "\xa9nam",
    "artist": "\xa9ART",
    "albumartist": "aART",
    "album": "\xa9alb",
    "date": "\xa9day",
    "genre": "\xa9gen",
    "composer": "\xa9wrt",
    "comment": "\xa9cmt",
}
MP4_ATOM_TO_KEY: Dict[str, str] = {v: k for k, v in MP4_TEXT_ATOMS.items()}
MP4_SKIP_ATOMS = {
    "covr",
    "trkn",
    "disk",
    "pgap",
    "tmpo",
    "rtng",
    "stik",
    "hdvd",
    "pcst",
    "shwm",
    "tvsn",
    "tves",
    "sfID",
    "atID",
    "cnID",
    "geID",
    "plID",
    "cmID",
    "akID",
    "apID",
}

# ID3 frame mappings
ID3_TEXT_FRAMES = {
    "title": ("TIT2", TIT2),
    "artist": ("TPE1", TPE1),
    "albumartist": ("TPE2", TPE2),
    "album": ("TALB", TALB),
    "tracknumber": ("TRCK", TRCK),
    "discnumber": ("TPOS", TPOS),
    "date": ("TDRC", TDRC),
    "genre": ("TCON", TCON),
    "composer": ("TCOM", TCOM),
}

# Vorbis comment canonical key aliases (read-side normalization)
VORBIS_ALIASES: Dict[str, str] = {
    "album_artist": "albumartist",
    "album artist": "albumartist",
    "track": "tracknumber",
    "totaltracks": "tracktotal",
    "disc": "discnumber",
    "totaldiscs": "disctotal",
    "year": "date",
    "description": "comment",
}
VORBIS_SEPARATE_TOTAL_EXTS = {".flac", ".ogg", ".opus"}
VORBIS_SKIP_KEYS = {"metadata_block_picture", "coverart", "coverartmime"}


def natural_sort_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def format_duration(seconds: float) -> str:
    if not seconds or seconds <= 0:
        return "--:--"
    total = int(round(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def parse_pair_string(value: str) -> Optional[Tuple[int, int]]:
    """Parse '3' -> (3, 0) or '3/12' -> (3, 12)."""
    s = value.strip()
    if not s:
        return None
    if "/" in s:
        left, right = s.split("/", 1)
        try:
            a = int(left.strip()) if left.strip() else 0
            b = int(right.strip()) if right.strip() else 0
            return (a, b)
        except ValueError:
            return None
    try:
        return (int(s), 0)
    except ValueError:
        return None


def format_pair_tuple(pair: Tuple[int, int]) -> str:
    a, b = pair
    if a <= 0 and b <= 0:
        return ""
    if b > 0:
        return f"{a}/{b}"
    return str(a)


def is_effective_artist_value(value: str) -> bool:
    cleaned = " ".join(value.strip().lower().split())
    return bool(cleaned) and cleaned != "unknown artist"


@dataclass
class AudioTrack:
    path: Path
    fmt: str = ""
    duration: float = 0.0
    bitrate: int = 0
    sample_rate: int = 0
    file_size: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    original_tags: Dict[str, str] = field(default_factory=dict)
    load_error: Optional[str] = None
    art_bytes: Optional[bytes] = None
    art_width: int = 0
    art_height: int = 0
    art_size: int = 0
    art_format: str = ""

    @property
    def art_info_label(self) -> str:
        if not self.art_bytes:
            return ""
        parts: List[str] = []
        if self.art_width > 0 and self.art_height > 0:
            parts.append(f"{self.art_width}×{self.art_height}")
        if self.art_size > 0:
            parts.append(format_size(self.art_size))
        info = " · ".join(parts)
        if self.art_format:
            return f"{info} ({self.art_format})" if info else self.art_format
        return info

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def is_dirty(self) -> bool:
        return self.tags != self.original_tags

    @property
    def effective_artist(self) -> str:
        aart = self.tags.get("albumartist", "").strip()
        if is_effective_artist_value(aart):
            return aart
        art = self.tags.get("artist", "").strip()
        if is_effective_artist_value(art):
            return art
        return ""

    @property
    def is_compilation(self) -> bool:
        val = self.tags.get("compilation", "").strip().lower()
        return val in ("1", "true", "yes")

    @property
    def has_effective_artist(self) -> bool:
        return bool(self.effective_artist)

    @property
    def missing_required(self) -> List[str]:
        """Return list of missing required metadata items.

        Required:
        - effective artist (artist OR albumartist, excluding 'unknown artist')
        - tracknumber
        - album
        - title
        """
        missing: List[str] = []
        if not self.tags.get("title", "").strip():
            missing.append("title")
        if not self.has_effective_artist:
            missing.append("artist")
        if not self.tags.get("album", "").strip():
            missing.append("album")
        if not self.tags.get("tracknumber", "").strip():
            missing.append("tracknumber")
        return missing

    @property
    def is_incomplete(self) -> bool:
        return bool(self.missing_required)

    def revert(self) -> None:
        self.tags = dict(self.original_tags)

    def set_tag(self, key: str, value: str) -> None:
        key = key.strip().lower()
        if not key:
            return
        if value == "" and key not in STANDARD_TAG_KEYS:
            self.tags.pop(key, None)
        else:
            self.tags[key] = value

    def extra_tag_keys(self) -> List[str]:
        return sorted(
            k for k in self.tags.keys() if k not in STANDARD_TAG_KEYS
        )


def _read_audio_info(mf: object, track: AudioTrack) -> None:
    info = getattr(mf, "info", None)
    if info is None:
        return
    track.duration = float(getattr(info, "length", 0.0) or 0.0)
    track.bitrate = int(getattr(info, "bitrate", 0) or 0)
    track.sample_rate = int(getattr(info, "sample_rate", 0) or 0)


def _attach_artwork(track: AudioTrack, data: Optional[bytes]) -> None:
    if not data:
        return
    track.art_bytes = data
    track.art_size = len(data)
    try:
        with Image.open(io.BytesIO(data)) as img:
            track.art_width, track.art_height = img.size
            track.art_format = (img.format or "").upper()
    except Exception:
        pass


def load_track(path: Path) -> AudioTrack:
    track = AudioTrack(path=path, fmt=path.suffix.lstrip(".").upper())
    try:
        track.file_size = path.stat().st_size
    except OSError:
        track.file_size = 0

    tags: Dict[str, str] = {k: "" for k in STANDARD_TAG_KEYS}
    ext = path.suffix.lower()

    try:
        if ext == ".mp3":
            mf = mutagen.File(path)
            if mf is not None:
                _read_audio_info(mf, track)
            try:
                id3 = ID3(path)
            except ID3NoHeaderError:
                id3 = ID3()

            for key, (frame_id, _) in ID3_TEXT_FRAMES.items():
                frame = id3.get(frame_id)
                if frame is not None and getattr(frame, "text", None):
                    raw_val = ", ".join(str(x) for x in frame.text if str(x))
                    if key == "tracknumber" and "/" in raw_val:
                        num, tot = [p.strip() for p in raw_val.split("/", 1)]
                        tags["tracknumber"] = num
                        tags["tracktotal"] = tot
                    elif key == "discnumber" and "/" in raw_val:
                        num, tot = [p.strip() for p in raw_val.split("/", 1)]
                        tags["discnumber"] = num
                        tags["disctotal"] = tot
                    else:
                        tags[key] = raw_val

            # Fallback to TYER if TDRC is absent
            if not tags["date"] and "TYER" in id3:
                frame = id3.get("TYER")
                if frame is not None and getattr(frame, "text", None):
                    tags["date"] = ", ".join(str(x) for x in frame.text if str(x))

            # Compilation flag (TCMP)
            tcmp = id3.get("TCMP")
            if tcmp is not None and getattr(tcmp, "text", None):
                tags["compilation"] = ", ".join(str(x) for x in tcmp.text if str(x))

            # Comments
            comms = id3.getall("COMM")
            if comms:
                # Prefer empty-description comment if available
                chosen = next((c for c in comms if not c.desc), comms[0])
                if chosen.text:
                    tags["comment"] = "\n".join(str(x) for x in chosen.text)

            # Custom TXXX tags
            for txxx in id3.getall("TXXX"):
                desc = (txxx.desc or "").strip().lower()
                if not desc:
                    continue
                val = ", ".join(str(x) for x in txxx.text if str(x))
                if desc in ("album artist", "album_artist", "albumartist"):
                    if not tags["albumartist"]:
                        tags["albumartist"] = val
                elif desc not in STANDARD_TAG_KEYS and val:
                    tags[desc] = val

            # Embedded artwork (APIC)
            apics = id3.getall("APIC")
            if apics:
                chosen_apic = next((a for a in apics if getattr(a, "type", 0) == 3), apics[0])
                if getattr(chosen_apic, "data", None):
                    _attach_artwork(track, bytes(chosen_apic.data))

        elif ext == ".m4a":
            mp4 = MP4(path)
            _read_audio_info(mp4, track)
            mtags = mp4.tags or {}
            for key, atom in MP4_TEXT_ATOMS.items():
                vals = mtags.get(atom)
                if vals:
                    tags[key] = ", ".join(str(v) for v in vals)
            if "trkn" in mtags and mtags["trkn"]:
                a, b = mtags["trkn"][0]
                tags["tracknumber"] = str(a) if a > 0 else ""
                tags["tracktotal"] = str(b) if b > 0 else ""
            if "disk" in mtags and mtags["disk"]:
                a, b = mtags["disk"][0]
                tags["discnumber"] = str(a) if a > 0 else ""
                tags["disctotal"] = str(b) if b > 0 else ""
            if "cpil" in mtags:
                tags["compilation"] = "1" if mtags["cpil"] else "0"
            if "covr" in mtags and mtags["covr"]:
                _attach_artwork(track, bytes(mtags["covr"][0]))

            for atom, vals in mtags.items():
                if atom in MP4_ATOM_TO_KEY or atom in MP4_SKIP_ATOMS:
                    continue
                if atom.startswith("----:"):
                    parts = atom.split(":", 2)
                    custom_name = parts[-1].strip().lower()
                    if custom_name and custom_name not in STANDARD_TAG_KEYS:
                        decoded = []
                        for v in vals:
                            if isinstance(v, (bytes, MP4FreeForm)):
                                decoded.append(bytes(v).decode("utf-8", errors="replace"))
                            else:
                                decoded.append(str(v))
                        joined = ", ".join(x for x in decoded if x)
                        if joined:
                            tags[custom_name] = joined

        elif ext in (".flac", ".ogg", ".opus"):
            mf = None
            if ext == ".flac":
                mf = FLAC(path)
            elif ext == ".opus":
                mf = OggOpus(path)
            else:
                # .ogg could be Vorbis or Opus
                mf = mutagen.File(path)
                if mf is None:
                    mf = OggVorbis(path)
                if isinstance(mf, OggOpus):
                    track.fmt = "OGG/OPUS"
            _read_audio_info(mf, track)

            raw_tags = mf.tags or {}
            tags["tracktotal"] = ""
            tags["disctotal"] = ""

            for raw_k, vals in raw_tags.items():
                k = raw_k.strip().lower()
                if k in VORBIS_SKIP_KEYS:
                    continue
                k = VORBIS_ALIASES.get(k, k)
                if isinstance(vals, (list, tuple)):
                    val = ", ".join(str(v) for v in vals if str(v))
                else:
                    val = str(vals)

                if k in STANDARD_TAG_KEYS or k in ("tracktotal", "disctotal"):
                    if not tags[k]:
                        tags[k] = val
                elif val:
                    tags[k] = val

            # Embedded artwork (FLAC pictures or Vorbis metadata_block_picture)
            if isinstance(mf, FLAC) and mf.pictures:
                chosen_pic = next((p for p in mf.pictures if p.type == 3), mf.pictures[0])
                if chosen_pic.data:
                    _attach_artwork(track, bytes(chosen_pic.data))
            else:
                b64_pics = raw_tags.get("metadata_block_picture")
                if b64_pics:
                    first_b64 = b64_pics[0] if isinstance(b64_pics, (list, tuple)) else b64_pics
                    try:
                        pic = Picture(base64.b64decode(first_b64))
                        if pic.data:
                            _attach_artwork(track, bytes(pic.data))
                    except Exception:
                        pass

    except Exception as exc:
        track.load_error = str(exc)

    track.tags = dict(tags)
    track.original_tags = dict(tags)
    return track


def save_track(track: AudioTrack) -> None:
    ext = track.path.suffix.lower()

    if ext == ".mp3":
        try:
            id3 = ID3(track.path)
        except ID3NoHeaderError:
            id3 = ID3()

        for key, (frame_id, frame_cls) in ID3_TEXT_FRAMES.items():
            id3.delall(frame_id)
            if key == "date":
                id3.delall("TYER")
            if key == "tracknumber":
                num = track.tags.get("tracknumber", "").strip()
                tot = track.tags.get("tracktotal", "").strip()
                val = f"{num}/{tot}" if (num and tot) else (f"0/{tot}" if tot else num)
            elif key == "discnumber":
                num = track.tags.get("discnumber", "").strip()
                tot = track.tags.get("disctotal", "").strip()
                val = f"{num}/{tot}" if (num and tot) else (f"0/{tot}" if tot else num)
            else:
                val = track.tags.get(key, "").strip()
            if val:
                id3.add(frame_cls(encoding=3, text=[val]))

        # Comment
        comment_val = track.tags.get("comment", "").strip()
        existing_comms = [c for c in id3.getall("COMM") if c.desc]
        id3.delall("COMM")
        for c in existing_comms:
            id3.add(c)
        if comment_val:
            id3.add(COMM(encoding=3, lang="eng", desc="", text=[comment_val]))

        # Compilation (TCMP)
        comp_val = track.tags.get("compilation", "").strip().lower()
        id3.delall("TCMP")
        if comp_val in ("1", "true", "yes"):
            id3.add(TCMP(encoding=3, text=["1"]))
        elif comp_val in ("0", "false", "no"):
            id3.add(TCMP(encoding=3, text=["0"]))

        # Custom TXXX tags (only touch keys that were loaded or added)
        managed_custom = {
            k for k in (set(track.tags.keys()) | set(track.original_tags.keys()))
            if k not in STANDARD_TAG_KEYS
        }
        if managed_custom:
            kept_txxx = [
                t for t in id3.getall("TXXX")
                if (t.desc or "").strip().lower() not in managed_custom
            ]
            id3.delall("TXXX")
            for t in kept_txxx:
                id3.add(t)
            for k in sorted(managed_custom):
                val = track.tags.get(k, "").strip()
                if val:
                    id3.add(TXXX(encoding=3, desc=k, text=[val]))

        id3.save(track.path, v2_version=3)

    elif ext == ".m4a":
        mp4 = MP4(track.path)
        if mp4.tags is None:
            mp4.add_tags()
        mtags = mp4.tags

        for key, atom in MP4_TEXT_ATOMS.items():
            val = track.tags.get(key, "").strip()
            if val:
                mtags[atom] = [val]
            else:
                mtags.pop(atom, None)

        # Track and disc tuples
        for num_key, tot_key, atom in (
            ("tracknumber", "tracktotal", "trkn"),
            ("discnumber", "disctotal", "disk"),
        ):
            num_s = track.tags.get(num_key, "").strip()
            tot_s = track.tags.get(tot_key, "").strip()
            if not num_s and not tot_s:
                mtags.pop(atom, None)
            else:
                try:
                    num_i = int(num_s) if num_s else 0
                    tot_i = int(tot_s) if tot_s else 0
                except ValueError:
                    raise ValueError(
                        f"M4A {num_key}/{tot_key} must be integers (got '{num_s}', '{tot_s}')"
                    )
                mtags[atom] = [(num_i, tot_i)]

        # Compilation boolean atom (cpil)
        comp_val = track.tags.get("compilation", "").strip().lower()
        if not comp_val:
            mtags.pop("cpil", None)
        elif comp_val in ("1", "true", "yes"):
            mtags["cpil"] = True
        elif comp_val in ("0", "false", "no"):
            mtags["cpil"] = False
        else:
            raise ValueError(
                f"M4A compilation must be 1/0 or true/false (got '{comp_val}')"
            )

        # Custom freeform atoms
        managed_custom = {
            k for k in (set(track.tags.keys()) | set(track.original_tags.keys()))
            if k not in STANDARD_TAG_KEYS
        }
        for k in managed_custom:
            # Remove any existing matching freeform atom case-insensitively
            matching = [
                a for a in list(mtags.keys())
                if a.startswith("----:") and a.split(":", 2)[-1].strip().lower() == k
            ]
            val = track.tags.get(k, "").strip()
            if val:
                atom_name = matching[0] if matching else f"----:com.apple.iTunes:{k}"
                for a in matching:
                    if a != atom_name:
                        mtags.pop(a, None)
                mtags[atom_name] = [MP4FreeForm(val.encode("utf-8"))]
            else:
                for a in matching:
                    mtags.pop(a, None)

        mp4.save()

    elif ext in (".flac", ".ogg", ".opus"):
        if ext == ".flac":
            mf = FLAC(track.path)
        elif ext == ".opus":
            mf = OggOpus(track.path)
        else:
            mf = mutagen.File(track.path)
            if mf is None:
                mf = OggVorbis(track.path)

        if mf.tags is None:
            mf.add_tags()

        def _del_vorbis(tag_key: str) -> None:
            if tag_key in mf.tags:
                del mf.tags[tag_key]

        # Standard tags (keep tracknumber/tracktotal and discnumber/disctotal separate)
        for key in STANDARD_TAG_KEYS:
            val = track.tags.get(key, "").strip()
            if key == "tracknumber":
                for alias in ("tracknumber", "track"):
                    _del_vorbis(alias)
                if val:
                    mf.tags["tracknumber"] = [val]
            elif key == "discnumber":
                for alias in ("discnumber", "disc"):
                    _del_vorbis(alias)
                if val:
                    mf.tags["discnumber"] = [val]
            else:
                # Clear aliases on write
                for alias, canon in VORBIS_ALIASES.items():
                    if canon == key:
                        _del_vorbis(alias)
                if val:
                    mf.tags[key] = [val]
                else:
                    _del_vorbis(key)

        for key, aliases in (
            ("tracktotal", ("tracktotal", "totaltracks")),
            ("disctotal", ("disctotal", "totaldiscs")),
        ):
            for alias in aliases:
                _del_vorbis(alias)
            val = track.tags.get(key, "").strip()
            if val:
                mf.tags[key] = [val]

        # Custom tags
        managed_custom = {
            k for k in (set(track.tags.keys()) | set(track.original_tags.keys()))
            if k not in STANDARD_TAG_KEYS and k not in ("tracktotal", "disctotal")
        }
        for k in managed_custom:
            val = track.tags.get(k, "").strip()
            if val:
                mf.tags[k] = [val]
            else:
                _del_vorbis(k)

        mf.save()

    track.original_tags = dict(track.tags)


def guess_tags_from_filename(path: Path) -> Dict[str, str]:
    """Best-effort parse of common music filename conventions."""
    stem = path.stem.strip()
    guessed: Dict[str, str] = {}

    # Pattern 1: "01 - Artist - Title" or "1-01 - Artist - Title"
    m = re.match(r"^(?:(\d+)-)?(\d{1,3})\s*[-._)]\s*(.+?)\s+-\s+(.+)$", stem)
    if m:
        if m.group(1):
            guessed["discnumber"] = str(int(m.group(1)))
        guessed["tracknumber"] = str(int(m.group(2)))
        guessed["artist"] = m.group(3).strip()
        guessed["title"] = m.group(4).strip()
        return guessed

    # Pattern 2: "01 - Title" or "01. Title" or "01 Title"
    m = re.match(r"^(?:(\d+)-)?(\d{1,3})\s*(?:[-._)]|\s)\s*(.+)$", stem)
    if m:
        if m.group(1):
            guessed["discnumber"] = str(int(m.group(1)))
        guessed["tracknumber"] = str(int(m.group(2)))
        guessed["title"] = m.group(3).strip()
        return guessed

    # Pattern 3: "Artist - Title"
    m = re.match(r"^(.+?)\s+-\s+(.+)$", stem)
    if m:
        guessed["artist"] = m.group(1).strip()
        guessed["title"] = m.group(2).strip()
        return guessed

    # Fallback: whole stem as title
    if stem:
        guessed["title"] = stem
    return guessed


def discover_audio_files(targets: Sequence[str], recursive: bool = False) -> List[Path]:
    paths: List[Path] = []
    if not targets:
        targets = ["."]

    for raw in targets:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(p)
        elif p.is_dir():
            iterator = p.rglob("*") if recursive else p.iterdir()
            found = [
                f for f in iterator
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            found.sort(key=lambda x: (str(x.parent).lower(), natural_sort_key(x)))
            paths.extend(found)

    # Deduplicate preserving order
    seen = set()
    unique: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Unicode display width helpers
# ---------------------------------------------------------------------------

def str_width(text: str) -> int:
    w = wcswidth(text)
    return w if w >= 0 else len(text)


def truncate_width(text: str, max_width: int, ellipsis: str = "…") -> str:
    if max_width <= 0:
        return ""
    if str_width(text) <= max_width:
        return text
    ell_w = str_width(ellipsis)
    if max_width <= ell_w:
        return ellipsis[:max_width]
    out: List[str] = []
    cur = 0
    limit = max_width - ell_w
    for ch in text:
        cw = wcwidth(ch)
        if cw < 0:
            cw = 1
        if cur + cw > limit:
            break
        out.append(ch)
        cur += cw
    return "".join(out) + ellipsis


def pad_width(text: str, width: int) -> str:
    truncated = truncate_width(text, width)
    w = str_width(truncated)
    if w < width:
        return truncated + " " * (width - w)
    return truncated


def safe_addstr(win: curses.window, y: int, x: int, text: str, attr: int = 0) -> None:
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y or x >= max_x:
        return
    if x < 0:
        text = text[-x:]
        x = 0
    avail = max_x - x
    if avail <= 0:
        return
    # Avoid writing to the very bottom-right cell which raises curses.error
    if y == max_y - 1:
        avail -= 1
    if avail <= 0:
        return
    clipped = truncate_width(text, avail, ellipsis="")
    try:
        win.addstr(y, x, clipped, attr)
    except curses.error:
        pass


# ---------------------------------------------------------------------------
# Inline Line Editor
# ---------------------------------------------------------------------------

class LineEditor:
    def __init__(self, initial: str = "", select_all: bool = False) -> None:
        self.text = initial
        self.cursor = len(initial)
        self.selected = select_all and bool(initial)

    def handle_key(self, ch: object) -> Optional[str]:
        """Process a key from get_wch().

        Returns:
            "commit"      on Enter
            "commit_next" on Tab
            "commit_prev" on Shift-Tab
            "cancel"      on Escape
            None          otherwise
        """
        if isinstance(ch, int):
            if ch in (curses.KEY_ENTER, 10, 13):
                return "commit"
            if ch == 27:
                return "cancel"
            if ch == curses.KEY_BTAB:
                return "commit_prev"
            if ch == curses.KEY_UP:
                return "commit_up"
            if ch == curses.KEY_DOWN:
                return "commit_down"
            if ch in (curses.KEY_BACKSPACE, 127, 8, curses.KEY_DC):
                if self.selected:
                    self.text = ""
                    self.cursor = 0
                    self.selected = False
                elif ch == curses.KEY_DC:
                    if self.cursor < len(self.text):
                        self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
                elif self.cursor > 0:
                    self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
                    self.cursor -= 1
            elif ch == curses.KEY_LEFT:
                if self.selected:
                    self.selected = False
                    self.cursor = 0
                else:
                    self.cursor = max(0, self.cursor - 1)
            elif ch == curses.KEY_RIGHT:
                if self.selected:
                    self.selected = False
                    self.cursor = len(self.text)
                else:
                    self.cursor = min(len(self.text), self.cursor + 1)
            elif ch == curses.KEY_HOME:
                self.selected = False
                self.cursor = 0
            elif ch == curses.KEY_END:
                self.selected = False
                self.cursor = len(self.text)
            return None

        if isinstance(ch, str):
            if ch in ("\n", "\r"):
                return "commit"
            if ch == "\x1b":
                return "cancel"
            if ch == "\t":
                return "commit_next"
            if ch in ("\x7f", "\x08"):
                if self.selected:
                    self.text = ""
                    self.cursor = 0
                    self.selected = False
                elif self.cursor > 0:
                    self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
                    self.cursor -= 1
            elif ch == "\x01":  # Ctrl-A
                self.selected = False
                self.cursor = 0
            elif ch == "\x05":  # Ctrl-E
                self.selected = False
                self.cursor = len(self.text)
            elif ch == "\x15":  # Ctrl-U
                if self.selected:
                    self.text = ""
                    self.cursor = 0
                    self.selected = False
                else:
                    self.text = self.text[self.cursor :]
                    self.cursor = 0
            elif ch == "\x0b":  # Ctrl-K
                if self.selected:
                    self.text = ""
                    self.cursor = 0
                    self.selected = False
                else:
                    self.text = self.text[: self.cursor]
            elif ch == "\x17":  # Ctrl-W (delete word backward)
                if self.selected:
                    self.text = ""
                    self.cursor = 0
                    self.selected = False
                else:
                    left = self.text[: self.cursor].rstrip()
                    idx = left.rfind(" ")
                    new_pos = 0 if idx < 0 else idx + 1
                    self.text = self.text[:new_pos] + self.text[self.cursor :]
                    self.cursor = new_pos
            elif ch.isprintable():
                if self.selected:
                    self.text = ch
                    self.cursor = len(ch)
                    self.selected = False
                else:
                    self.text = self.text[: self.cursor] + ch + self.text[self.cursor :]
                    self.cursor += len(ch)
        return None

    def visible_slice(self, width: int) -> Tuple[str, int]:
        """Return (visible_text, cursor_x_offset) fitting within display width."""
        if width <= 1:
            return ("", 0)
        # Compute start index so cursor is visible
        prefix_w = str_width(self.text[: self.cursor])
        start = 0
        if prefix_w >= width - 1:
            target = prefix_w - (width - 2)
            acc = 0
            for idx, c in enumerate(self.text):
                cw = max(1, wcwidth(c))
                acc += cw
                if acc >= target:
                    start = idx + 1
                    break
        # Slice forward from start up to width
        out: List[str] = []
        acc = 0
        for c in self.text[start:]:
            cw = max(1, wcwidth(c))
            if acc + cw > width:
                break
            out.append(c)
            acc += cw
        cursor_offset = str_width(self.text[start : self.cursor])
        cursor_offset = min(cursor_offset, width - 1)
        return ("".join(out), cursor_offset)


# ---------------------------------------------------------------------------
# Terminal Artwork Rendering (Kitty Graphics Protocol + Unicode half-blocks)
# ---------------------------------------------------------------------------

KITTY_ART_IMAGE_ID = 42


def supports_kitty_graphics() -> bool:
    if os.environ.get("TMUX"):
        return False
    if os.environ.get("KITTY_WINDOW_ID") or os.environ.get("GHOSTTY_RESOURCES_DIR"):
        return True
    term = os.environ.get("TERM", "").lower()
    term_prog = os.environ.get("TERM_PROGRAM", "").lower()
    if any(k in term for k in ("kitty", "ghostty")):
        return True
    if any(k in term_prog for k in ("kitty", "ghostty", "wezterm")):
        return True
    return False


def supports_truecolor() -> bool:
    ct = os.environ.get("COLORTERM", "").lower()
    if "truecolor" in ct or "24bit" in ct:
        return True
    term_prog = os.environ.get("TERM_PROGRAM", "").lower()
    return any(k in term_prog for k in ("iterm", "ghostty", "kitty", "wezterm", "warp", "vscode"))


def _rgb_to_ansi256(r: int, g: int, b: int) -> int:
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return int(round(((r - 8) / 247) * 24)) + 232
    ri = int(round(r / 255 * 5))
    gi = int(round(g / 255 * 5))
    bi = int(round(b / 255 * 5))
    return 16 + 36 * ri + 6 * gi + bi


def encode_png_for_kitty(art_bytes: bytes, max_px: int = 512) -> bytes:
    with Image.open(io.BytesIO(art_bytes)) as img:
        img = img.convert("RGB")
        if img.width > max_px or img.height > max_px:
            img.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def build_kitty_delete_bytes(image_id: int = KITTY_ART_IMAGE_ID) -> bytes:
    return f"\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\".encode("ascii")


def build_kitty_graphics_bytes(
    png_bytes: bytes,
    row: int,
    col: int,
    cols: int,
    rows: int,
    image_id: int = KITTY_ART_IMAGE_ID,
) -> bytes:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    chunk_size = 4096
    out = bytearray()
    out.extend(f"\x1b[{row + 1};{col + 1}H".encode("ascii"))
    total = len(b64)
    offset = 0
    first = True
    while offset < total:
        chunk = b64[offset : offset + chunk_size]
        offset += chunk_size
        more = 1 if offset < total else 0
        if first:
            header = f"\x1b_Ga=T,f=100,i={image_id},c={cols},r={rows},C=1,q=2,m={more};"
            first = False
        else:
            header = f"\x1b_Gm={more},q=2;"
        out.extend(header.encode("ascii"))
        out.extend(chunk.encode("ascii"))
        out.extend(b"\x1b\\")
    return bytes(out)


def build_unicode_halfblocks_bytes(
    art_bytes: bytes,
    row: int,
    col: int,
    cols: int,
    rows: int,
    truecolor: bool = True,
) -> bytes:
    with Image.open(io.BytesIO(art_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize((cols, rows * 2), Image.Resampling.LANCZOS)
        px = img.load()

    parts: List[str] = []
    for r in range(rows):
        parts.append(f"\x1b[{row + r + 1};{col + 1}H")
        y_top = r * 2
        y_bot = r * 2 + 1
        for c in range(cols):
            tr, tg, tb = px[c, y_top]
            br, bg, bb = px[c, y_bot]
            if truecolor:
                parts.append(f"\x1b[38;2;{tr};{tg};{tb};48;2;{br};{bg};{bb}m▀")
            else:
                fg = _rgb_to_ansi256(tr, tg, tb)
                bg_c = _rgb_to_ansi256(br, bg, bb)
                parts.append(f"\x1b[38;5;{fg};48;5;{bg_c}m▀")
        parts.append("\x1b[0m")
    return "".join(parts).encode("utf-8")


# ---------------------------------------------------------------------------
# Terminal UI Application
# ---------------------------------------------------------------------------

class TagEditorApp:
    # Color pair IDs (all use default terminal background -1)
    C_DEFAULT = 0
    C_VIOLET = 1
    C_YELLOW = 2

    def __init__(self, tracks: List[AudioTrack], root_label: str) -> None:
        self.all_tracks = tracks
        self.root_label = root_label

        self.focus: str = "left"  # "left" or "right"
        self.song_idx: int = 0
        self.song_scroll: int = 0
        self.tag_idx: int = 0
        self.tag_scroll: int = 0

        self.selected_paths: set[Path] = set()
        self.only_incomplete: bool = False
        self.search_query: str = ""

        # Modal / inline modes: None, "edit_tag", "search", "add_tag", "confirm_quit", "help"
        self.mode: Optional[str] = None
        self.editor: Optional[LineEditor] = None
        self.editing_tag_key: Optional[str] = None
        self.edit_cursor_pos: Tuple[int, int] = (0, 0)

        self.status_msg: str = ""
        self.status_is_error: bool = False

        # Artwork rendering state
        self._use_kitty: bool = supports_kitty_graphics()
        self._use_truecolor: bool = supports_truecolor()
        self._pending_art: Optional[Tuple[AudioTrack, int, int, int, int]] = None
        self._kitty_shown_key: Optional[Tuple[Path, int, int, int, int]] = None
        self._png_cache: Dict[Path, bytes] = {}
        self._unicode_art_cache: Dict[Tuple[Path, int, int, int, int], bytes] = {}

    # -- Track filtering & selection helpers --

    def visible_tracks(self) -> List[AudioTrack]:
        q = self.search_query.strip().lower()
        out: List[AudioTrack] = []
        for t in self.all_tracks:
            if self.only_incomplete and not t.is_incomplete:
                continue
            if q:
                hay = " ".join(
                    [t.name, t.tags.get("title", ""), t.tags.get("artist", ""), t.tags.get("album", "")]
                ).lower()
                if q not in hay:
                    continue
            out.append(t)
        return out

    def clamp_song_idx(self) -> List[AudioTrack]:
        vis = self.visible_tracks()
        if not vis:
            self.song_idx = 0
            self.song_scroll = 0
        else:
            self.song_idx = max(0, min(self.song_idx, len(vis) - 1))
        return vis

    def active_tracks(self) -> List[AudioTrack]:
        """Tracks targeted by the right-hand tag table (multi-select or current song)."""
        vis = self.clamp_song_idx()
        if not vis:
            return []
        if self.selected_paths:
            selected = [t for t in self.all_tracks if t.path in self.selected_paths]
            if selected:
                return selected
        return [vis[self.song_idx]]

    def current_song(self) -> Optional[AudioTrack]:
        vis = self.clamp_song_idx()
        if not vis:
            return None
        return vis[self.song_idx]

    def current_tag_rows(self, targets: List[AudioTrack]) -> List[Tuple[str, str, str, bool]]:
        """Return list of (key, label, display_value, is_mixed) for the right panel."""
        extra_keys: set[str] = set()
        for t in targets:
            extra_keys.update(t.extra_tag_keys())

        ordered_keys = [k for k, _, _ in STANDARD_TAGS] + sorted(extra_keys)
        rows: List[Tuple[str, str, str, bool]] = []
        for k in ordered_keys:
            label = STANDARD_TAG_LABELS.get(k, k)
            vals = [t.tags.get(k, "") for t in targets]
            unique_vals = set(vals)
            if len(unique_vals) == 1:
                rows.append((k, label, vals[0], False))
            else:
                rows.append((k, label, "<multiple values>", True))
        return rows

    def dirty_tracks(self) -> List[AudioTrack]:
        return [t for t in self.all_tracks if t.is_dirty]

    def set_status(self, msg: str, error: bool = False) -> None:
        self.status_msg = msg
        self.status_is_error = error

    # -- Color initialization (strictly terminal theme ANSI colors) --

    def init_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(self.C_VIOLET, curses.COLOR_MAGENTA, -1)
        curses.init_pair(self.C_YELLOW, curses.COLOR_YELLOW, -1)

    def tag_name_attr(self, key: str, targets: List[AudioTrack]) -> int:
        # Highlight required tag name in bold violet if missing on any target track
        if key == "title" and any(not t.tags.get("title", "").strip() for t in targets):
            return curses.color_pair(self.C_VIOLET) | curses.A_BOLD
        if key == "album" and any(not t.tags.get("album", "").strip() for t in targets):
            return curses.color_pair(self.C_VIOLET) | curses.A_BOLD
        if key == "tracknumber" and any(not t.tags.get("tracknumber", "").strip() for t in targets):
            return curses.color_pair(self.C_VIOLET) | curses.A_BOLD
        if key in ("artist", "albumartist") and any(
            not t.has_effective_artist for t in targets
        ):
            return curses.color_pair(self.C_VIOLET) | curses.A_BOLD

        return curses.color_pair(self.C_VIOLET)

    # -- Main loop --

    def _clear_kitty_art(self) -> None:
        if self._use_kitty and self._kitty_shown_key is not None:
            try:
                sys.stdout.buffer.write(build_kitty_delete_bytes())
                sys.stdout.buffer.flush()
            except OSError:
                pass
            self._kitty_shown_key = None

    def _flush_artwork(self, stdscr: curses.window) -> None:
        if self._pending_art is None:
            self._clear_kitty_art()
            return

        track, row, col, cols, rows = self._pending_art
        if not track.art_bytes or cols <= 0 or rows <= 0:
            self._clear_kitty_art()
            return

        key = (track.path, row, col, cols, rows)
        try:
            if self._use_kitty:
                if self._kitty_shown_key != key:
                    if self._kitty_shown_key is not None:
                        sys.stdout.buffer.write(build_kitty_delete_bytes())
                    png = self._png_cache.get(track.path)
                    if png is None:
                        png = encode_png_for_kitty(track.art_bytes)
                        self._png_cache[track.path] = png
                    sys.stdout.buffer.write(
                        build_kitty_graphics_bytes(png, row, col, cols, rows)
                    )
                    sys.stdout.buffer.flush()
                    self._kitty_shown_key = key
            else:
                rendered = self._unicode_art_cache.get(key)
                if rendered is None:
                    rendered = build_unicode_halfblocks_bytes(
                        track.art_bytes,
                        row,
                        col,
                        cols,
                        rows,
                        truecolor=self._use_truecolor,
                    )
                    if len(self._unicode_art_cache) > 32:
                        self._unicode_art_cache.clear()
                    self._unicode_art_cache[key] = rendered
                sys.stdout.buffer.write(rendered)
                sys.stdout.buffer.flush()
        except Exception:
            pass

    def run(self, stdscr: curses.window) -> None:
        self.init_colors()
        curses.curs_set(0)
        stdscr.keypad(True)
        curses.set_escdelay(25)

        try:
            while True:
                self.draw(stdscr)
                if self.mode in ("edit_tag", "search", "add_tag"):
                    curses.curs_set(1)
                    cy, cx = self.edit_cursor_pos
                    try:
                        stdscr.move(cy, cx)
                    except curses.error:
                        pass
                else:
                    curses.curs_set(0)

                try:
                    ch = stdscr.get_wch()
                except curses.error:
                    continue
                except KeyboardInterrupt:
                    if self.dirty_tracks():
                        self.mode = "confirm_quit"
                        continue
                    break

                if ch == curses.KEY_RESIZE:
                    self._clear_kitty_art()
                    continue

                if not self.handle_input(ch):
                    break
        finally:
            self._clear_kitty_art()

    # -- Drawing --

    def draw(self, stdscr: curses.window) -> None:
        self._pending_art = None
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        if max_y < 10 or max_x < 44:
            safe_addstr(
                stdscr,
                0,
                0,
                "Terminal too small (need at least 44x10).",
                curses.color_pair(self.C_VIOLET) | curses.A_BOLD,
            )
            stdscr.refresh()
            self._clear_kitty_art()
            return

        # Top header bar (row 0)
        self._draw_header(stdscr, max_x)

        # Layout split
        body_top = 1
        body_bottom = max_y - 1  # leave row max_y-1 for bottom bar / prompt
        body_h = body_bottom - body_top

        left_w = max(24, min(48, int(max_x * 0.36)))
        right_x = left_w + 1
        right_w = max_x - right_x

        self._draw_left_pane(stdscr, body_top, 0, body_h, left_w)
        self._draw_right_pane(stdscr, body_top, right_x, body_h, right_w)

        # Status & footer
        self._draw_footer(stdscr, max_y, max_x)

        # Overlays
        if self.mode == "help":
            self._pending_art = None
            self._draw_help_modal(stdscr, max_y, max_x)
        elif self.mode == "confirm_quit":
            self._pending_art = None
            self._draw_confirm_quit_modal(stdscr, max_y, max_x)

        stdscr.refresh()
        self._flush_artwork(stdscr)

    @staticmethod
    def _album_level_artist(track: AudioTrack) -> str:
        aart = track.tags.get("albumartist", "").strip()
        if aart:
            cleaned = " ".join(aart.lower().split())
            return "" if cleaned == "unknown artist" else cleaned
        art = track.tags.get("artist", "").strip()
        cleaned = " ".join(art.lower().split())
        return "" if cleaned == "unknown artist" else cleaned

    def album_mismatch_reasons(self) -> List[str]:
        if len(self.all_tracks) <= 1:
            return []
        reasons: List[str] = []
        albums = {t.tags.get("album", "").strip() for t in self.all_tracks}
        if len(albums) > 1:
            reasons.append("album")
        eff_artists = {self._album_level_artist(t) for t in self.all_tracks}
        if len(eff_artists) > 1:
            reasons.append("effective artist")
        compilations = {t.is_compilation for t in self.all_tracks}
        if len(compilations) > 1:
            reasons.append("compilation")
        return reasons

    def minority_mismatch_paths(self) -> set[Path]:
        if len(self.all_tracks) <= 1:
            return set()
        flagged: set[Path] = set()
        extractors = (
            lambda t: t.tags.get("album", "").strip(),
            self._album_level_artist,
            lambda t: t.is_compilation,
        )
        for fn in extractors:
            vals = [fn(t) for t in self.all_tracks]
            counts = Counter(vals)
            if len(counts) > 1:
                min_count = min(counts.values())
                for t, v in zip(self.all_tracks, vals):
                    if counts[v] == min_count:
                        flagged.add(t.path)
        return flagged

    def _draw_header(self, stdscr: curses.window, max_x: int) -> None:
        total = len(self.all_tracks)
        incomplete = sum(1 for t in self.all_tracks if t.is_incomplete)
        dirty = len(self.dirty_tracks())

        title = f" {self.root_label} "
        safe_addstr(stdscr, 0, 0, title, curses.A_BOLD)

        mismatches = self.album_mismatch_reasons()
        title_end = str_width(title)
        if mismatches:
            safe_addstr(stdscr, 0, title_end, " ⚠️", curses.color_pair(self.C_YELLOW) | curses.A_BOLD)
            warn_rest = f" album mismatch: {', '.join(mismatches)} "
            safe_addstr(stdscr, 0, title_end + 3, warn_rest, curses.color_pair(self.C_VIOLET) | curses.A_BOLD)
            title_end += 3 + str_width(warn_rest)

        stats_parts: List[str] = [f"{total} files"]
        if incomplete:
            stats_parts.append(f"{incomplete} incomplete")
        if dirty:
            stats_parts.append(f"{dirty} unsaved")
        if self.selected_paths:
            stats_parts.append(f"{len(self.selected_paths)} selected")
        if self.only_incomplete:
            stats_parts.append("FILTER: incomplete")
        if self.search_query:
            stats_parts.append(f"/{self.search_query}")

        stats = " │ ".join(stats_parts) + " "
        start_x = max(title_end + 1, max_x - str_width(stats))
        if start_x < max_x:
            attr = curses.color_pair(self.C_VIOLET) if (incomplete or dirty) else curses.A_NORMAL
            safe_addstr(stdscr, 0, start_x, stats, attr)

    def _draw_box(
        self,
        win: curses.window,
        y: int,
        x: int,
        h: int,
        w: int,
        title: str,
        focused: bool,
    ) -> None:
        if h < 2 or w < 4:
            return
        border_attr = curses.A_BOLD if focused else curses.A_NORMAL
        # Top border
        inner_w = w - 2
        if title:
            title_str = truncate_width(f" {title} ", max(0, inner_w - 2))
            rem = max(0, inner_w - str_width(title_str) - 1)
            top_line = "╭─" + title_str + ("─" * rem) + "╮"
        else:
            top_line = "╭" + ("─" * inner_w) + "╮"
        safe_addstr(win, y, x, top_line, border_attr)

        # Sides
        for row in range(y + 1, y + h - 1):
            safe_addstr(win, row, x, "│", border_attr)
            safe_addstr(win, row, x + w - 1, "│", border_attr)

        # Bottom border
        bot_line = "╰" + ("─" * inner_w) + "╯"
        safe_addstr(win, y + h - 1, x, bot_line, border_attr)

    def _draw_left_pane(
        self, win: curses.window, y: int, x: int, h: int, w: int
    ) -> None:
        vis = self.clamp_song_idx()
        self._draw_box(win, y, x, h, w, "", focused=(self.focus == "left"))

        list_h = h - 2
        inner_w = w - 2
        if list_h <= 0 or inner_w <= 2:
            return

        if not vis:
            msg = "No matching audio files" if self.all_tracks else "No audio files in directory"
            safe_addstr(win, y + 1, x + 2, truncate_width(msg, inner_w - 2), curses.A_NORMAL)
            return

        # Keep selection in viewport
        if self.song_idx < self.song_scroll:
            self.song_scroll = self.song_idx
        elif self.song_idx >= self.song_scroll + list_h:
            self.song_scroll = self.song_idx - list_h + 1

        mismatch_paths = self.minority_mismatch_paths()

        for row_i in range(list_h):
            idx = self.song_scroll + row_i
            if idx >= len(vis):
                break
            track = vis[idx]
            is_cur = idx == self.song_idx
            is_sel = track.path in self.selected_paths
            is_mismatch = track.path in mismatch_paths

            sel_prefix = "●" if is_sel else " "
            warn_prefix = "⚠️" if is_mismatch else "  "
            name_w = max(1, inner_w - 7)
            name_text = " " + pad_width(track.name, name_w) + " "

            if track.load_error:
                attr = curses.color_pair(self.C_VIOLET) | curses.A_BOLD
            elif track.is_incomplete:
                # Required: missing effective artist, track number, album, or title -> violet
                attr = curses.color_pair(self.C_VIOLET)
            else:
                attr = curses.A_NORMAL

            if is_cur:
                attr |= curses.A_REVERSE
                if self.focus == "left":
                    attr |= curses.A_BOLD
            elif is_sel:
                attr |= curses.A_BOLD

            sel_attr = curses.color_pair(self.C_VIOLET) | curses.A_BOLD if is_sel else curses.A_NORMAL
            warn_attr = curses.color_pair(self.C_YELLOW) | curses.A_BOLD if is_mismatch else curses.A_NORMAL
            safe_addstr(win, y + 1 + row_i, x + 1, sel_prefix, sel_attr)
            safe_addstr(win, y + 1 + row_i, x + 2, warn_prefix, warn_attr)
            safe_addstr(win, y + 1 + row_i, x + 4, name_text, attr)

            mark_x = x + 4 + name_w + 2
            if track.is_dirty:
                safe_addstr(win, y + 1 + row_i, mark_x, " *", curses.color_pair(self.C_VIOLET) | curses.A_BOLD)
            else:
                safe_addstr(win, y + 1 + row_i, mark_x, "  ", curses.A_NORMAL)

    def _draw_right_pane(
        self, win: curses.window, y: int, x: int, h: int, w: int
    ) -> None:
        targets = self.active_tracks()
        if not targets:
            self._draw_box(win, y, x, h, w, "", focused=(self.focus == "right"))
            safe_addstr(win, y + 2, x + 2, "Select a song to view and edit tags.", curses.A_NORMAL)
            return

        self._draw_box(win, y, x, h, w, "", focused=(self.focus == "right"))

        inner_x = x + 2
        inner_w = w - 4
        if inner_w < 20 or h < 8:
            return

        # Technical metadata / error / missing line
        cur_y = y + 1
        if len(targets) == 1:
            t = targets[0]
            info_bits = [t.fmt]
            if t.duration:
                info_bits.append(format_duration(t.duration))
            if t.bitrate:
                info_bits.append(f"{t.bitrate // 1000} kbps")
            if t.sample_rate:
                info_bits.append(f"{t.sample_rate / 1000:.1f} kHz")
            info_bits.append(format_size(t.file_size))
            info_str = " · ".join(info_bits)
            safe_addstr(win, cur_y, inner_x, truncate_width(info_str, inner_w), curses.A_NORMAL)

            if t.load_error:
                err_str = f"Error: {t.load_error}"
                safe_addstr(
                    win,
                    cur_y + 1,
                    inner_x,
                    truncate_width(err_str, inner_w),
                    curses.color_pair(self.C_VIOLET) | curses.A_BOLD,
                )
                table_top = cur_y + 2
            elif t.is_incomplete:
                miss_str = "Missing: " + ", ".join(t.missing_required)
                safe_addstr(
                    win,
                    cur_y + 1,
                    inner_x,
                    truncate_width(miss_str, inner_w),
                    curses.color_pair(self.C_VIOLET),
                )
                table_top = cur_y + 2
            else:
                table_top = cur_y + 1
        else:
            table_top = cur_y

        # Unicode Table for tags
        table_bottom = y + h - 2
        table_h = table_bottom - table_top + 1
        if table_h < 5:
            return

        rows = self.current_tag_rows(targets)
        self.tag_idx = max(0, min(self.tag_idx, len(rows) - 1))

        # Column widths inside Unicode table: │ <tag> │ <value> │
        max_label_w = max(12, max(str_width(r[1]) for r in rows))
        key_col_w = min(max_label_w + 5, max(17, inner_w // 3))
        val_col_w = inner_w - key_col_w - 3  # 3 vertical bars: │ ... │ ... │
        if val_col_w < 8:
            val_col_w = 8
            key_col_w = max(8, inner_w - val_col_w - 3)

        tbl_attr = curses.A_NORMAL
        # Top border of table
        safe_addstr(
            win,
            table_top,
            inner_x,
            "┌" + ("─" * key_col_w) + "┬" + ("─" * val_col_w) + "┐",
            tbl_attr,
        )
        # Header row
        hdr_y = table_top + 1
        safe_addstr(win, hdr_y, inner_x, "│", tbl_attr)
        safe_addstr(
            win,
            hdr_y,
            inner_x + 1,
            " " + pad_width("Tag", key_col_w - 2) + " ",
            curses.A_BOLD,
        )
        safe_addstr(win, hdr_y, inner_x + 1 + key_col_w, "│", tbl_attr)
        safe_addstr(
            win,
            hdr_y,
            inner_x + 2 + key_col_w,
            " " + pad_width("Value", val_col_w - 2) + " ",
            curses.A_BOLD,
        )
        safe_addstr(win, hdr_y, inner_x + 2 + key_col_w + val_col_w, "│", tbl_attr)

        # Header separator
        safe_addstr(
            win,
            table_top + 2,
            inner_x,
            "├" + ("─" * key_col_w) + "┼" + ("─" * val_col_w) + "┤",
            tbl_attr,
        )

        # Visible data rows
        data_top = table_top + 3
        data_h = table_bottom - data_top  # leave 1 row for bottom border
        if data_h <= 0:
            return

        if self.tag_idx < self.tag_scroll:
            self.tag_scroll = self.tag_idx
        elif self.tag_idx >= self.tag_scroll + data_h:
            self.tag_scroll = self.tag_idx - data_h + 1

        for row_i in range(data_h):
            r_idx = self.tag_scroll + row_i
            row_y = data_top + row_i
            if r_idx >= len(rows):
                # Empty table row padding or stop early
                break

            key, label, value, is_mixed = rows[r_idx]
            is_tag_dirty = any(
                t.tags.get(key, "") != t.original_tags.get(key, "") for t in targets
            )
            is_cur_row = r_idx == self.tag_idx
            is_editing_this = (
                self.mode == "edit_tag"
                and self.editor is not None
                and is_cur_row
            )

            safe_addstr(win, row_y, inner_x, "│", tbl_attr)

            # Tag name cell (colored; highlight includes 1 space padding, asterisk is 1 space after)
            label_w = max(1, key_col_w - 5)
            name_text = " " + pad_width(label, label_w) + " "
            name_attr = self.tag_name_attr(key, targets)
            if is_cur_row and self.focus == "right" and not is_editing_this:
                name_attr |= curses.A_REVERSE
            safe_addstr(win, row_y, inner_x + 1, name_text, name_attr)
            if is_tag_dirty:
                safe_addstr(
                    win,
                    row_y,
                    inner_x + 1 + label_w + 2,
                    " * ",
                    curses.color_pair(self.C_VIOLET) | curses.A_BOLD,
                )
            else:
                safe_addstr(win, row_y, inner_x + 1 + label_w + 2, "   ", curses.A_NORMAL)

            safe_addstr(win, row_y, inner_x + 1 + key_col_w, "│", tbl_attr)

            # Value cell
            val_x = inner_x + 2 + key_col_w
            inner_val_w = max(1, val_col_w - 2)

            if is_editing_this and self.editor is not None:
                vis_text, cur_off = self.editor.visible_slice(inner_val_w)
                if self.editor.selected and self.editor.text:
                    safe_addstr(win, row_y, val_x, " " + (" " * inner_val_w) + " ", curses.A_UNDERLINE)
                    safe_addstr(win, row_y, val_x + 1, vis_text, curses.A_REVERSE | curses.A_BOLD)
                else:
                    cell = " " + pad_width(vis_text, inner_val_w) + " "
                    safe_addstr(
                        win,
                        row_y,
                        val_x,
                        cell,
                        curses.A_UNDERLINE | curses.A_BOLD,
                    )
                self.edit_cursor_pos = (row_y, val_x + 1 + cur_off)
            else:
                val_text = " " + pad_width(value, inner_val_w) + " "
                val_attr = curses.A_NORMAL
                if is_cur_row and self.focus == "right":
                    val_attr = curses.A_REVERSE
                safe_addstr(win, row_y, val_x, val_text, val_attr)

            safe_addstr(win, row_y, val_x + val_col_w, "│", tbl_attr)

        # Bottom border right after last rendered row
        rendered_count = min(data_h, len(rows) - self.tag_scroll)
        bot_y = data_top + rendered_count
        safe_addstr(
            win,
            bot_y,
            inner_x,
            "└" + ("─" * key_col_w) + "┴" + ("─" * val_col_w) + "┘",
            tbl_attr,
        )

        # Embedded artwork section (below tag table when single track has artwork)
        if len(targets) == 1 and targets[0].art_bytes:
            t = targets[0]
            art_header_y = bot_y + 1
            pane_inner_bottom = y + h - 1  # exclusive (bottom box border is at y+h-1)
            avail_rows = pane_inner_bottom - art_header_y
            if avail_rows >= 1:
                prefix = "Artwork: "
                safe_addstr(
                    win,
                    art_header_y,
                    inner_x,
                    prefix,
                    curses.color_pair(self.C_VIOLET) | curses.A_BOLD,
                )
                info = t.art_info_label
                safe_addstr(
                    win,
                    art_header_y,
                    inner_x + str_width(prefix),
                    truncate_width(info, max(1, inner_w - str_width(prefix))),
                    curses.A_NORMAL,
                )
            if avail_rows >= 3:
                art_rows = min(14, avail_rows - 1)
                aspect = (
                    (t.art_width / t.art_height)
                    if (t.art_width > 0 and t.art_height > 0)
                    else 1.0
                )
                # Terminal cells are roughly 1:2 (width:height), so multiply width by 2
                art_cols = max(4, min(inner_w, int(round(art_rows * 2.0 * aspect))))
                if art_cols >= 4 and art_rows >= 2:
                    self._pending_art = (t, art_header_y + 1, inner_x, art_cols, art_rows)

    def _draw_footer(self, win: curses.window, max_y: int, max_x: int) -> None:
        hint_y = max_y - 1

        if self.mode == "search" and self.editor is not None:
            prompt = " / Search: "
            safe_addstr(win, hint_y, 0, prompt, curses.color_pair(self.C_VIOLET) | curses.A_BOLD)
            avail = max(1, max_x - str_width(prompt) - 1)
            vis_text, cur_off = self.editor.visible_slice(avail)
            safe_addstr(win, hint_y, str_width(prompt), vis_text, curses.A_NORMAL)
            self.edit_cursor_pos = (hint_y, str_width(prompt) + cur_off)
            return
        if self.mode == "add_tag" and self.editor is not None:
            prompt = " + New tag name: "
            safe_addstr(win, hint_y, 0, prompt, curses.color_pair(self.C_VIOLET) | curses.A_BOLD)
            avail = max(1, max_x - str_width(prompt) - 1)
            vis_text, cur_off = self.editor.visible_slice(avail)
            safe_addstr(win, hint_y, str_width(prompt), vis_text, curses.A_NORMAL)
            self.edit_cursor_pos = (hint_y, str_width(prompt) + cur_off)
            return
        if self.status_is_error and self.status_msg:
            safe_addstr(
                win,
                hint_y,
                1,
                truncate_width(self.status_msg, max_x - 2),
                curses.color_pair(self.C_VIOLET) | curses.A_BOLD,
            )
            return

        hint_items = [
            ("Space:", " select"),
            ("A:", " select all"),
            ("e:", " edit"),
            ("a:", " append"),
            ("s:", " save"),
            ("S:", " save all"),
            ("q:", " quit"),
            ("g:", " guess missing tags from filename (track, disc, artist, title)"),
        ]
        violet_attr = curses.color_pair(self.C_VIOLET)
        cur_x = 1
        limit_x = max_x - 1
        for idx, (key_part, desc_part) in enumerate(hint_items):
            if cur_x >= limit_x:
                break
            if idx > 0:
                sep = " │ "
                safe_addstr(win, hint_y, cur_x, truncate_width(sep, limit_x - cur_x, ellipsis=""), violet_attr)
                cur_x += str_width(sep)
                if cur_x >= limit_x:
                    break
            safe_addstr(win, hint_y, cur_x, truncate_width(key_part, limit_x - cur_x, ellipsis=""), violet_attr)
            cur_x += str_width(key_part)
            if cur_x >= limit_x:
                break
            safe_addstr(win, hint_y, cur_x, truncate_width(desc_part, limit_x - cur_x, ellipsis=""), curses.A_NORMAL)
            cur_x += str_width(desc_part)

    def _draw_help_modal(self, win: curses.window, max_y: int, max_x: int) -> None:
        lines = [
            "Navigation",
            "  ↑/↓ or j/k       Move up / down",
            "  ←/→, h/l, Tab    Switch between Songs and Tags panes",
            "  Home/End, g/G    Jump to top / bottom",
            "",
            "Editing",
            "  Enter / e        Edit selected tag (with text selected)",
            "  a                Append to selected tag (requires matching values)",
            "  Tab / Shift-Tab  While editing: apply & jump to next/prev tag",
            "  x / Delete       Clear selected tag value",
            "  +                Add a custom tag",
            "  u                Revert unsaved changes on active file(s)",
            "  s / Ctrl-S       Save active file(s) to disk",
            "  S                Save ALL modified files to disk",
            "",
            "Batch & Smart Tools",
            "  Space            Toggle select song for multi-file batch edit",
            "  A / c            Select all visible / Clear selection",
            "  #                Auto-number tracks (1/N..N/N) in current order",
            "  g                Guess missing tags from filename",
            "  =                Copy Artist → Album Artist (or vice versa)",
            "  /                Search / filter songs",
            "  f                Toggle showing only incomplete (yellow) songs",
            "",
            "Press any key to close this help.",
        ]
        box_w = min(max_x - 4, 68)
        box_h = min(max_y - 2, len(lines) + 2)
        top = max(0, (max_y - box_h) // 2)
        left = max(0, (max_x - box_w) // 2)

        # Clear modal area
        blank = " " * box_w
        for r in range(box_h):
            safe_addstr(win, top + r, left, blank, curses.A_NORMAL)

        self._draw_box(win, top, left, box_h, box_w, "Keybindings & Help", focused=True)
        for i, line in enumerate(lines[: box_h - 2]):
            attr = (
                curses.color_pair(self.C_VIOLET) | curses.A_BOLD
                if line and not line.startswith(" ")
                else curses.A_NORMAL
            )
            safe_addstr(win, top + 1 + i, left + 2, truncate_width(line, box_w - 4), attr)

    def _draw_confirm_quit_modal(self, win: curses.window, max_y: int, max_x: int) -> None:
        dirty_n = len(self.dirty_tracks())
        box_w = min(max_x - 4, 54)
        box_h = 6
        top = max(0, (max_y - box_h) // 2)
        left = max(0, (max_x - box_w) // 2)

        blank = " " * box_w
        for r in range(box_h):
            safe_addstr(win, top + r, left, blank, curses.A_NORMAL)

        self._draw_box(win, top, left, box_h, box_w, "Unsaved Changes", focused=True)
        msg1 = f"You have {dirty_n} file(s) with unsaved tag changes."
        msg2 = "(y) Save all & quit   (n) Discard & quit   (c) Cancel"
        safe_addstr(win, top + 2, left + 2, truncate_width(msg1, box_w - 4), curses.color_pair(self.C_VIOLET))
        safe_addstr(win, top + 3, left + 2, truncate_width(msg2, box_w - 4), curses.A_BOLD)

    # -- Input handling --

    def handle_input(self, ch: object) -> bool:
        if self.mode == "help":
            self.mode = None
            return True

        if self.mode == "confirm_quit":
            if ch in ("y", "Y"):
                if self._save_tracks(self.dirty_tracks()):
                    return False
                self.mode = None
            elif ch in ("n", "N"):
                return False
            elif ch in ("c", "C", "\x1b", 27):
                self.mode = None
            return True

        if self.mode == "edit_tag":
            return self._handle_edit_tag_input(ch)

        if self.mode == "search":
            return self._handle_search_input(ch)

        if self.mode == "add_tag":
            return self._handle_add_tag_input(ch)

        return self._handle_normal_input(ch)

    def _start_edit_current_tag(self, append: bool = False) -> None:
        targets = self.active_tracks()
        if not targets:
            return
        rows = self.current_tag_rows(targets)
        if not rows:
            return
        self.tag_idx = max(0, min(self.tag_idx, len(rows) - 1))
        key, label, value, is_mixed = rows[self.tag_idx]
        if append:
            if is_mixed:
                self.set_status(
                    f"Cannot append '{label}': values differ across selected files",
                    error=True,
                )
                return
            self.editing_tag_key = key
            self.editor = LineEditor(value, select_all=False)
        else:
            self.editing_tag_key = key
            self.editor = LineEditor("" if is_mixed else value, select_all=not is_mixed)
        self.mode = "edit_tag"
        self.focus = "right"

    def _commit_tag_edit(self) -> None:
        if self.editor is None or self.editing_tag_key is None:
            return
        targets = self.active_tracks()
        new_val = self.editor.text.strip()
        for t in targets:
            t.set_tag(self.editing_tag_key, new_val)
        label = STANDARD_TAG_LABELS.get(self.editing_tag_key, self.editing_tag_key)
        if len(targets) > 1:
            self.set_status(f"Updated '{label}' on {len(targets)} files (press s to save)")
        else:
            self.set_status(f"Updated '{label}' (press s to save)")

    def _handle_edit_tag_input(self, ch: object) -> bool:
        assert self.editor is not None
        res = self.editor.handle_key(ch)
        if res == "cancel":
            self.mode = None
            self.editor = None
            self.editing_tag_key = None
            self.set_status("Edit cancelled")
        elif res in ("commit", "commit_next", "commit_prev", "commit_up", "commit_down"):
            self._commit_tag_edit()
            self.mode = None
            self.editor = None
            self.editing_tag_key = None
            if res in ("commit_next", "commit_prev"):
                targets = self.active_tracks()
                rows = self.current_tag_rows(targets)
                if rows:
                    delta = 1 if res == "commit_next" else -1
                    self.tag_idx = (self.tag_idx + delta) % len(rows)
                    self._start_edit_current_tag()
            elif res in ("commit_up", "commit_down"):
                targets = self.active_tracks()
                rows = self.current_tag_rows(targets)
                if rows:
                    if res == "commit_up" and self.tag_idx > 0:
                        self.tag_idx -= 1
                    elif res == "commit_down" and self.tag_idx + 1 < len(rows):
                        self.tag_idx += 1
        return True

    def _handle_search_input(self, ch: object) -> bool:
        assert self.editor is not None
        res = self.editor.handle_key(ch)
        if res == "cancel":
            self.mode = None
            self.editor = None
        elif res in ("commit", "commit_next", "commit_prev"):
            self.search_query = self.editor.text.strip()
            self.song_idx = 0
            self.song_scroll = 0
            self.mode = None
            self.editor = None
            if self.search_query:
                self.set_status(f"Filtered by '{self.search_query}' (press / then Enter to clear)")
            else:
                self.set_status("Cleared search filter")
        else:
            # Live search update as user types
            self.search_query = self.editor.text.strip()
            self.song_idx = 0
            self.song_scroll = 0
        return True

    def _handle_add_tag_input(self, ch: object) -> bool:
        assert self.editor is not None
        res = self.editor.handle_key(ch)
        if res == "cancel":
            self.mode = None
            self.editor = None
        elif res in ("commit", "commit_next", "commit_prev"):
            new_key = self.editor.text.strip().lower()
            self.mode = None
            self.editor = None
            if new_key:
                targets = self.active_tracks()
                for t in targets:
                    if new_key not in t.tags:
                        t.tags[new_key] = ""
                rows = self.current_tag_rows(targets)
                for i, (k, _, _, _) in enumerate(rows):
                    if k == new_key:
                        self.tag_idx = i
                        break
                self._start_edit_current_tag()
        return True

    def _save_tracks(self, tracks: List[AudioTrack]) -> bool:
        if not tracks:
            self.set_status("No unsaved changes to write")
            return True
        saved = 0
        errors: List[str] = []
        for t in tracks:
            try:
                save_track(t)
                saved += 1
            except Exception as exc:
                errors.append(f"{t.name}: {exc}")
        if errors:
            self.set_status(f"Saved {saved}, error: {errors[0]}", error=True)
            return False
        self.set_status(f"Saved {saved} file(s) successfully")
        return True

    def _handle_normal_input(self, ch: object) -> bool:
        self.status_is_error = False
        # Quit
        if ch in ("q", "Q"):
            if self.dirty_tracks():
                self.mode = "confirm_quit"
                return True
            return False

        # Help
        if ch == "?":
            self.mode = "help"
            return True

        # Switch panes
        if ch == "\t":
            self.focus = "right" if self.focus == "left" else "left"
            return True
        if ch in (curses.KEY_LEFT, "h"):
            self.focus = "left"
            return True
        if ch in (curses.KEY_RIGHT, "l"):
            self.focus = "right"
            return True

        # Search & filter
        if ch == "/":
            self.editor = LineEditor(self.search_query)
            self.mode = "search"
            return True
        if ch == "f":
            self.only_incomplete = not self.only_incomplete
            self.song_idx = 0
            self.song_scroll = 0
            state = "ON (incomplete only)" if self.only_incomplete else "OFF (all songs)"
            self.set_status(f"Incomplete filter: {state}")
            return True

        # Save
        if ch in ("s", "\x13"):  # s or Ctrl-S: only currently selected file(s)
            targets = [t for t in self.active_tracks() if t.is_dirty]
            self._save_tracks(targets)
            return True
        if ch == "S":
            self._save_tracks(self.dirty_tracks())
            return True

        # Revert
        if ch == "u":
            targets = [t for t in self.active_tracks() if t.is_dirty]
            if not targets:
                self.set_status("No unsaved changes on active file(s)")
            else:
                for t in targets:
                    t.revert()
                self.set_status(f"Reverted changes on {len(targets)} file(s)")
            return True

        # Multi-select
        if ch == " ":
            cur = self.current_song()
            if cur is not None:
                if cur.path in self.selected_paths:
                    self.selected_paths.remove(cur.path)
                else:
                    self.selected_paths.add(cur.path)
                vis = self.visible_tracks()
                if self.song_idx + 1 < len(vis):
                    self.song_idx += 1
                self.set_status(f"{len(self.selected_paths)} file(s) selected")
            return True
        if ch == "A":
            vis = self.visible_tracks()
            self.selected_paths = {t.path for t in vis}
            self.set_status(f"Selected all {len(self.selected_paths)} visible file(s)")
            return True
        if ch == "c":
            self.selected_paths.clear()
            self.set_status("Cleared selection")
            return True

        # Smart tools
        if ch == "#":
            vis = self.visible_tracks()
            targets = (
                [t for t in vis if t.path in self.selected_paths]
                if self.selected_paths
                else vis
            )
            if targets:
                total = len(targets)
                for idx, t in enumerate(targets, start=1):
                    t.set_tag("tracknumber", str(idx))
                    t.set_tag("tracktotal", str(total))
                self.set_status(f"Auto-numbered {total} track(s) (press S to save all)")
            return True

        if ch == "g":
            targets = self.active_tracks()
            updated = 0
            for t in targets:
                guessed = guess_tags_from_filename(t.path)
                changed = False
                for k, v in guessed.items():
                    if not t.tags.get(k, "").strip() and v:
                        t.set_tag(k, v)
                        changed = True
                if changed:
                    updated += 1
            if updated:
                self.set_status(f"Guessed missing tags for {updated} file(s) (press s to save)")
            else:
                self.set_status("No missing tags could be guessed from filename")
            return True

        if ch == "=":
            targets = self.active_tracks()
            updated = 0
            for t in targets:
                art = t.tags.get("artist", "").strip()
                aart = t.tags.get("albumartist", "").strip()
                if art and not aart:
                    t.set_tag("albumartist", art)
                    updated += 1
                elif aart and not art:
                    t.set_tag("artist", aart)
                    updated += 1
            if updated:
                self.set_status(f"Synced Artist ↔ Album Artist on {updated} file(s)")
            else:
                self.set_status("Artist and Album Artist already set (or both empty)")
            return True

        if ch == "a":
            self._start_edit_current_tag(append=True)
            return True

        if ch == "+":
            if self.active_tracks():
                self.editor = LineEditor("")
                self.mode = "add_tag"
            return True

        # Navigation within focused pane
        if self.focus == "left":
            vis = self.visible_tracks()
            if ch in (curses.KEY_UP, "k"):
                if self.song_idx > 0:
                    self.song_idx -= 1
            elif ch in (curses.KEY_DOWN, "j"):
                if self.song_idx + 1 < len(vis):
                    self.song_idx += 1
            elif ch == curses.KEY_PPAGE:
                self.song_idx = max(0, self.song_idx - 10)
            elif ch == curses.KEY_NPAGE:
                self.song_idx = min(max(0, len(vis) - 1), self.song_idx + 10)
            elif ch == curses.KEY_HOME:
                self.song_idx = 0
            elif ch in (curses.KEY_END, "G"):
                self.song_idx = max(0, len(vis) - 1)
            elif ch in ("\n", "\r", curses.KEY_ENTER, "e"):
                if vis:
                    self.focus = "right"
                    # Jump cursor to first missing required tag if any
                    targets = self.active_tracks()
                    if len(targets) == 1 and targets[0].missing_required:
                        first_miss = targets[0].missing_required[0]
                        rows = self.current_tag_rows(targets)
                        for i, (k, _, _, _) in enumerate(rows):
                            if k == first_miss:
                                self.tag_idx = i
                                break
                    if ch == "e":
                        self._start_edit_current_tag()
        else:
            targets = self.active_tracks()
            rows = self.current_tag_rows(targets) if targets else []
            if ch in (curses.KEY_UP, "k"):
                if self.tag_idx > 0:
                    self.tag_idx -= 1
            elif ch in (curses.KEY_DOWN, "j"):
                if self.tag_idx + 1 < len(rows):
                    self.tag_idx += 1
            elif ch == curses.KEY_HOME:
                self.tag_idx = 0
            elif ch in (curses.KEY_END, "G"):
                self.tag_idx = max(0, len(rows) - 1)
            elif ch in ("\n", "\r", curses.KEY_ENTER, "e"):
                self._start_edit_current_tag()
            elif ch in ("x", curses.KEY_DC, curses.KEY_BACKSPACE, "\x7f"):
                if rows and targets:
                    key, label, _, _ = rows[self.tag_idx]
                    for t in targets:
                        t.set_tag(key, "")
                    self.set_status(f"Cleared '{label}' (press s to save)")

        return True


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tagedit",
        description="Terminal audio tag editor for MP3, M4A, Opus, Ogg, and FLAC.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to open (defaults to current directory).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Scan directories recursively.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    files = discover_audio_files(args.paths, recursive=args.recursive)
    tracks = [load_track(p) for p in files]

    if not args.paths:
        root_label = Path.cwd().name or str(Path.cwd())
    elif len(args.paths) == 1:
        root_label = str(Path(args.paths[0]).resolve().name or args.paths[0])
    else:
        root_label = f"{len(args.paths)} paths"

    locale.setlocale(locale.LC_ALL, "")
    app = TagEditorApp(tracks, root_label=root_label)
    # Work around macOS system ncurses 5.7 bug with xterm-kitty's `rep` capability
    # dropping repeated multibyte UTF-8 box-drawing characters (`─`).
    if sys.platform == "darwin" and os.environ.get("TERM") == "xterm-kitty":
        os.environ["TERM"] = "xterm-256color"
    curses.wrapper(app.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
