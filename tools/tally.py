"""Work out every number in the README from the run logs in logs/.

    uv run python tools/tally.py                 # all logs in logs/; writes media/numbers.json
    uv run python tools/tally.py <log folder>    # another folder of logs; writes numbers.json there

Models not in the published results get their own column, named by their model id.

Reads each Inspect log (.eval, a zip file) directly: the run settings from header.json and each run's score and time
from summaries.json. Scores are the share of the goal reached (0 to 1).
"""
from __future__ import annotations

import json
import statistics
import struct
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LEVELS = {  # Inspect task name -> (level, given block positions?)
    "sort_blocks": ("1", False),
    "stack_blocks": ("2", False),
    "level3_messy_sort": ("3", False),
    "level3_messy_sort_coords": ("3", True),
    "level4_stack_no_top_camera": ("4", False),
    "level4_stack_no_top_camera_coords": ("4", True),
    "level5_copy_the_picture": ("5", False),
    "level5_copy_the_picture_coords": ("5", True),
}
NAMES = {
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
    "gpt-6-luna": "GPT-6 Luna",
    "gpt-6.1-sol": "GPT-6.1 Sol",
}
ORDER = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "gpt-6-luna", "gpt-6.1-sol"]


def member(path: Path, name: str) -> bytes:
    """One file from inside a log. Inspect compresses logs with zstd, which Python's zipfile can't read before 3.14:
    this uses the zstandard package (installed by `uv sync`), or the `zstd` command if the package is missing."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(name)
        if info.compress_type != 93:
            return z.read(name)
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        head = f.read(30)
        n, e = struct.unpack("<HH", head[26:30])
        f.seek(info.header_offset + 30 + n + e)
        raw = f.read(info.compress_size)
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompressobj().decompress(raw)
    except ImportError:
        return subprocess.run(["zstd", "-dcq"], input=raw, capture_output=True, check=True).stdout


def column(model: str, effort: str | None, level: str) -> str:
    """Levels 3-5 ran Sonnet twice: once with default settings (Oct 9) and once with medium reasoning effort, like
    the OpenAI models (Oct 11). Both count in one Sonnet column, so its levels 3-5 cells average 6 runs."""
    return model


def main() -> None:
    runs = defaultdict(list)          # (column, level, positions given) -> [(score, seconds), ...]
    settings = defaultdict(set)
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs"
    for log in sorted(log_dir.glob("*.eval")):
        header = json.loads(member(log, "header.json"))
        ev = header["eval"]
        if ev["task"] not in LEVELS or header["status"] != "success":
            continue
        level, coords = LEVELS[ev["task"]]
        model = ev["model"].split("/")[-1]
        effort = (ev.get("model_generate_config") or {}).get("reasoning_effort")
        col = column(model, effort, level)
        settings[col].add(effort or "default")
        for s in json.loads(member(log, "summaries.json")):
            score = list(s["scores"].values())[0]["value"]
            runs[(col, level, coords)].append((score, s["total_time"]))

    out = {"columns": [], "levels_1_2": {}, "levels_3_5": {}, "seconds_per_run": {}}
    seen = {k[0] for k in runs}
    cols = [c for c in ORDER if c in seen] + sorted(seen - set(ORDER))
    for c in cols:
        out["columns"].append({"id": c, "name": NAMES.get(c.split()[0], c.split()[0]), "settings": sorted(settings[c])})
    for (c, level, coords), v in sorted(runs.items()):
        mean = sum(x[0] for x in v) / len(v)
        group = "levels_1_2" if level in "12" else "levels_3_5"
        key = level + (" positions" if coords else " pictures")
        out[group].setdefault(c, {})[key] = {"mean": mean, "runs": len(v), "scores": sorted(x[0] for x in v)}
        if not coords:
            out["seconds_per_run"].setdefault(c, {})[level] = statistics.median(x[1] for x in v)
    for c, d in out["levels_3_5"].items():
        for kind in ("pictures", "positions"):
            vals = [d[f"{lv} {kind}"]["mean"] for lv in "345" if f"{lv} {kind}" in d]
            if len(vals) == 3:
                d[f"average {kind}"] = sum(vals) / 3

    dest = log_dir / "numbers.json" if len(sys.argv) > 1 else ROOT / "media" / "numbers.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {dest}")

    pct = lambda x: f"{round(100 * (x['mean'] if isinstance(x, dict) else x)):>4}"
    print("column".ljust(26), "  L1  L2 |  L3   L4   L5  avg | with positions: L3  L4  L5  avg")
    for c in cols:
        a, b = out["levels_1_2"].get(c, {}), out["levels_3_5"].get(c, {})
        l12 = "".join(pct(a[k]) if k in a else "   -" for k in ("1 pictures", "2 pictures"))
        pic = " ".join(pct(b[k]) if k in b else "   -" for k in ("3 pictures", "4 pictures", "5 pictures",
                                                                           "average pictures"))
        pos = " ".join(pct(b[k]) if k in b else "   -" for k in ("3 positions", "4 positions", "5 positions",
                                                                           "average positions"))
        print(c.ljust(26), l12, "|", pic, "|", pos)
    print("\nmedian seconds per run (pictures only)")
    for c in cols:
        t = out["seconds_per_run"].get(c, {})
        print(c.ljust(26), " ".join(f"L{lv} {round(t[lv]):>4}" if lv in t else f"L{lv}    -" for lv in "12345"))


if __name__ == "__main__":
    main()
