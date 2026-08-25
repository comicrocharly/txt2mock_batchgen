#!/usr/bin/env python3
"""
txt2mock_batchgen — Mass generation of mockup images via the ComfyUI API.

Reads a JSON file of prompts (see llm_instructions.md) and, for every object,
generates one image using a ComfyUI workflow exported in API format
(included: zimage-turbo-api.json).

Items file format (default: items.json next to this script):
  {
    "items": [
      {"id": 51, "name": "4-Cup Aluminum Moka Pot", "prompt": "Professional product photography of ..."},
      {"id": 52, "name": "Coffee Filter 62", "prompt": "..."}
    ]
  }
- "id":     primary key of the DB row (used in the output file name: <id>-<slug>.png)
- "name":   object name (slugged for the file name)
- "prompt": image generation prompt

The script starts ComfyUI automatically if it is not already running (local
host) and shuts it down at the end to free memory (unless --keep-comfyui).

Batch per entity: with --batch N, every object gets N images and EVERY
single generation rolls its own seed (fully random, or S, S+1, S+2, ...
when --seed S is given), so no two images in a batch look the same.

Usage:
  python3 generate_comfyui.py                        # generate everything in items.json
  python3 generate_comfyui.py --items other.json     # alternative items file
  python3 generate_comfyui.py --ids 61-72           # only these primary keys
  python3 generate_comfyui.py --only 1,5,7          # positions in the file (1-based)
  python3 generate_comfyui.py --batch 3             # 3 images per entity (seed rolled per generation)
  python3 generate_comfyui.py --seed 42             # seeds 42, 43, 44, ... (reproducible)
  python3 generate_comfyui.py --width 768 --height 768
  python3 generate_comfyui.py --host http://127.0.0.1:8188
  python3 generate_comfyui.py --keep-comfyui        # do not shut ComfyUI down at the end
"""

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlencode, urlparse

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ITEMS = BASE_DIR / "items.json"
DEFAULT_WORKFLOW = BASE_DIR / "zimage-turbo-api.json"
DEFAULT_OUTDIR = BASE_DIR / "output"
DEFAULT_COMFYUI = Path.home() / "ComfyUI-Installs" / "Comfy_Env" / "ComfyUI"
COMFYUI_LOG = BASE_DIR / "comfyui-auto.log"


# ---------------------------------------------------------------- items.json

def load_items(path: Path):
    """Load the items file. Accepts an object {"items": [...]} or a bare list."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(
            f"Items file not found: {path}\n"
            "Generate items.json with an LLM following llm_instructions.md, "
            "or pass one with --items."
        )
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list) or not data:
        sys.exit(f"Invalid items format in {path} (expected: list of objects with id/name/prompt).")
    items = []
    for i, obj in enumerate(data, 1):
        if not isinstance(obj, dict) or not str(obj.get("prompt", "")).strip():
            print(f"Note: object {i} has no valid 'prompt', skipped.")
            continue
        items.append((
            obj.get("id"),                      # primary key (may be None)
            str(obj.get("name") or f"item-{i}").strip(),
            str(obj["prompt"]).strip(),
        ))
    return items


def _expand_numbers(spec):
    """'61-72,90,110' -> {61..72, 90, 110}. Ranges are inclusive."""
    nums = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            nums.update(range(int(a), int(b) + 1))
        else:
            nums.add(int(part))
    return nums


def select_items(items, only_spec=None, ids_spec=None):
    """Filter items.

    --ids:  primary keys of the DB rows (e.g. 61-72 -> ids 61..72).
    --only: 1-based positions in the file (1..N).
    """
    by_id = _expand_numbers(ids_spec) if ids_spec else None
    by_pos = _expand_numbers(only_spec) if only_spec else None
    if by_id is None and by_pos is None:
        return items
    if by_id is not None:
        known = {it[0] for it in items if it[0] is not None}
        missing = by_id - known
        if missing:
            print(f"Note: ids not present in the file, ignored: {sorted(missing)}")
    if by_id is not None and by_pos is not None:
        sel = [it for i, it in enumerate(items, 1) if (it[0] in by_id) or (i in by_pos)]
    elif by_id is not None:
        sel = [it for it in items if it[0] in by_id]
    else:
        sel = [it for i, it in enumerate(items, 1) if i in by_pos]
    if not sel:
        sys.exit("--only/--ids: no object matches.")
    return sel


# ------------------------------------------------------------------- HTTP / API

def _with_retry(fn, what, retries=6, pause=2.0):
    """Retry on transient connection errors
    (ComfyUI under GPU load sometimes resets active connections)."""
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(min(pause * (attempt + 1), 8))
    raise last


def http_json(url, data=None, method="GET", retries=6, pause=2.0):
    def _do():
        req = urllib.request.Request(url, method=method)
        req.add_header("Content-Type", "application/json")
        body = json.dumps(data).encode() if data is not None else None
        with urllib.request.urlopen(req, body) as r:
            return json.loads(r.read().decode())
    if retries <= 1:
        return _do()
    return _with_retry(_do, url, retries, pause)


# -------------------------------------------------------------------- workflow

def find_prompt_node(workflow):
    """Find the CLIPTextEncode node to use as the positive prompt."""
    candidates = {}
    for node_id, node in workflow.items():
        if node.get("class_type") == "CLIPTextEncode":
            txt = node.get("inputs", {}).get("text", "")
            candidates[node_id] = (txt is not None and len(txt.strip()) > 0)
    if not candidates:
        sys.exit("No CLIPTextEncode node found in the API workflow.")
    filled = [nid for nid, f in candidates.items() if f]
    choice = filled[0] if filled else next(iter(candidates))
    print(f"Using node {choice} (CLIPTextEncode) as the positive prompt.")
    return choice


def set_size(workflow, width, height):
    """Override the size on the EmptySD3LatentImage / EmptyLatentImage node."""
    for nid, node in workflow.items():
        ct = node.get("class_type", "")
        if ct in ("EmptySD3LatentImage", "EmptyLatentImage") and node.get("inputs"):
            node["inputs"]["width"] = width
            node["inputs"]["height"] = height
            return nid
    return None


def set_seed(workflow, seed):
    for nid, node in workflow.items():
        if node.get("class_type") == "KSampler" and "seed" in node.get("inputs", {}):
            node["inputs"]["seed"] = seed


# --------------------------------------------------------------- ComfyUI server

def comfyui_reachable(host):
    try:
        http_json(f"{host}/system_stats", retries=1)
        return True
    except Exception:
        return False


def start_comfyui(host, comfyui_dir):
    """Start ComfyUI in the background; returns the Popen (caller manages it)."""
    if not (comfyui_dir / "main.py").exists():
        sys.exit(
            f"ComfyUI not found in {comfyui_dir}\n"
            "Pass --comfyui-dir (path to a ComfyUI installation) or start ComfyUI "
            f"yourself on {host}."
        )
    python = comfyui_dir / ".venv" / "bin" / "python"
    if not python.exists():
        sys.exit(f"ComfyUI venv not found: {python}")
    port = urlparse(host).port or 8188
    COMFYUI_LOG.write_text("", encoding="utf-8")
    log_fh = open(COMFYUI_LOG, "ab")
    cmd = [str(python), "main.py", "--listen", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(cmd, cwd=comfyui_dir, stdout=log_fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    print(f"Starting ComfyUI (pid {proc.pid}) -> log: {COMFYUI_LOG}")
    deadline = time.time() + 240
    last_dots = 0
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"ComfyUI exited at startup (exit {proc.returncode}). "
                     f"Last log lines:\n{COMFYUI_LOG.read_text(encoding='utf-8', errors='replace')[-2000:]}")
        if comfyui_reachable(host):
            print("ComfyUI ready.")
            return proc
        if int(time.time()) // 15 != last_dots:
            last_dots = int(time.time()) // 15
            print("  waiting for ComfyUI...")
        time.sleep(2)
    proc.terminate()
    sys.exit("Timeout starting ComfyUI (240 s). Log: " + str(COMFYUI_LOG))


def stop_comfyui(proc):
    """Kill the ComfyUI process group (TERM, then KILL)."""
    if proc is None:
        return
    print("Shutting down ComfyUI to free memory...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    for _ in range(20):
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()
    print("ComfyUI stopped.")


# ------------------------------------------------------------------- generation

def wait_and_download(host, prompt_id, outdir, base_name):
    """Wait for generation to finish and download the produced images."""
    url = f"{host}/history/{prompt_id}"
    for _ in range(600):  # ~20 min (poll every 2 s, with retries on reset)
        time.sleep(2)
        try:
            h = http_json(url)
        except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError) as e:
            sys.exit(
                f"ComfyUI is no longer responding on {host} (crashed? OOM/VRAM?).\n"
                f"  Last error: {e}\n"
                f"  If the script started it: log at {COMFYUI_LOG}"
            )
        entry = h.get(prompt_id)
        if not entry:
            continue
        if entry.get("status", {}).get("status_str") == "error":
            sys.exit(f"ComfyUI error: {json.dumps(entry.get('status'), indent=2)}")
        outputs = entry.get("outputs", {})
        if outputs:
            saved = []
            for node_id, out in outputs.items():
                for img in out.get("images", []):
                    if img.get("type", "output") != "output":
                        continue
                    fname = img.get("name") or img.get("filename")
                    if not fname:
                        continue
                    q = urlencode({
                        "filename": fname,
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })
                    suffix = f"_{fname}" if len(out.get("images", [])) > 1 else ""
                    dest = outdir / f"{base_name}{suffix}.{Path(fname).suffix or '.png'}"
                    view_url = f"{host}/view?{q}"
                    _with_retry(lambda: urllib.request.urlretrieve(view_url, dest), str(dest))
                    saved.append(str(dest))
            return saved
    sys.exit("Timeout waiting for generation (20 min).")


_seed_counter = 0


def roll_seed(base_seed):
    """Roll a new 32-bit seed for a single generation.

    Without --seed: fully random on every call.
    With --seed S: S, S+1, S+2, ... (still a different seed per generation,
    reproducible)."""
    global _seed_counter
    if base_seed is None:
        return random.getrandbits(32)
    seed = (base_seed + _seed_counter) % (2**32)
    _seed_counter += 1
    return seed


def generate(host, template, prompt_node, items, args):
    """Batch per entity: for each item, generate --batch images, each with a
    freshly rolled seed -> POST /prompt -> download PNG."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    total = len(items) * args.batch
    done = 0
    for pos, (pk, name, prompt) in enumerate(items, 1):
        print(f"\n[{pos}/{len(items)}] {name}" + (f" (id {pk})" if pk is not None else ""))
        wf = json.loads(json.dumps(template))  # deep copy
        wf[str(prompt_node)]["inputs"]["text"] = prompt
        if args.width or args.height:
            w = args.width or wf_size_default(wf)
            h = args.height or w
            set_size(wf, w, h)

        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or f"item-{pos}"
        base = f"{pk}-{slug}" if pk is not None else slug
        for k in range(1, args.batch + 1):
            seed = roll_seed(args.seed)
            set_seed(wf, seed)
            out_base = base if args.batch == 1 else f"{base}_{k}"
            print(f"  gen {k}/{args.batch} seed={seed}")
            r = http_json(f"{host}/prompt", {"prompt": wf}, method="POST")
            prompt_id = r.get("prompt_id")
            if not prompt_id:
                sys.exit(f"Request rejected: {r}")
            saved = wait_and_download(host, prompt_id, args.outdir, out_base)
            done += 1
            print(f"  [{done}/{total}] -> {', '.join(saved)}")
    print(f"\nDone. {total} images in:", args.outdir)


def wf_size_default(wf):
    for node in wf.values():
        if node.get("class_type") in ("EmptySD3LatentImage", "EmptyLatentImage"):
            return node["inputs"].get("width", 1024)
    return 1024


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="txt2mock_batchgen: items.json -> ComfyUI API -> PNG")
    ap.add_argument("--host", default="http://127.0.0.1:8188")
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS,
                    help="JSON file with the prompts (default: items.json)")
    ap.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW,
                    help="ComfyUI workflow in API format (default: zimage-turbo-api.json)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--ids", type=str, default=None,
                    help="primary keys to generate (e.g. 61-72, 51,55)")
    ap.add_argument("--only", type=str, default=None,
                    help="positions in the file, 1-based (e.g. 1,4,7 or 11-60)")
    ap.add_argument("--seed", type=int, default=None,
                    help="base seed (default: fully random; with --seed S the seeds are S, S+1, S+2, ...)")
    ap.add_argument("--batch", type=int, default=1,
                    help="images per entity (default 1); every generation in the batch "
                         "gets a freshly rolled seed, files named <id>-<slug>_1..N.png")
    ap.add_argument("--width", type=int, default=None, help="override width (default: from the workflow)")
    ap.add_argument("--height", type=int, default=None, help="override height (default: from the workflow)")
    ap.add_argument("--comfyui-dir", type=Path, default=DEFAULT_COMFYUI,
                    help="ComfyUI installation (default: ~/ComfyUI-Installs/Comfy_Env/ComfyUI)")
    ap.add_argument("--keep-comfyui", action="store_true",
                    help="do not shut down the ComfyUI instance started by the script")
    args = ap.parse_args()

    # 0. validate the API workflow
    if not args.workflow.exists():
        sys.exit(
            f"API workflow not found: {args.workflow}\n"
            "In the ComfyUI UI: menu (bottom left) -> Save -> 'Workflow (API format)'."
        )
    template = json.loads(args.workflow.read_text(encoding="utf-8"))
    prompt_node = find_prompt_node(template)

    # 1. server: if it is not running and the host is local, the script starts it
    proc = None
    if comfyui_reachable(args.host):
        print(f"ComfyUI already running on {args.host} -> using it and NOT shutting it down at the end.")
    else:
        if urlparse(args.host).hostname not in ("127.0.0.1", "localhost"):
            sys.exit(f"ComfyUI not reachable on {args.host} (remote host: start it yourself and retry).")
        proc = start_comfyui(args.host, args.comfyui_dir)

    try:
        # 2. items from the JSON file
        items = load_items(args.items)
        print(f"{len(items)} objects found in {args.items}")
        items = select_items(items, args.only, args.ids)
        pks = [it[0] for it in items if it[0] is not None]
        print(f"{len(items)} to generate" + (f" (ids: {min(pks)}..{max(pks)})" if pks else ""))
        # 3. generation
        generate(args.host, template, prompt_node, items, args)
    finally:
        if proc is not None and not args.keep_comfyui:
            stop_comfyui(proc)


if __name__ == "__main__":
    main()
