#!/usr/bin/env python3
"""
txt2mock_batchgen — Mass generation of mockup images via the ComfyUI API.

Reads the prompts from a JSON items file (see llm_instructions.md) or from a
markdown prompts file, and for every object generates one (or N, with
--batch) images using a ComfyUI workflow exported in API format
(included: workflow.json).

The script is model-agnostic: it works with ANY text-to-image workflow
exported in API format. The positive-prompt node and the seed nodes are
detected automatically by tracing the workflow graph (sampler -> positive/
conditioning input -> encoding node with a 'text' input).

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

Alternative input: a markdown file with blocks like
  ## 1. Name (id 51)
  ```
  prompt text
  ```
(default: prompts-comfyui.md next to this script).

ComfyUI management: if it is already running (e.g. open in the browser) it is
used but NOT shut down at the end; if the script started it, it shuts it down
to free memory (unless --keep-comfyui).

Batch per entity: with --batch N, every object gets N images and EVERY
single generation rolls its own seed (fully random, or S, S+1, S+2, ...
when --seed S is given), so no two images in a batch look the same.

Usage:
  python3 generate_comfyui.py                        # everything in items.json
  python3 generate_comfyui.py --items other.json     # alternative items file
  python3 generate_comfyui.py --md prompts.md        # markdown prompts file
  python3 generate_comfyui.py --ids 61-72           # only these primary keys
  python3 generate_comfyui.py --ids 51,55,80        # single primary keys
  python3 generate_comfyui.py --only 1,5,7          # positions in the file (1-based)
  python3 generate_comfyui.py --batch 3             # 3 images per entity (seed rolled per generation)
  python3 generate_comfyui.py --seed 42             # seeds 42, 43, 44, ... (reproducible)
  python3 generate_comfyui.py --width 768 --height 768
  python3 generate_comfyui.py --host http://127.0.0.1:8188
  python3 generate_comfyui.py -i ComfyUI            # use instance "ComfyUI" (default)
  python3 generate_comfyui.py --models-dir ~/ComfyUI-Shared/models
  python3 generate_comfyui.py --check               # start the instance, verify the models, exit (no generation)
  python3 generate_comfyui.py --keep-comfyui        # do not shut ComfyUI down at the end
"""

import argparse
import json
from collections import deque
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlencode, urlparse

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ITEMS = BASE_DIR / "items.json"
DEFAULT_MD = BASE_DIR / "prompts-comfyui.md"
DEFAULT_WORKFLOW = BASE_DIR / "workflow.json"
DEFAULT_OUTDIR = BASE_DIR / "output"
COMFYUI_INSTALLS_ROOT = Path.home() / "ComfyUI-Installs"
DEFAULT_INSTANCE = "ComfyUI"
DEFAULT_MODELS_SHARED = Path.home() / "ComfyUI-Shared" / "models"  # auto-used when present
COMFYUI_LOG = BASE_DIR / "comfyui-auto.log"


# ------------------------------------------------------------- input files
# A normalized item is a tuple: (position, pk, name, prompt)
#   position: 1-based index in the input file
#   pk:       primary key of the DB row (may be None)

def load_items_json(path: Path):
    """Load the JSON items file. Accepts an object {"items": [...]} or a bare list."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(
            f"Items file not found: {path}\n"
            "Generate items.json with an LLM following llm_instructions.md, "
            "or pass one with --items (or a markdown file with --md)."
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
            i,
            obj.get("id"),                      # primary key (may be None)
            str(obj.get("name") or f"item-{i}").strip(),
            str(obj["prompt"]).strip(),
        ))
    if not items:
        sys.exit(f"No valid items (with a 'prompt') found in {path}.")
    return items


def load_items_md(path: Path):
    """Extract (position, pk, name, prompt) from the blocks of the markdown file.

    Each block: "## N. Name (id DB)" ... followed by a ``` block with the prompt."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        sys.exit(f"Markdown prompts file not found: {path}\nPass one with --md or a JSON file with --items.")
    items = []
    for m in re.finditer(
        r"^##\s+(\d+)\.\s+(.+?)(?:\s*\(id\s+(\d+)\))?\s*$\n(?:.*?\n)?\s*```\n(.*?)\n```",
        text,
        re.M | re.S,
    ):
        items.append((
            int(m.group(1)),
            int(m.group(3)) if m.group(3) else None,
            m.group(2).strip(),
            m.group(4).strip(),
        ))
    if not items:
        sys.exit(f"No prompt blocks found in {path}.")
    return items


def load_items(items_path=None, md_path=None):
    """Resolve the input file: explicit flags win; otherwise items.json, then prompts-comfyui.md."""
    if items_path is not None:
        return load_items_json(items_path)
    if md_path is not None:
        return load_items_md(md_path)
    if DEFAULT_ITEMS.exists():
        return load_items_json(DEFAULT_ITEMS)
    if DEFAULT_MD.exists():
        return load_items_md(DEFAULT_MD)
    sys.exit(
        f"No input file found: neither {DEFAULT_ITEMS} nor {DEFAULT_MD}.\n"
        "Generate items.json with an LLM following llm_instructions.md, or pass --items/--md."
    )


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
    If both are passed, the union of the two sets.
    """
    by_id = _expand_numbers(ids_spec) if ids_spec else None
    by_pos = _expand_numbers(only_spec) if only_spec else None
    if by_id is None and by_pos is None:
        return items
    if by_id is not None:
        known = {it[1] for it in items if it[1] is not None}
        missing = by_id - known
        if missing:
            print(f"Note: ids not present in the file, ignored: {sorted(missing)}")
    if by_id is not None and by_pos is not None:
        sel = [it for it in items if (it[1] in by_id) or (it[0] in by_pos)]
    elif by_id is not None:
        sel = [it for it in items if it[1] in by_id]
    else:
        sel = [it for it in items if it[0] in by_pos]
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


class PromptRejected(Exception):
    """ComfyUI answered with an HTTP error (400/500): permanent, do not retry."""


def http_json(url, data=None, method="GET", retries=6, pause=2.0):
    def _do():
        req = urllib.request.Request(url, method=method)
        req.add_header("Content-Type", "application/json")
        body = json.dumps(data).encode() if data is not None else None
        try:
            with urllib.request.urlopen(req, body) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode(errors="replace")
            except Exception:
                detail = str(e)
            raise PromptRejected(f"HTTP {e.code}: {detail.strip()[:2000]}")
    if retries <= 1:
        return _do()
    return _with_retry(_do, url, retries, pause)


# -------------------------------------------------------------------- workflow

def _find_text_nodes(wf):
    """All nodes with a string 'text' input (CLIPTextEncode, TextEncodeQwenImage, ...)."""
    return [nid for nid, node in wf.items()
            if isinstance(node.get("inputs", {}).get("text"), str)]


def _build_reverse_links(wf):
    """Reverse graph: src_id -> [(dst_id, input_key)] for every [node, slot] link."""
    rev = {}
    for nid, node in wf.items():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and val[0] in wf:
                rev.setdefault(val[0], []).append((nid, key))
    return rev


def _first_upstream_text(rev, start_ids, text_ids, max_depth=30):
    """BFS upstream from start_ids until reaching a text node."""
    seen = set(start_ids)
    dq = deque((s, 0) for s in start_ids)
    while dq:
        nid, depth = dq.popleft()
        if nid in text_ids:
            return nid
        if depth >= max_depth:
            continue
        for src, _key in rev.get(nid, []):
            if src not in seen:
                seen.add(src)
                dq.append((src, depth + 1))
    return None


def find_prompt_node(wf):
    """Locate the positive prompt text node, workflow-agnostic.

    Strategy: on samplers (KSampler*/SamplerCustom*/SamplerNode) exposing a
    'positive' or 'conditioning' input, trace the link upstream to the first
    node with a 'text' input. Fallback: workflow with a single text node;
    otherwise the first non-empty text node (with a warning)."""
    text_ids = _find_text_nodes(wf)
    if not text_ids:
        sys.exit("No node with a 'text' input in the workflow: not a text-to-image API workflow?")
    rev = _build_reverse_links(wf)
    for nid, node in wf.items():
        ct = node.get("class_type", "")
        if not (ct.startswith(("KSampler", "SamplerCustom")) or ct in ("SamplerNode", "SamplerNodeAdv")):
            continue
        for key in ("positive", "conditioning"):
            link = node.get("inputs", {}).get(key)
            if isinstance(link, list) and len(link) == 2 and link[0] in wf:
                found = _first_upstream_text(rev, {link[0]}, set(text_ids))
                if found:
                    print(f"Using node {found} ({wf[found].get('class_type')}) as the positive prompt "
                          f"(traced from {wf[nid].get('class_type')}.{key}).")
                    return found
    # Fallback (workflow without a recognized sampler)
    if len(text_ids) == 1:
        print(f"Using node {text_ids[0]} ({wf[text_ids[0]].get('class_type')}) as the positive prompt (only text node).")
        return text_ids[0]
    filled = [nid for nid in text_ids if wf[nid]["inputs"]["text"].strip()]
    if len(filled) == 1:
        print(f"Using node {filled[0]} ({wf[filled[0]].get('class_type')}) as the positive prompt (only filled text node).")
        return filled[0]
    choice = (filled or text_ids)[0]
    print(f"WARNING: multiple text nodes {sorted(text_ids)} and no identifiable sampler: "
          f"using {choice} (it might NOT be the positive prompt!)")
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


def find_seed_nodes(wf):
    """All nodes with a numeric 'seed' input (KSampler*, Seed, ...)."""
    return [nid for nid, node in wf.items()
            if isinstance(node.get("inputs", {}).get("seed"), int)]


def set_seed(workflow, seed):
    for nid in find_seed_nodes(workflow):
        workflow[nid]["inputs"]["seed"] = seed


# --------------------------------------------------------------- ComfyUI server

def comfyui_reachable(host):
    try:
        http_json(f"{host}/system_stats", retries=1)
        return True
    except Exception:
        return False


def write_model_paths_cfg(models_dir: Path):
    """Write an extra_model_paths yaml (nested format, same as Comfy Desktop)
    mapping each subfolder of models_dir to a ComfyUI model folder.
    Returns the temp file path."""
    lines = ["txt2mock:", f"  base_path: '{models_dir.resolve()}'"]
    for sub in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        lines.append(f"  {sub.name}: {sub.name}/")
    cfg = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    cfg.write("\n".join(lines) + "\n")
    cfg.close()
    return cfg.name


def read_instance_launch_args(instance_name, comfyui_dir):
    """Read the instance launch options (e.g. '--lowvram --cpu-vae') saved by
    ComfyUI Desktop in installations.json. Returns a string or None."""
    for entry in _desktop_instance_entries():
        if instance_name and entry.get("name") == instance_name:
            return (entry.get("launchArgs") or "").strip() or None
        install_path = entry.get("installPath")
        if install_path and (Path(install_path) / "ComfyUI").resolve() == comfyui_dir.resolve():
            return (entry.get("launchArgs") or "").strip() or None
    return None


def _desktop_instance_entries():
    path = Path.home() / ".local" / "share" / "comfyui-desktop-2" / "installations.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
        return entries if isinstance(entries, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def start_comfyui(host, comfyui_dir, models_dir=None, launch_args=None):
    """Start ComfyUI in the background; returns the Popen (caller manages it)."""
    if not (comfyui_dir / "main.py").exists():
        sys.exit(
            f"ComfyUI not found in {comfyui_dir}\n"
            "Pass --comfyui-dir (path to a ComfyUI installation) or -i <instance> "
            f"(looked for under {COMFYUI_INSTALLS_ROOT}/), or start ComfyUI "
            f"yourself on {host}."
        )
    python = comfyui_dir / ".venv" / "bin" / "python"
    if not python.exists():
        sys.exit(f"ComfyUI venv not found: {python}")
    port = urlparse(host).port or 8188
    COMFYUI_LOG.write_text("", encoding="utf-8")
    log_fh = open(COMFYUI_LOG, "ab")
    # instance startup options first (from ComfyUI Desktop), then our own args
    # (ours win in case of conflict, e.g. --port)
    instance_args = shlex.split(launch_args) if launch_args else []
    cmd = [str(python), "main.py"] + instance_args + ["--listen", "127.0.0.1", "--port", str(port)]
    if models_dir is not None:
        cmd += ["--extra-model-paths-config", write_model_paths_cfg(models_dir)]
        print(f"Models: extra-model-paths from {models_dir}")
    print("Startup args: " + " ".join(cmd[2:]))
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


def check_models(host, wf):
    """Verify the models referenced by the workflow (UNET/CLIP/VAE loaders) are
    available in the ComfyUI instance. No generation is queued."""
    loader_input = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name", "VAELoader": "vae_name"}
    needed = {}
    for node in wf.values():
        ct = node.get("class_type")
        if ct in loader_input:
            name = node.get("inputs", {}).get(loader_input[ct])
            if isinstance(name, str) and name:
                needed.setdefault(ct, []).append(name)
    if not needed:
        print("Check: workflow has no UNETLoader/CLIPLoader/VAELoader nodes -> nothing to verify.")
        return
    try:
        info = http_json(f"{host}/object_info")
    except Exception as e:
        sys.exit(f"Check failed: cannot read /object_info on {host}: {e}")
    ok = True
    for ct, names in needed.items():
        key = loader_input[ct]
        entry = info.get(ct, {}).get("input", {}).get("required", {}).get(key, [])
        avail = entry[0] if isinstance(entry, list) and entry and isinstance(entry[0], list) else entry
        avail = list(avail or [])
        for name in names:
            if name in avail:
                print(f"  OK      {ct}.{key}: {name}")
            else:
                ok = False
                print(f"  MISSING {ct}.{key}: {name} (available: {avail or 'none'})")
    if not ok:
        sys.exit(
            "Check failed: models not found in the ComfyUI instance.\n"
            "  Check the --models-dir folder / the instance model config, or the log: " + str(COMFYUI_LOG)
        )
    print("Check: all workflow models are available in the instance.")


def generate(host, template, prompt_node, items, args):
    """Batch per entity: for each item, generate --batch images, each with a
    freshly rolled seed -> POST /prompt -> download PNG."""
    args.outdir.mkdir(parents=True, exist_ok=True)
    total = len(items) * args.batch
    done = 0
    for pos, (num, pk, name, prompt) in enumerate(items, 1):
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
            try:
                r = http_json(f"{host}/prompt", {"prompt": wf}, method="POST")
            except PromptRejected as e:
                sys.exit(
                    f"ComfyUI rejected the prompt: {e}\n"
                    "  Usual cause: the workflow's models (UNET/CLIP/VAE) are not in the\n"
                    "  model folders of the ComfyUI installation. Point --models-dir at a\n"
                    f"  shared models folder (e.g. {DEFAULT_MODELS_SHARED}), or check the log: {COMFYUI_LOG}"
                )
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
    ap = argparse.ArgumentParser(description="txt2mock_batchgen: items.json / prompts.md -> ComfyUI API -> PNG")
    ap.add_argument("--host", default="http://127.0.0.1:8188")
    ap.add_argument("--items", type=Path, default=None,
                    help="JSON file with the prompts (default: items.json if present)")
    ap.add_argument("--md", type=Path, default=None,
                    help="markdown file with the prompts (default: prompts-comfyui.md if present)")
    ap.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW,
                    help="ComfyUI workflow in API format (default: workflow.json)")
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
    ap.add_argument("-i", "--instance", default=DEFAULT_INSTANCE,
                    help="ComfyUI instance name under ~/ComfyUI-Installs/ "
                         f"(default: {DEFAULT_INSTANCE}); the checkout is "
                         "~/ComfyUI-Installs/<instance>/ComfyUI")
    ap.add_argument("--comfyui-dir", type=Path, default=None,
                    help="full path to the ComfyUI checkout, overriding --instance")
    ap.add_argument("--models-dir", type=str, default=None,
                    help="shared models folder with ComfyUI model subfolders "
                         f"(default: {DEFAULT_MODELS_SHARED} if it exists); passed to ComfyUI via "
                         "--extra-model-paths-config when the script starts it. "
                         "Use an empty string to disable the auto-detected folder.")
    ap.add_argument("--check", action="store_true",
                    help="only: start the instance if needed, verify the workflow models are "
                         "available, and exit (no generation)")
    ap.add_argument("--keep-comfyui", action="store_true",
                    help="do not shut down the ComfyUI instance started by the script")
    args = ap.parse_args()

    comfyui_dir = args.comfyui_dir or COMFYUI_INSTALLS_ROOT / args.instance / "ComfyUI"
    print(f"Instance: {args.instance} ({comfyui_dir})")
    launch_args = read_instance_launch_args(args.instance, comfyui_dir)
    print(f"Startup options: {launch_args or '(none configured)'}")

    # models folder: explicit --models-dir wins ('' disables); otherwise the shared
    # models folder auto-detected (same folder the ComfyUI Desktop instance uses)
    if args.models_dir is None:
        models_dir = DEFAULT_MODELS_SHARED if DEFAULT_MODELS_SHARED.is_dir() else None
        if models_dir is not None:
            print(f"Models: auto-detected shared folder {models_dir} "
                  f"(override: --models-dir PATH, disable: --models-dir '')")
    else:
        models_dir = Path(args.models_dir) if args.models_dir.strip() else None
        if models_dir is not None and not models_dir.is_dir():
            sys.exit(f"--models-dir: not a folder: {args.models_dir}")

    # 0. validate the API workflow
    if not args.workflow.exists():
        sys.exit(
            f"API workflow not found: {args.workflow}\n"
            "In the ComfyUI UI: menu (bottom left) -> Save -> 'Workflow (API format)'."
        )
    template = json.loads(args.workflow.read_text(encoding="utf-8"))
    prompt_node = find_prompt_node(template)
    if not find_seed_nodes(template):
        print("WARNING: no node with a 'seed' input: --seed and per-generation seed rolls will be ignored.")

    # 1. server: if it is not running and the host is local, the script starts it
    proc = None
    if comfyui_reachable(args.host):
        print(f"ComfyUI already running on {args.host} -> using it and NOT shutting it down at the end.")
    else:
        if urlparse(args.host).hostname not in ("127.0.0.1", "localhost"):
            sys.exit(f"ComfyUI not reachable on {args.host} (remote host: start it yourself and retry).")
        proc = start_comfyui(args.host, comfyui_dir, models_dir, launch_args)

    try:
        if args.check:
            check_models(args.host, template)
            print("Check passed: ComfyUI instance up and running.")
            return
        # 2. items from the input file (JSON or markdown)
        items = load_items(args.items, args.md)
        print(f"{len(items)} objects found in the items file")
        items = select_items(items, args.only, args.ids)
        pks = [it[1] for it in items if it[1] is not None]
        print(f"{len(items)} to generate" + (f" (ids: {min(pks)}..{max(pks)})" if pks else ""))
        # 3. generation
        generate(args.host, template, prompt_node, items, args)
    finally:
        if proc is not None:
            crashed = proc.poll() is not None
            if not args.keep_comfyui:
                stop_comfyui(proc)
                if crashed:
                    print("Note: ComfyUI had already died during the generation.")
                    print(COMFYUI_LOG.read_text(encoding='utf-8', errors='replace')[-2000:])


if __name__ == "__main__":
    main()
