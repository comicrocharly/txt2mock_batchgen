# txt2mock_batchgen

**Massive** mockup image generation via the **ComfyUI API**, designed to give
a developer a lot of images in one shot (demos, UI tests, placeholder
catalogs).

The flow in 3 steps:

```
 DB table (id, name, description)
        │  1) llm_instructions.md
        ▼
 items.json  ──────────►  2) generate_comfyui.py
 (id, name, prompt)              │  (ComfyUI API, workflow included)
                                 ▼
                       output/<id>-<slug>.png
```

## Project contents

| File | What |
|------|------|
| `generate_comfyui.py` | The script: items.json -> ComfyUI API -> PNG |
| `workflow.json` | **Z-Image Turbo** workflow already exported in API format |
| `llm_instructions.md` | Instructions for an LLM: DB rows -> `items.json` |
| `example-items.json` | Ready-made example items file |
| `output/` | Where the PNGs end up |

## Prerequisites

1. **ComfyUI** installed (default: instance `ComfyUI`, i.e.
   `~/ComfyUI-Installs/ComfyUI/ComfyUI`; override with `-i <instance>` or
   `--comfyui-dir`), with the workflow's models in the right
   (the shared folder `~/ComfyUI-Shared/models` is auto-detected, see below).
   The instance **startup options** configured in the ComfyUI Desktop app
   (e.g. `--lowvram --cpu-vae`) are read from
   `~/.local/share/comfyui-desktop-2/installations.json` and echoed/applied
   when the script starts the instance (the script's own args win in case of
   conflict, e.g. `--port`).
   folders:
   - UNET: `unsloth/z_image_turbo-Q6_K.gguf`
   - CLIP: `qwen_3_4b.safetensors`
   - VAE: `ae.safetensors`
2. Python 3.9+ (stdlib only, **zero dependencies to install**).

If ComfyUI is not running and the host is local, the script **starts it
itself** and shuts it down at the end (use `--keep-comfyui` to keep it open).

## Usage

### Step 1 — create the prompts file

Give an LLM the contents of `llm_instructions.md` + the rows of your DB table
(`id`, `name`, `description`) and ask for the `items.json` file. Every item
has:

```json
{ "id": 51, "name": "4-Cup Aluminum Moka Pot", "prompt": "Professional product photography of ..." }
```

`id` is the primary key (used in the file name), `name` and `prompt` as above.

Alternative input: a **markdown** prompts file with blocks like
`## 1. Name (id 51)` + a ``` block with the prompt (pass it with `--md`;
if no `--items` is given, `items.json` is preferred, then `prompts-comfyui.md`).

### Step 2 — generate

```bash
cd txt2mock_batchgen

# quick check: only 2 items, without running everything
python3 generate_comfyui.py --items example-items.json --only 1,2

# generate everything in items.json (default: ./items.json, output in ./output)
python3 generate_comfyui.py

# only certain primary keys
python3 generate_comfyui.py --ids 61-72
python3 generate_comfyui.py --ids 51,55,80

# positions in the file (1-based)
python3 generate_comfyui.py --only 1-10

python3 generate_comfyui.py --md prompts-comfyui.md

# ComfyUI instance name under ~/ComfyUI-Installs/ (default: ComfyUI):
python3 generate_comfyui.py -i ComfyUI

# check only: start the instance if needed, verify the workflow models, exit
# (no generation) — useful to test the auto-start:
python3 generate_comfyui.py --check

# models live in a shared folder, not inside the ComfyUI install
# (auto-detected: ~/ComfyUI-Shared/models, see below):
python3 generate_comfyui.py --models-dir ~/ComfyUI-Shared/models
python3 generate_comfyui.py --models-dir ''   # disable the auto-detected folder
```

`--models-dir` (optional): a folder with ComfyUI model subfolders
(`diffusion_models`, `text_encoders`, `vae`, ...). When the script
starts ComfyUI it passes it via `--extra-model-paths-config` (nested yaml,
same format ComfyUI Desktop uses), so a fresh install with empty model
folders works with shared models. **Default: `~/ComfyUI-Shared/models`
when it exists** (the same folder the ComfyUI Desktop instance points at);
override with `--models-dir PATH`, disable with `--models-dir ''`.

### Troubleshooting

- **"Value not in list: unet_name: ... not in []"** (ComfyUI rejects the
  prompt): the workflow's models are not visible to the running ComfyUI
  instance. Either put/symlink them in the install's model folders, point
  `--models-dir` at a shared models folder, or run `--check` to test:
  it starts the instance and reports OK/MISSING per model, without
  generating anything.

### Batch per entity

`--batch N` makes the script generate N images **per entity**, and **every
single generation rolls its own seed** (fully random by default, or
`S, S+1, S+2, ...` when `--seed S` is given), so all images of a batch
differ from each other instead of being repeated with one shared seed:

```bash
# 3 images per entity, every generation with a freshly rolled seed
python3 generate_comfyui.py --batch 3

# reproducible batch: seeds 42, 43, 44, ... across generations
python3 generate_comfyui.py --batch 3 --seed 42
```

Files land in `output/` named `<id>-<slug>.png` (or `<slug>.png` without id),
e.g. `51-4-cup-aluminum-moka-pot.png`. With `--batch N > 1` each generation
of an entity is numbered: `51-4-cup-aluminum-moka-pot_1.png`, `_2.png`, `_3.png`.

## Want a different model/workflow?

1. Build your workflow in ComfyUI (it must contain a `CLIPTextEncode` node,
   a `KSampler` and an image-save node).
2. Save it with: bottom-left menu -> **Save -> "Workflow (API format)"**.
3. `python3 generate_comfyui.py --workflow my-workflow.json`

The script finds the prompt node (CLIPTextEncode) and the KSampler by itself.

## Notes

- Seed: **rolled per generation**. Default: fully random every generation;
  with `--seed S`: seeds are `S, S+1, S+2, ...` in generation order
  (reproducible).
- If ComfyUI was already open (e.g. in the browser), the script uses it and
  does not shut it down.
- Generation timeout: 20 min per item; auto-start log in `comfyui-auto.log`.
- The items file is **detachable**: the script knows nothing about the DB
  source, it works with any list of objects generated by the LLM.
