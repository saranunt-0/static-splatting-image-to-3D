"""Generates notebooks/single_image_to_3dgs_colab.ipynb.

Edit the cells here, then run:  python scripts/build_notebook.py
(Keeping the notebook as code makes diffs reviewable and the JSON always valid.)
"""

import json
from pathlib import Path

REPO_URL = "https://github.com/saranunt-0/static-splatting-image-to-3D"
OUT = Path(__file__).resolve().parents[1] / "notebooks" / "single_image_to_3dgs_colab.ipynb"

cells = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


md(r"""
# 🖼️ ➜ 🧊 One photo to 3D Gaussian Splatting ("static splatting")

Turn **a single photo** (e.g. a corridor) into a 3D Gaussian Splatting scene you can look around in —
for **small camera motion around the spot where the photo was taken** (a few % of the scene depth).

| Step | What happens | Tool |
|---|---|---|
| 1. Geometry | metric depth, camera field of view and normals from the photo | **MoGe-2** (Microsoft, MIT) · or Depth Anything V2 |
| 2. Lift | one surface-aligned Gaussian per pixel; depth edges cut cleanly | this repo |
| 3. Hidden layer | background behind every occluding edge + beyond the photo border | planar depth continuation + **LaMa** inpainting |
| 4. Refine | fit the Gaussians to the photo with the geometry held in place | pure-PyTorch 3DGS rasterizer |
| 5. Export | `.ply` (Blender / Unreal / Unity / SuperSplat) · `.html` viewer · preview videos | PlayCanvas splat-transform |

Optional: run **Apple SHARP** (feed-forward single-image 3DGS, Dec 2025) on the same photo to compare.
SHARP's weights are licensed for **non-commercial research only**.

**Use a GPU runtime** (*Runtime ▸ Change runtime type ▸ T4 GPU*). CPU works but is ~20× slower.
""")

md("## Step 1 · Install (≈2–3 min)")
code(r"""
#@title Clone the code and install dependencies { display-mode: "form" }
INSTALL_SHARP = False  #@param {type:"boolean"}
BRANCH = "main"  #@param {type:"string"}
import os, subprocess, sys
REPO = "/content/static-splatting-image-to-3D"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, '""" + REPO_URL + r"""', REPO], check=True)
os.chdir(REPO)
sys.path.insert(0, REPO)
args = [sys.executable, "-m", "ssplat.setup_env"] + (["--sharp"] if INSTALL_SHARP else [])
subprocess.run(args, check=True)
import torch
print("✅ ready —", "GPU: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "⚠️ CPU only (slow)")
""")

md("## Step 2 · Your photo")
code(r"""
#@title Upload a photo (or leave empty to use the URL below) { display-mode: "form" }
IMAGE_URL = ""  #@param {type:"string"}
from pathlib import Path
import urllib.request
Path("/content/input").mkdir(exist_ok=True)
if IMAGE_URL:
    IMAGE = Path("/content/input") / IMAGE_URL.split("?")[0].rsplit("/", 1)[-1]
    urllib.request.urlretrieve(IMAGE_URL, IMAGE)
else:
    from google.colab import files
    up = files.upload()
    name = next(iter(up))
    IMAGE = Path("/content/input") / name
    IMAGE.write_bytes(up[name])
from PIL import Image
from IPython.display import display
im = Image.open(IMAGE); im.thumbnail((640, 640)); display(im)
print(IMAGE)
""")

md(r"""
## Step 3 · Settings
* **MAX_BASELINE** — how far you plan to move the camera, as a fraction of the median scene depth.
  It sets how much hidden background is synthesised. 0.05–0.1 is the sweet spot for one photo.
* **FOV_DEG** — horizontal field of view; 0 = let MoGe-2 estimate it (recommended), else use EXIF / your value.
* **REFINE_STEPS** — single-view refinement (sharpens the photo view; geometry stays on the depth map).
""")
code(r"""
#@title Settings { display-mode: "form" }
PRESET = "quality"  #@param ["quality", "balanced", "fast"]
DEPTH_MODEL = "moge2"  #@param ["moge2", "dav2", "moge3"]
FOV_DEG = 0  #@param {type:"number"}
MAX_BASELINE = 0.08  #@param {type:"number"}
REFINE_STEPS = 500  #@param {type:"integer"}
MAX_LONG_EDGE = 1024  #@param {type:"integer"}
ORIENTATION = "3dgs"  #@param ["3dgs", "z_up", "y_up"]
FORMATS = "ply,html,spz,glb"  #@param {type:"string"}
WORK = Path("/content/work") / IMAGE.stem
SETTINGS = [f"depth.model={DEPTH_MODEL}", f"depth.fov_deg={FOV_DEG}", f"occlusion.max_baseline={MAX_BASELINE}",
            f"optim.steps={REFINE_STEPS}", f"lift.max_long_edge={MAX_LONG_EDGE}",
            f"export.orientation={ORIENTATION}", f"export.formats={FORMATS}"]
print(WORK, SETTINGS)
""")

md("## Step 4 · Run (≈1–3 min on a T4)")
code(r"""
#@title Reconstruct { display-mode: "form" }
from ssplat.config import from_preset
from ssplat.run import apply_overrides, run
cfg = apply_overrides(from_preset(PRESET), SETTINGS)
summary = run(cfg, IMAGE, WORK)
""")

md("## Step 5 · Results")
code(r"""
#@title Novel views (3×3 grid around the photo's viewpoint; centre = the photo) { display-mode: "form" }
from IPython.display import Image as IPImage, HTML, display
display(IPImage(filename=str(WORK / "preview_grid.png"), width=900))
""")
code(r"""
#@title Preview videos { display-mode: "form" }
import base64
for name, path in summary.get("videos", {}).items():
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    display(HTML(f'<p><b>{name}</b></p><video width=640 controls loop autoplay muted>'
                 f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'))
""")
code(r"""
#@title What the pipeline saw: depth, occlusion edges, hidden layer { display-mode: "form" }
for n in ["disparity.png", "occlusion_edges.png", "hidden_layer.png"]:
    p = WORK / "debug" / n
    if p.exists():
        print(n); display(IPImage(filename=str(p), width=640))
""")

md("## Step 6 · Download")
code(r"""
#@title Zip the export + videos and download { display-mode: "form" }
import shutil
from google.colab import files
zip_path = shutil.make_archive(str(WORK), "zip", root_dir=WORK, base_dir=".")
files.download(zip_path)
""")

md(r"""
## Optional · Compare with Apple SHARP
Requires `INSTALL_SHARP` in Step 1. Research-only licence (see the repo's docs/RESEARCH.md).
""")
code(r"""
#@title Run SHARP on the same photo { display-mode: "form" }
cfg_s = apply_overrides(from_preset(PRESET), ["method=sharp", f"lift.max_long_edge={MAX_LONG_EDGE}",
                                              f"export.formats={FORMATS}"])
summary_s = run(cfg_s, IMAGE, WORK.parent / (WORK.name + "_sharp"))
display(IPImage(filename=str(WORK.parent / (WORK.name + "_sharp") / "preview_grid.png"), width=900))
""")

md(r"""
### Opening the result
* **`scene_viewer.html`** — double-click, opens in any browser.
* **`scene.ply`** — [SuperSplat](https://superspl.at/editor) (drag & drop), Blender (KIRI 3DGS add-on),
  Unreal / Unity Gaussian-splatting plugins. The photo's camera is at the origin looking along +Z
  (`3dgs` orientation); pick `z_up` for Blender/Unreal or `y_up` for three.js/Unity.
* Stay near the original viewpoint: everything the photo did not see is synthesised.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    c["source"] = [line + "\n" for line in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {OUT}")
