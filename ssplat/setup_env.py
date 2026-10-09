"""Install everything the pipeline needs (Colab or any Linux machine; GPU optional).

    python -m ssplat.setup_env            # core + MoGe-2 + splat-transform (default)
    python -m ssplat.setup_env --sharp    # also Apple SHARP (research-only licence)
    python -m ssplat.setup_env --no-moge --skip-node   # minimal: Depth Anything V2 (ONNX) + .ply only

No CUDA compilation is needed: the Gaussian rasterizer is pure PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from .export import NODE_DIR, SPLAT_TRANSFORM_VERSION

# Pinned upstream revisions (see docs/RESEARCH.md for why these models).
MOGE_REPO = "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"
UTILS3D_REPO = "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
SHARP_REPO = "git+https://github.com/apple/ml-sharp.git@aed6527499ef91cba3b54c18d49a870f25947190"
CORE = ["numpy", "scipy", "pillow", "opencv-python-headless", "plyfile", "imageio", "imageio-ffmpeg",
        "matplotlib", "huggingface_hub", "tqdm"]


def sh(cmd, **kw) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def pip(*args: str) -> None:
    sh([sys.executable, "-m", "pip", "install", "-q", *args])


def has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def install_node_and_splat_transform() -> str:
    node = shutil.which("node")
    if node:
        version = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        if int(version.lstrip("v").split(".")[0]) >= 22:
            sh(["npm", "install", "-g", "--loglevel=error", "--prefix", str(NODE_DIR),
                f"@playcanvas/splat-transform@{SPLAT_TRANSFORM_VERSION}"])
            return f"system node {version}"
    # Colab ships an older Node; splat-transform needs >= 22. Fetch the official build.
    shasums = urllib.request.urlopen("https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt").read().decode()
    tarball = re.search(r"(node-v22\.[\d.]+-linux-x64\.tar\.xz)", shasums).group(1)
    NODE_DIR.mkdir(parents=True, exist_ok=True)
    dst = NODE_DIR.parent / tarball
    if not dst.exists():
        urllib.request.urlretrieve(f"https://nodejs.org/dist/latest-v22.x/{tarball}", dst)
    with tarfile.open(dst) as tf:
        tf.extractall(NODE_DIR.parent)
    extracted = NODE_DIR.parent / tarball.replace(".tar.xz", "")
    if NODE_DIR.exists():
        shutil.rmtree(NODE_DIR)
    extracted.rename(NODE_DIR)
    env = {**os.environ, "PATH": f"{NODE_DIR / 'bin'}:{os.environ['PATH']}"}
    sh([str(NODE_DIR / "bin" / "npm"), "install", "-g", "--loglevel=error", "--prefix", str(NODE_DIR),
        f"@playcanvas/splat-transform@{SPLAT_TRANSFORM_VERSION}"], env=env)
    return tarball


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-moge", action="store_true", help="Skip MoGe (use depth.model=dav2)")
    ap.add_argument("--sharp", action="store_true", help="Install Apple SHARP (research-only licence)")
    ap.add_argument("--skip-node", action="store_true", help="Skip Node.js/splat-transform (.ply export only)")
    args = ap.parse_args(argv)
    report = {}
    try:
        import torch  # noqa: F401 - Colab already has a CUDA build; never replace it
    except ImportError:
        pip("torch", "torchvision")
    pip(*CORE, "onnxruntime-gpu" if has_cuda() else "onnxruntime")
    report["core"] = "ok"
    if not args.no_moge:
        # --no-deps: MoGe-3's extras (FlexGEMM sparse conv, gradio) are not needed for MoGe-2.
        pip("--no-deps", MOGE_REPO)
        pip(UTILS3D_REPO)
        report["moge"] = "ok (MoGe-2 ready; MoGe-3 additionally needs FlexGEMM on a CUDA GPU)"
    if args.sharp:
        pip(SHARP_REPO)
        report["sharp"] = "ok (Apple ML Research licence: non-commercial research only)"
    if not args.skip_node:
        try:
            report["node"] = install_node_and_splat_transform()
        except Exception as e:  # pragma: no cover - network dependent
            report["node"] = f"FAILED ({e}); only .ply export will be available"
    report["cuda"] = has_cuda()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
