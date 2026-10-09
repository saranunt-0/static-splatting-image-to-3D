#!/usr/bin/env bash
# Real-corridor test images used in docs/RESEARCH.md §6.4, fetched from their public research
# repositories (not redistributed here; check each source's licence before other uses), then
# the pipeline is run on each.
#
#   bash bench/real_corridors.sh [work_dir] [extra --set options...]
#
# NYUv2 #50 (basement corridor, with Kinect depth): Metric3D repo, data/nyu_demo
# Subway platform: Lotus repo, assets/in-the-wild_example/10.jpg
# House hallway (RealEstate10K-style frame): ViewCrafter repo, test/images_sparse/real1
set -euo pipefail
WORK=${1:-work/real}
shift || true
DATA=$WORK/data
mkdir -p "$DATA"
raw() { curl -sSL --fail -o "$2" "https://raw.githubusercontent.com/$1"; }
raw YvanYin/Metric3D/main/data/nyu_demo/rgb/rgb_00050.jpg "$DATA/nyu50.jpg"
raw YvanYin/Metric3D/main/data/nyu_demo/depth/sync_depth_00050.png "$DATA/nyu50_depth.png"
raw EnVision-Research/Lotus/main/assets/in-the-wild_example/10.jpg "$DATA/subway.jpg"
raw Drexubery/ViewCrafter/main/test/images_sparse/real1/31698000.png "$DATA/hallway.png"

# NYUv2 frames carry a white registration border: use the standard (Eigen) crop.
# FoV of the crop from the Kinect calibration (fx = 518.86 px, width 560 px) = 56.7 deg.
python - "$DATA" <<'EOF'
import sys, numpy as np, cv2
from scipy import ndimage
from PIL import Image
d = sys.argv[1]
img = np.asarray(Image.open(f"{d}/nyu50.jpg").convert("RGB"))[45:471, 41:601]
Image.fromarray(img).save(f"{d}/nyu50_crop.png")
z = cv2.imread(f"{d}/nyu50_depth.png", -1).astype(np.float32)[45:471, 41:601] / 1000.0
bad = ~((z > 0.1) & (z < 10))
_, (iy, ix) = ndimage.distance_transform_edt(bad, return_indices=True)
np.save(f"{d}/nyu50_depth_filled.npy", ndimage.median_filter(z[iy, ix], size=3).astype(np.float32))
EOF

COMMON=(--preset balanced --set lift.max_long_edge=640 --set optim.steps=200 --set render.frames=72 "$@")
python -m ssplat.run --input "$DATA/nyu50_crop.png" --work "$WORK/nyu" "${COMMON[@]}" --set depth.fov_deg=56.7
python -m ssplat.run --input "$DATA/nyu50_crop.png" --work "$WORK/nyu_measured_depth" "${COMMON[@]}" \
    --set depth.model=file --set depth.depth_file="$DATA/nyu50_depth_filled.npy" --set depth.fov_deg=56.7
python -m ssplat.run --input "$DATA/subway.jpg" --work "$WORK/subway" "${COMMON[@]}"
python -m ssplat.run --input "$DATA/hallway.png" --work "$WORK/hallway" "${COMMON[@]}"
