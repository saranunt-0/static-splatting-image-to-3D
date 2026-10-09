# One photo → 3D Gaussian Splatting: survey, design decisions and tests (October 2026)

**Goal.** Build a 3D Gaussian Splatting (3DGS) scene from **one image** of a static interior (corridor),
viewable from viewpoints **near** the one the photo was taken from. Runs on Colab (GPU) and,
more slowly, on CPU.

**Original proposal.** (1) monocular depth model → initial point cloud instead of video + COLMAP;
(2) train 3DGS on the one image.

## TL;DR

| Question | Answer |
|---|---|
| Is "depth → point cloud → train 3DGS on the one image" sound? | **Partly.** Step 1 is the right backbone. Step 2 alone barely helps novel views: one photo carries no depth information, so training can only sharpen the input view. Left unconstrained, it degrades geometry (Gaussians slide along their viewing rays). The real failure mode is elsewhere: **holes behind every occluding edge and beyond the frame** as soon as the camera moves. |
| What this repository does instead | depth (MoGe-2) → **one surface-aligned Gaussian per pixel** (pixel-footprint covariance, depth edges cut by a parallax test) → **hidden background layer** behind occluding edges + **border extension** (planar depth continuation + LaMa inpainting) → **geometry-anchored single-view refinement** → standard `.ply`. |
| Depth model | **MoGe-2** (Microsoft, NeurIPS 2025, **MIT**): metric point map **+ camera FoV + normals**. Fallback: Depth Anything V2 (ONNX, relative depth, needs a FoV). MoGe-3 (2026) optional on CUDA. |
| Strongest alternative | **Apple SHARP** (Dec 2025): a network trained end-to-end for exactly this task (single photo → 3DGS for nearby views, <1 s). Likely higher quality, but **research-only licence**, and its weights could not be downloaded in this sandbox. Wired in as `--set method=sharp` for a direct comparison on Colab. |
| If you need to *walk down* the corridor | Different problem class: generative video/multi-view diffusion → 3DGS (NVIDIA **Lyra**, **GEN3C**, Stability **Stable Virtual Camera**, WonderWorld, World Labs Marble). Hallucinates unseen space; heavy GPU; mostly non-commercial licences. |

## 1. What one photo can and cannot give you

* **Recoverable:** colour of every visible surface point; its depth (only through a learned prior,
  i.e. a monocular depth model); the camera FoV (again a prior, or EXIF).
* **Not recoverable, must be synthesised:** everything behind occluders (door frames, pillars,
  furniture) and everything outside the frame. The amount grows with camera motion:
  two surfaces at disparities d₁ > d₂ separate by `f·b·(d₁ − d₂)` pixels for a sideways move `b`.
* **"No camera movement"** in the request means the *input* is a single static shot. The *viewer*
  must still move a little, otherwise 3D is pointless. Every method here is valid for motion of
  a few percent of the scene depth (head-motion parallax, "3D photo"). SHARP's authors scope
  their model the same way ("nearby views").

## 2. Approaches considered

| Family | Examples (year) | How it works | Fit for this task |
|---|---|---|---|
| Depth-lifted + optimise (original proposal) | many in-house pipelines; DepthSplat / FSGS-style depth priors | monocular depth → points → 3DGS training | Transparent and licence-friendly. Needs explicit occlusion handling and geometry anchoring (this repo adds both). |
| Layered depth "3D photo" | 3D Photo Inpainting (Shih et al., CVPR 2020), SLIDE (ICCV 2021) | layered depth image; background behind depth edges inpainted (colour + depth) | Same viewing range as ours. The idea is reused here as a Gaussian hidden layer. |
| Feed-forward single-image 3DGS | **SHARP** (Apple, 2025), Flash3D (2024), MLGS (ACM MM 2026), studentSplat (2026), diffusion "splatter images" (2025) | a network regresses Gaussians (SHARP: 2 depth layers, metric) in one pass | Best quality per second for nearby views. SHARP: <1 s; reported LPIPS −25–34 % vs the best prior model. Research-only licence. |
| Generative world models | **Lyra** (NVIDIA, ICLR 2026) & Lyra 2.0, **GEN3C** (CVPR 2025), **Stable Virtual Camera** (2025), ViewCrafter, FlashWorld, WonderWorld, Scene Splatter, ExScene | camera-controlled video diffusion generates many views → 3DGS (or distilled directly) | Needed for large motion (walking into the corridor). Content is hallucinated and can drift from the photo. 24–80 GB GPUs. |

**Decision.** For "static splatting" (one photo, small motion) the depth-lifted pipeline is
kept as the backbone, because it is simple, inspectable, permissively licensed and runs anywhere.
Two components are added, because analysis and the tests below show they matter more than
training: occlusion handling and geometry anchoring. SHARP is included as a switchable
baseline so the two can be compared on the same photo.

## 3. Depth / geometry model

What the lifting step needs: **depth with the right shape, and the camera focal length.**
A relative-depth model (affine-invariant *disparity*) has an unknown shift and no FoV, so
unprojecting it bends planes. Corridors, made of long planes, show this immediately.

| Model | Output | Licence | Notes |
|---|---|---|---|
| **MoGe-2** (Microsoft, NeurIPS 2025) | metric point map, depth, **FoV**, normals, sky mask | **MIT** (DINOv2 parts Apache-2.0) | Its paper reports the best average rank against UniDepth V2, Depth Pro, MASt3R, Depth Anything V1/V2, Metric3D and others on 10 benchmarks, including NYUv2, iBims-1 and HAMMER. Depth Pro has sharper boundaries on some sets. **Default here.** |
| MoGe-3 (2026 preprint, weights released) | as MoGe-2, finer detail (sparse volumetric refinement) | MIT | Needs FlexGEMM/Triton (CUDA). Optional `depth.model=moge3`. Self-reported gains over MoGe-2. |
| Depth Pro (Apple, 2024) | metric depth + focal | Apple research licence | sharp edges; non-commercial |
| Depth Anything 3 (ByteDance, Nov 2025) | any-view geometry, metric variant | mixed (Large/Giant CC-BY-NC; metric-large reported Apache-2.0) | Strongest as a *multi-view* model. Its 3DGS head is pose-conditioned, not monocular. |
| UniDepth V2 | metric depth + camera | CC-BY-NC | — |
| **Depth Anything V2** (2024) | relative disparity only | Small Apache-2.0, Base/Large CC-BY-NC | Offline / CPU fallback (`depth.model=dav2`, ONNX from a GitHub release). Needs a FoV (EXIF, or `depth.fov_deg`). |

Sources: MoGe-2 paper ([arXiv 2507.02546](https://arxiv.org/abs/2507.02546)) and
[repo](https://github.com/microsoft/MoGe); MoGe-3 ([arXiv 2607.17967](https://arxiv.org/abs/2607.17967));
Depth Anything 3 ([arXiv 2511.10647](https://arxiv.org/abs/2511.10647)). All comparisons are the
authors' own. No independent indoor-only benchmark was found.

## 4. Why not gsplat / the Inria rasterizer

One image means about one Gaussian per pixel (≤ 1 M) and a few hundred optimisation steps. The
fragment-based rasterizer in `ssplat/rasterize.py` is pure PyTorch:
* no CUDA compilation (that step dominated setup time in the video pipeline);
* runs on CPU, so it was developed and tested in a GPU-less sandbox;
* implements the same image model as the reference renderer: EWA projection, 0.3 px² dilation,
  α = min(0.99, o·G), α < 1/255 skipped, early stop at T < 1e-4. Exported `.ply` files therefore
  look the same in standard viewers.
* `tests/test_rasterize.py` checks it against a brute-force per-pixel implementation (exact
  match) and its gradients against finite differences.

Speed on this machine's 4 CPU cores: ~1 s per optimisation step for 512×384 (200 k Gaussians). A
GPU is 1–2 orders of magnitude faster.

## 5. The method in detail

1. **Geometry.** MoGe-2 → depth `D`, intrinsics `K`, sky/invalid mask (clamped far).
2. **Flying pixels.** Depth networks blur depth edges over a few pixels. Lifted to 3D, those pixels
   hang in mid-air and smear as streaks. In zones where the 3×3 disparity range forms an occlusion
   edge, disparity is snapped to the near or far side, decided on the locally averaged disparity
   (a spatially coherent decision, no dithering). `geometry.sharpen_depth_edges`.
3. **Occlusion edges from physics.** A neighbour pair is cut iff a camera move of
   `b = occlusion.max_baseline × median depth` would open a gap ≥ `edge_parallax_px` (1 px), plus
   a 3 % relative jump. Steep but smooth surfaces are not cut: a corridor floor near the vanishing
   point has tiny per-pixel disparity steps. `geometry.occlusion_edges`.
4. **Surface Gaussians.** For pixel `p`, take the 3D steps to its right and lower neighbours
   *on the same surface* (one-sided differences that never cross a cut): `J = [∂P/∂u, ∂P/∂v]`.
   Then `Σ = σ²JJᵀ + thin normal term`. This is the pixel's footprint on the surface. From the
   input camera every Gaussian covers exactly its pixel. From a new viewpoint neighbours still
   overlap, so walls and floors stay closed (no cracks, even at grazing angles), and cut edges
   separate cleanly (no rubber sheets). `geometry.footprint_gaussians`.
5. **Hidden layer + border** (`occlusion.py`):
   * *strip width* behind each foreground edge pixel = the gap it can open, `⌈f·b·Δd⌉` + margin;
   * *strip depth* = planar continuation of the **far side's** disparity from the nearest
     trusted background pixel. Disparity is affine in pixel coordinates on any plane, so walls,
     floors and door recesses continue exactly behind the occluder. Background slopes use
     normalised (mask-aware) smoothing so the foreground never leaks into them;
   * *border* = canvas padded by the parallax the motion can reveal (+ the look-at rotation),
     depth continued from the image border the same way;
   * *colour* = LaMa inpainting. The foreground next to each strip is masked too, so LaMa
     copies background, not foreground, texture.
6. **Single-view refinement** (`optimize.py`) of the visible layer only. Loss: L1 + 0.2 D-SSIM
   on the photo, + depth loss to `D`, + anchor loss on each Gaussian's distance along its ray,
   + a small flatness term. The hidden layer is frozen: the photo carries no information about it.
7. **Export**: standard 3DGS `.ply` (SH degree 0), camera at the origin; `.html`/`.spz`/`.glb`
   via PlayCanvas splat-transform. Preview grid and swing/circle/dolly videos.

## 6. Experiments

See §6 results below (synthetic corridor with exact ground truth, NYUv2 corridor with measured
depth, two real corridor photos).
