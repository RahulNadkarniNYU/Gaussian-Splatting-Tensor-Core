#!/bin/bash
set -e  # stop on error (remove if you want it to continue)

DATA="/scratch/sr7463/BDML_project/datasets"
SCENE="bonsai"
CKPT="/scratch/sr7463/BDML_project/checkpoints_baseline"
OUTDIR="logs/${SCENE}"

mkdir -p "${OUTDIR}"

BASE_CMD=(
  python render.py
  -s "${DATA}/${SCENE}/"
  -m "${CKPT}/${SCENE}_train_gausssplat"
  --eval
)

echo "Running experiments for scene: ${SCENE}"

# 1. Baseline
"${BASE_CMD[@]}" \
  | tee "${OUTDIR}/baseline.log"

# 2. Opacity culling only
"${BASE_CMD[@]}" \
  --use_opacity_culling \
  --min_opacity 0.01 \
  | tee "${OUTDIR}/opacity.log"

# 3. Opacity + frustum
"${BASE_CMD[@]}" \
  --use_opacity_culling \
  --use_frustum_culling \
  | tee "${OUTDIR}/opacity_frustum.log"

# 4. Opacity + radius
"${BASE_CMD[@]}" \
  --use_opacity_culling \
  --use_radius_culling \
  | tee "${OUTDIR}/opacity_radius.log"

# 5. Opacity + LOD
"${BASE_CMD[@]}" \
  --use_opacity_culling \
  --use_lod_culling \
  | tee "${OUTDIR}/opacity_lod.log"

# 6. All culling
"${BASE_CMD[@]}" \
  --use_opacity_culling \
  --use_all_culling \
  | tee "${OUTDIR}/all_culling.log"

echo "All runs completed."