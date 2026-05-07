#!/usr/bin/env bash
set -euo pipefail

# Prepare external assets needed by the PhysEval tracking pipeline.
#
# This script intentionally downloads third-party assets into ignored local
# paths instead of committing them to this repository.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM2_REPO_DIR="${SAM2_REPO_DIR:-${ROOT_DIR}/repo/sam2}"
SAM2_MODEL="${SAM2_MODEL:-base_plus}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8n.pt}"
INSTALL_SAM2=0
SKIP_YOLO=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup_sam2_assets.sh [options]

Options:
  --sam2-model tiny|small|base_plus|large
      SAM2.1 checkpoint to download. Default: base_plus.
  --sam2-repo-dir PATH
      Where to clone facebookresearch/sam2. Default: repo/sam2.
  --install-sam2
      Run "pip install -e repo/sam2" after cloning. Optional because CUDA/PyTorch
      setup is environment-specific.
  --skip-yolo
      Do not download YOLO weights.
  -h, --help
      Show this help.

Environment overrides:
  SAM2_REPO_DIR=/path/to/sam2
  SAM2_MODEL=base_plus
  YOLO_WEIGHTS=yolov8n.pt
USAGE
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    echo "Missing value for ${option}" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sam2-model)
      require_value "$1" "${2:-}"
      SAM2_MODEL="$2"
      shift 2
      ;;
    --sam2-repo-dir)
      require_value "$1" "${2:-}"
      SAM2_REPO_DIR="$2"
      shift 2
      ;;
    --install-sam2)
      INSTALL_SAM2=1
      shift
      ;;
    --skip-yolo)
      SKIP_YOLO=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

download_file() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then
    echo "[ok] exists: $out"
    return
  fi
  echo "[download] $url -> $out"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

case "$SAM2_MODEL" in
  tiny)
    SAM2_CKPT="sam2.1_hiera_tiny.pt"
    ;;
  small)
    SAM2_CKPT="sam2.1_hiera_small.pt"
    ;;
  base_plus|base-plus|b+)
    SAM2_CKPT="sam2.1_hiera_base_plus.pt"
    ;;
  large)
    SAM2_CKPT="sam2.1_hiera_large.pt"
    ;;
  *)
    echo "Unsupported SAM2_MODEL: $SAM2_MODEL" >&2
    exit 2
    ;;
esac

SAM2_CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/${SAM2_CKPT}"
YOLO_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/${YOLO_WEIGHTS}"

echo "[setup] workspace: $ROOT_DIR"

if [[ -d "$SAM2_REPO_DIR/.git" ]]; then
  echo "[ok] SAM2 repo exists: $SAM2_REPO_DIR"
else
  mkdir -p "$(dirname "$SAM2_REPO_DIR")"
  echo "[clone] https://github.com/facebookresearch/sam2.git -> $SAM2_REPO_DIR"
  git clone https://github.com/facebookresearch/sam2.git "$SAM2_REPO_DIR"
fi

download_file "$SAM2_CKPT_URL" "${SAM2_REPO_DIR}/checkpoints/${SAM2_CKPT}"

if [[ "$INSTALL_SAM2" -eq 1 ]]; then
  echo "[install] pip install -e ${SAM2_REPO_DIR}"
  python -m pip install -e "$SAM2_REPO_DIR"
fi

if [[ "$SKIP_YOLO" -eq 0 ]]; then
  download_file "$YOLO_URL" "${ROOT_DIR}/${YOLO_WEIGHTS}"
fi

cat <<EOF

Done.

Expected paths:
  SAM2 repo:       ${SAM2_REPO_DIR}
  SAM2 checkpoint: ${SAM2_REPO_DIR}/checkpoints/${SAM2_CKPT}
  YOLO weights:    ${ROOT_DIR}/${YOLO_WEIGHTS}

Default batch command:
  python scripts/run_batch_simple.py \\
    --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \\
    --video-root data/t2v_videos/<model_name> \\
    --output-dir batch_eval_results/<model_name> \\
    --detector yolo_then_motion
EOF
