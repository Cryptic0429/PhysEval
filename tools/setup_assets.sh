#!/usr/bin/env bash
set -euo pipefail

# Download ignored third-party assets for evaluation or component experiments.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${ROOT_DIR}/repo"
SAM2_REPO_DIR="${SAM2_REPO_DIR:-${REPO_DIR}/sam2}"
SAM2_MODEL="${SAM2_MODEL:-base_plus}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-yolov8n.pt}"
ASSET_MODE=core
INSTALL_SAM2=0
INSTALL_EDITABLE=0
SKIP_YOLO=0
NO_PROXY_MODE=0

usage() {
  cat <<'USAGE'
Usage: bash tools/setup_assets.sh [options]

Options:
  --mode core|compare
      core: YOLO and SAM2; compare also includes SAM1 and Track-Anything/XMem.
      Default: core.
  --sam2-model tiny|small|base_plus|large
      SAM2.1 checkpoint to download. Default: base_plus.
  --sam2-repo-dir PATH
      Where to clone facebookresearch/sam2. Default: repo/sam2.
  --install-sam2
      Install SAM2 in editable mode with the selected Python environment.
  --install-editable
      Install SAM2 and, in compare mode, Segment Anything in editable mode,
      reusing the selected PyTorch build without build isolation.
  --skip-yolo
      Do not download YOLO weights.
  --no-proxy
      Ignore proxy environment variables and Git proxy configuration for this run.
  -h, --help
      Show this help.

Environment overrides: SAM2_REPO_DIR, SAM2_MODEL, YOLO_WEIGHTS.
USAGE
}

require_value() {
  if [[ -z "${2:-}" || "$2" == --* ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) require_value "$1" "${2:-}"; ASSET_MODE="$2"; shift 2 ;;
    --sam2-model) require_value "$1" "${2:-}"; SAM2_MODEL="$2"; shift 2 ;;
    --sam2-repo-dir) require_value "$1" "${2:-}"; SAM2_REPO_DIR="$2"; shift 2 ;;
    --install-sam2) INSTALL_SAM2=1; shift ;;
    --install-editable) INSTALL_EDITABLE=1; shift ;;
    --skip-yolo) SKIP_YOLO=1; shift ;;
    --no-proxy) NO_PROXY_MODE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$ASSET_MODE" != core && "$ASSET_MODE" != compare ]]; then
  echo "Unsupported asset mode: $ASSET_MODE" >&2
  exit 2
fi

case "$SAM2_MODEL" in
  tiny) SAM2_CKPT=sam2.1_hiera_tiny.pt; SAM2_CFG=configs/sam2.1/sam2.1_hiera_t.yaml ;;
  small) SAM2_CKPT=sam2.1_hiera_small.pt; SAM2_CFG=configs/sam2.1/sam2.1_hiera_s.yaml ;;
  base_plus|base-plus|b+) SAM2_CKPT=sam2.1_hiera_base_plus.pt; SAM2_CFG=configs/sam2.1/sam2.1_hiera_b+.yaml ;;
  large) SAM2_CKPT=sam2.1_hiera_large.pt; SAM2_CFG=configs/sam2.1/sam2.1_hiera_l.yaml ;;
  *) echo "Unsupported SAM2_MODEL: $SAM2_MODEL" >&2; exit 2 ;;
esac

GIT_COMMAND=(git)
CURL_OPTIONS=()
WGET_OPTIONS=()
if [[ "$NO_PROXY_MODE" -eq 1 ]]; then
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  GIT_COMMAND+=( -c http.proxy= -c https.proxy= )
  CURL_OPTIONS+=( --noproxy '*' )
  WGET_OPTIONS+=( --no-proxy )
  echo "[network] Proxy settings are disabled for this setup run."
fi

download_file() {
  local url="$1" output="$2"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]]; then
    echo "[ok] exists: $output"
    return
  fi
  echo "[download] $url -> $output"
  if command -v curl >/dev/null 2>&1; then
    curl "${CURL_OPTIONS[@]}" -L --fail --retry 3 -o "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget "${WGET_OPTIONS[@]}" -O "$output" "$url"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

clone_repo() {
  local url="$1" output="$2"
  if [[ -d "$output/.git" ]]; then
    echo "[ok] repo exists: $output"
    return
  fi
  mkdir -p "$(dirname "$output")"
  "${GIT_COMMAND[@]}" clone "$url" "$output"
}

echo "[setup] workspace: $ROOT_DIR; mode: $ASSET_MODE"
clone_repo https://github.com/facebookresearch/sam2.git "$SAM2_REPO_DIR"
download_file "https://dl.fbaipublicfiles.com/segment_anything_2/092824/${SAM2_CKPT}" "${SAM2_REPO_DIR}/checkpoints/${SAM2_CKPT}"
if [[ "$SKIP_YOLO" -eq 0 ]]; then
  download_file "https://github.com/ultralytics/assets/releases/download/v8.3.0/${YOLO_WEIGHTS}" "${ROOT_DIR}/${YOLO_WEIGHTS}"
fi

if [[ "$ASSET_MODE" == compare ]]; then
  clone_repo https://github.com/facebookresearch/segment-anything.git "$REPO_DIR/segment-anything"
  clone_repo https://github.com/gaomingqi/Track-Anything.git "$REPO_DIR/Track-Anything"
  download_file https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth "$REPO_DIR/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
  download_file https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth "$REPO_DIR/Track-Anything/checkpoints/XMem.pth"
fi

if [[ "$INSTALL_EDITABLE" -eq 1 ]]; then
  python -m pip install --no-build-isolation -e "$SAM2_REPO_DIR"
  if [[ "$ASSET_MODE" == compare ]]; then
    python -m pip install --no-build-isolation -e "$REPO_DIR/segment-anything"
  fi
elif [[ "$INSTALL_SAM2" -eq 1 ]]; then
  python -m pip install -e "$SAM2_REPO_DIR"
fi

cat <<EOF

Assets are ready.
  SAM2 repo:       ${SAM2_REPO_DIR}
  SAM2 config:     ${SAM2_CFG}
  SAM2 checkpoint: ${SAM2_REPO_DIR}/checkpoints/${SAM2_CKPT}
  YOLO weights:    ${ROOT_DIR}/${YOLO_WEIGHTS}

Evaluate:
  python physeval.py eval --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \\
    --video-root data/t2v_videos/<model_name> --output-dir batch_eval_results/<model_name> \\
    --model-cfg ${SAM2_CFG} --model-weights ${SAM2_REPO_DIR}/checkpoints/${SAM2_CKPT}

Comparison preflight:
  python physeval.py compare --video-root data/t2v_videos/<model_name> --model-name <model_name> --dry-run
EOF
