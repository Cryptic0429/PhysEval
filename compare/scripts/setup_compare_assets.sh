#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${ROOT_DIR}/repo"
INSTALL_EDITABLE=0
NO_PROXY_MODE=0

usage() {
  cat <<'USAGE'
Usage: bash compare/scripts/setup_compare_assets.sh [--install-editable] [--no-proxy]

Downloads/clones the ignored third-party assets for:
  - YOLOv8n
  - SAM2.1 Hiera base+
  - SAM1 ViT-B
  - Track-Anything / XMem

Options:
  --install-editable  Install SAM2 and Segment Anything in editable mode.
  --no-proxy          Ignore proxy environment variables and Git proxy config
                      for this invocation only.
  -h, --help          Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-editable)
      INSTALL_EDITABLE=1
      shift
      ;;
    --no-proxy)
      NO_PROXY_MODE=1
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
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]]; then
    echo "[ok] $output"
    return
  fi
  echo "[download] $url"
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
  local url="$1"
  local output="$2"
  if [[ -d "$output/.git" ]]; then
    echo "[ok] $output"
    return
  fi
  mkdir -p "$(dirname "$output")"
  "${GIT_COMMAND[@]}" clone "$url" "$output"
}

mkdir -p "$REPO_DIR"
clone_repo https://github.com/facebookresearch/sam2.git "$REPO_DIR/sam2"
clone_repo https://github.com/facebookresearch/segment-anything.git "$REPO_DIR/segment-anything"
clone_repo https://github.com/gaomingqi/Track-Anything.git "$REPO_DIR/Track-Anything"

download_file \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt \
  "$REPO_DIR/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
download_file \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  "$REPO_DIR/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
download_file \
  https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth \
  "$REPO_DIR/Track-Anything/checkpoints/XMem.pth"
download_file \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt \
  "$ROOT_DIR/yolov8n.pt"

if [[ "$INSTALL_EDITABLE" -eq 1 ]]; then
  # SAM2 declares torch as a build dependency. Reuse the already selected
  # CUDA environment instead of downloading a second Torch wheel into a
  # temporary isolated build environment.
  python -m pip install --no-build-isolation -e "$REPO_DIR/sam2"
  python -m pip install --no-build-isolation -e "$REPO_DIR/segment-anything"
fi

cat <<EOF

Assets are ready under:
  $REPO_DIR

Next:
  cd "$ROOT_DIR"
  python physeval.py compare --video-root data/t2v_videos/<model_name> --model-name <model_name> --dry-run
EOF
