"""Internal SAM2 subprocess boundary used by the public evaluation pipeline."""
import argparse
from pathlib import Path

WORKSPACE_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
SAM2_PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run SAM2 video tracking inside the workspace and save shared frames + tracking outputs."
    )

    parser.add_argument("--workspace-root", type=str, default=str(WORKSPACE_ROOT_FOR_IMPORTS),
                        help="Workspace root. Default: inferred from this script.")
    parser.add_argument("--scene-name", type=str, required=True,
                        help="Scene / video id, e.g. video_001")
    parser.add_argument("--video-path", type=str, required=True,
                        help="Input video path")
    parser.add_argument("--copy-video-to-workspace", action="store_true",
                        help="Copy input video into data/custom/<scene>/raw/input.mp4")

    parser.add_argument("--sam2-repo", type=str, default=str(WORKSPACE_ROOT_FOR_IMPORTS / "repo" / "sam2"),
                        help="Path to SAM2 repo root")
    parser.add_argument("--model-cfg", type=str, required=True,
                        help="SAM2 model config path, e.g. configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--model-weights", type=str, required=True,
                        help="Path to SAM2 checkpoint")

    parser.add_argument("--init-frame-idx", type=int, default=0,
                        help="Manual prompt frame. In auto mode this is replaced by the detected frame.")
    parser.add_argument("--init-obj-id", type=int, default=1)
    parser.add_argument("--init-point-x", type=float, default=None)
    parser.add_argument("--init-point-y", type=float, default=None)
    parser.add_argument("--init-label", type=int, default=1,
                        help="1 = foreground point")
    parser.add_argument("--init-box", type=float, nargs=4, default=None,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Optional manual box prompt in image coordinates.")

    parser.add_argument("--auto-init", action="store_true",
                        help="Automatically find the main moving object and use it as the SAM2 prompt.")
    parser.add_argument("--auto-init-num-objects", type=int, default=1,
                        help="Number of moving objects to initialize automatically. Default tracks one object.")
    parser.add_argument("--auto-init-scan-frames", type=int, default=60,
                        help="Number of frames used to estimate motion.")
    parser.add_argument("--auto-init-scan-mode", type=str, default="uniform",
                        choices=["start", "middle", "uniform"],
                        help="Which part of the video to scan for automatic initialization.")
    parser.add_argument("--no-bidirectional-propagation", action="store_true",
                        help="Only propagate forward from the init frame. By default nonzero init frames track both directions.")
    parser.add_argument("--auto-init-threshold", type=float, default=25.0,
                        help="Grayscale difference threshold for motion segmentation.")
    parser.add_argument("--auto-init-min-area", type=int, default=40,
                        help="Minimum moving component area in pixels.")
    parser.add_argument("--auto-init-max-area-ratio", type=float, default=0.20,
                        help="Reject moving components larger than this fraction of the image.")
    parser.add_argument("--auto-init-box-pad", type=int, default=8,
                        help="Padding around the detected moving-object box.")
    parser.add_argument("--auto-init-min-fill-ratio", type=float, default=0.08,
                        help="Reject sparse/slender motion components below this component bbox fill ratio.")
    parser.add_argument("--auto-init-max-aspect-ratio", type=float, default=6.0,
                        help="Reject very elongated motion components, e.g. springs or rods.")
    parser.add_argument("--auto-init-min-separation-ratio", type=float, default=0.03,
                        help="Minimum point separation for selecting multiple auto-init objects, as image diagonal ratio.")
    parser.add_argument("--auto-init-object-refine", action=argparse.BooleanOptionalAction, default=True,
                        help="Refine motion candidates to the surrounding whole visible object before prompting SAM2.")
    parser.add_argument("--auto-init-box-expand", type=float, default=2.8,
                        help="Expansion factor around a motion component when searching for the whole object.")
    parser.add_argument("--auto-init-target-hint", type=str, default="auto",
                        help="Optional target hint from metadata, e.g. standard_tennis_ball, basketball, hockey_puck.")
    parser.add_argument("--disable-auto-init-shadow-filter", action="store_true",
                        help="Disable the shadow-like motion penalty used during auto initialization.")
    parser.add_argument("--no-auto-init-negative-points", action="store_true",
                        help="Do not add automatic background negative points around the auto-init box.")
    parser.add_argument("--no-auto-init-box", action="store_true",
                        help="Use only a point prompt for auto initialization.")
    parser.add_argument("--save-auto-init-debug", action="store_true",
                        help="Save a debug image showing the detected auto-init prompt.")
    parser.add_argument("--yolo-weights", type=str, default=None,
                        help="YOLO weights, e.g. yolo11n.pt or a custom trained .pt.")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                        help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.50,
                        help="YOLO NMS IoU threshold.")
    parser.add_argument("--yolo-classes", type=str, default=None,
                        help="Optional comma-separated YOLO class names or ids to allow.")

    parser.add_argument("--save-vis-video", action="store_true",
                        help="Save tracking visualization video")
    parser.add_argument("--save-mask-png", action="store_true",
                        help="Save binary mask png for each frame")
    parser.add_argument("--overwrite-frames", action="store_true",
                        help="Re-extract frames even if frames dir already has jpgs")
    parser.add_argument("--overwrite-outputs", action="store_true",
                        help="Overwrite tracking json/video/masks if already exist")

    parser.add_argument("--vos-optimized", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=SAM2_PRECISION_CHOICES,
        help=(
            "SAM2 inference precision. 'auto' uses BF16 when supported and FP16 otherwise; "
            "'fp32' disables autocast. The selected policy covers initialization, prompt "
            "injection, and every propagation iteration."
        ),
    )

    parser.add_argument("--prompt-manifest", default=None, help="Validated manual or renderer_gt prompt manifest.")
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    from .sam2 import run_tracking

    return run_tracking(args)


if __name__ == "__main__":
    main()
