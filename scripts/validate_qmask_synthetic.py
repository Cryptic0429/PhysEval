#!/usr/bin/env python3
"""Controlled mask sanity checks, including a stable wrong-target negative control."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics_eval.quality.diagnostics import extract_quality_evidence
from physics_eval.quality.protocol import load_protocol
from physics_eval.quality.scoring import compute_qmask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a new output file")
    rows = []
    for case in ("correct_mask", "stable_wrong_target_30px_offset", "alternating_width", "centroid_jitter"):
        records, masks, errors, ious = [], {1:{}}, [], []
        for i in range(40):
            target = np.zeros((80,180), dtype=bool)
            target[30:40, 30+i:40+i] = True
            dx = 30 if case == "stable_wrong_target_30px_offset" else (15 if case == "centroid_jitter" and i%2 else 0)
            width = (3 if i%2 else 28) if case == "alternating_width" else 10
            mask = np.zeros_like(target)
            mask[30:40,30+i+dx:30+i+dx+width] = True
            ys,xs = np.nonzero(mask)
            center = [float(xs.mean()), float(ys.mean())]
            masks[1][i] = mask
            records.append(dict(object_id=1, frame_idx=i, valid=True, area=int(mask.sum()),
                                centroid=center, bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]))
            errors.append(float(np.linalg.norm(np.array(center)-[34.5+i,34.5])))
            ious.append(float(np.count_nonzero(mask & target)/np.count_nonzero(mask | target)))
        evidence=extract_quality_evidence(records,masks,total_frames=40)
        grouped_score=compute_qmask(evidence,load_protocol(),[1])["score"]
        rows.append({"case":case,"centroid_rmse_px":float(np.sqrt(np.mean(np.square(errors)))),
                     "ground_truth_mask_iou":float(np.mean(ious)),"grouped_v2_candidate_score":grouped_score})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
