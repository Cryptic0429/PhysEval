"""Check confidence-only weighting on the frozen synthetic design."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics_eval.evaluators.spring import eval_spring, CONFIDENCE_PROTOCOL
from physics_eval.utils.period_quality import DEFAULT_RULES
from scripts.spring_synthetic_cases import cases, signal
from scripts.report_io import write_csv, write_json


def evaluate_case(c):
    t, s = signal(c)
    track = pd.DataFrame(dict(t=t, frame=np.rint(t*c['fps']).astype(int),
                             cx_px=s, cy_px=np.zeros(len(t)), object_id='1'))
    metadata = dict(Known_Parameters_JSON='{"mass_kg":0.25}', Motion_Axis='x')
    old = eval_spring(track, metadata)
    new = eval_spring(track, metadata, period_protocol=CONFIDENCE_PROTOCOL)
    return old, new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit('Use a fresh output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design = list(cases())
    write_json(args.output_dir/'design.json', dict(protocol=CONFIDENCE_PROTOCOL,
               rules=asdict(DEFAULT_RULES), cases=design, period_policy='frozen_legacy'))
    rows, details = [], []
    eligible = {'valid', 'weak_valid'}
    for c in design:
        old, new = evaluate_case(c)
        a, b = json.loads(old['Extra_JSON']), json.loads(new['Extra_JSON'])
        assert (old['Status'] in eligible) == (new['Status'] in eligible)
        assert old['Measured_Value'] == new['Measured_Value']
        assert a.get('period_s') == b.get('period_s')
        assert a.get('period_method') == b.get('period_method')
        period = b.get('period_s')
        periodic = c['kind'] not in {'linear', 'quadratic', 'random_walk', 'white_noise', 'chirp'}
        row = {**c, 'old_status':old['Status'], 'new_status':new['Status'],
               'period_s':period, 'k':new['Measured_Value'], 'source':b.get('period_method'),
               'quality_status':b.get('period_quality_status'), 'quality':b.get('period_confidence'),
               'relative_period_error':abs(period/c['period']-1) if period and periodic else None,
               'reasons':';'.join(b.get('period_quality_reasons', []))}
        rows.append(row); details.append({**row, 'diagnostics':b.get('period_quality')})
    bad = {'linear','quadratic','random_walk','white_noise','chirp','short'}
    summary = dict(n_cases=len(rows), eligibility_and_measurements_unchanged=True,
        old_status=dict(Counter(r['old_status'] for r in rows)),new_status=dict(Counter(r['new_status'] for r in rows)),
        transitions=dict(Counter(r['old_status']+' -> '+r['new_status'] for r in rows)),
        false_valid_nonperiodic_or_short=sum(r['new_status']=='valid' and r['kind'] in bad for r in rows),
        valid_period_errors_over_5_percent=sum(r['new_status']=='valid' and (r['relative_period_error'] or 0)>.05 for r in rows))
    write_csv(args.output_dir/'cases.csv',rows)
    write_json(args.output_dir/'diagnostics.json',details)
    write_json(args.output_dir/'summary.json',summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
