"""Recompute saved spring tracks in a fresh directory, freezing all other scoring."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from physics_eval.evaluators.spring import eval_spring, apply_period_confidence, CONFIDENCE_PROTOCOL
from physics_eval.utils.tracking import normalize_tracking_json
from physics_eval.utils.period_quality import DEFAULT_RULES
from scripts import score_results as scorer
from scripts.report_io import write_csv,write_json


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=ROOT.parent/'results')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise SystemExit('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'protocol.json',dict(protocol=CONFIDENCE_PROTOCOL,rules=asdict(DEFAULT_RULES),qmask='grouped_v2_candidate',weak_weight=.8,
        eligibility_policy='legacy_membership_preserved'))
    argv=sys.argv;sys.argv=['score','--no-plots']
    try: options=scorer.parse_args()
    finally: sys.argv=argv
    paths=sorted(args.results_root.glob('*/*/*/result.json'))
    manifest={str(p.resolve()):sha(p) for p in paths}
    all_old=[];all_new=[];changes=[];unavailable=[];legacy_mismatch=[]
    for p in paths:
        data=json.loads(p.read_text(encoding='utf-8'));model=p.relative_to(args.results_root).parts[0]
        old=scorer.score_one(p,scorer.DEFAULT_TOLERANCES,options);old['model']=model;all_old.append(old)
        if data.get('metric')!='spring_constant':
            new=scorer.score_one(p,scorer.DEFAULT_TOLERANCES,options);new['model']=model
            assert old==new
            all_new.append(new);continue
        tracking=p.parent/'tracking_points.json'
        reason=''
        saved_legacy = {**data['physics_result'], 'Extra_JSON':json.dumps(data.get('physics_extra',{}))}
        if tracking.exists(): manifest[str(tracking.resolve())]=sha(tracking)
        try:
            df=normalize_tracking_json(tracking)
            physics=apply_period_confidence(saved_legacy,df,data['metadata'])
            extra=json.loads(physics.pop('Extra_JSON'))
            legacy=eval_spring(df,data['metadata'])
            old_k=data['physics_result'].get('Measured_Value'); replay_k=legacy.get('Measured_Value')
            if legacy['Status']!=data['physics_result']['Status'] or (old_k is not None and replay_k is not None and not np.isclose(old_k,replay_k,rtol=1e-8,atol=1e-10)):
                legacy_mismatch.append(dict(model=model,prompt_id=data['prompt_id'],old_status=data['physics_result']['Status'],replay_status=legacy['Status'],old_k=old_k,replay_k=replay_k))
        except Exception as exc:
            reason=f'{type(exc).__name__}: {exc}'
            unavailable.append(dict(model=model,prompt_id=data['prompt_id'],reason=reason))
            physics=apply_period_confidence(saved_legacy,None,data['metadata'])
            extra=json.loads(physics.pop('Extra_JSON'));extra['recompute_error']=reason
        assert (physics['Status'] in scorer.PHYSICS_VALID_STATUSES) == (data['physics_result']['Status'] in scorer.PHYSICS_VALID_STATUSES)
        for field in ('Measured_Value','Relative_Error','Absolute_Error','Measured_Unit'):
            assert physics.get(field)==data['physics_result'].get(field), (p,field)
        for field in ('period_s','period_method'):
            assert extra.get(field)==data.get('physics_extra',{}).get(field), (p,field)
        new_data=copy.deepcopy(data);new_data['physics_result']=physics;new_data['physics_extra']=extra
        assert new_data['mask_qc']==data['mask_qc'] and new_data['tracking_quality']==data['tracking_quality']
        write_json(out/'candidate_results'/p.relative_to(args.results_root),new_data)
        with patch.object(scorer,'load_json',side_effect=lambda path: new_data if path==p else json.loads(path.read_text(encoding='utf-8'))):
            new=scorer.score_one(p,scorer.DEFAULT_TOLERANCES,options)
        new['model']=model;all_new.append(new)
        assert old['effective_for_score']==new['effective_for_score'], p
        assert old['base_score']==new['base_score'], p
        assert old['mask_quality_score']==new['mask_quality_score'], p
        original=data.get('physics_extra',{});d=extra.get('period_quality',{})
        changes.append(dict(model=model,prompt_id=data['prompt_id'],old_method=original.get('period_method'),new_method=extra.get('period_method'),
            old_T=original.get('period_s'),new_T=extra.get('period_s'),old_k=data['physics_result'].get('Measured_Value'),new_k=physics.get('Measured_Value'),
            target=data['metadata'].get('Target_Value'),old_status=old['status'],new_status=new['status'],
            old_weak_weight=scorer.STATUS_MULTIPLIERS.get(old['status'],0),new_weak_weight=scorer.STATUS_MULTIPLIERS.get(new['status'],0),
            old_score=old['adjusted_score'],new_score=new['adjusted_score'],old_effective=old['effective_for_score'],new_effective=new['effective_for_score'],
            observed_cycles=extra.get('observed_cycles'),supported_cycles=d.get('supported_cycles'),fit_r2=d.get('fit_r2'),interval_cv=d.get('interval_cv'),
            coverage=d.get('coverage'),quality_status=extra.get('period_quality_status'),
            quality_reasons=';'.join(extra.get('period_quality_reasons') or []),
            old_confidence=data['physics_result'].get('Fit_R2'),new_confidence=extra.get('period_confidence'),
            reasons=physics.get('Failure_Reason'),recompute_error=reason))
        if len(changes)%50==0: print(f'Processed {len(changes)} spring records',flush=True)
    summaries=[]
    for model in sorted(set(r['model'] for r in all_old)):
        a=[r for r in all_old if r['model']==model and r['metric']=='spring_constant'];b=[r for r in all_new if r['model']==model and r['metric']=='spring_constant']
        sa=scorer.summarize_metric('spring_constant',a);sb=scorer.summarize_metric('spring_constant',b)
        both=[(x,y) for x,y in zip(a,b) if x['effective_for_score'] and y['effective_for_score']]
        summary=dict(model=model,n=len(a),old_effective=sa['n_effective_for_score'],new_effective=sb['n_effective_for_score'],
                     old_effective_score=sa['effective_video_score'],new_effective_score=sb['effective_video_score'],
                     old_end_to_end=sa['end_to_end_score'],new_end_to_end=sb['end_to_end_score'],common_effective_n=len(both),
                     common_old_score=float(np.mean([x['adjusted_score'] for x,y in both])) if both else None,
                     common_new_score=float(np.mean([y['adjusted_score'] for x,y in both])) if both else None)
        for prefix,rows in [('old',a),('new',b)]:
            for status in ('valid','weak_valid','invalid','failed'):summary[prefix+'_'+status]=sum(r['status']==status for r in rows)
            vals=[r['adjusted_score'] for r in rows]
            for q in (0,25,50,75,100):summary[f'{prefix}_score_p{q}']=float(np.percentile(vals,q))
        summary['scores_up']=sum(y['adjusted_score']>x['adjusted_score']+1e-12 for x,y in zip(a,b))
        summary['scores_down']=sum(y['adjusted_score']<x['adjusted_score']-1e-12 for x,y in zip(a,b))
        summaries.append(summary)
    peaks=[r for r in changes if r['old_method']=='peaks']
    unchanged=all(sha(Path(p))==h for p,h in manifest.items())
    verification=dict(n_all=len(paths),n_spring=len(changes),n_unavailable=len(unavailable),
        protocol=CONFIDENCE_PROTOCOL,
        effective_membership_exactly_unchanged=all(a['effective_for_score']==b['effective_for_score'] for a,b in zip(all_old,all_new)),
        spring_measurements_and_sources_exactly_unchanged=all(r['old_T']==r['new_T'] and r['old_k']==r['new_k'] and r['old_method']==r['new_method'] for r in changes),
        old_peaks_n=len(peaks),old_peaks_status=dict(Counter(r['old_status'] for r in peaks)),
        new_status_of_old_peaks=dict(Counter(r['new_status'] for r in peaks)),
        transitions=dict(Counter(r['old_status']+' -> '+r['new_status'] for r in changes)),
        legacy_replay_mismatch_n=len(legacy_mismatch),input_files_unchanged=unchanged,
        non_spring_exactly_unchanged=sum(r['metric']!='spring_constant' for r in all_old),
        old_total_effective=sum(r['effective_for_score'] for r in all_old),new_total_effective=sum(r['effective_for_score'] for r in all_new))
    assert unchanged
    for name,rows in [('per_video',changes),('status_changes',[r for r in changes if r['old_status']!=r['new_status']]),('old_peak_records',peaks),('model_summary',summaries),('unavailable',unavailable),('legacy_replay_mismatch',legacy_mismatch)]:
        write_json(out/(name+'.json'),rows);write_csv(out/(name+'.csv'),rows)
    write_json(out/'verification.json',verification);write_json(out/'input_manifest.json',manifest)
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()
