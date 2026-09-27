from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from physics_eval.evaluators.spring import apply_period_confidence, eval_spring, CONFIDENCE_PROTOCOL
from physics_eval.utils.period_quality import estimate_period_quality
from scripts.spring_synthetic_cases import cases, signal
from scripts.validate_spring_confidence import evaluate_case


class SpringConfidenceTests(unittest.TestCase):
    def test_all_synthetic_membership_measurements_and_source_preserved(self):
        for c in cases():
            with self.subTest(case=c):
                a,b=evaluate_case(c);ea,eb=json.loads(a['Extra_JSON']),json.loads(b['Extra_JSON'])
                self.assertEqual(a['Status'] in {'valid','weak_valid'},b['Status'] in {'valid','weak_valid'})
                for key in ('Measured_Value','Measured_Unit','Relative_Error','Absolute_Error'):
                    self.assertEqual(a.get(key),b.get(key))
                for key in ('period_s','period_method'):self.assertEqual(ea.get(key),eb.get(key))
                if c['kind'] in {'linear','quadratic','white_noise','random_walk','chirp','short'}:
                    self.assertNotEqual(b['Status'],'valid')

    def test_fixed_period_quality_same_for_both_sources(self):
        t=np.arange(0,4,1/30);s=np.sin(2*np.pi*t)
        a=estimate_period_quality(t,s,source='peaks',fixed_period_s=1.)
        b=estimate_period_quality(t,s,source='autocorrelation',fixed_period_s=1.)
        self.assertEqual(a.status,'valid');self.assertEqual(a.status,b.status)
        self.assertEqual(a.confidence,b.confidence);self.assertEqual(a.period_s,b.period_s)

    def test_fixed_period_fps_length_stability(self):
        for fps in (24,30,60):
            for duration in (4,8,16):
                t=np.arange(0,duration,1/fps)
                e=estimate_period_quality(t,np.sin(2*np.pi*t),source='peaks',fixed_period_s=1.)
                self.assertEqual(e.period_s,1.);self.assertEqual(e.status,'valid')

    def test_bad_evidence_stays_eligible_with_no_high_confidence(self):
        base=dict(Status='valid',Measured_Value=9.87,Fit_R2=.99,
                  Extra_JSON=json.dumps(dict(period_s=1.,period_method='autocorrelation')))
        result=apply_period_confidence(base,None,{})
        self.assertEqual(result['Status'],'weak_valid')
        self.assertEqual(result['Measured_Value'],9.87)
        extra=json.loads(result['Extra_JSON'])
        self.assertEqual(extra['period_confidence'],0.)
        self.assertEqual(extra['period_quality_status'],'unavailable')
        self.assertEqual(base['Status'],'valid')

    def test_failed_rows_are_not_promoted(self):
        a=dict(Status='failed',Measured_Value=None,Failure_Reason='insufficient_frames',Extra_JSON='{}')
        b=apply_period_confidence(a,None,{})
        self.assertEqual(b['Status'],'failed');self.assertEqual(b['Failure_Reason'],a['Failure_Reason'])

    def test_good_four_cycle_signal_upgrades_weight_only(self):
        c=dict(kind='clean',period=1.,fps=30,cycles=4.,noise=0.,decay=0.,missing=0.,seed=0)
        a,b=evaluate_case(c)
        self.assertEqual(a['Status'],'weak_valid');self.assertEqual(b['Status'],'valid')
        self.assertEqual(a['Measured_Value'],b['Measured_Value'])

    def test_wrong_harmonic_not_promoted(self):
        t=np.arange(0,8,1/30)
        e=estimate_period_quality(t,np.sin(2*np.pi*t),source='peaks',fixed_period_s=2.)
        self.assertNotEqual(e.status,'valid');self.assertEqual(e.period_s,2.)

if __name__=='__main__':unittest.main()
