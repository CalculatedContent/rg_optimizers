"""Scientific invariants: pairing, zero information, and severity weighting."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
import numpy as np
import pandas as pd
from weighted import paired_contributions,seed_scores,interval

def fixture():
    return pd.DataFrame([
        dict(seed=1,batch_size=128,multiplier=1,arm='adamw',alpha_fc1=1.7),
        dict(seed=1,batch_size=256,multiplier=1,arm='adamw',alpha_fc1=1.95),
        dict(seed=1,batch_size=128,multiplier=1,arm='graft',alpha_fc1=1.9),
        dict(seed=1,batch_size=256,multiplier=1,arm='graft',alpha_fc1=1.9)])

class WeightedTests(unittest.TestCase):
    def score(self,f):
        return seed_scores(paired_contributions(f,'adamw','graft',['fc1'],'absolute_baseline_distance')).iloc[0]
    def test_severity_and_signed_penalty(self):
        self.assertAlmostEqual(self.score(fixture()).improvement,(.3*.2-.05*.05)/.35)
    def test_row_order_cannot_change_pairing(self):
        self.assertAlmostEqual(self.score(fixture()).improvement,self.score(fixture().sample(frac=1,random_state=3)).improvement)
    def test_missing_pair_rejected(self):
        with self.assertRaises(ValueError):self.score(fixture().iloc[:-1])
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):self.score(pd.concat([fixture(),fixture().iloc[:1]]))
    def test_zero_weight_not_called_success(self):
        f=fixture();f.loc[f.arm=='adamw','alpha_fc1']=2
        self.assertEqual(self.score(f).score_status,'ZERO_WEIGHT')
        self.assertTrue(np.isnan(self.score(f).improvement))
    def test_nonfinite_not_dropped(self):
        f=fixture();f.loc[2,'alpha_fc1']=np.nan
        self.assertEqual(self.score(f).score_status,'NONFINITE')
        self.assertEqual(interval([1,2,np.nan,3,4])['status'],'UNDEFINED')
    def test_overshoot_counted_as_worsening(self):
        f=fixture();f.loc[f.arm=='graft','alpha_fc1']=3
        self.assertLess(self.score(f).improvement,0)
    def test_sub2_weight_is_baseline_only(self):
        f=fixture();f.loc[1,'alpha_fc1']=2.1
        c=paired_contributions(f,'adamw','graft',['fc1'],'sub2_baseline_deficit')
        self.assertEqual(c.loc[c.batch_size==256,'weight'].iloc[0],0)

if __name__=='__main__':unittest.main()
