import unittest
import pandas as pd
from plot_verified import aggregate
class Check(unittest.TestCase):
 def test_three_layers(self):
  d=pd.DataFrame([dict(dataset='MNIST',optimizer='SGDM',seed=1,epoch=5,layer='FC'+str(i),alpha=2,ks_d=.03,n_tail=10,test_accuracy=.9) for i in range(1,4)])
  self.assertAlmostEqual(aggregate(d).alpha_bar.iloc[0],2)
 def test_incomplete_rejected(self):
  d=pd.DataFrame([dict(dataset='MNIST',optimizer='SGDM',seed=1,epoch=5,layer='FC1',alpha=2,ks_d=.03,n_tail=10,test_accuracy=.9)])
  with self.assertRaises(ValueError):aggregate(d)
if __name__=='__main__':unittest.main()
