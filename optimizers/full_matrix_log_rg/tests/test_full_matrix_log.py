import unittest
import torch
from full_matrix_log_rg import FullMatrixLogConfig, FullMatrixLogRG, full_matrix_log_geometry, remove_inward_matrix_log_flow

class FullMatrixLogTests(unittest.TestCase):
    def test_isotropic_zero(self):
        g=full_matrix_log_geometry(torch.eye(4),4,normalization_dimension=4)
        self.assertLess(float(g.potential),1e-10)
        self.assertLess(g.gradient_norm_sq,1e-10)
    def test_trace_zero_can_still_be_anisotropic(self):
        w=torch.diag(torch.tensor([2.0,0.5])); d=float(torch.sum(w.square()))
        g=full_matrix_log_geometry(w,2,normalization_dimension=d)
        self.assertAlmostEqual(float(torch.sum(g.log_eigenvalues)),0.0,places=5)
        self.assertGreater(float(g.potential),0.1)
    def test_inward_flow_cancelled(self):
        w=torch.diag(torch.tensor([2.0,1.0,0.5])); g=full_matrix_log_geometry(w,3,normalization_dimension=3)
        r=remove_inward_matrix_log_flow(-0.01*g.gradient,g,projection_strength=1.0,max_correction_ratio=None)
        self.assertLess(r.base_drift,0.0); self.assertAlmostEqual(r.corrected_drift,0.0,places=6)
    def test_outward_untouched(self):
        w=torch.diag(torch.tensor([2.0,1.0,0.5])); g=full_matrix_log_geometry(w,3,normalization_dimension=3)
        delta=0.01*g.gradient; r=remove_inward_matrix_log_flow(delta,g,max_correction_ratio=None)
        self.assertFalse(r.applied); self.assertTrue(torch.equal(r.corrected_delta,delta))
    def test_sgd_wrapper(self):
        model=torch.nn.Linear(4,4,bias=False); base=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        opt=FullMatrixLogRG(base,model.named_parameters(),FullMatrixLogConfig(max_correction_ratio=None))
        x=torch.randn(8,4); y=torch.randn(8,4); opt.zero_grad(); torch.nn.functional.mse_loss(model(x),y).backward(); opt.step()
        self.assertIn('weight',opt.last_stats)

if __name__=='__main__': unittest.main()
