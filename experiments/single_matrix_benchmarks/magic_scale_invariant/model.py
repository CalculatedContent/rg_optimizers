"""Independent positive radial invariance, no epsilon or spectral penalty."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ScaleInvariantLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        # Match the initial ordinary Linear effective weights, then learn gain.
        self.gain = nn.Parameter(self.weight.detach().norm() / math.sqrt(out_features))
    def effective_weight(self):
        return self.gain * math.sqrt(self.out_features) * self.weight / self.weight.norm()
    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)

def build():
    return nn.Sequential(ScaleInvariantLinear(10,784),nn.ReLU(),
                         ScaleInvariantLinear(784,784),nn.ReLU(),
                         ScaleInvariantLinear(784,2))

def verify():
    torch.manual_seed(3470)
    m=build().double()
    x=torch.randn(32,10,dtype=torch.float64)
    y=torch.randint(0,2,(32,))
    reference=m(x).detach()
    for layer in (m[0],m[2],m[4]):
        with torch.no_grad():
            for b in (.125, 2., 8.):
                layer.weight.mul_(b)
                torch.testing.assert_close(m(x),reference,rtol=1e-11,atol=1e-12)
                layer.weight.div_(b)
    F.cross_entropy(m(x),y).backward()
    for layer in (m[0],m[2],m[4]):
        w,g=layer.weight,layer.weight.grad
        residual=(w*g).sum().abs()/(w.norm()*g.norm()).clamp_min(1e-30)
        assert residual.item()<1e-11, residual.item()
    print("Independent radial invariance and gradient orthogonality verified")

if __name__=="__main__":
    verify()
