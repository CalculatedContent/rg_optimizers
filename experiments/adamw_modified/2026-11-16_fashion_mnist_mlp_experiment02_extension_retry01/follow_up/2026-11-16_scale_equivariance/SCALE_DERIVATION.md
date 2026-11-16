# What the scale audit establishes

Let c>0 and change parameter coordinates by W'=cW. Assume gradients transform as g'=g/c, either under an explicitly reparameterized loss L'(W')=L(W'/c), or under a separately established symmetry of the loss. The ordinary biased ReLU MLP used here does not automatically satisfy L(cW)=L(W). That additional premise must not be inferred from an optimizer test.

With transformed initial state, momentum obeys b'=b/c, Adam first moment m'=m/c and second moment v'=v/c² at every step. The step count and bias corrections are unchanged. For epsilon'=epsilon/c, the Adam preconditioned vector u=mhat/(sqrt(vhat)+epsilon) is unchanged. Its unit direction is therefore unchanged.

The matrix graft is ΔW=−η_s ||b||_F u/||u||_F. Under η_s'=c²η_s, its transformed step is cΔW. SGDM has the same target-rate exponent. A dimensionless constant multiplying η_s preserves this law if that constant is independent of c. Equal multiplicative decay d also commutes with W'=cW: dW'=c(dW). The statement assumes a nonzero direction and no numerical clamp activation; zero-gradient and underflow limits need their own convention.

The actual graft retains native Adam updates for biases. Those require η_b'=cη_b and epsilon'=epsilon/c, rather than η_b'=c²η_b. Consequently the complete hybrid optimizer has a mixed rate transformation. Applying one common SGDM rate law to every parameter fails on the native Adam biases. Native Adam itself is covariant under its own exponent-one rate law. It is inaccurate to call Adam universally non-invariant.

The implementation audit exercises the frozen kernel with nonzero optimizer moments and nonzero parameter values at c=1/4,1/2,2,4. It checks the full parameter result with transformed states, target rates, Adam bias rates and epsilon. A separate finite-epsilon direction control shows why leaving epsilon fixed is not an exact mathematical symmetry. The reference Adam/independent SGDM buffer contract additionally checks fifteen nonzero matrix updates on actual model gradients.

Scope: this is a conditional optimizer-coordinate covariance test. A proof of an RG coarse-graining flow, unit Jacobian volume, trace-log cancellation or alpha=2 does not follow from it. The training experiment tests the magnitude intervention and its spectral/performance consequences while retaining these distinctions. The same kernel is used; no historical training source is changed.

Context: [van Laarhoven, 2017](https://arxiv.org/abs/1706.05350) studies weight scale and effective learning rate in normalized networks. Its normalization premise is different from the unnormalized MLP here. The displayed graft derivation is our explicit calculation, not a result attributed to that paper.
