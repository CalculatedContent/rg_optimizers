"""Active-set projection onto the full matrix-log no-return cone."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from .geometry import MatrixLogGeometry

@dataclass
class ActiveSetQPSolution:
    multipliers: torch.Tensor
    active_indices: torch.Tensor
    iterations: int
    converged: bool
    kkt_residual: float

@dataclass
class ConeProjectionResult:
    corrected_delta: torch.Tensor
    correction: torch.Tensor
    base_potential_drift: float
    corrected_potential_drift: float
    correction_ratio: float
    applied: bool
    capped: bool
    eligible_mode_count: int
    initial_inward_mode_count: int
    active_set_size: int
    iterations: int
    converged: bool
    kkt_residual: float
    max_signed_violation_before: float
    max_signed_violation_after: float

def _solve(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(matrix, vector)
    except (RuntimeError, NotImplementedError):
        cpu_m=matrix.detach().cpu().double(); cpu_v=vector.detach().cpu().double()
        try: out=torch.linalg.solve(cpu_m,cpu_v)
        except RuntimeError: out=torch.linalg.lstsq(cpu_m,cpu_v.unsqueeze(-1)).solution.squeeze(-1)
        return out.to(matrix.device,matrix.dtype)

def matrix_log_mode_drifts(delta: torch.Tensor, geometry: MatrixLogGeometry) -> torch.Tensor:
    work=delta.T if geometry.transposed else delta
    work=work.to(geometry.reference_work.device,geometry.reference_work.dtype)
    mode_delta=torch.sum((work@geometry.mode_right_vectors)*geometry.mode_left_vectors,dim=0)
    radial_delta=2.0*torch.sum(geometry.reference_work*work)/geometry.frobenius_norm_sq
    return 2.0*mode_delta/geometry.singular_values-radial_delta

def matrix_log_mode_gram(geometry: MatrixLogGeometry) -> torch.Tensor:
    rank=int(geometry.retained_rank); s=geometry.singular_values; S=geometry.frobenius_norm_sq
    return torch.diag(4.0/s.square())-(4.0/S)*torch.ones((rank,rank),device=s.device,dtype=s.dtype)

def matrix_log_correction_from_coefficients(coefficients: torch.Tensor, geometry: MatrixLogGeometry, *, output_like: torch.Tensor) -> torch.Tensor:
    work=2.0*((geometry.mode_left_vectors*(coefficients/geometry.singular_values).unsqueeze(0))@geometry.mode_right_vectors.T)
    work=work-(2.0*torch.sum(coefficients)/geometry.frobenius_norm_sq)*geometry.reference_work
    result=work.T if geometry.transposed else work
    return result.to(output_like.device,output_like.dtype)

def _cap(correction,delta,max_ratio,eps):
    base=torch.linalg.vector_norm(delta.float()); corr=torch.linalg.vector_norm(correction.float()); capped=False
    if max_ratio is not None:
        allowed=float(max_ratio)*float(base.detach().cpu()); current=float(corr.detach().cpu())
        if current>allowed and current>eps:
            correction=correction*(allowed/current); corr=torch.linalg.vector_norm(correction.float()); capped=True
    denom=float(base.detach().cpu()); ratio=float(corr.detach().cpu())/denom if denom>eps else 0.0
    return correction,ratio,capped

def _kkt_residual(lam: torch.Tensor, grad: torch.Tensor) -> float:
    primal=float(torch.max(torch.clamp(-lam,min=0.0)).detach().cpu()) if lam.numel() else 0.0
    dual=float(torch.max(torch.clamp(-grad,min=0.0)).detach().cpu()) if grad.numel() else 0.0
    comp=float(torch.max(torch.abs(lam*grad)).detach().cpu()) if lam.numel() else 0.0
    return max(primal,dual,comp)

@torch.no_grad()
def solve_active_set_nonnegative_qp(gram: torch.Tensor, linear: torch.Tensor, *, ridge_relative=1e-8, tolerance=1e-8, max_iterations=128) -> ActiveSetQPSolution:
    """Solve min 1/2 λᵀGλ + qᵀλ with λ>=0 using an active set."""
    if gram.ndim!=2 or gram.shape[0]!=gram.shape[1]: raise ValueError('gram must be square')
    if linear.ndim!=1 or linear.numel()!=gram.shape[0]: raise ValueError('linear must match gram')
    n=int(linear.numel())
    if n==0:
        z=torch.zeros_like(linear); return ActiveSetQPSolution(z,torch.empty(0,dtype=torch.long,device=z.device),0,True,0.0)
    G=0.5*(gram+gram.T)
    scale=torch.max(torch.abs(torch.diagonal(G))).clamp_min(torch.finfo(G.dtype).eps)
    G=G+float(ridge_relative)*scale*torch.eye(n,device=G.device,dtype=G.dtype)
    lam=torch.zeros_like(linear)
    active=set(int(i) for i in torch.nonzero(linear < -float(tolerance),as_tuple=False).reshape(-1).tolist())
    converged=False; iterations=0; seen=set()
    for iterations in range(1,int(max_iterations)+1):
        key=tuple(sorted(active))
        if key in seen and active:
            # Break cycles by removing the weakest multiplier after the solve below.
            pass
        seen.add(key)
        lam.zero_()
        if active:
            idx=torch.tensor(sorted(active),device=linear.device,dtype=torch.long)
            sub=G.index_select(0,idx).index_select(1,idx)
            candidate=_solve(sub,-linear[idx])
            if bool(torch.any(candidate <= float(tolerance)).item()):
                remove_offset=int(torch.argmin(candidate).detach().cpu())
                active.remove(int(idx[remove_offset].detach().cpu()))
                continue
            lam[idx]=candidate
        grad=G@lam+linear
        inactive=[i for i in range(n) if i not in active]
        if inactive:
            ii=torch.tensor(inactive,device=linear.device,dtype=torch.long)
            vals=grad[ii]
            minval,off=torch.min(vals,dim=0)
            if float(minval.detach().cpu()) < -float(tolerance):
                active.add(inactive[int(off.detach().cpu())]); continue
        active_res=float(torch.max(torch.abs(grad[torch.tensor(sorted(active),device=linear.device,dtype=torch.long)])).detach().cpu()) if active else 0.0
        if active_res <= 10.0*float(tolerance):
            converged=True; break
    grad=G@lam+linear
    residual=_kkt_residual(lam,grad)
    if residual <= 20.0*float(tolerance): converged=True
    return ActiveSetQPSolution(lam,torch.nonzero(lam>float(tolerance),as_tuple=False).reshape(-1),iterations,converged,residual)

@torch.no_grad()
def project_matrix_log_cone(delta_base: torch.Tensor, geometry: MatrixLogGeometry, *, projection_strength=1.0, max_correction_ratio=None, gram_ridge_relative=1e-8, tolerance=1e-7, max_iterations=128, log_deadband=1e-6, eps=1e-12) -> ConeProjectionResult:
    """Minimum-norm projection onto sign(ell_i) d ell_i >= 0."""
    if not 0.0<=float(projection_strength)<=1.0: raise ValueError
    drifts=matrix_log_mode_drifts(delta_base,geometry)
    logs=geometry.log_eigenvalues.to(drifts)
    eligible=torch.abs(logs)>float(log_deadband)
    signs=torch.sign(logs)
    signed=signs*drifts
    initial=eligible&(signed < -float(tolerance))
    eligible_count=int(torch.count_nonzero(eligible)); inward_count=int(torch.count_nonzero(initial))
    base_potential=float(torch.sum(geometry.gradient.float()*delta_base.float()).detach().cpu())
    before=float(torch.max(torch.clamp(-signed[eligible],min=0.0)).detach().cpu()) if eligible_count else 0.0
    if inward_count==0:
        zero=torch.zeros_like(delta_base)
        return ConeProjectionResult(delta_base,zero,base_potential,base_potential,0.0,False,False,eligible_count,0,0,0,True,0.0,before,before)
    idx=torch.nonzero(eligible,as_tuple=False).reshape(-1)
    q=signs[idx]*drifts[idx]
    rawG=matrix_log_mode_gram(geometry).index_select(0,idx).index_select(1,idx)
    s=signs[idx]; G=s[:,None]*rawG*s[None,:]
    solution=solve_active_set_nonnegative_qp(G,q,ridge_relative=gram_ridge_relative,tolerance=tolerance,max_iterations=max_iterations)
    coeff=torch.zeros_like(drifts); coeff[idx]=solution.multipliers*s
    full=matrix_log_correction_from_coefficients(coeff,geometry,output_like=delta_base)
    correction=float(projection_strength)*full
    correction,ratio,capped=_cap(correction,delta_base,max_correction_ratio,eps)
    corrected=delta_base+correction
    after_drifts=matrix_log_mode_drifts(corrected,geometry); signed_after=signs*after_drifts
    after=float(torch.max(torch.clamp(-signed_after[eligible],min=0.0)).detach().cpu()) if eligible_count else 0.0
    corrected_potential=float(torch.sum(geometry.gradient.float()*corrected.float()).detach().cpu())
    return ConeProjectionResult(corrected,correction,base_potential,corrected_potential,ratio,bool(float(torch.linalg.vector_norm(correction.float()))>eps),capped,eligible_count,inward_count,int(solution.active_indices.numel()),solution.iterations,solution.converged,solution.kkt_residual,before,after)
