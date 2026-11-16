"""State-rescaling and native small-N estimator audits. No training launcher."""
import copy, contextlib, io, hashlib, inspect
import numpy as np

def small_n(eigenvalues):
    from weightwatcher.WW_powerlaw import WWFit
    x=np.sort(np.asarray(eigenvalues,float))
    fit=object.__new__(WWFit)
    fit.data=x.copy();fit.N=len(x)
    fit.fit_powerlaw_smallN(k_min=8,lambda_prior=0.)
    return dict(alpha=float(fit.alpha),xmin=float(fit.xmin),sigma=float(fit.sigma),D=float(fit.D),n_tail=len(fit.data))

def matrix_errors(s,reference,scaled,c):
    return {n:float(s.frobenius(getattr(scaled,n).weight-c*getattr(reference,n).weight)/s.frobenius(c*getattr(reference,n).weight).clamp_min(1e-30)) for n in s.LAYERS}

def run_audits(s,root):
    torch=s.torch
    original_rates=s.RATES.copy()
    records=[]
    try:
        for arm in s.ARMS:
            torch.manual_seed(32997)
            net=s.MLP().double()
            spec={'seed':32997,'batch_size':256,'multiplier':1.,'arm':arm}
            opt=s.make_opt(net,spec)
            for step in range(4):
                for p in net.parameters():p.grad=torch.randn_like(p)*.01
                s.apply_update(net,opt,spec,.4,256,True)
            for c in [.25,.5,2.,4.]:
                a=copy.deepcopy(net);oa=s.make_opt(a,spec);oa.load_state_dict(copy.deepcopy(opt.state_dict()))
                b=copy.deepcopy(net);ob=s.make_opt(b,spec);ob.load_state_dict(copy.deepcopy(opt.state_dict()))
                with torch.no_grad():
                    for p in b.parameters():p.mul_(c)
                for pa,pb in zip(a.parameters(),b.parameters()):
                    pa.grad=torch.randn_like(pa)*.01;pb.grad=pa.grad/c
                    st=ob.state[pb]
                    for key in ['momentum_buffer','exp_avg']:
                        if key in st:st[key].div_(c)
                    if 'exp_avg_sq' in st:st['exp_avg_sq'].div_(c*c)
                for group in ob.param_groups:
                    if 'eps' in group:group['eps']/=c
                s.RATES=original_rates.copy();s.apply_update(a,oa,spec,.4,256,True)
                # Graft has separate SGD target rates and Adam bias rates.
                if arm=='sgdm':mult=c*c
                else:mult=c
                if arm=='magnitude_graft':s.RATES=[r*c for r in original_rates]
                s.apply_update(b,ob,{**spec,'multiplier':mult},.4,256,True)
                errs=matrix_errors(s,a,b,c)
                bias=max(float(s.frobenius(getattr(b,n).bias-c*getattr(a,n).bias)/s.frobenius(c*getattr(a,n).bias)) for n in s.LAYERS)
                assert max(errs.values())<1e-10 and bias<1e-10
                records.append({'arm':arm,'scale':c,'matrix_relative_errors':errs,'bias_relative_error':bias,'passed':True})
        # Independent direction check isolates finite epsilon; fixed epsilon is
        # intentionally incorrect for coordinate-scale covariance.
        rng=np.random.default_rng(10);m=rng.normal(size=(16,16))*1e-8;v=np.exp(rng.normal(size=(16,16)))*1e-16
        eps=1e-8;c=4.
        def direction(mm,vv,ee):
            x=mm/(np.sqrt(vv)+ee);return x/np.linalg.norm(x)
        correct=np.linalg.norm(direction(m/c,v/c**2,eps/c)-direction(m,v,eps))
        wrong=np.linalg.norm(direction(m/c,v/c**2,eps)-direction(m,v,eps))
        assert correct<1e-12 and wrong>1e-3
        # A shared eta -> c^2 eta transformation gives native Adam an extra c.
        adam_wrong_rate_relative_error=abs(c-1)
        assert adam_wrong_rate_relative_error>1
        negative={'fixed_epsilon_direction_error':float(wrong),'correct_epsilon_direction_error':float(correct),
                  'adam_under_sgdm_rate_law_relative_error':adam_wrong_rate_relative_error,
                  'graft_native_bias_under_sgdm_rate_law_relative_error':adam_wrong_rate_relative_error,
                  'meaning':'Graft matrix target follows eta_s -> c^2 eta_s; its untouched Adam bias requires eta_b -> c eta_b. A single common c^2 rate law is NOT a full-optimizer symmetry.'}
        s.writej(root/'Scale_Audit.json',{'passed':True,'scope':'optimizer coordinate covariance under explicitly transformed states/rates/epsilon; not proof of loss invariance or RG trace-log',
            'transformation':{'weights':'c W','gradient':'g/c','momentum':'m/c','second_moment':'v/c^2','epsilon':'eps/c','sgdm_target_rate':'c^2 eta_s','adam_bias_rate':'c eta_b','decay':'same multiplicative factor'},
            'nonzero_state_checks':len(records),'records':records,'negative_controls':negative,'all_parameters_sgdm_scaling_claim':False})
    finally:s.RATES=original_rates
    from weightwatcher.WW_powerlaw import WWFit
    rng=np.random.default_rng(431);errors=[];native_differences=0
    for alpha in [1.5,2.,3.,5.]:
        for i in range(25):
            x=1+rng.pareto(alpha-1,10)
            a=small_n(x);b=small_n(x*1e4)
            tail=x[x>=a['xmin']];reference=1+(len(tail)-1)/np.log(tail/a['xmin']).sum()
            err=max(abs(reference-a['alpha']),abs(a['alpha']-b['alpha']))
            assert err<1e-9 and a['n_tail']>=8
            errors.append(err)
            with contextlib.redirect_stdout(io.StringIO()):native=WWFit(x)
            if abs(native.alpha-a['alpha'])>1e-10:native_differences+=1
    source=inspect.getfile(WWFit)
    import json
    expected=json.loads((root/'Protocol.json').read_text())['fc3_estimator']['source_sha256']
    assert hashlib.sha256(open(source,'rb').read()).hexdigest()==expected,'Unexpected WeightWatcher estimator source'
    s.writej(root/'FC3_Estimator_Audit.json',{'passed':True,'controlled_spectra':len(errors),'maximum_formula_or_scale_error':max(errors),
        'method':'native WWFit.fit_powerlaw_smallN(k_min=8,lambda_prior=0), invoked directly to preserve its J-selected xmin',
        'formula':'1+(n_tail-1)/sum(log(lambda/xmin))','alpha_two_prior':False,'native_constructor_differences':native_differences,
        'source_sha256':hashlib.sha256(open(source,'rb').read()).hexdigest(),'constructor_note':'In pinned 0.7.7 constructor, argmin(Ds) overwrites the small-N method J selection; save both estimates, never alter historical fits.',
        'uncertainty':'Ten eigenvalues remain a small, correlated sample; the formula correction does not establish an accurate population exponent.'})
