"""Eight-process PJRT/XLA trainer with Gloo CPU table exchange (no SPMD/NCCL)."""
import datetime
import gc
import json
import math
from pathlib import Path
import time
import torch
import torch.distributed as dist

from .config import (TRAINING_STAGES, LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, WS_POST_YARN_EXT)
from .schedule import TrainingSchedule
from .gpt import GPT, ForwardScheduleConfig
from .data import distributed_data_generator, ScheduledBatches, PinnedBatchStaging
from .host_table import HostTable
from .ngram_math import is_update_step
from .optimizer import Optimizer
from .sampled_softmax import CandidateBuilder, candidate_count_at
from .tail import TailAverages
from .tracking import INTERVAL, PROBE_TOKENS, queue_snapshot

PREFLIGHT_STEPS = [0,1,2,3,320,321,322,323,336,337,338,339,514,515,
                   681,682,683,684,965,966,967,968,1107,1108,1109,1110,1174,1175,1176,1177]


def write_json(path,value):
    path=Path(path); temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n'); temporary.replace(path)


def schedule():
    return TrainingSchedule(TRAINING_STAGES,1122,20,torch.device('cpu'),
                            LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT)


class ForwardConfig:
    def __init__(self, model, sched, device):
        self.model,self.schedule,self.device=model,sched,device
        self.windows=TRAINING_STAGES[0].window_sizes

    def at(self,step,sampled=None,final=False):
        stage,_=self.schedule.lookup(step)
        short,long=stage.window_sizes
        old_short,old_long=self.windows
        if (short,long)!=self.windows:
            self.model.yarn_wide.apply(old_long*128,long*128)
            for yarn in (self.model.yarn,self.model.yarn_paired_head):
                yarn.apply(old_short*128,short*128)
        self.windows=(short,long)
        # Upstream extends final attention window WITHOUT a further YaRN frequency update.
        if final: long=WS_POST_YARN_EXT
        return ForwardScheduleConfig(self.schedule.mtp_weights[step].to(self.device),
            self.schedule.prefix_weights[step].to(self.device),short*128,long*128,
            min(stage.train_max_seq_len,2560),sampled)


def device_batch(batch,table,model,device,training):
    began=time.perf_counter()
    ids,cache,slots=table.lookup(torch.from_numpy(batch.ngram_ids_cpu))
    lookup_seconds=time.perf_counter()-began
    # Fixed shape per token stage, irrespective of the number of unique row ids.
    padded=torch.zeros(2*batch.inputs.numel(),768,dtype=torch.bfloat16)
    padded[:cache.size(0)].copy_(cache)
    model.ngram_cache=padded.to(device)
    sink=(torch.zeros_like(model.ngram_cache,requires_grad=True) if training else None)
    args=(batch.inputs.to(device),batch.targets.to(device),batch.cum_seqlens.to(device),slots.to(device))
    table.last_lookup={'unique_rows':ids.numel(),'active_row_bytes':cache.numel()*cache.element_size(),
                       'host_lookup_seconds':lookup_seconds,
                       'cache_bytes_uploaded':padded.numel()*padded.element_size(),
                       'batch_staging_seconds':time.perf_counter()-began}
    return ids,slots,sink,args


def memory_sample(xm,device):
    # Runtime peak is recorded when exposed; checkpoint usage alone is not a peak.
    info=xm.get_memory_info(device)
    return {key:int(value) for key,value in info.items()}


def evaluate(model,table,data,device,cfg,*,step,tokens_seen,batch_tokens,batches,weight_state):
    """Read-only evaluation; fresh loader and no optimizer/RNG/schedule changes."""
    import torch_xla.core.xla_model as xm
    world=table.world
    was_training=model.training
    model.eval(); started=time.perf_counter(); nll_sum=0.; correct=0
    loader=distributed_data_generator(str(data/'fineweb_val_*.bin'),batch_tokens,3072,PinnedBatchStaging(),False)
    try:
        with torch.no_grad():
            for _ in range(batches):
                batch=next(loader)
                _,_,_,args=device_batch(batch,table,model,device,False)
                losses=model(*args,cfg)
                # Float64 host accumulation; correct-token counts remain exact integers.
                xm.mark_step(wait=True)
                sums=torch.tensor([float(losses.float().sum().cpu()),int(model.last_eval_correct.cpu())],dtype=torch.float64)
                if world>1: dist.all_reduce(sums,op=dist.ReduceOp.SUM)
                nll_sum+=float(sums[0]);correct+=int(sums[1])
                del losses,args
    finally:
        loader.close();model.train(was_training)
    count=batch_tokens*batches; nll=nll_sum/count
    if not math.isfinite(nll): raise FloatingPointError('Nonfinite validation loss')
    full=count==10485760
    qualified=(full and weight_state=='tail_averaged_final' and step==1194
               and tokens_seen==328663040 and cfg.ws_long==2560)
    return {'kind':'validation','step':step,'tokens_seen':tokens_seen,
            'evaluation_tokens':count,'validation_tokens':count,'val_nll':nll,
            'val_perplexity':math.exp(nll),'val_error_count':count-correct,
            'val_token_error':1-correct/count,'val_accuracy':correct/count,
            'full_benchmark_evaluation':full,'target_reached':qualified and nll<=3.28,
            'weight_state':weight_state,'ws_short':cfg.ws_short,'ws_long':cfg.ws_long,
            'validation_seconds':time.perf_counter()-started}


def worker(index,options):
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.debug.metrics as metrics
    rank,world=xr.global_ordinal(),xr.world_size()
    if xr.device_type()!='TPU' or world!=8 or xr.is_spmd():
        raise RuntimeError('Requires eight TPU processes in PJRT MPMD mode; SPMD/CPU unsupported')
    device=xm.xla_device()
    # ANVIL's sensitive FP32 polynomial recurrence needs full matmul precision.
    from torch_xla.backends import set_mat_mul_precision
    set_mat_mul_precision('highest')
    root=Path(options['output'])
    dist.init_process_group('gloo',init_method='file://'+str(root/'gloo-init'),
        rank=rank,world_size=world,timeout=datetime.timedelta(hours=2))
    try:
        attributes=xr.global_runtime_device_attributes()
        # Runtime reports the actual kind; never infer eight chips from a v4/v5p core suffix.
        kinds={str(a.get('device_kind','')).lower() for a in attributes}
        if not kinds or any(not ('v5' in k and ('lite' in k or 'v5e' in k)) for k in kinds):
            raise RuntimeError('Expected v5e runtime devices; observed '+repr(attributes))
        report={'rank':rank,'world_size':world,'device':str(device),'attributes':attributes,
                'torch':torch.__version__,'torch_xla':torch_xla.__version__,
                'mode':'tpu-port','placement':'host-backed-ngram','execution':'PJRT MPMD + CPU Gloo'}
        probe=torch.ones(8,device=device)
        probe=xm.all_reduce(xm.REDUCE_SUM,probe)
        xm.mark_step(wait=True)
        if not torch.equal(probe.cpu(),torch.full((8,),8.)):
            raise RuntimeError('TPU collective preflight failed')
        write_json(root/f'hardware-rank-{rank:02d}.json',report)
        if options['action']=='check': return

        # CPU init is explicitly broadcast, including the signed n-gram pool.
        torch.manual_seed(options['seed'])
        model=GPT(50257,11,6,128,768,262144,ngram_dim=768,world_size=8,device=torch.device('cpu'))
        model.cast_matrix_weights_bf16()
        for parameter in model.parameters(): dist.broadcast(parameter.data,0)
        dist.broadcast(model.ngram_sign_pool,0)
        from .prefix_prediction import build_prefix_table_bucket
        prefix=build_prefix_table_bucket(model.vocab_size,rank,world)
        dist.all_reduce(prefix,op=dist.ReduceOp.MAX)
        model.prefix_table.copy_(prefix)
        # Build canonical mask only once per run, prior to host-table allocation.
        if rank==0:
            import numpy as np
            from .canonical_mask import build_canonical_mask
            np.save(root/'canonical-mask.npy',build_canonical_mask(model.vocab_size))
        dist.barrier()
        import numpy as np
        canonical=torch.from_numpy(np.load(root/'canonical-mask.npy'))
        # Training ignores canonical masking; all diagnostic/final validation uses it.
        model=model.to(device)
        model.canon_mask=canonical.to(device)
        for yarn in model.yarns:
            yarn.device=device
            yarn.angular_freq=yarn.angular_freq.to(device)
        sched=schedule(); forward_cfg=ForwardConfig(model,sched,device)
        table=HostTable(rank,world)
        def reduce_mean(gradient):
            return xm.all_reduce(xm.REDUCE_SUM,gradient,scale=1/world)
        optimizer=Optimizer(model,sched,reduce_mean)
        tail=TailAverages(optimizer.params,sched.total_steps)
        builder=CandidateBuilder(model.vocab_size,24576,49152)
        builder.reset(rank,world)
        initial=TRAINING_STAGES[0]
        data=Path(options['data_root'])/'data/fineweb10B'
        loader=distributed_data_generator(str(data/'fineweb_train_*.bin'),initial.batch_size,
            initial.train_max_seq_len,PinnedBatchStaging())
        is_preflight=options['action']=='preflight'
        steps=PREFLIGHT_STEPS if is_preflight else list(range(sched.total_steps))
        batches=ScheduledBatches(loader,sched,steps)
        xm.mark_step(wait=True)
        started=time.perf_counter(); timings=[]; tokens_seen=0
        def monitor(step,cfg,weight_state):
            validation=evaluate(model,table,data,device,cfg,step=step,tokens_seen=tokens_seen,
                batch_tokens=PROBE_TOKENS,batches=1,weight_state=weight_state)
            if rank==0:
                with (root/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(validation)+'\n')
                queue_snapshot(root,model,validation)
            # All ranks observe the same model step before proceeding.
            dist.barrier()
        monitor(0,forward_cfg.at(0),'initial')
        for step in steps:
            tic=time.perf_counter()
            batch=batches.take(step)
            ids,slots,sink,args=device_batch(batch,table,model,device,True)
            count=candidate_count_at(sched,step)
            sampled=None
            if count:
                candidates,targets,prefixes=builder.build(count,batch.targets_cpu.numpy(),prefix.numpy())
                sampled=tuple(torch.from_numpy(a.copy()).to(device) for a in (candidates,targets,prefixes))
            cfg=forward_cfg.at(step,sampled)
            model.train()
            loss=model(*args,cfg,sink)
            # Reference training differentiates the SUM, not the mean.
            loss.sum().backward()
            xm.mark_step(wait=True)
            mean=xm.all_reduce(xm.REDUCE_SUM,loss.detach().mean(),scale=1/world)
            loss_value=float(mean.cpu())
            if not math.isfinite(loss_value): raise FloatingPointError('Nonfinite training loss')
            sparse_started=time.perf_counter()
            table.accumulate(ids,slots,sink.grad)
            sparse_gradient_seconds=time.perf_counter()-sparse_started
            optimizer.step(step)
            sparse_started=time.perf_counter()
            if is_update_step(step): table.update(step,.008*sched.get_lr(step))
            sparse_update_seconds=time.perf_counter()-sparse_started
            tail.tick(step)
            xm.mark_step(wait=True)
            elapsed=time.perf_counter()-tic
            stage_index=next(i for i,(_,end) in enumerate(sched.boundaries) if step<end)
            tokens_seen+=TRAINING_STAGES[stage_index].batch_size
            event={'kind':'preflight_step' if is_preflight else 'train','step':step+1,
                   'stage':stage_index,'candidates':count,'train_loss':loss_value,
                   'batch_tokens':TRAINING_STAGES[stage_index].batch_size,
                   'tokens_seen':tokens_seen,'elapsed_seconds':time.perf_counter()-started,
                   'step_seconds':elapsed,'loss_definition':'summed MTP/prefix objective; not perplexity NLL',
                   'sparse_gradient_to_host_seconds':sparse_gradient_seconds,
                   'sparse_update_seconds':sparse_update_seconds,**table.last_lookup}
            if is_preflight or (step+1)%INTERVAL==0:
                memory={'step':step+1,**memory_sample(xm,device)}
                with (root/f'memory-rank-{rank:02d}.jsonl').open('a') as handle:
                    handle.write(json.dumps(memory)+'\n')
            timings.append(event)
            if rank==0:
                with (root/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(event)+'\n')
                print(json.dumps(event),flush=True)
            del loss,sink,args,sampled
            if not is_preflight and (step+1)%INTERVAL==0:
                monitor(step+1,cfg,'raw_training')
        if not is_preflight:
            tail.ship()
        # Release optimizer and tail state before the 262k-token/rank validation.
        for parameter in model.parameters(): parameter.grad=None
        del optimizer,tail,batches,loader
        gc.collect(); xm.mark_step(wait=True)
        cfg=forward_cfg.at(1194,final=True)
        val_steps=1 if is_preflight else 5
        result=evaluate(model,table,data,device,cfg,step=steps[-1]+1,tokens_seen=tokens_seen,
            batch_tokens=2097152,batches=val_steps,
            weight_state='preflight_final' if is_preflight else 'tail_averaged_final')
        result.update(mode='tpu-port',kind='preflight' if is_preflight else 'final_validation',
                training_steps=len(steps),
                elapsed_seconds=time.perf_counter()-started,
                timings=timings if is_preflight else None)
        with (root/f'memory-rank-{rank:02d}.jsonl').open('a') as handle:
            handle.write(json.dumps({'step':steps[-1]+1,'phase':'final_validation',**memory_sample(xm,device)})+'\n')
        if rank==0:
            with (root/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(result)+'\n')
            queue_snapshot(root,model,result)
        (root/f'xla-metrics-rank-{rank:02d}.txt').write_text(metrics.metrics_report())
        if is_preflight:
            write_json(root/f'preflight-rank-{rank:02d}.json',result)
            dist.barrier()
            if rank==0: write_json(root/'PREFLIGHT_COMPLETE.json',result)
            return
        # Adapt the unchanged exporter via its public shard interface. It copies one chunk at a time.
        import importlib.util
        reference=Path(options['reference'])
        spec=importlib.util.spec_from_file_location('leaderboard_weight_export',reference/'weight_export.py')
        exporter=importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)
        exporter.export_rank(root/'weights',model,table,rank=rank,world_size=world,step=1194)
        dist.barrier()
        if rank==0:
            exporter.verify_export(root/'weights',world,84602880)
            result['weights_complete']=True
            write_json(root/'FINAL_RESULT.json',result)
            print(json.dumps(result),flush=True)
    finally:
        dist.destroy_process_group()
