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
    ids,cache,slots=table.lookup(torch.from_numpy(batch.ngram_ids_cpu))
    # Fixed shape per token stage, irrespective of the number of unique row ids.
    padded=torch.zeros(2*batch.inputs.numel(),768,dtype=torch.bfloat16)
    padded[:cache.size(0)].copy_(cache)
    model.ngram_cache=padded.to(device)
    sink=(torch.zeros_like(model.ngram_cache,requires_grad=True) if training else None)
    args=(batch.inputs.to(device),batch.targets.to(device),batch.cum_seqlens.to(device),slots.to(device))
    return ids,slots,sink,args


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
        # Training uses no canonical mask; install the mask only for final validation.
        model=model.to(device)
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
            loss_value=float(loss.detach().mean().cpu())
            if not math.isfinite(loss_value): raise FloatingPointError('Nonfinite training loss')
            table.accumulate(ids,slots,sink.grad)
            optimizer.step(step)
            if is_update_step(step): table.update(step,.008*sched.get_lr(step))
            tail.tick(step)
            xm.mark_step(wait=True)
            elapsed=time.perf_counter()-tic
            stage_index=next(i for i,(_,end) in enumerate(sched.boundaries) if step<end)
            tokens_seen+=TRAINING_STAGES[stage_index].batch_size
            event={'kind':'preflight_step' if is_preflight else 'train','step':step+1,
                   'stage':stage_index,'candidates':count,'train_loss':loss_value,
                   'batch_tokens':TRAINING_STAGES[stage_index].batch_size,
                   'tokens_seen':tokens_seen,'elapsed_seconds':time.perf_counter()-started,
                   'step_seconds':elapsed}
            timings.append(event)
            if rank==0:
                with (root/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(event)+'\n')
                print(json.dumps(event),flush=True)
            del loss,sink,args,sampled
        if not is_preflight:
            tail.ship()
        # Release optimizer and tail state before the 262k-token/rank validation.
        for parameter in model.parameters(): parameter.grad=None
        del optimizer,tail,batches,loader
        gc.collect(); xm.mark_step(wait=True)
        model.canon_mask=canonical.to(device)
        cfg=forward_cfg.at(1194,final=True)
        model.eval()
        val_loader=distributed_data_generator(str(data/'fineweb_val_*.bin'),2097152,3072,PinnedBatchStaging(),False)
        val_steps=1 if is_preflight else 5
        val_sum=0.
        with torch.no_grad():
            for _ in range(val_steps):
                batch=next(val_loader)
                _,_,_,args=device_batch(batch,table,model,device,False)
                losses=model(*args,cfg)
                mean=xm.all_reduce(xm.REDUCE_SUM,losses.mean(),scale=1/world)
                xm.mark_step(wait=True)
                val_sum+=float(mean.cpu())
                del losses,args
        val_nll=val_sum/val_steps
        if not math.isfinite(val_nll): raise FloatingPointError('Nonfinite validation loss')
        result={'mode':'tpu-port','kind':'preflight' if is_preflight else 'final_validation',
                'training_steps':len(steps),'tokens_seen':tokens_seen,'validation_tokens':2097152*val_steps,
                'val_nll':val_nll,'val_perplexity':math.exp(val_nll),
                'full_benchmark_evaluation':not is_preflight,
                'target_reached':not is_preflight and val_nll<=3.28,
                'elapsed_seconds':time.perf_counter()-started,
                'timings':timings if is_preflight else None}
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
