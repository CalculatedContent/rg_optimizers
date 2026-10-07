"""Full dense TPU hardware ladder. Synthetic data; never a benchmark or preflight."""
import gc
import math
import time
from dataclasses import dataclass
import torch

from .gpt import GPT, ForwardScheduleConfig
from .optimizer import Optimizer
from .tail import TailAverages


@dataclass
class SyntheticBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    seqlens: torch.Tensor
    slots: torch.Tensor
    cache: torch.Tensor
    sink: torch.Tensor | None


def synthetic_batch(tokens,device='cpu',training=True):
    if tokens<=0 or tokens%16: raise ValueError('Probe tokens must be a positive multiple of 16')
    # Local CPU RNG makes batches identical across ranks, independent of model RNG.
    generator=torch.Generator(device='cpu').manual_seed(1337)
    sequence=torch.randint(0,50257,(tokens+1,),generator=generator,dtype=torch.int32)
    ends=[*range(0,tokens,2560),tokens]
    cache=torch.zeros((2*tokens,768),dtype=torch.bfloat16,device=device)
    sink=torch.zeros_like(cache,requires_grad=True) if training else None
    return SyntheticBatch(sequence[:-1].to(device),sequence[1:].long().to(device),
        torch.tensor(ends,dtype=torch.int32,device=device),
        torch.arange(2*tokens,dtype=torch.int32,device=device),cache,sink)


def compile_sample():
    from torch_xla.debug import metrics
    value=metrics.metric_data('CompileTime')
    # PyTorch/XLA time-metric accumulators are nanoseconds, not seconds.
    return (0,0.) if value is None else (int(value[0]),float(value[1])/1e9)


def record_compile(event,before):
    after=compile_sample()
    event.update(compile_count=after[0]-before[0],compile_seconds=after[1]-before[1])


def run(report,device,xm):
    from .runtime import schedule
    # Retain the real maximum-length buffers, including the zero canonical mask.
    # No HostTable, tokenizer, canonical-mask builder or WeightWatcher is called.
    event=report.begin('model_move')
    torch.manual_seed(1337)
    model=GPT(50257,11,6,128,768,262144,ngram_dim=768,world_size=8,device=torch.device('cpu'))
    model.cast_matrix_weights_bf16()
    model=model.to(device)
    for yarn in model.yarns:
        yarn.device=device; yarn.angular_freq=yarn.angular_freq.to(device)
    xm.mark_step(wait=True)
    event['status']='complete'; report.memory(xm,device,event)

    event=report.begin('optimizer_and_tail')
    sched=schedule()
    def reduce_mean(gradient):
        return xm.all_reduce(xm.REDUCE_SUM,gradient,scale=1/8)
    optimizer=Optimizer(model,sched,reduce_mean)
    tail=TailAverages(optimizer.params,sched.total_steps)
    xm.mark_step(wait=True)
    event['status']='complete'; report.memory(xm,device,event)
    cfg=ForwardScheduleConfig(torch.tensor([1.],device=device),torch.tensor([0.],device=device),
                              640,1408,2560,None)
    report.data['timing_note']=('First step includes compilation; second is labeled steady but Adam on '
        'optimizer.step(1) adds a distinct graph and may compile. Wall time includes host synchronization.')

    for tokens in (128,49152):
        event=report.begin(f'train_{tokens}_batch',tokens_per_rank=tokens)
        batch=synthetic_batch(tokens,device)
        model.ngram_cache=batch.cache
        model.train(); xm.mark_step(wait=True)
        event['status']='complete'; report.memory(xm,device,event)
        for step,label in ((0,'compile'),(1,'steady')):
            event=report.begin(f'train_{tokens}_{label}',tokens_per_rank=tokens,
                               timing_label=label,optimizer_step=step)
            compiled_before=compile_sample()
            began=time.perf_counter()
            try:
                loss=model(batch.inputs,batch.targets,batch.seqlens,batch.slots,cfg,batch.sink)
                loss.sum().backward()
                xm.mark_step(wait=True)
                value=float(loss.detach().float().mean().cpu())
                event.update(loss=value if math.isfinite(value) else None,finite_loss=math.isfinite(value))
                if not math.isfinite(value):
                    report.data['finite_loss']=False
                    raise FloatingPointError('Nonfinite probe training loss')
                optimizer.step(step)
                xm.mark_step(wait=True)
                record_compile(event,compiled_before)
                event['status']='complete'; report.data['finite_loss']=True
            finally:
                event['seconds']=time.perf_counter()-began
                report.flush()
            report.memory(xm,device,event)
            batch.sink.grad=None
            del loss
        # Accumulate Adam gradients within each pair, but not across batch sizes.
        for parameter in model.parameters(): parameter.grad=None
        del batch

    event=report.begin('release_optimizer_and_tail')
    del optimizer,tail
    gc.collect(); xm.mark_step(wait=True)
    event['status']='complete'; report.memory(xm,device,event)

    event=report.begin('eval_262144_batch',tokens_per_rank=262144)
    batch=synthetic_batch(262144,device,training=False)
    model.ngram_cache=batch.cache
    model.eval(); xm.mark_step(wait=True)
    event['status']='complete'; report.memory(xm,device,event)
    event=report.begin('eval_262144_compile',tokens_per_rank=262144,timing_label='compile')
    compiled_before=compile_sample()
    began=time.perf_counter()
    try:
        with torch.no_grad():
            loss=model(batch.inputs,batch.targets,batch.seqlens,batch.slots,cfg)
            xm.mark_step(wait=True)
            value=float(loss.float().mean().cpu())
        event.update(loss=value if math.isfinite(value) else None,finite_loss=math.isfinite(value))
        if not math.isfinite(value):
            report.data['finite_loss']=False
            raise FloatingPointError('Nonfinite probe evaluation loss')
        record_compile(event,compiled_before)
        event['status']='complete'; report.data['finite_loss']=True
    finally:
        event['seconds']=time.perf_counter()-began
        report.flush()
    report.memory(xm,device,event)
