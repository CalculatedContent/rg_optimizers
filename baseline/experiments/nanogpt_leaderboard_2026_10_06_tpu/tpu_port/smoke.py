"""Full learned dense shapes, tiny token batch on CPU; not a TPU qualification."""
import torch
from .gpt import GPT, ForwardScheduleConfig


def smoke():
    torch.set_num_threads(2)
    torch.manual_seed(1337)
    model=GPT(50257,11,6,128,768,16,ngram_dim=768,world_size=8,device=torch.device('cpu'))
    model.cast_matrix_weights_bf16()
    inputs=torch.arange(16,dtype=torch.int32)
    cfg=ForwardScheduleConfig(torch.tensor([1.,.5,.25]),torch.tensor([.25]),128,384,896)
    sink=torch.zeros(32,768,dtype=torch.bfloat16,requires_grad=True)
    losses=model(inputs,inputs.long(),torch.tensor([0,8,16],dtype=torch.int32),torch.arange(32),cfg,sink)
    losses.sum().backward()
    missing=[name for name,p in model.named_parameters() if p.grad is None]
    bad=[name for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if missing or bad or not torch.isfinite(sink.grad).all():
        raise RuntimeError(f'Gradient failure: missing={missing}, nonfinite={bad}')
    shapes={name:list(p.shape) for name,p in model.named_parameters()}
    return {'status':'cpu_forward_backward_passed','dense_parameters':sum(p.numel() for p in model.parameters()),
            'ngram_table_parameters':84602880*768,'token_rows':16,
            'shapes':shapes,'training_loss':float(losses.detach().mean()),
            'tpu_execution_verified':False,'full_table_allocated':False,'convergence_verified':False}
