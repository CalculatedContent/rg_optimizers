"""Stock GPT-2 Small dimensions; importable without PyTorch for launch plans."""
from dataclasses import dataclass

ARCHITECTURE = 'gpt2-small-stock-v1'


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError('GPT dimensions must be positive')
        if self.n_embd % self.n_head:
            raise ValueError('n_embd must be divisible by n_head')
