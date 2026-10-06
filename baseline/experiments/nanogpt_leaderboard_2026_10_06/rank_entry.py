"""Optional post-run export adapter; the vendored training source stays unchanged."""
import inspect
import os
from pathlib import Path
import runpy
import sys


def main():
    import torch.distributed as dist
    from weight_export import export_rank
    entry = Path(__file__).resolve().parent/'source/train_gpt.py'
    root = Path(os.environ['RG_LEADERBOARD_EXPORT_ROOT'])
    original = dist.destroy_process_group
    exported = False

    def finish(*args, **kwargs):
        nonlocal exported
        frame = inspect.currentframe().f_back
        try:
            # This call is after the training loop and final validation, outside all timed regions.
            if Path(frame.f_code.co_filename).resolve() == entry and frame.f_code.co_name == 'main':
                state = frame.f_locals
                if state['step'] != state['training_schedule'].total_steps:
                    raise RuntimeError('Refusing to export an incomplete run as final weights')
                export_rank(root, state['uncompiled_model'], state['ngram_table'],
                            rank=state['env'].rank, world_size=state['env'].world_size,
                            step=state['step'])
                exported = True
        finally:
            del frame
        return original(*args, **kwargs)

    dist.destroy_process_group = finish
    previous_argv, previous_path = sys.argv, sys.path[:]
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry)]  # upstream read_source must receive the upstream entry point
    try:
        runpy.run_path(str(entry), run_name='__main__')
        if not exported:
            raise RuntimeError('Upstream shutdown did not call the final weight exporter')
    finally:
        dist.destroy_process_group = original
        sys.argv = previous_argv
        sys.path[:] = previous_path


if __name__ == '__main__':
    main()
