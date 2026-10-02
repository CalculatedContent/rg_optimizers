"""One fresh scientific process. No retry loop, continuation or resume option."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import threading
import time

from .continuous_support import atomic_json, publisher, publish_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--device', default='tpu', choices=['tpu', 'cpu'])
    args = parser.parse_args()
    from .muonclip import install_muonclip_extension
    install_muonclip_extension()
    from .config import load_config
    from .training import run_optimizer_replicates
    cfg = load_config(args.config)
    if not cfg.get('continuous', {}).get('enabled') or cfg.get('continuation'):
        raise ValueError('A fresh continuous-run config is required')
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    # An exclusive claim survives VM reboots and remains even after failure.
    with (root/'CONTINUOUS_STARTED.json').open('x') as f:
        import json
        json.dump(dict(pid=os.getpid(), started_unix=time.time(), start_step=0,
                       automatic_restart=False), f)
    stop = root/'STOP'
    cfg['training']['stop_file'] = str(stop.resolve())
    sink = publisher(cfg)
    if sink:
        sink.file(args.config, 'config.yaml')
        sink.file(root/'CONTINUOUS_STARTED.json', 'CONTINUOUS_STARTED.json')
    def request_stop(*unused):
        stop.touch()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    timer = threading.Timer(float(cfg['continuous']['max_wall_hours'])*3600, request_stop)
    timer.daemon = True
    timer.start()
    status = {'status':'running', 'pid':os.getpid(), 'start_step':0, 'restarts':0}
    try:
        run_optimizer_replicates(cfg=cfg, config_path=args.config,
            optimizer_name='muon_clip', seeds=(1337,), data_root=args.data_root,
            results_root=root, device=args.device, resume=False, overwrite=False)
        status['status'] = 'completed'
    except SystemExit as exc:
        status.update(status='stopped_at_checkpoint' if exc.code == 75 else 'failed', exit_code=exc.code)
        raise
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        timer.cancel()
        status['ended_unix'] = time.time()
        atomic_json(root/'CONTINUOUS_STATUS.json', status)
        if sink:
            sink.json(status, 'CONTINUOUS_STATUS.json')
            run_dir = root/'muon_clip'/'seed_1337'
            if run_dir.exists():
                publish_metadata(cfg, run_dir)


if __name__ == '__main__':
    main()
