#!/usr/bin/env python3
"""Prepare the shared corpus on Cloud Shell before any TPU is requested."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

PROJECT = 'tpu-builders-504820'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--gcs-uri', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(output.parent/'hf-cache')
    # Use the very same split writer and validator as training, without importing
    # the package's torch/plotting exports into the small Cloud Shell environment.
    source = Path(__file__).resolve().parents[1]/'src/rg_nanogpt_one_head/data.py'
    spec = importlib.util.spec_from_file_location('corpus_writer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.validate_prepared_data(output, cfg)
    except FileNotFoundError:
        needed = 2*sum(int(cfg['dataset'][s+'_tokens']) for s in ('train','val','test'))
        if shutil.disk_usage(output).free < needed + 3*1024**3:
            raise RuntimeError('Insufficient Cloud Shell temporary disk space for corpus and download buffers')
    print('Preparing shared corpus on Cloud Shell CPU. No TPUs have been requested.', flush=True)
    module.prepare_fineweb_edu(cfg, output)
    receipts = {}
    for name in ('train.bin', 'val.bin', 'test.bin', 'meta.json'):
        path = output/name
        subprocess.run(['gcloud','storage','cp',str(path),args.gcs_uri+'/'+name,
                        '--project='+PROJECT], check=True)
        receipts[name] = {'bytes':path.stat().st_size, 'sha256':module._sha256(path)}
    manifest = {'files':receipts, 'dataset':cfg['dataset'],
                'dataset_config_sha256':hashlib.sha256(json.dumps(cfg['dataset'],sort_keys=True).encode()).hexdigest()}
    complete = output/'COMPLETE.json'
    complete.write_text(json.dumps(manifest, indent=2)+'\n')
    # Gcloud verifies transfers; publish the completion marker only after all
    # files succeeded. TPU workers also verify SHA256 after downloading.
    subprocess.run(['gcloud','storage','cp',str(complete),args.gcs_uri+'/COMPLETE.json',
                    '--project='+PROJECT], check=True)
    print('Shared cloud corpus complete. TPU provisioning can now begin.', flush=True)


if __name__ == '__main__':
    main()
