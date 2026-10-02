#!/usr/bin/env python3
"""Prepare or recover the pinned corpus on the TPU VM's durable data disk."""
import json
import os
from pathlib import Path

from google.api_core.exceptions import NotFound
import yaml

from rg_nanogpt_one_head.continuous_support import CloudPublisher, sha_file
from rg_nanogpt_one_head.data import prepare_fineweb_edu, validate_prepared_data


def main():
    cfg = yaml.safe_load(Path('configs/muonclip_continuous8.yaml').read_text())
    data = Path('/mnt/disks/rg-data/continuous8/data')
    data.mkdir(exist_ok=True)
    archive = CloudPublisher(os.environ['RG_CONTINUOUS_DATA_URI'])
    run = CloudPublisher(os.environ['RG_CONTINUOUS_GCS_URI'])
    run.json({'status':'preparing_data','location':'TPU VM CPU and persistent disk'},'SETUP_STATUS.json')
    try:
        manifest = json.loads(archive.bucket.blob(archive.prefix+'/COMPLETE.json').download_as_text())
    except NotFound:
        manifest = None
    if manifest is not None:
        if manifest['dataset'] != cfg['dataset']:
            raise RuntimeError('Archived corpus does not match the experiment configuration')
        for name in ('train.bin','val.bin','test.bin','meta.json'):
            path = data/name
            archive.bucket.blob(archive.prefix+'/'+name).download_to_filename(str(path),checksum='crc32c',timeout=600)
            receipt = manifest['files'][name]
            if path.stat().st_size != receipt['bytes'] or sha_file(path) != receipt['sha256']:
                raise RuntimeError('Archived data checksum failed: '+name)
        validate_prepared_data(data,cfg)
    else:
        print('Preparing 5B-token corpus on the TPU VM CPU; Cloud Shell can disconnect.',flush=True)
        prepare_fineweb_edu(cfg,data)
        receipts = {}
        run.json({'status':'uploading_data'},'SETUP_STATUS.json')
        for name in ('train.bin','val.bin','test.bin','meta.json'):
            receipts[name] = archive.file(data/name,name)
            receipts[name]['sha256'] = sha_file(data/name)
        manifest = {'dataset':cfg['dataset'],'files':receipts}
        archive.json(manifest,'COMPLETE.json')
    run.json({'uri':os.environ['RG_CONTINUOUS_DATA_URI'],'manifest':manifest},'DATA_SOURCE.json')
    run.snapshot_text_file('/mnt/disks/rg-data/continuous8/run.log','run.log')
    print('Corpus complete: document-disjoint splits and SHA256 verified; cloud copy saved.',flush=True)


if __name__ == '__main__':
    main()
