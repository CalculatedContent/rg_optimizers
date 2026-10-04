"""Verified rolling cloud checkpoints and all scalar/spectral records for a fresh run."""
import json
import os
from pathlib import Path

from rg_nanogpt_one_head.continuous_support import CloudPublisher


class RunBackup:
    def __init__(self, output, uri, sink=None):
        self.output = Path(output)
        self.sink = sink or CloudPublisher(uri)
        saved = self.output/'cloud_checkpoint.json'
        self.sequence = json.loads(saved.read_text())['next_slot_sequence'] if saved.exists() else 0
        self.sent = set()

    def publish(self, checkpoint, step):
        # Three slots bound storage without deleting any prior experiment objects.
        # Never overwrite the slot referenced by the last successful pointer.
        name = f'muonclip/checkpoints/slot_{self.sequence % 3}.pt'
        receipt = self.sink.file(checkpoint, name)
        pointer = {**receipt, 'step': step, 'file': name,
                   'local_file': Path(checkpoint).name,
                   'next_slot_sequence': self.sequence+1,
                   'note': 'Validate object generation and CRC32C when recovering this rolling checkpoint.'}
        if step == 0:
            self.sink.file(checkpoint, 'muonclip/checkpoints/initial.pt')
        for folder in ('metrics', 'ww_metrics', 'diagnostics'):
            for path in sorted((self.output/folder).glob('*.json')):
                if path not in self.sent:
                    self.sink.file(path, 'muonclip/'+path.relative_to(self.output).as_posix())
                    self.sent.add(path)
        self.sink.file(self.output/'manifest.json', 'muonclip/manifest.json')
        self.sink.json(pointer, 'muonclip/checkpoints/LATEST_VERIFIED.json')
        tmp = self.output/'cloud_checkpoint.tmp'
        tmp.write_text(json.dumps(pointer, indent=2)); tmp.replace(self.output/'cloud_checkpoint.json')
        self.sequence += 1
        log = os.environ.get('RG_GPT2_RUN_LOG')
        if log and Path(log).is_file(): self.sink.snapshot_text_file(log, 'run.log')
        print(f'[cloud-backup] verified step={step}; three rolling slots; all metric records retained', flush=True)
