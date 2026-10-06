"""CPU-readable final weights, including the table omitted by upstream state_dict.

This exports weights for analysis, not a resumable optimizer checkpoint. It adds
no callbacks to model forwards, optimizer updates, or the timed training loop.
"""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, payload):
    import torch
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    return {'file':path.name, 'bytes':path.stat().st_size, 'sha256':file_hash(path)}


def export_rank(root, model, table, *, rank, world_size, step, chunk_rows=262144):
    root = Path(root)
    folder = root/f'rank-{rank:02d}'
    folder.mkdir(parents=True)  # refuse to overwrite an earlier export
    files = []
    if rank == 0:
        weights = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        # These inference buffers are intentionally non-persistent upstream.
        evaluation = {name:getattr(model,name).detach().cpu().clone()
                      for name in ('canon_mask', 'prefix_table')}
        files.append(save(folder/'model.pt', {'model':weights, 'evaluation_buffers':evaluation,
                           'step':step, 'ngram_table':'All rank-*/ngram-*.pt chunks are also required',
                           'optimizer_state_saved':False}))
        del weights, evaluation
    rows, width = table.shard.shape
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        weights = table.shard[start:stop].detach().cpu().clone()
        receipt = save(folder/f'ngram-{start:09d}.pt',
                       {'first_row':table.first_row + start, 'weights':weights})
        files.append({**receipt, 'first_row':table.first_row + start, 'rows':stop-start})
        del weights
    manifest = {'rank':rank, 'world_size':world_size, 'step':step,
                'first_row':table.first_row, 'rows':rows, 'width':width,
                'files':files, 'status':'complete', 'optimizer_state_saved':False}
    temporary = folder/'EXPORT.tmp'
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(folder/'EXPORT.json')


def verify_export(root, world_size, total_rows):
    root = Path(root)
    next_row = 0
    expected_step = None
    ranks = []
    for rank in range(world_size):
        folder = root/f'rank-{rank:02d}'
        data = json.loads((folder/'EXPORT.json').read_text())
        if data['rank'] != rank or data['world_size'] != world_size or data['status'] != 'complete':
            raise RuntimeError('Incomplete or mismatched weight-export rank')
        if data['first_row'] != next_row:
            raise RuntimeError('Missing or overlapping n-gram rows')
        if expected_step is not None and data['step'] != expected_step:
            raise RuntimeError('Weight-export steps disagree')
        expected_step = data['step']
        chunk_row = next_row
        if rank == 0 and not any(f['file']=='model.pt' for f in data['files']):
            raise RuntimeError('Dense model weights missing')
        for item in data['files']:
            if item['file'].startswith('ngram-'):
                if item['first_row'] != chunk_row or item['rows'] <= 0:
                    raise RuntimeError('Missing or overlapping n-gram chunks')
                chunk_row += item['rows']
            path = folder/item['file']
            if path.stat().st_size != item['bytes'] or file_hash(path) != item['sha256']:
                raise RuntimeError('Weight-export integrity failure: '+str(path))
        next_row += data['rows']
        if chunk_row != next_row:
            raise RuntimeError('N-gram chunks do not cover their rank')
        ranks.append(data)
    if next_row != total_rows:
        raise RuntimeError('N-gram export does not cover the full table')
    result = {'status':'complete', 'step':expected_step, 'ranks':ranks,
              'optimizer_state_saved':False, 'purpose':'post-run weight analysis; not automatic resume'}
    temporary = root/'WEIGHTS_COMPLETE.tmp'
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(root/'WEIGHTS_COMPLETE.json')
    return result
