"""Offline capacity audit, not a TPU trainer or hardware qualification.

All commands use the standard library and allocate no model tensors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3
# v4/v5p names count TensorCores, not chips. v5e names count chips.
# v5e's documentation says 16 GB. Using 16 GiB is an optimistic upper
# bound; even this bound fails the planning gate for v5e-8.
HARDWARE = {
    'v5e-8': {'chips': 8, 'hosts': 1, 'hbm_gib_per_chip': 16,
              'hbm_basis': 'optimistic upper bound (Google documents 16 GB)'},
    'v4-8': {'chips': 4, 'hosts': 1, 'hbm_gib_per_chip': 32},
    'v4-32': {'chips': 16, 'hosts': 4, 'hbm_gib_per_chip': 32},
    'v5p-8': {'chips': 4, 'hosts': 1, 'hbm_gib_per_chip': 95},
    'v5p-16': {'chips': 8, 'hosts': 2, 'hbm_gib_per_chip': 95},
}
ALIASES = {'v5litepod-8': 'v5e-8'}
# Planning allowance, NOT a measured peak. Covers dense model/optimizer,
# row caches, communication, activations and XLA temporaries provisionally.
# Qualification must measure actual peaks, including compile/eval/export.
PLANNING_RESERVE_PER_CHIP = 16 * GIB


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract():
    return json.loads((HERE / 'port_contract.json').read_text())


def verify_reference(reference=None):
    contract = load_contract()
    reference = Path(reference) if reference is not None else HERE.parent / contract['reference_folder']
    for name, key in [('upstream.json', 'reference_metadata_sha256'),
                      ('upstream_git_files.json', 'reference_file_manifest_sha256')]:
        if sha256(reference / name) != contract[key]:
            raise ValueError('Pinned reference metadata changed: ' + name)
    metadata = json.loads((reference / 'upstream.json').read_text())
    for field in ('source_commit', 'architecture', 'optimizer', 'total_steps',
                  'stage_boundaries', 'stage_batch_tokens', 'total_train_tokens',
                  'validation_tokens', 'target_validation_nll'):
        if metadata[field] != contract[field]:
            raise ValueError('TPU contract differs from reference: ' + field)
    manifest = json.loads((reference / 'upstream_git_files.json').read_text())
    for relative, info in manifest.items():
        if sha256(reference / 'vendor' / relative) != info['sha256']:
            raise ValueError('Pinned reference source changed: ' + relative)
    actual = {p.relative_to(reference / 'vendor').as_posix()
              for p in (reference / 'vendor').rglob('*.py')}
    if actual != {p for p in manifest if p.endswith('.py')}:
        raise ValueError('Unexpected Python source in reference vendor directory')
    bounds, batches = contract['stage_boundaries'], contract['stage_batch_tokens']
    if len(bounds) != len(batches) + 1 or bounds[0] != 0 or bounds[-1] != contract['total_steps']:
        raise ValueError('Invalid stage boundaries')
    tokens = sum((end - start) * batch for start, end, batch in zip(bounds, bounds[1:], batches))
    if tokens != contract['total_train_tokens']:
        raise ValueError('Training token budget changed')
    return {'verified_reference_files': len(manifest), 'source_commit': contract['source_commit']}


def capacity(accelerator_type):
    name = ALIASES.get(accelerator_type, accelerator_type)
    if name not in HARDWARE:
        raise ValueError('Unknown accelerator type; add its documented chip count and HBM first')
    hardware = HARDWARE[name]
    contract = load_contract()
    rows = contract['architecture']['ngram_rows']
    width = contract['architecture']['ngram_width']
    chips = hardware['chips']
    if rows % chips:
        raise ValueError('N-gram rows must divide evenly over the planned chip shards')
    # BF16 table + FP32 second moment + int32 last event + int32 row map,
    # as represented in upstream NgramTable. No first moment (beta1=0).
    weights = rows * width * 2
    row_state = rows * (4 + 4 + 4)
    fixed_per_chip = (weights + row_state) // chips
    hbm_per_chip = hardware['hbm_gib_per_chip'] * GIB
    eligible = fixed_per_chip + PLANNING_RESERVE_PER_CHIP <= hbm_per_chip
    return {
        'accelerator_type': name,
        **hardware,
        'ngram_weight_bytes': weights,
        'ngram_row_state_bytes': row_state,
        'ngram_weight_gib': weights / GIB,
        'fixed_ngram_gib_per_chip': fixed_per_chip / GIB,
        'remaining_hbm_gib_per_chip': (hbm_per_chip - fixed_per_chip) / GIB,
        'planning_reserve_gib_per_chip': PLANNING_RESERVE_PER_CHIP / GIB,
        'eligible_for_allocation_review': eligible,
        'ready_to_train': False,
        'tpu_execution_verified': False,
        'reason': ('Passes provisional capacity gate only; trainer and live TPU qualification pending'
                   if eligible else 'Insufficient headroom for this full-size model under the planning gate'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify-reference')
    commands.add_parser('plan')
    check = commands.add_parser('check-capacity')
    check.add_argument('--accelerator-type', required=True, choices=sorted(set(HARDWARE) | set(ALIASES)))
    args = parser.parse_args(argv)
    try:
        verified = verify_reference()
        if args.command == 'verify-reference':
            result = verified
        elif args.command == 'plan':
            result = {'contract': load_contract(), 'reference': verified,
                      'capacity_options': [capacity(name) for name in HARDWARE]}
        else:
            result = capacity(args.accelerator_type)
        print(json.dumps(result, indent=2))
        if args.command == 'check-capacity' and not result['eligible_for_allocation_review']:
            return 2
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
