"""Capacity and source-fidelity gates; these do not test TPU execution."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('leaderboard_tpu_capacity', HERE / 'capacity.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_reference_and_contract_are_frozen():
    assert audit.verify_reference()['verified_reference_files'] == 58
    contract = audit.load_contract()
    assert contract['total_steps'] == 1194
    assert contract['total_train_tokens'] == 328663040
    assert contract['validation_tokens'] == 10485760
    assert contract['target_validation_nll'] == 3.28
    assert contract['ready_to_train'] is False


@pytest.mark.parametrize('name', ['v5e-8', 'v5litepod-8', 'v4-8'])
def test_small_existing_allocations_fail(name):
    result = audit.capacity(name)
    assert result['eligible_for_allocation_review'] is False
    assert result['ngram_weight_bytes'] == 129950023680


@pytest.mark.parametrize('name,chips,hosts', [('v5p-8', 4, 1), ('v5p-16', 8, 2), ('v4-32', 16, 4)])
def test_larger_allocations_are_only_candidates(name, chips, hosts):
    result = audit.capacity(name)
    assert result['chips'] == chips
    assert result['hosts'] == hosts
    assert result['eligible_for_allocation_review'] is True
    assert result['ready_to_train'] is False
    assert result['tpu_execution_verified'] is False


def test_unknown_topology_rejected():
    with pytest.raises(ValueError, match='Unknown accelerator'):
        audit.capacity('v5p-32')


@pytest.mark.parametrize('filename', ['upstream.json', 'upstream_git_files.json', 'vendor/train_gpt.py'])
def test_reference_tamper_rejected(tmp_path, filename):
    source = HERE.parent / audit.load_contract()['reference_folder']
    copy = tmp_path / 'reference'
    shutil.copytree(source, copy, ignore=shutil.ignore_patterns('__pycache__', 'results'))
    path = copy / filename
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        audit.verify_reference(copy)


def test_cli_capacity_failure_is_nonzero(capsys):
    assert audit.main(['check-capacity', '--accelerator-type', 'v5litepod-8']) == 2
    assert json.loads(capsys.readouterr().out)['ready_to_train'] is False


def test_no_training_command():
    with pytest.raises(SystemExit) as error:
        audit.main(['run'])
    assert error.value.code == 2
