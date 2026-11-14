"""One fresh paired seed at the previously failing condition; no new sweep."""
import json
import time
import datetime
import concurrent.futures
import multiprocessing
import study as s


def main():
    p = s.OUT
    protocol = json.loads((p / 'Replication_Protocol.json').read_text())
    assert s.sha(p / 'study.py') == protocol['source_sha256']
    for name, digest in protocol['data_hashes'].items():
        assert s.sha(p / 'data' / name) == digest
    assert s.torch.__version__ == protocol['versions']['torch']
    assert s.np.__version__ == protocol['versions']['numpy']
    assert s.ww.__version__ == protocol['versions']['weightwatcher']
    assert not (p / 'Replication_Launch.json').exists(), 'Do not duplicate this launch'
    specs = protocol['specifications']; initial = []
    for spec in specs:
        s.torch.manual_seed(spec['seed']); initial.append(s.weight_hash(s.MLP()))
    assert len(set(initial)) == 1
    s.DATA = s.load_data()
    start = time.time(); deadline = start + protocol['maximum_training_seconds']
    s.writej(p / 'Replication_Launch.json', {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'planned': 3, 'specifications': specs, 'initial_hash': initial[0],
        'source_sha256': protocol['source_sha256'], 'data_hashes_match': True,
        'versions_match': True, 'deadline_epoch_seconds': deadline, 'workers': 3})
    print('LAUNCHED 3 paired runs; seed31006, batch128, LR x2', flush=True)
    returned = []; failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('fork')) as pool:
        futures = {pool.submit(s.train, spec, deadline): spec for spec in specs}
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                returned.append({'spec': result['spec'], 'status': result['status'], 'last_epoch': result['last_epoch']})
            except Exception as exc:
                failures.append({'spec': futures[future], 'exception': repr(exc)})
            status = {'planned': 3, 'completed': sum(r['status'] == 'complete' for r in returned),
                      'results': returned, 'failures': failures, 'wall_seconds': time.time() - start}
            s.writej(p / 'Replication_Status.json', status)
            print(json.dumps(status), flush=True)
    assert s.sha(p / 'study.py') == protocol['source_sha256']
    s.writej(p / 'Replication_Completion.json', {**status, 'training_active': False,
             'original_experiment_unchanged': True, 'next_experiment_started': False,
             'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print('REPLICATION TRAINING FINISHED', flush=True)


if __name__ == '__main__':
    main()
