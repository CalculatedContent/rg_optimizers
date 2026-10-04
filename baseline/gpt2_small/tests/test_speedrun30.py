import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

BASE = Path(__file__).resolve().parents[1]/"speedrun30"
sys.path.insert(0, str(BASE))
import train as speed
import worker as supervisor


def small_model():
    return speed.reference.GPT(speed.reference.GPTConfig(
        block_size=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16))


def test_accumulated_update_matches_reference_adamw():
    torch.set_num_threads(1)
    model = small_model()
    direct = copy.deepcopy(model)
    opt = speed.optimizer_for(model, torch.device("cpu"))
    # Upstream optimizer grouping/AdamW, using its own function as the oracle.
    speed.reference.master_process = True
    ref_opt = direct.configure_optimizers(0.1, speed.PEAK_LR, (0.9, 0.95), "cpu", 0)
    generator = torch.Generator().manual_seed(83)
    for step in range(3):
        x = torch.randint(32, (8,8), generator=generator)
        y = torch.randint(32, (8,8), generator=generator)
        opt.zero_grad(set_to_none=False)
        ref_opt.zero_grad(set_to_none=False)
        for i in range(4):
            _, loss = model(x[2*i:2*i+2], y[2*i:2*i+2])
            (loss/4).backward()
        speed.clip_gradients(model)
        _, loss = direct(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.)
        for optimizer in (opt, ref_opt):
            for group in optimizer.param_groups:
                group["lr"] = speed.learning_rate(step)
            optimizer.step()
        for a,b in zip(model.parameters(), direct.parameters()):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)


def write_shard(path, count):
    header = np.zeros(256, dtype="<i4")
    header[:3] = [20240520, 1, count]
    with path.open("wb") as f:
        header.tofile(f)
        (np.arange(count)%32).astype("<u2").tofile(f)


def test_lazy_stream_matches_upstream_shard_windows(tmp_path):
    import hashlib
    names = ["fineweb_train_000001.bin", "fineweb_train_000002.bin"]
    files = {}
    for name in names:
        path = tmp_path/name
        write_shard(path, 57)
        files[name] = {"sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "size":path.stat().st_size}
        speed.write_json(path.with_suffix(".verified.json"), files[name])
    source = speed.FineWeb(tmp_path, time.time()+60)
    source.manifest["files"] = files
    actual = speed.TrainStream(source, batch=2, context=4)
    expected = speed.reference.DistributedDataLoader(str(tmp_path/"*.bin"), 2, 4, 0, 1)
    for _ in range(20):
        x,y = actual.next_batch()
        rx,ry = expected.next_batch()
        assert torch.equal(x,rx) and torch.equal(y,ry)


def test_incorrect_download_is_never_promoted(tmp_path, monkeypatch):
    import io
    source = speed.FineWeb(tmp_path, time.time()+60)
    monkeypatch.setattr(speed.urllib.request, "urlopen", lambda *a,**k:io.BytesIO(b"bad"))
    with pytest.raises(RuntimeError, match="differs"):
        source.array("fineweb_val_000000.bin")
    assert not (tmp_path/"fineweb_val_000000.bin").exists()


def test_schedule_and_reference_comparison_do_not_extrapolate():
    assert speed.learning_rate(0) == pytest.approx(0.0006/700)
    assert speed.learning_rate(699) == pytest.approx(0.0006)
    assert speed.learning_rate(19560) == pytest.approx(0.)
    b = speed.reference_bracket(250)
    assert b["published_lower"] == b["published_upper"]
    assert b["published_lower"]["val_nll"] == 6.171
    b = speed.reference_bracket(270)
    assert b["published_lower"]["step"] == 250
    assert b["published_upper"]["step"] == 500


def test_wallclock_limit_kills_stalled_child():
    start = time.monotonic()
    result = supervisor.bounded([sys.executable, "-c", "import time; time.sleep(30)"], 0.15)
    assert result["timed_out"]
    assert result["exit_code"] != 0
    assert time.monotonic()-start < 3


def test_fresh_run_does_not_stop_existing_active_speedrun(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("speedrun_launcher", BASE/"cloudshell.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pointer = tmp_path/"SPEEDRUN30_LATEST.json"
    pointer.write_text(json.dumps({"unit":"already-running"}))
    monkeypatch.setattr(module, "BASE", tmp_path)
    monkeypatch.setattr(module, "LATEST", pointer)
    monkeypatch.setattr(module.os, "geteuid", lambda:0)
    monkeypatch.setattr(module.os.path, "ismount", lambda x:True)
    monkeypatch.setattr(module, "active", lambda unit:True)
    observed = []
    monkeypatch.setattr(module, "status_remote", lambda:observed.append("status"))
    monkeypatch.setattr(module, "stop_current", lambda:pytest.fail("must not stop an active speedrun"))
    module.start_remote("a"*40)
    assert observed == ["status"]


def test_long_run_refuses_concurrent_reference_job(monkeypatch, tmp_path):
    path = BASE.parent/"scripts/run_muonclip.py"
    spec = importlib.util.spec_from_file_location("speedrun_idle_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "BASE", tmp_path)
    (tmp_path/"SPEEDRUN30_LATEST.json").write_text(json.dumps({"unit":"speedrun.service"}))
    monkeypatch.setattr(module, "active", lambda unit:unit == "speedrun.service")
    with pytest.raises(RuntimeError, match="reference run is active"):
        module.assert_idle()
