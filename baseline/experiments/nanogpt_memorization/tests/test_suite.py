import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("memorization_run", HERE / "run.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
CFG = json.loads((HERE / "configs/suite.json").read_text())


def study(condition, stage="smoke", seed=1337):
    return m.Study(CFG, condition, stage, seed, batch=32)


def test_pack_only_scores_suffix_and_has_correct_shift():
    r = m.Record("a", "a", (7, 8), (9, 10))
    x, y = m.pack([r], 8)
    assert x.tolist() == [[0, 0, 0, 0, 0, 7, 8, 9]]
    assert y.tolist() == [[-100] * 6 + [9, 10]]
    with pytest.raises(ValueError):
        m.pack([r], 2)


def test_exact_lifetime_doses_and_no_post_withdrawal_presentations():
    s = study("verbatim")
    observed = Counter(r.eid for step in range(s.steps) for r in s.sample(step))
    for r in s.audit:
        assert observed[r.eid] == int(r.group.split("_")[-1])
    assert all(r.eid == "background" for r in s.sample(s.boundary))


def test_counterfactual_replaces_only_injection_slots():
    present, absent = study("verbatim"), study("verbatim_absent")
    assert present.audit == absent.audit
    assert present.injections == absent.injections
    for step in range(present.steps):
        for i, (a, b) in enumerate(zip(present.sample(step), absent.sample(step))):
            if step * 32 + i not in present.injections:
                assert a == b
            else:
                assert a.eid.startswith("canary") and b.eid == "background"
    assert not any(r.eid.startswith("canary") for r in absent.sample(0))


def test_sampling_reproducible_and_not_consumed_by_probes():
    a, b = study("associations"), study("associations")
    expected = a.sample(1)
    a.probes(final=True); a.sample(0)
    assert a.sample(1) == expected == b.sample(1)
    assert study("associations", seed=2027).sample(1) != expected


def test_rule_split_is_pair_disjoint_and_exhaustive():
    s = study("rule_half_noise")
    train = {r.eid for r in s.train}
    val = {r.eid for r in s.audit if r.group == "validation_rule"}
    test = {r.eid for r in s.audit if r.group == "test_rule"}
    assert not train & val and not train & test and not val & test
    assert len(train | val | test) == CFG["modulus"] ** 2
    assert not any(r.group == "test_rule" for r in s.probes())
    assert any(r.group == "test_rule" for r in s.probes(final=True))


def test_label_noise_is_frozen_and_holdouts_unchanged():
    clean, half, full = [study(c) for c in ("rule_clean", "rule_half_noise", "rule_random")]
    assert [r.eid for r in clean.train] == [r.eid for r in full.train]
    assert sum(r.group.endswith("randomized") for r in half.train) == round(len(half.train)/2)
    assert all(r.group.endswith("randomized") for r in full.train)
    assert [r for r in clean.audit if r.group.endswith("_rule")] == [r for r in full.audit if r.group.endswith("_rule")]
    assert any(a.target != b.target for a, b in zip(clean.train, full.train))


def test_association_controls_are_disjoint_and_template_is_unseen():
    s = study("associations")
    seen = {r.prompt[2:-1] for r in s.train}
    unseen = {r.prompt[2:-1] for r in s.audit if r.group == "unseen_key_control"}
    assert len(seen) == len(unseen) == CFG["associations"] and not seen & unseen
    assert {r.prompt[1] for r in s.sample(0)} == {4, 5}
    assert all(r.prompt[1] == 6 for r in s.audit if r.group == "seen_key_new_template")


def test_forgetting_arms_have_identical_acquisition_but_different_interference():
    a, b = study("forgetting_disjoint"), study("forgetting_conflict")
    assert a.train == b.train and a.sample(0) == b.sample(0)
    assert {r.prompt for r in a.train}.isdisjoint({r.prompt for r in a.second})
    assert [r.prompt for r in b.train] == [r.prompt for r in b.second]
    assert all(x.target != y.target for x, y in zip(b.train, b.second))
    assert all(r.eid.startswith("B_") for r in a.sample(a.boundary))


def test_prefix_ablation_only_shortens_prompt_not_suffix():
    s = study("verbatim")
    probes = s.probes(final=True)
    for record in s.audit:
        rows = [r for r in probes if r.eid == record.eid]
        assert {len(r.prompt) for r in rows} == set(CFG["prefix_lengths_final"])
        assert all(r.target == record.target for r in rows)


def test_shared_aux_control_matches_decay_not_just_coefficient():
    src = {"optimizer_profiles": {
        "adamw": {"learning_rate": 6e-4, "min_learning_rate": 6e-5, "weight_decay": .1,
                  "warmup_fraction": .01, "lr_schedule_epochs": 1, "beta1": .9, "beta2": .95, "epsilon": 1e-8},
        "muon": {"matrix_learning_rate": .02, "matrix_min_learning_rate": .002}}}
    mu = m.resolve_profile(src, "muon", "shared_aux_decay")
    ad = m.resolve_profile(src, "adamw", "shared_aux_decay")
    assert mu["aux_learning_rate"] == ad["learning_rate"]
    assert mu["warmup_fraction"] == ad["warmup_fraction"]
    assert mu["matrix_learning_rate"] * mu["matrix_weight_decay"] == pytest.approx(ad["learning_rate"] * ad["weight_decay"])
    assert mu["matrix_min_learning_rate"] * mu["matrix_weight_decay"] == pytest.approx(ad["min_learning_rate"] * ad["weight_decay"])
    assert "aux_learning_rate" not in src["optimizer_profiles"]["muon"]


def test_interrupted_jsonl_only_allows_last_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1}\n{"step":')
    assert m.read_rows(path) == [{"step": 1}]
    path.write_text('{broken}\n{"step": 1}\n')
    with pytest.raises(json.JSONDecodeError):
        m.read_rows(path)


def test_evaluation_distinguishes_exact_and_token_recall():
    torch = pytest.importorskip("torch")
    class Uniform:
        cfg = SimpleNamespace(block_size=8)
        training = True
        def train(self, value=True): self.training = value
        def eval(self): self.training = False
        def __call__(self, x): return torch.zeros((*x.shape, 16)), None
        def generate_greedy(self, prompt, n):
            return torch.cat([prompt, torch.ones((len(prompt), n), dtype=torch.long)], dim=1)
    model = Uniform()
    out = m.evaluate(model, [m.Record("a", "g", (3,), (1, 1)), m.Record("b", "g", (3,), (2, 1))], 2, "cpu")
    assert out["g"]["mean"]["nll"] == pytest.approx(np.log(16))
    assert out["g"]["mean"]["exact_match"] == .5
    assert out["g"]["mean"]["continuation_token_accuracy"] == .75
    assert model.training


def test_all_conditions_produce_valid_bounded_batches():
    for condition in CFG["conditions"]:
        s = study(condition)
        for step in range(s.steps):
            x, y = m.pack(s.sample(step), 256)
            assert x.shape == y.shape == (32, 256)
            assert (x >= 0).all() and (x < 50257).all()
            assert ((y == -100) | ((y >= 0) & (y < 50257))).all()
            assert (y != -100).any(axis=1).all()


def test_weightwatcher_call_is_clipped_and_does_not_change_rng_or_weights(monkeypatch, tmp_path):
    """Mocked API contract test, NOT a test of actual WW numerical fits."""
    torch = pytest.importorskip("torch")
    pd = pytest.importorskip("pandas")
    from types import ModuleType
    model = torch.nn.Linear(32, 32, bias=False)
    before = model.weight.detach().clone()
    module = ModuleType("rg_nanogpt_one_head.model")
    module.transformer_matrix_items = lambda model: [("L00_W_Q", "W_Q", 0, model.weight)]
    monkeypatch.setitem(sys.modules, "rg_nanogpt_one_head.model", module)
    fake = ModuleType("weightwatcher")
    calls = []
    class Watcher:
        def __init__(self, model): self.model = model
        def analyze(self, **kwargs):
            calls.append(kwargs)
            np.random.rand(10); torch.rand(10)
            return pd.DataFrame([dict(longname="L00_W_Q", alpha=2.1, raw_alpha=2.4, D=.1,
                                      rand_distance=.2, ERG_gap=1, num_traps=0, num_pl_spikes=24)])
    fake.WeightWatcher = Watcher
    monkeypatch.setitem(sys.modules, "weightwatcher", fake)
    monkeypatch.setattr(m.importlib.metadata, "version", lambda name: "0.7.7")
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    m.weightwatch(model, tmp_path / "ww.csv", CFG["weightwatcher"], 1337, 0)
    assert torch.equal(before, model.weight)
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert calls == [{k:v for k,v in CFG["weightwatcher"].items() if k != "version"}]
    table = pd.read_csv(tmp_path / "ww.csv")
    assert table["alpha_clip_xmax"].iloc[0] == 2.1
    assert table["alpha_raw"].iloc[0] == 2.4
    assert table["matrix_name"].iloc[0] == "L00_W_Q"
