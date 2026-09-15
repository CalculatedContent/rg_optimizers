"""Unit tests for adaptive spectral-guard control logic."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import spectral_guard_train as g


def table(values):
    rows = []
    for i, (name, clip, raw) in enumerate(values):
        rows.append({
            "step": 125,
            "matrix_name": name,
            "alpha_clip_xmax": clip,
            "alpha_raw": raw,
        })
    return pd.DataFrame(rows)


def controller():
    names = ["L00_W_Q", "L00_W_K", "L00_W_V", "L00_W_O", "L00_W_MLP_IN", "L00_W_MLP_OUT"]
    return {
        "lr_scale": {n: 1.0 for n in names},
        "anchor_lambda": {n: 0.0 for n in names},
        "dropout": {"0": 0.0},
    }


def test_alpha_gate_uses_worse_of_raw_and_clipped_fit():
    d = table([
        ("L00_W_Q", 2.2, 2.3),
        ("L00_W_K", 1.94, 4.0),
    ])
    s = g.classify_alpha(d, target=2.0001, floor=1.95)
    assert not s["passed"]
    assert s["worst_matrix"] == "L00_W_K"
    assert s["offenders"][0]["severity"] == "hard"


def test_muon_adaptation_targets_only_offending_matrix():
    c = controller()
    s = g.classify_alpha(
        table([
            ("L00_W_Q", 1.90, 3.0),
            ("L00_W_K", 2.20, 2.20),
        ]),
        target=2.0001,
        floor=1.95,
    )
    before = g.adapt_controller(
        c,
        s,
        optimizer_name="muon",
        mild_backoff=0.75,
        hard_backoff=0.5,
        min_lr_scale=0.01,
        anchor_start=0.5,
        anchor_growth=2.0,
        max_anchor=32.0,
        dropout_step=0.025,
        max_dropout=0.25,
    )
    assert before["lr_scale"]["L00_W_Q"] == 1.0
    assert c["lr_scale"]["L00_W_Q"] == pytest.approx(0.5)
    assert c["anchor_lambda"]["L00_W_Q"] == pytest.approx(0.5)
    assert c["lr_scale"]["L00_W_K"] == 1.0
    assert c["dropout"]["0"] == 0.0


def test_sgd_hard_violation_adds_block_dropout_and_anchor():
    c = controller()
    s = g.classify_alpha(
        table([
            ("L00_W_O", 1.80, 3.1),
            ("L00_W_MLP_OUT", 1.99, 2.5),
        ]),
        target=2.0001,
        floor=1.95,
    )
    g.adapt_controller(
        c,
        s,
        optimizer_name="sgd_momentum",
        mild_backoff=0.75,
        hard_backoff=0.5,
        min_lr_scale=0.01,
        anchor_start=0.5,
        anchor_growth=2.0,
        max_anchor=32.0,
        dropout_step=0.025,
        max_dropout=0.25,
    )
    assert c["lr_scale"]["L00_W_O"] == pytest.approx(0.5)
    assert c["lr_scale"]["L00_W_MLP_OUT"] == pytest.approx(0.75)
    assert c["dropout"]["0"] == pytest.approx(0.05)
    assert c["anchor_lambda"]["L00_W_O"] == pytest.approx(0.5)


def test_repeated_adaptation_respects_caps():
    c = controller()
    s = g.classify_alpha(
        table([("L00_W_Q", 1.0, 1.0)]),
        target=2.0001,
        floor=1.95,
    )
    for _ in range(10):
        old = deepcopy(c)
        try:
            g.adapt_controller(
                c,
                s,
                optimizer_name="sgd_momentum",
                mild_backoff=0.75,
                hard_backoff=0.5,
                min_lr_scale=0.01,
                anchor_start=0.5,
                anchor_growth=2.0,
                max_anchor=32.0,
                dropout_step=0.025,
                max_dropout=0.25,
            )
        except RuntimeError:
            assert old == c
            break
    assert c["lr_scale"]["L00_W_Q"] >= 0.01
    assert c["anchor_lambda"]["L00_W_Q"] <= 32.0
    assert c["dropout"]["0"] <= 0.25


def test_set_block_dropout_updates_attention_and_mlp_modules():
    block = SimpleNamespace(
        attn=SimpleNamespace(dropout=0.0, resid_dropout=SimpleNamespace(p=0.0)),
        mlp=SimpleNamespace(dropout=SimpleNamespace(p=0.0)),
    )
    model = SimpleNamespace(blocks=[block])
    g.set_block_dropout(model, {"0": 0.125})
    assert block.attn.dropout == pytest.approx(0.125)
    assert block.attn.resid_dropout.p == pytest.approx(0.125)
    assert block.mlp.dropout.p == pytest.approx(0.125)
