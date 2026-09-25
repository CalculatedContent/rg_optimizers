from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class RandomCanaryExperiment:
    """Controlled random-token memorization embedded in ordinary FineWeb training.

    Each canary is a full block_size+1 random GPT-2 token sequence. Dose is the
    exact number of times that sequence replaces one ordinary FineWeb training
    window. Dose-0 canaries are never injected and are negative controls.
    """

    def __init__(self, cfg: dict, *, seed: int, total_steps: int, run_dir: Path):
        spec = cfg["memorization"]
        self.seed = int(seed)
        self.block_size = int(cfg["model"]["block_size"])
        self.batch_size = int(cfg["training"]["batch_size"])
        self.grad_accum = int(cfg["training"]["grad_accum_steps"])
        self.prefix = int(spec["prefix_tokens"])
        self.suffix = int(spec["suffix_tokens"])
        self.doses = tuple(int(x) for x in spec["doses"])
        self.per_dose = int(spec["canaries_per_dose"])
        self.vocab = int(cfg["model"]["vocab_size"])
        self.eot = 50256 if self.vocab == 50257 else self.vocab - 1
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if self.prefix + self.suffix > self.block_size:
            raise ValueError("memorization prefix+suffix exceeds block_size")
        if 0 not in self.doses:
            raise ValueError("memorization doses must include zero control")

        # Same canary content for both optimizer arms of a seed.
        rng = np.random.default_rng(np.random.SeedSequence([
            int(spec["data_seed"]), self.seed, 1701
        ]))
        allowed = np.arange(self.vocab - 1, dtype=np.int64)
        self.canaries = []
        for dose in self.doses:
            for index in range(self.per_dose):
                seq = rng.choice(allowed, size=self.block_size + 1, replace=True)
                self.canaries.append({
                    "id": f"dose{dose}_canary{index}",
                    "dose": dose,
                    "tokens": torch.tensor(seq, dtype=torch.long),
                })

        # Spread exact presentations uniformly/randomly through the acquisition
        # window. One injection occupies one (update,microbatch,row) slot.
        acquisition_fraction = float(spec.get("acquisition_fraction", 0.5))
        acquisition_steps = max(1, min(total_steps, int(round(total_steps * acquisition_fraction))))
        slots = acquisition_steps * self.grad_accum * self.batch_size
        presentations = [(i, c["id"]) for i, c in enumerate(self.canaries) for _ in range(c["dose"])]
        if len(presentations) > slots:
            raise ValueError("canary presentations exceed available acquisition slots")
        srng = np.random.default_rng(np.random.SeedSequence([
            int(spec["data_seed"]), self.seed, 1702
        ]))
        chosen = srng.choice(slots, size=len(presentations), replace=False)
        self.schedule = {}
        by_id = {c["id"]: c for c in self.canaries}
        for slot, (_, cid) in zip(chosen.tolist(), presentations):
            step = slot // (self.grad_accum * self.batch_size)
            rem = slot % (self.grad_accum * self.batch_size)
            micro = rem // self.batch_size
            row = rem % self.batch_size
            self.schedule[(int(step), int(micro), int(row))] = by_id[cid]

        manifest = {
            "schema_version": 1,
            "background": "FineWeb-Edu next-token language modeling",
            "seed": self.seed,
            "doses": list(self.doses),
            "canaries_per_dose": self.per_dose,
            "prefix_tokens": self.prefix,
            "suffix_tokens": self.suffix,
            "acquisition_steps": acquisition_steps,
            "total_presentations": len(presentations),
            "canaries": [{"id": c["id"], "dose": c["dose"]} for c in self.canaries],
            "schedule": [
                {"step": k[0], "micro": k[1], "row": k[2], "id": v["id"]}
                for k, v in sorted(self.schedule.items())
            ],
        }
        (self.run_dir / "random_canary_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        self.csv_path = self.run_dir / "random_canary_metrics.csv"

    @classmethod
    def from_config(cls, cfg, *, seed, total_steps, run_dir):
        spec = cfg.get("memorization")
        if not spec or not bool(spec.get("enabled", False)):
            return None
        return cls(cfg, seed=seed, total_steps=total_steps, run_dir=run_dir)

    def inject(self, x: torch.Tensor, y: torch.Tensor, *, completed_step: int, micro_index: int):
        x = x.clone()
        y = y.clone()
        for row in range(x.shape[0]):
            c = self.schedule.get((int(completed_step), int(micro_index), int(row)))
            if c is None:
                continue
            seq = c["tokens"]
            x[row] = seq[:-1]
            y[row] = seq[1:]
        return x, y

    @torch.inference_mode()
    def _score_one(self, model, tokens: torch.Tensor, device: torch.device):
        prefix = tokens[:self.prefix].to(device)
        target = tokens[self.prefix:self.prefix + self.suffix].to(device)

        # Teacher-forced conditional NLL and token accuracy over the suffix.
        full = torch.cat([prefix, target]).unsqueeze(0)
        logits, _ = model(full[:, :-1], None)
        pred_logits = logits[:, self.prefix - 1:self.prefix - 1 + self.suffix, :]
        nll = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            target.reshape(-1),
            reduction="mean",
        )
        teacher = pred_logits.argmax(-1).reshape(-1).eq(target).float().mean()

        # Free-running continuation recall.
        generated = prefix.unsqueeze(0)
        out = []
        for _ in range(self.suffix):
            glogits, _ = model(generated, None)
            nxt = glogits[:, -1, :].argmax(-1, keepdim=True)
            out.append(nxt)
            generated = torch.cat([generated, nxt], dim=1)
        generated_suffix = torch.cat(out, dim=1).reshape(-1)
        hit = generated_suffix.eq(target)
        return float(nll.cpu()), float(teacher.cpu()), float(hit.float().mean().cpu()), float(hit.all().cpu())

    def evaluate(self, model, *, device: torch.device, step: int, epoch: float):
        was_training = model.training
        model.eval()
        rows = []
        try:
            for c in self.canaries:
                nll, teacher, token_acc, exact = self._score_one(model, c["tokens"], device)
                rows.append({
                    "step": int(step), "epoch": float(epoch), "id": c["id"],
                    "dose": int(c["dose"]), "nll": nll,
                    "teacher_accuracy": teacher,
                    "token_accuracy": token_acc, "exact_match": exact,
                })
        finally:
            model.train(was_training)

        exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

        exposed = [r for r in rows if r["dose"] > 0]
        zero = [r for r in rows if r["dose"] == 0]
        return {
            "exact_match": sum(r["exact_match"] for r in exposed) / max(1, len(exposed)),
            "token_accuracy": sum(r["token_accuracy"] for r in exposed) / max(1, len(exposed)),
            "zero_exact_match": sum(r["exact_match"] for r in zero) / max(1, len(zero)),
        }
