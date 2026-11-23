"""Select the graft reference rate on validation, then evaluate one paired seed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RATES = (0.05, 0.1, 0.2, 0.4, 0.8)
ORIGINAL = 0.1
SCREEN = 4000
BUDGET = 20000
CALIBRATION_SEED = 2027
CONFIRMATION_SEED = 1340


def rate_dir(output, phase, rate):
    return output / phase / f"rate_{rate:g}"


def worker(args):
    import run
    run.runtime.DEVICE = args.device
    run.runtime.initialize()
    seed = CALIBRATION_SEED if args.phase == "calibration" else CONFIRMATION_SEED
    data = run.base.load_data(args.data)
    if args.phase == "calibration":
        del data["test"]
    run.RATIO = args.rate / run.ADAM["peak_lr"]
    folder = rate_dir(args.output, args.phase, args.rate)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = run.design()
    cfg["recipes"]["graft"]["reference"] = dict(run.SGDM, peak_lr=args.rate)
    cfg.update(phase=args.phase, study_seed=seed, available_splits=sorted(data),
               driver_sha256=run.base.sha(Path(__file__)))
    manifest = folder / "design.json"
    run.previous.freeze(manifest, cfg)
    try:
        run.trial(folder, data, "graft", seed, run.SCHEDULE, args.steps, run.base.sha(manifest))
    except FloatingPointError:
        if args.phase == "calibration":
            run.base.write_json(folder / "screen.json", dict(reference_lr=args.rate,
                                step=args.steps, status="diverged", validation_loss=None))
            return
        raise
    job = folder / "graft" / f"seed_{seed}"
    if args.phase == "calibration":
        snapshot = run.torch.load(job / "last.pt", map_location="cpu", weights_only=False)
        item = next(r for r in snapshot["progress"]["history"] if r["step"] == args.steps)
        score = dict(reference_lr=args.rate, step=item["step"], status="complete",
                     validation_loss=item["validation_loss"])
        run.base.write_json(folder / f"score_{args.steps}.json", score)
        run.base.write_json(folder / "screen.json", score)


def stage(args, phase, rates, steps):
    def one(rate):
        folder = rate_dir(args.output, phase, rate)
        folder.mkdir(exist_ok=True, parents=True)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", phase, "--rate", str(rate),
               "--steps", str(steps), "--device", args.device, "--output", str(args.output),
               "--data", str(args.data)]
        with (folder / "training.log").open("a") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, rates))


def scores(args, rates):
    return [json.loads((rate_dir(args.output, "calibration", r) / "screen.json").read_text()) for r in rates]


def study(args):
    import run
    args.output.mkdir(parents=True, exist_ok=True)
    design = dict(reference_rates=list(RATES), native_peak_lr=0.001, original_reference_rate=ORIGINAL,
                  calibration_seed=CALIBRATION_SEED, confirmation_seed=CONFIRMATION_SEED,
                  screen_steps=SCREEN, final_steps=BUDGET, driver_sha256=run.base.sha(Path(__file__)),
                  method="Advance the two alternative rates with lowest validation loss at 4000 plus the original rate. Choose the best alternative by final validation loss at 20000. Ties favor the lower rate. Test only this alternative on seed 1340; archived baselines use the same seed.")
    with run.previous.lock(args.output/"study.lock"):
        run.previous.freeze(args.output/"plan.json", design)
        stage(args, "calibration", RATES, SCREEN)
        short = scores(args, RATES)
        rank = lambda x: (x["validation_loss"], x["reference_lr"])
        alts = sorted((s for s in short if s["reference_lr"] != ORIGINAL and s["status"]=="complete"), key=rank)
        if not alts:
            raise RuntimeError("No alternative reference rate completed the screen")
        finalists = [ORIGINAL] + [s["reference_lr"] for s in alts[:2]]
        stage(args, "calibration", finalists, BUDGET)
        full = scores(args, finalists)
        options = [s for s in full if s["reference_lr"]!=ORIGINAL and s["status"]=="complete"]
        if not options:
            raise RuntimeError("No alternative reference rate reached the full validation budget")
        winner = min(options, key=rank)["reference_lr"]
        run.previous.freeze(args.output/"selection.json", dict(winner=winner, candidates=full,
                            screen=short, calibration_seed=CALIBRATION_SEED, confirmation_seed=CONFIRMATION_SEED))
        stage(args, "confirmation", [winner], BUDGET)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", choices=("cpu","mps","tpu"), default="cpu")
    p.add_argument("--workers", type=int, default=1, help="Concurrent CPU trials; use one process on accelerators")
    p.add_argument("--phase", choices=("calibration","confirmation"), help=argparse.SUPPRESS)
    p.add_argument("--rate", type=float, choices=RATES, help=argparse.SUPPRESS)
    p.add_argument("--steps", type=int, choices=(SCREEN,BUDGET), help=argparse.SUPPRESS)
    opt = p.parse_args()
    if opt.workers < 1 or (opt.device != "cpu" and opt.workers != 1):
        p.error("Choose a positive worker count and a single worker on an accelerator")
    opt.data = opt.data.resolve()
    opt.output = opt.output.resolve()
    if opt.phase:
        worker(opt)
    else:
        study(opt)
