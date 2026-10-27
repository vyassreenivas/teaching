"""Command line entry point.

    python -m confed.cli estimate configs/experiment_pilot.yaml
    python -m confed.cli run      configs/experiment_smoke.yaml --backend mock
    python -m confed.cli judge    outputs/smoke --backend mock
    python -m confed.cli analyze  outputs/smoke
    python -m confed.cli all      configs/experiment_smoke.yaml --backend anthropic
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from . import __version__
from .analyze import analyze
from .judge import judge_all
from .llm import make_backend
from .sim import build_conditions, run_team

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"

# Published per-million-token prices (USD) used only for the cost estimate.
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00),
          "claude-haiku-5-5": (0.10, 0.50)}


def _yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def _load(exp_path: Path):
    exp = _yaml(exp_path)
    return (exp, _yaml(CONFIGS / exp["venture"]), _yaml(CONFIGS / "founders.yaml"),
            _yaml(CONFIGS / "agendas.yaml"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def cmd_estimate(args):
    exp, venture, founders, _ = _load(Path(args.config))
    conds = build_conditions(exp)
    weeks = min(exp.get("max_weeks", 99), len(venture["weeks"]))
    agents = len(founders["founders"]) + 1
    k = exp.get("turns_per_agent_per_week", 2)
    talk = weeks * k * agents
    per_team = talk + weeks + weeks * len(founders["founders"])
    judge = weeks * k
    # Rough token model: average prompt is half the final transcript
    # (~180 tokens per turn) plus ~600 tokens of system/instructions; ~400
    # output tokens per call including thinking. Ignores cache discounts, so
    # this is an upper-end figure.
    avg_in = 600 + 180 * talk / 2
    calls = len(conds) * (per_team + judge)
    pin, pout = PRICES.get(exp["models"]["agents"], (4.0, 20.0))
    cost = calls * (avg_in * pin + 400 * pout) / 1e6
    print(f"{exp['name']}: {len(conds)} teams, {weeks} weeks, "
          f"{per_team} sim calls + {judge} judge calls per team")
    print(f"Total calls: {calls:,}   Est. cost before caching: ~${cost:,.0f}")


def cmd_run(args):
    exp_path = Path(args.config)
    exp, venture, founders, agendas = _load(exp_path)
    run_dir = Path(args.out or ROOT / "outputs" / exp["name"])
    teams_dir = run_dir / "teams"
    teams_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"version": __version__, "git_commit": _git_commit(),
                "backend": args.backend, "python": platform.python_version(),
                "experiment": exp, "argv": sys.argv}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for name in ("agendas.yaml", "founders.yaml", exp["venture"]):
        (run_dir / name).write_text((CONFIGS / name).read_text())  # frozen copy

    backend = make_backend(args.backend, run_dir / "calls.jsonl", exp.get("seed", 0))
    conds = build_conditions(exp)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda c: run_team(c, backend=backend, venture_cfg=venture,
                                         founders_cfg=founders, agendas_cfg=agendas,
                                         exp=exp, out_dir=teams_dir), conds))
    print(f"Ran {len(conds)} teams -> {teams_dir}")
    return run_dir


def cmd_judge(args):
    run_dir = Path(args.run_dir)
    exp = json.loads((run_dir / "manifest.json").read_text())["experiment"]
    agendas = _yaml(run_dir / "agendas.yaml")
    backend = make_backend(args.backend, run_dir / "calls.jsonl", exp.get("seed", 0))
    paths = judge_all(run_dir / "teams", run_dir / "judged", backend=backend,
                      agendas_cfg=agendas, exp=exp, workers=args.workers)
    print(f"Judged {len(paths)} teams -> {run_dir / 'judged'}")


def cmd_analyze(args):
    print(f"Wrote {analyze(Path(args.run_dir))}")


def cmd_all(args):
    run_dir = cmd_run(args)
    args.run_dir = run_dir
    cmd_judge(args)
    cmd_analyze(args)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="confed")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, with_config=True):
        if with_config:
            p.add_argument("config")
        p.add_argument("--backend", choices=["mock", "anthropic"], default="mock")
        p.add_argument("--workers", type=int, default=8)

    p = sub.add_parser("estimate"); p.add_argument("config"); p.set_defaults(f=cmd_estimate)
    p = sub.add_parser("run"); common(p); p.add_argument("--out"); p.set_defaults(f=cmd_run)
    p = sub.add_parser("judge"); p.add_argument("run_dir"); common(p, False); p.set_defaults(f=cmd_judge)
    p = sub.add_parser("analyze"); p.add_argument("run_dir"); p.set_defaults(f=cmd_analyze)
    p = sub.add_parser("all"); common(p); p.add_argument("--out"); p.set_defaults(f=cmd_all)

    args = ap.parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()
