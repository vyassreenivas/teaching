"""Blind role-fidelity coding.

For every confederate turn, a separate judge model reads the turn plus that
week's meeting so far and guesses which agenda (if any) it shows. The judge is
never told the assigned agenda. Fidelity = the judge recovers the assigned
agenda; strength = how clearly. Plotting these by week gives the drift curve.

This is the proposed "manipulation check" for AI confederates.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import prompts
from .llm import Backend


def judge_schema(agenda_keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "agenda": {"type": "string", "enum": agenda_keys},
            "strength_0to3": {"type": "integer"},
            "admits_hidden_goal": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": ["agenda", "strength_0to3", "admits_hidden_goal", "rationale"],
        "additionalProperties": False,
    }


def judge_team(team_path: Path, *, backend: Backend, agendas_cfg: dict,
               exp: dict, model: str, out_dir: Path) -> Path:
    out_path = out_dir / team_path.name
    if out_path.exists():
        return out_path
    team = json.loads(team_path.read_text())
    agendas = agendas_cfg["agendas"]
    menu = prompts.agenda_menu(agendas)
    schema = judge_schema(list(agendas))
    models = exp["models"]

    codes = []
    for idx, t in enumerate(team["turns"]):
        if not t["is_confederate"]:
            continue
        # Context = this week's earlier turns only, so the judge codes the
        # turn locally and is not anchored by Alex's earlier behavior.
        week_ctx = "\n".join(f"{u['speaker']}: {u['text']}"
                             for u in team["turns"][:idx] if u["week"] == t["week"])
        r = backend.call(role="judge", model=model,
                         effort=models["effort_judge"], system=prompts.JUDGE_SYSTEM,
                         user=prompts.judge_prompt(week_ctx or "(start of meeting)",
                                                   t["text"], menu),
                         schema=schema, max_tokens=2000,
                         meta={"team": team["condition"]["team_id"], "turn": idx})
        codes.append({"turn": idx, "week": t["week"], **(r.data or {})})

    out_path.write_text(json.dumps({"condition": team["condition"], "judge": model,
                                    "codes": codes}, indent=1))
    return out_path


def judge_models(exp: dict) -> list[str]:
    """A panel of judges if configured, else the single judge."""
    m = exp["models"]
    return m.get("judges") or [m["judge"]]


def judge_all(teams_dir: Path, judged_dir: Path, *, backend: Backend,
              agendas_cfg: dict, exp: dict, workers: int = 8) -> list[Path]:
    """Code every team with every judge model. Each judge writes to its own
    folder, judged/<model>/, so judges never see each other's codes."""
    paths = sorted(teams_dir.glob("team_*.json"))
    jobs = []
    for model in judge_models(exp):
        out_dir = judged_dir / model
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs += [(p, model, out_dir) for p in paths]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda j: judge_team(j[0], backend=backend,
                                                  agendas_cfg=agendas_cfg, exp=exp,
                                                  model=j[1], out_dir=j[2]), jobs))
