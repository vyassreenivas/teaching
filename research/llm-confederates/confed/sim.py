"""Run one synthetic founding team through the venture, week by week.

Each week: new data arrives, the four cofounders talk in a shuffled order, a
note-taker records the decision, and each human founder fills out a private
check-in about Alex. Every agent sees the full transcript so far, so the
confederate's prompt gets longer each week. That growing context is exactly
the condition under which role drift would show up.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from . import prompts
from .llm import Backend


@dataclass
class Condition:
    team_id: str
    agenda: str
    dose: str | None
    warned: bool
    replicate: int
    seed: int


def decision_schema(week: dict) -> dict:
    props: dict = {"decision_summary": {"type": "string"}}
    for name, spec in week["fields"].items():
        p = {"type": spec["type"], "description": spec["description"]}
        if "enum" in spec:
            p["enum"] = spec["enum"]
        props[name] = p
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


def survey_schema(agenda_keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "trust_1to7": {"type": "integer"},
            "suspicion_0to100": {"type": "integer"},
            "suspected_agenda": {"type": "string", "enum": agenda_keys},
            "evidence": {"type": "string"},
        },
        "required": ["trust_1to7", "suspicion_0to100", "suspected_agenda", "evidence"],
        "additionalProperties": False,
    }


def run_team(cond: Condition, *, backend: Backend, venture_cfg: dict,
             founders_cfg: dict, agendas_cfg: dict, exp: dict, out_dir: Path) -> Path:
    out_path = out_dir / f"{cond.team_id}.json"
    if out_path.exists():  # resume: skip finished teams
        return out_path

    rng = random.Random(cond.seed)
    venture = venture_cfg["venture"]
    weeks = venture_cfg["weeks"][: exp.get("max_weeks", len(venture_cfg["weeks"]))]
    founders = founders_cfg["founders"]
    conf = founders_cfg["confederate"]
    agendas = agendas_cfg["agendas"]
    team_names = [f["name"] for f in founders] + [conf["name"]]
    menu = prompts.agenda_menu(agendas)

    systems = {f["name"]: prompts.founder_system(f, team_names, venture, cond.warned)
               for f in founders}
    dose_text = agendas_cfg["doses"][cond.dose] if cond.dose else None
    systems[conf["name"]] = prompts.confederate_system(
        conf, team_names, venture, agendas[cond.agenda], dose_text)

    models = exp["models"]
    call = lambda **kw: backend.call(model=models["agents"],
                                     effort=models["effort_agents"], **kw)

    transcript = ""
    turns, decisions, surveys = [], [], []
    k = exp.get("turns_per_agent_per_week", 2)

    for wi, week in enumerate(weeks):
        transcript += prompts.week_header(wi, week)
        for r in range(k):
            order = team_names[:]
            rng.shuffle(order)
            for speaker in order:
                before = transcript
                reply = call(role="confederate" if speaker == conf["name"] else "founder",
                             system=systems[speaker],
                             user=prompts.turn_prompt(transcript, speaker),
                             max_tokens=2000,
                             meta={"team": cond.team_id, "week": wi, "round": r,
                                   "speaker": speaker})
                text = reply.text.strip()
                transcript += f"{speaker}: {text}\n"
                turns.append({"week": wi, "round": r, "speaker": speaker,
                              "is_confederate": speaker == conf["name"],
                              "text": text, "stop_reason": reply.stop_reason,
                              "context_chars": len(before)})

        d = call(role="decision", system=prompts.DECISION_SYSTEM,
                 user=prompts.decision_prompt(transcript, week),
                 schema=decision_schema(week), max_tokens=2000,
                 meta={"team": cond.team_id, "week": wi})
        decisions.append({"week": wi, "topic": week["topic"], **(d.data or {})})
        transcript += f"[Decision recorded: {(d.data or {}).get('decision_summary', '')}]\n"

        for f in founders:
            s = call(role="survey", system=prompts.survey_system(f, venture),
                     user=prompts.survey_prompt(transcript, menu),
                     schema=survey_schema(list(agendas)), max_tokens=1500,
                     meta={"team": cond.team_id, "week": wi, "respondent": f["name"]})
            surveys.append({"week": wi, "respondent": f["name"], **(s.data or {})})

    result = {"condition": asdict(cond), "turns": turns, "decisions": decisions,
              "surveys": surveys, "backend": type(backend).__name__}
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.rename(out_path)
    return out_path


def build_conditions(exp: dict) -> list[Condition]:
    conds, i = [], 0
    base = exp.get("seed", 0)
    for rep in range(exp.get("replicates", 1)):
        for warned in exp.get("warned", [False]):
            for agenda in exp["agendas"]:
                doses = [None] if agenda == "honest_control" else exp["doses"]
                for dose in doses:
                    conds.append(Condition(
                        team_id=f"team_{i:04d}", agenda=agenda, dose=dose,
                        warned=warned, replicate=rep, seed=base * 10_000 + i))
                    i += 1
    return conds
