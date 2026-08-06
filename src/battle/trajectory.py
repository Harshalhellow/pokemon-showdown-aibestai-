from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.heuristic_agent import ActionDecision
from src.battle.state import BattleState


BattleRequest = dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BattleTrajectory:
    battle_id: str
    started_at: str = field(default_factory=_utc_now)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def add_decision(
        self,
        state: BattleState,
        request: BattleRequest,
        decision: ActionDecision,
    ) -> None:
        self.decisions.append(
            {
                "turn": state.turn,
                "request_id": request.get("rqid"),
                "state": state.snapshot(),
                "legal_actions": [
                    scored_action.action
                    for scored_action in decision.ranked_actions
                ],
                "chosen_action": decision.action,
                "action_scores": [
                    {
                        "action": scored_action.action,
                        "score": scored_action.score,
                        "reasons": list(scored_action.reasons),
                    }
                    for scored_action in decision.ranked_actions
                ],
            }
        )

    def finish(self, state: BattleState) -> dict[str, Any]:
        return {
            "battle_id": self.battle_id,
            "format": state.format_name,
            "started_at": self.started_at,
            "ended_at": _utc_now(),
            "our_side": state.our_side,
            "players": {
                side_id: side.name
                for side_id, side in sorted(state.sides.items())
            },
            "winner": state.winner,
            "tied": state.tied,
            "result": state.result_for_our_side(),
            "turns": state.turn,
            "decisions": self.decisions,
        }


class TrajectoryRecorder:
    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file
        self._active: dict[str, BattleTrajectory] = {}

    def record_decision(
        self,
        state: BattleState,
        request: BattleRequest,
        decision: ActionDecision,
    ) -> None:
        trajectory = self._active.setdefault(
            state.room_id,
            BattleTrajectory(state.room_id),
        )
        trajectory.add_decision(state, request, decision)

    def finish_battle(self, state: BattleState) -> Path | None:
        trajectory = self._active.pop(state.room_id, None)

        if trajectory is None:
            return None

        record = trajectory.finish(state)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with self.output_file.open("a", encoding="utf-8") as output:
            json.dump(record, output, separators=(",", ":"))
            output.write("\n")

        return self.output_file
