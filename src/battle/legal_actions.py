from __future__ import annotations

import random
from typing import Any


BattleRequest = dict[str, Any]


def _is_fainted(pokemon: BattleRequest) -> bool:
    condition = pokemon.get("condition", "")
    return "fnt" in condition.split()


def _available_switches(request: BattleRequest) -> list[str]:
    team = request.get("side", {}).get("pokemon", [])

    return [
        f"switch {slot}"
        for slot, pokemon in enumerate(team, start=1)
        if not pokemon.get("active", False)
        and not _is_fainted(pokemon)
    ]


def legal_actions(request: BattleRequest) -> list[str]:
    if request.get("wait", False):
        return []

    team = request.get("side", {}).get("pokemon", [])

    if request.get("teamPreview", False):
        slots = "".join(str(slot) for slot in range(1, len(team) + 1))
        return [f"team {slots}"]

    switches = _available_switches(request)
    force_switch = request.get("forceSwitch", [])

    if force_switch is True or (
        isinstance(force_switch, list) and any(force_switch)
    ):
        return switches or ["pass"]

    active_requests = request.get("active") or []

    if not active_requests:
        return []

    active = active_requests[0]
    moves = active.get("moves", [])
    can_mega_evolve = active.get("canMegaEvo", False)
    can_ultra_burst = active.get("canUltraBurst", False)
    z_moves = active.get("canZMove") or []

    choices: list[str] = []

    for slot, move in enumerate(moves, start=1):
        if not move.get("disabled", False):
            choices.append(f"move {slot}")

            if can_mega_evolve:
                choices.append(f"move {slot} mega")

            if can_ultra_burst:
                choices.append(f"move {slot} ultra")

        if slot <= len(z_moves) and z_moves[slot - 1] is not None:
            choices.append(f"move {slot} zmove")

    if not active.get("trapped", False):
        choices.extend(switches)

    return choices or ["default"]


def choose_uniform_action(request: BattleRequest) -> str | None:
    choices = legal_actions(request)

    if not choices:
        return None

    return random.choice(choices)