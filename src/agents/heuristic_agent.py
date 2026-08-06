from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poke_env.data import GenData

from src.battle.legal_actions import legal_actions
from src.battle.state import BattleState, PokemonState, to_id


BattleRequest = dict[str, Any]

_TYPE_IMMUNITY_ABILITIES = {
    "dryskin": "Water",
    "flashfire": "Fire",
    "levitate": "Ground",
    "lightningrod": "Electric",
    "motordrive": "Electric",
    "sapsipper": "Grass",
    "stormdrain": "Water",
    "voltabsorb": "Electric",
    "waterabsorb": "Water",
}


@dataclass(frozen=True)
class ScoredAction:
    action: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ActionDecision:
    action: str | None
    ranked_actions: tuple[ScoredAction, ...]


class HeuristicAgent:
    def __init__(self, generation: int = 7) -> None:
        self.generation = generation
        self.data = GenData.from_gen(generation)

    def choose_action(
        self,
        request: BattleRequest,
        state: BattleState,
    ) -> ActionDecision:
        actions = legal_actions(request)

        if not actions:
            return ActionDecision(action=None, ranked_actions=())

        ranked_actions = tuple(
            sorted(
                (self.score_action(action, request, state) for action in actions),
                key=lambda scored_action: scored_action.score,
                reverse=True,
            )
        )

        return ActionDecision(
            action=ranked_actions[0].action,
            ranked_actions=ranked_actions,
        )

    def score_action(
        self,
        action: str,
        request: BattleRequest,
        state: BattleState,
    ) -> ScoredAction:
        if action.startswith("move "):
            return self._score_move(action, request, state)
        if action.startswith("switch "):
            return self._score_switch(action, request, state)
        if action.startswith("team "):
            return ScoredAction(action, 0.0, ("only team-preview order",))
        if action == "pass":
            return ScoredAction(action, 0.0, ("no switch is available",))
        return ScoredAction(action, 0.0, ("Showdown fallback action",))

    def _score_move(
        self,
        action: str,
        request: BattleRequest,
        state: BattleState,
    ) -> ScoredAction:
        action_parts = action.split()
        move_slot = int(action_parts[1]) - 1
        modifier = action_parts[2] if len(action_parts) > 2 else None
        active_requests = request.get("active") or []

        if not active_requests:
            return ScoredAction(action, 0.0, ("missing active request",))

        moves = active_requests[0].get("moves", [])

        if move_slot >= len(moves):
            return ScoredAction(action, -100.0, ("invalid move slot",))

        move_request = moves[move_slot]
        move_id = self._resolve_move_id(move_request.get("id", ""))
        move_data = self.data.moves.get(move_id)

        if move_data is None:
            return self._score_unknown_move(action, move_request, modifier)

        category = move_data.get("category", "Status")

        if category == "Status":
            score, reasons = self._status_move_score(
                move_id,
                move_data,
                state,
            )
        else:
            score, reasons = self._damaging_move_score(
                move_id,
                move_data,
                state,
            )

        if modifier == "zmove":
            score += 2.5
            reasons.append("Z-Move power/utility bonus")
        elif modifier == "mega":
            score += 1.5
            reasons.append("Mega Evolution bonus")
        elif modifier == "ultra":
            score += 1.5
            reasons.append("Ultra Burst bonus")

        current_pp = move_request.get("pp")
        maximum_pp = move_request.get("maxpp")

        if current_pp == 1 and maximum_pp and maximum_pp > 1:
            score -= 0.25
            reasons.append("last PP conservation penalty")

        return ScoredAction(action, round(score, 3), tuple(reasons))

    def _damaging_move_score(
        self,
        move_id: str,
        move_data: dict[str, Any],
        state: BattleState,
    ) -> tuple[float, list[str]]:
        base_power = float(move_data.get("basePower", 0))
        variable_power = base_power == 0

        if variable_power:
            base_power = 60.0

        accuracy_value = move_data.get("accuracy", 100)
        accuracy = 1.0 if accuracy_value is True else float(accuracy_value) / 100
        move_type = move_data.get("type", "Normal")
        effectiveness = self._move_effectiveness(
            move_type,
            state.opponent_active_pokemon,
        )
        own_types = self._pokemon_types(state.our_active_pokemon)
        stab = 1.5 if move_type in own_types else 1.0
        expected_power = base_power * accuracy * stab * effectiveness
        score = 4.0 + expected_power / 30
        reasons = [
            (
                f"variable-power estimate for {move_id}: {base_power:g}"
                if variable_power
                else f"{base_power:g} base power"
            ),
            f"{accuracy * 100:g}% accuracy",
        ]

        if stab > 1:
            reasons.append("same-type attack bonus")

        if effectiveness == 0:
            score = -10.0
            reasons.append("known immunity")
        elif effectiveness > 1:
            reasons.append(f"{effectiveness:g}x super effective")
        elif effectiveness < 1:
            reasons.append(f"{effectiveness:g}x resisted")

        opponent = state.opponent_active_pokemon

        if opponent and opponent.hp_fraction is not None:
            if opponent.hp_fraction <= 0.25 and expected_power > 0:
                score += 1.5
                reasons.append("opponent is in possible KO range")

        priority = move_data.get("priority", 0)

        if priority > 0:
            score += min(priority, 2) * 0.75
            reasons.append("priority move")

        return score, reasons

    def _status_move_score(
        self,
        move_id: str,
        move_data: dict[str, Any],
        state: BattleState,
    ) -> tuple[float, list[str]]:
        score = 3.0
        reasons = ["status or utility move"]
        own_pokemon = state.our_active_pokemon
        opponent = state.opponent_active_pokemon

        if "heal" in move_data:
            hp_fraction = own_pokemon.hp_fraction if own_pokemon else None

            if hp_fraction is None:
                score = 5.0
                reasons.append("healing move")
            elif hp_fraction >= 0.95:
                score = -2.0
                reasons.append("healing would be wasted near full HP")
            else:
                score = 5.0 + 5.0 * (1.0 - hp_fraction)
                reasons.append(f"heals at {hp_fraction * 100:.0f}% HP")

        inflicted_status = move_data.get("status")

        if inflicted_status:
            if opponent and opponent.status:
                score -= 2.0
                reasons.append("opponent is already statused")
            else:
                score += 3.0
                reasons.append(f"can inflict {inflicted_status}")

        side_condition = move_data.get("sideCondition")

        if side_condition:
            condition_id = to_id(side_condition)
            target = move_data.get("target")
            target_side = (
                state.our_side
                if target == "allySide"
                else state.opponent_side
            )
            existing = (
                state.sides[target_side].conditions
                if target_side
                else set()
            )

            if condition_id in existing:
                score = -1.0
                reasons.append("side condition is already active")
            else:
                score += 4.0
                reasons.append(f"sets {condition_id}")

        boosts = move_data.get("boosts", {})
        self_effect = move_data.get("self", {})
        self_boosts = self_effect.get("boosts", {}) if self_effect else {}
        positive_boosts = sum(
            amount
            for amount in [*boosts.values(), *self_boosts.values()]
            if amount > 0
        )

        if positive_boosts:
            score += min(positive_boosts, 4) * 1.25
            reasons.append("raises useful stats")

        if move_id in {"protect", "detect", "kingsshield", "spikyshield"}:
            score += 1.5
            reasons.append("protective move")

        if move_data.get("weather") or move_data.get("terrain"):
            score += 2.0
            reasons.append("controls field conditions")

        return score, reasons

    def _score_switch(
        self,
        action: str,
        request: BattleRequest,
        state: BattleState,
    ) -> ScoredAction:
        switch_slot = int(action.split()[1]) - 1
        team = request.get("side", {}).get("pokemon", [])

        if switch_slot >= len(team):
            return ScoredAction(action, -100.0, ("invalid switch slot",))

        switch_request = team[switch_slot]
        candidate = self._request_pokemon_state(switch_request)
        hp_fraction = candidate.hp_fraction or 0.0
        current_threat = self._worst_stab_threat(
            state.opponent_active_pokemon,
            state.our_active_pokemon,
        )
        candidate_threat = self._worst_stab_threat(
            state.opponent_active_pokemon,
            candidate,
        )
        matchup_improvement = current_threat - candidate_threat
        score = 3.0 + 2.0 * hp_fraction + 4.0 * matchup_improvement
        reasons = [
            f"switch target has {hp_fraction * 100:.0f}% HP",
            f"estimated incoming type pressure {candidate_threat:g}x",
        ]

        if matchup_improvement > 0:
            reasons.append("improves defensive type matchup")
        elif current_threat <= 1 and candidate_threat >= current_threat:
            score -= 2.0
            reasons.append("unnecessary switch penalty")

        if state.our_side:
            side_conditions = state.sides[state.our_side].conditions
            hazard_count = len(
                side_conditions
                & {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
            )

            if hazard_count:
                score -= 0.75 * hazard_count
                reasons.append("entry-hazard penalty")

        return ScoredAction(action, round(score, 3), tuple(reasons))

    def _score_unknown_move(
        self,
        action: str,
        move_request: BattleRequest,
        modifier: str | None,
    ) -> ScoredAction:
        score = 4.0
        reasons = [f"unknown move data for {move_request.get('id', 'move')}"]

        if modifier in {"mega", "ultra"}:
            score += 1.5
            reasons.append(f"{modifier} transformation bonus")
        elif modifier == "zmove":
            score += 2.5
            reasons.append("Z-Move bonus")

        return ScoredAction(action, score, tuple(reasons))

    def _move_effectiveness(
        self,
        move_type: str,
        target: PokemonState | None,
    ) -> float:
        if target and target.ability:
            immune_type = _TYPE_IMMUNITY_ABILITIES.get(target.ability)

            if immune_type == move_type:
                return 0.0

        target_types = self._pokemon_types(target)

        if not target_types:
            return 1.0

        multiplier = 1.0

        for target_type in target_types:
            type_row = self.data.type_chart.get(target_type.upper(), {})
            multiplier *= type_row.get(move_type.upper(), 1.0)

        return multiplier

    def _worst_stab_threat(
        self,
        attacker: PokemonState | None,
        defender: PokemonState | None,
    ) -> float:
        attacker_types = self._pokemon_types(attacker)

        if not attacker_types or defender is None:
            return 1.0

        return max(
            self._move_effectiveness(attacker_type, defender)
            for attacker_type in attacker_types
        )

    def _pokemon_types(self, pokemon: PokemonState | None) -> list[str]:
        if pokemon is None or not pokemon.species:
            return []

        pokedex_entry = self.data.pokedex.get(to_id(pokemon.species))
        return pokedex_entry.get("types", []) if pokedex_entry else []

    @staticmethod
    def _request_pokemon_state(request_pokemon: BattleRequest) -> PokemonState:
        name = request_pokemon.get("ident", "").partition(":")[2].strip()
        pokemon = PokemonState(name=name)
        pokemon.update_from_request(request_pokemon)
        return pokemon

    def _resolve_move_id(self, move_id: str) -> str:
        normalized = to_id(move_id)

        if normalized in self.data.moves:
            return normalized
        if normalized.startswith("hiddenpower"):
            return "hiddenpower"
        return normalized
