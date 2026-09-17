"""
HarderHeuristicAgent — the original HeuristicAgent with two critical exploit-patches.

Problem: a smart opponent can trivially beat the original HeuristicAgent by:
  1. Switching in a Pokémon that is immune to our best move — the agent keeps
     clicking that move forever (immunity lock).
  2. Running Wish + Protect — the agent keeps attacking into the heal loop and
     can never actually KO anything.

These are *opponent-modelling* failures: the original agent scores each turn in
isolation without tracking what happened last turn.  The two patches below add
the minimum history-tracking needed to escape both traps, making this a much
tougher benchmark than the plain heuristic.

Both patches operate as *score overrides* applied on top of the normal heuristic
score, keeping the rest of the decision-making identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from poke_env.data import GenData

from src.battle.legal_actions import legal_actions
from src.battle.state import BattleState, PokemonState, to_id
from src.agents.heuristic_agent import (
    HeuristicAgent,
    ScoredAction,
    ActionDecision,
    _TYPE_IMMUNITY_ABILITIES,
)

BattleRequest = dict[str, Any]

# How many consecutive turns of zero-effective attacks before we force a switch
_IMMUNITY_LOCK_THRESHOLD = 2

# How many turns the opponent can recover HP before we stop face-tanking and
# start playing around it
_STALL_COUNTER_THRESHOLD = 2


@dataclass
class _BattleMemory:
    """Per-battle history used by the two patches."""

    # Patch 1: counts consecutive turns where every attacking move we chose
    # dealt zero typed damage (i.e. we were locked into an immunity).
    zero_damage_turns: int = 0
    # The move slot (0-based) we clicked last turn — used to detect a repeat
    last_move_slot: int | None = None

    # Patch 2: stall detection.  We record the opponent's HP fraction at the
    # start of each turn.  If it went *up* (opponent healed) we increment.
    opponent_hp_last_turn: float | None = None
    stall_turns: int = 0

    # Track what the opponent's active species was last turn so we reset
    # counters on an opponent switch.
    opponent_species_last_turn: str = ""


class HarderHeuristicAgent(HeuristicAgent):
    """
    HeuristicAgent + two exploit-patches:
      1. Immunity-lock escape
      2. Wish/Protect stall counter
    """

    def __init__(self, generation: int = 7) -> None:
        super().__init__(generation)
        # Keyed by room_id so one agent instance can handle multiple battles.
        self._memory: dict[str, _BattleMemory] = {}

    # ------------------------------------------------------------------
    # Public API (same signature as HeuristicAgent.choose_action)
    # ------------------------------------------------------------------

    def choose_action(
        self,
        request: BattleRequest,
        state: BattleState,
        *,
        room_id: str = "",
    ) -> ActionDecision:
        mem = self._memory.setdefault(room_id, _BattleMemory())
        self._update_memory(mem, state, request)

        actions = legal_actions(request)
        if not actions:
            return ActionDecision(action=None, ranked_actions=())

        # Score with the parent heuristic first
        base_scored = [
            super().score_action(action, request, state) for action in actions
        ]

        # Apply patches as score adjustments
        patched = [
            self._apply_patches(sa, request, state, mem) for sa in base_scored
        ]

        ranked = tuple(
            sorted(patched, key=lambda s: s.score, reverse=True)
        )

        # Record what we decided
        chosen = ranked[0].action
        if chosen and chosen.startswith("move "):
            mem.last_move_slot = int(chosen.split()[1]) - 1
        else:
            mem.last_move_slot = None

        return ActionDecision(action=chosen, ranked_actions=ranked)

    # ------------------------------------------------------------------
    # Memory update (called before scoring)
    # ------------------------------------------------------------------

    def _update_memory(
        self,
        mem: _BattleMemory,
        state: BattleState,
        request: BattleRequest,
    ) -> None:
        opponent = state.opponent_active_pokemon

        # Reset counters if the opponent switched (new species)
        opp_species = (opponent.species if opponent else "") or ""
        if opp_species != mem.opponent_species_last_turn:
            mem.zero_damage_turns = 0
            mem.stall_turns = 0
            mem.opponent_hp_last_turn = None
            mem.opponent_species_last_turn = opp_species

        # Patch 2: did opponent HP go up since last turn?
        current_opp_hp = opponent.hp_fraction if opponent else None
        if (
            mem.opponent_hp_last_turn is not None
            and current_opp_hp is not None
            and current_opp_hp > mem.opponent_hp_last_turn + 0.05  # healed meaningfully
        ):
            mem.stall_turns += 1
        else:
            mem.stall_turns = max(0, mem.stall_turns - 1)

        mem.opponent_hp_last_turn = current_opp_hp

        # Patch 1: did our last move do zero effective damage?
        if mem.last_move_slot is not None and opponent is not None:
            eff = self._move_effectiveness_for_slot(
                mem.last_move_slot, request, opponent
            )
            if eff == 0.0:
                mem.zero_damage_turns += 1
            else:
                mem.zero_damage_turns = 0
        else:
            mem.zero_damage_turns = 0

    # ------------------------------------------------------------------
    # Patch application
    # ------------------------------------------------------------------

    def _apply_patches(
        self,
        sa: ScoredAction,
        request: BattleRequest,
        state: BattleState,
        mem: _BattleMemory,
    ) -> ScoredAction:
        score = sa.score
        reasons = list(sa.reasons)

        # ---- Patch 1: Immunity-lock escape ----
        # If we've wasted >= threshold turns into an immunity, heavily penalise
        # every move that is also immune, and give a large bonus to switches.
        if mem.zero_damage_turns >= _IMMUNITY_LOCK_THRESHOLD:
            if sa.action.startswith("move "):
                slot = int(sa.action.split()[1]) - 1
                opponent = state.opponent_active_pokemon
                eff = self._move_effectiveness_for_slot(slot, request, opponent)
                if eff == 0.0:
                    score -= 20.0
                    reasons.append(
                        f"immunity-lock penalty (wasted {mem.zero_damage_turns} turns)"
                    )
            elif sa.action.startswith("switch "):
                score += 5.0
                reasons.append("immunity-lock: forced switch bonus")

        # ---- Patch 2: Stall counter ----
        # If the opponent has been healing back every other turn, stop face-
        # tanking and either use a setup/hazard move or switch to a wallbreaker.
        if mem.stall_turns >= _STALL_COUNTER_THRESHOLD:
            if sa.action.startswith("move "):
                # Bonus for hazard/status/boost moves (they don't get healed away)
                slot = int(sa.action.split()[1]) - 1
                move_data = self._move_data_for_slot(slot, request)
                if move_data:
                    category = move_data.get("category", "Status")
                    if category == "Status":
                        has_hazard = move_data.get("sideCondition")
                        has_boost = move_data.get("boosts") or (
                            move_data.get("self") or {}
                        ).get("boosts")
                        has_status = move_data.get("status")
                        if has_hazard or has_boost or has_status:
                            score += 4.0
                            reasons.append(
                                "stall counter: prefer setup/hazard/status vs healer"
                            )
                    else:
                        # Slight penalty for raw damage into a healer
                        score -= 1.0
                        reasons.append("stall counter: raw damage penalised vs healer")

            elif sa.action.startswith("switch "):
                # Bonus for switching in a wallbreaker (high Atk or SpA base stat)
                slot = int(sa.action.split()[1]) - 1
                team = request.get("side", {}).get("pokemon", [])
                if slot < len(team):
                    species_id = to_id(
                        team[slot].get("details", "").split(",", 1)[0].strip()
                        or team[slot].get("ident", "").partition(":")[2].strip()
                    )
                    entry = self.data.pokedex.get(species_id, {})
                    base_stats = entry.get("baseStats", {})
                    best_offense = max(
                        base_stats.get("atk", 0), base_stats.get("spa", 0)
                    )
                    if best_offense >= 100:
                        score += 3.0
                        reasons.append(
                            "stall counter: switch to wallbreaker bonus"
                        )

        return ScoredAction(sa.action, round(score, 3), tuple(reasons))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_effectiveness_for_slot(
        self,
        slot: int,
        request: BattleRequest,
        opponent: PokemonState | None,
    ) -> float:
        """Return the type-effectiveness of the move in the given slot."""
        move_data = self._move_data_for_slot(slot, request)
        if move_data is None:
            return 1.0
        move_type = move_data.get("type", "Normal")
        return self._move_effectiveness(move_type, opponent)

    def _move_data_for_slot(
        self,
        slot: int,
        request: BattleRequest,
    ) -> dict[str, Any] | None:
        """Return the Showdown move-data dict for the given 0-based slot."""
        active_requests = request.get("active") or []
        if not active_requests:
            return None
        moves = active_requests[0].get("moves", [])
        if slot >= len(moves):
            return None
        move_id = self._resolve_move_id(moves[slot].get("id", ""))
        return self.data.moves.get(move_id)
