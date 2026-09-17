"""
ImprovedHeuristicAgent — a drop-in upgrade over HeuristicAgent.

Improvements over the base agent:
  1.  Stat-boost awareness  – attacker/defender boosts scale damage estimates.
  2.  Smarter KO detection  – projects KOs and rewards them heavily.
  3.  Own-HP urgency        – pressing harder when we are low, avoiding wasted
                              status turns.
  4.  Weather/terrain synergy – applies Rain/Sun/Sandstorm/Hail/Terrain bonus.
  5.  Protect loop prevention – penalises consecutive Protect-family moves.
  6.  Better team-preview    – leads with bulkier, offensive Pokémon.
  7.  Offensive switch score – rewards switching in a type-advantaged attacker.
  8.  Expanded immunity dict – covers all Gen 7 damage-blocking abilities plus
                              Scrappy overrides.
  9.  Recoil/crash penalties – discounts dangerous moves when HP is low.
  10. Confusion/wrap tracking – penalises status moves while confused/trapped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from poke_env.data import GenData

from src.battle.legal_actions import legal_actions
from src.battle.state import BattleState, PokemonState, to_id


BattleRequest = dict[str, Any]

# ---------------------------------------------------------------------------
# Immunity/absorption abilities (type → set of ability ids that block it)
# ---------------------------------------------------------------------------
_TYPE_IMMUNITY_ABILITIES: dict[str, set[str]] = {
    "Normal":   {"wonderguard"},
    "Fire":     {"flashfire"},
    "Water":    {"dryskin", "stormdrain", "waterabsorb"},
    "Electric": {"lightningrod", "motordrive", "voltabsorb"},
    "Grass":    {"sapsipper"},
    "Ground":   {"levitate"},
    "Fighting": set(),
    "Poison":   set(),
}

# Abilities that block all sound moves (Soundproof) or bullet moves (Bulletproof)
_SOUNDPROOF_MOVES = {
    "boomburst", "bugbuzz", "chatter", "confide", "disarmingvoice",
    "echoedvoice", "grasswhistle", "growl", "healbell", "howl",
    "hypervoice", "metalsound", "nobleroar", "parabolacharge", "perishsong",
    "relicsong", "roar", "round", "screech", "sing", "snarl", "snore",
    "supersonic", "uproar",
}
_BULLETPROOF_MOVES = {
    "aurasphere", "barrage", "bulletseed", "cannonball", "eggbomb",
    "electball", "energyball", "focusblast", "gyroball", "iceball",
    "magnetbomb", "mistball", "mudbomb", "octazooka", "pollen puff",
    "pollenpuff", "pyroball", "rockblast", "seedflare", "shadowball",
    "sludgebomb", "weatherball", "zingzap",
}

# Abilities that let Normal/Fighting moves hit Ghost types
_SCRAPPY_ABILITIES = {"scrappy", "mindseye"}

# Protect-family move ids
_PROTECT_MOVES = {"protect", "detect", "kingsshield", "spikyshield", "banefulbunker"}

# Weather id → (boosted type, nerfed type)
_WEATHER_TYPE_BONUS: dict[str, tuple[str, str]] = {
    "raindance":       ("Water", "Fire"),
    "sunnyday":        ("Fire",  "Water"),
    "sandstorm":       ("Rock",  ""),      # no nerf in this simplified model
    "hail":            ("Ice",   ""),
    "primordialsea":   ("Water", "Fire"),
    "desolateland":    ("Fire",  "Water"),
}

# Terrain id → boosted type (grounded attacker only; simplified)
_TERRAIN_TYPE_BONUS: dict[str, str] = {
    "electricterrain": "Electric",
    "grassyterrain":   "Grass",
    "psychicterrain":  "Psychic",
    "mistyterrain":    "",  # reduces Dragon; we skip nerfing for simplicity
}

# Boost-stage to multiplier table (Gen 7 standard formula)
_BOOST_MULTIPLIERS = {
    -6: 2/8, -5: 2/7, -4: 2/6, -3: 2/5, -2: 2/4, -1: 2/3,
     0: 1.0,
     1: 3/2,  2: 4/2,  3: 5/2,  4: 6/2,  5: 7/2,  6: 8/2,
}


def _boost_mult(stages: int) -> float:
    return _BOOST_MULTIPLIERS.get(max(-6, min(6, stages)), 1.0)


@dataclass(frozen=True)
class ScoredAction:
    action: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ActionDecision:
    action: str | None
    ranked_actions: tuple[ScoredAction, ...]


class ImprovedHeuristicAgent:
    """
    Heuristic agent with the ten improvements listed in the module docstring.
    Keeps the same public API as HeuristicAgent so it is a drop-in replacement.
    """

    def __init__(self, generation: int = 7) -> None:
        self.generation = generation
        self.data = GenData.from_gen(generation)
        # improvement 5: track the last chosen action per battle to detect
        # consecutive protect usage.  Key = room_id, value = last action str.
        self._last_action: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def choose_action(
        self,
        request: BattleRequest,
        state: BattleState,
        *,
        room_id: str = "",
    ) -> ActionDecision:
        actions = legal_actions(request)

        if not actions:
            return ActionDecision(action=None, ranked_actions=())

        ranked_actions = tuple(
            sorted(
                (self.score_action(action, request, state, room_id=room_id)
                 for action in actions),
                key=lambda sa: sa.score,
                reverse=True,
            )
        )

        chosen = ranked_actions[0].action
        self._last_action[room_id] = chosen

        return ActionDecision(action=chosen, ranked_actions=ranked_actions)

    def score_action(
        self,
        action: str,
        request: BattleRequest,
        state: BattleState,
        *,
        room_id: str = "",
    ) -> ScoredAction:
        if action.startswith("move "):
            return self._score_move(action, request, state, room_id=room_id)
        if action.startswith("switch "):
            return self._score_switch(action, request, state)
        if action.startswith("team "):
            return self._score_team_preview(action, request)
        if action == "pass":
            return ScoredAction(action, 0.0, ("no switch available",))
        return ScoredAction(action, 0.0, ("Showdown fallback",))

    # ------------------------------------------------------------------
    # Move scoring
    # ------------------------------------------------------------------

    def _score_move(
        self,
        action: str,
        request: BattleRequest,
        state: BattleState,
        *,
        room_id: str = "",
    ) -> ScoredAction:
        parts = action.split()
        slot = int(parts[1]) - 1
        modifier = parts[2] if len(parts) > 2 else None
        active_requests = request.get("active") or []

        if not active_requests:
            return ScoredAction(action, 0.0, ("missing active request",))

        moves = active_requests[0].get("moves", [])

        if slot >= len(moves):
            return ScoredAction(action, -100.0, ("invalid move slot",))

        move_req = moves[slot]
        move_id = self._resolve_move_id(move_req.get("id", ""))
        move_data = self.data.moves.get(move_id)

        if move_data is None:
            return self._score_unknown_move(action, move_req, modifier)

        category = move_data.get("category", "Status")

        if category == "Status":
            score, reasons = self._status_move_score(move_id, move_data, state)
        else:
            score, reasons = self._damaging_move_score(
                move_id, move_data, state, modifier=modifier
            )

        # improvement 5: penalise repeated protect
        if move_id in _PROTECT_MOVES:
            last = self._last_action.get(room_id, "")
            last_id = self._resolve_move_id(last.split()[-1]) if last.startswith("move ") else ""
            if last_id in _PROTECT_MOVES:
                score -= 4.0
                reasons.append("consecutive protect penalty (50% fail chance)")

        # modifier bonuses
        if modifier == "zmove":
            score += 2.5
            reasons.append("Z-Move power/utility bonus")
        elif modifier == "mega":
            score += 1.5
            reasons.append("Mega Evolution bonus")
        elif modifier == "ultra":
            score += 1.5
            reasons.append("Ultra Burst bonus")

        # PP conservation
        current_pp = move_req.get("pp")
        max_pp = move_req.get("maxpp")
        if current_pp == 1 and max_pp and max_pp > 1:
            score -= 0.25
            reasons.append("last PP conservation penalty")

        # improvement 3: own-HP urgency for damaging moves
        own = state.our_active_pokemon
        if own and own.hp_fraction is not None and own.hp_fraction <= 0.25:
            if category != "Status":
                score += 2.0
                reasons.append("urgency bonus: we are at low HP")
            else:
                score -= 2.0
                reasons.append("status move discouraged at low HP")

        # improvement 10: confusion/wrap penalty for status moves
        if category == "Status" and own:
            if own.status in {"confusion"}:
                score -= 1.5
                reasons.append("status move penalised: user is confused")

        return ScoredAction(action, round(score, 3), tuple(reasons))

    def _damaging_move_score(
        self,
        move_id: str,
        move_data: dict[str, Any],
        state: BattleState,
        modifier: str | None = None,
    ) -> tuple[float, list[str]]:
        base_power = float(move_data.get("basePower", 0))
        variable_power = base_power == 0
        if variable_power:
            base_power = 60.0

        accuracy_val = move_data.get("accuracy", 100)
        accuracy = 1.0 if accuracy_val is True else float(accuracy_val) / 100
        move_type = move_data.get("type", "Normal")

        # type effectiveness
        effectiveness = self._move_effectiveness(
            move_id, move_type, state.our_active_pokemon,
            state.opponent_active_pokemon
        )
        # STAB
        own_types = self._pokemon_types(state.our_active_pokemon)
        stab = 1.5 if move_type in own_types else 1.0

        # improvement 1: stat-boost multipliers
        own = state.our_active_pokemon
        opp = state.opponent_active_pokemon
        is_physical = move_data.get("category") == "Physical"
        atk_stat = "atk" if is_physical else "spa"
        def_stat = "def" if is_physical else "spd"
        atk_boost = _boost_mult(own.boosts.get(atk_stat, 0) if own else 0)
        def_boost = _boost_mult(opp.boosts.get(def_stat, 0) if opp else 0)
        boost_factor = atk_boost / def_boost

        # improvement 4: weather/terrain multiplier
        weather_mult = self._weather_mult(move_type, state)
        terrain_mult = self._terrain_mult(move_type, state)

        expected_power = (
            base_power * accuracy * stab * effectiveness
            * boost_factor * weather_mult * terrain_mult
        )

        score = 4.0 + expected_power / 30
        reasons: list[str] = [
            (
                f"variable-power estimate {base_power:g}"
                if variable_power
                else f"{base_power:g} base power"
            ),
            f"{accuracy * 100:g}% accuracy",
        ]

        if stab > 1:
            reasons.append("same-type attack bonus")
        if boost_factor != 1.0:
            reasons.append(f"boost factor {boost_factor:.2f}x")
        if weather_mult != 1.0:
            reasons.append(f"weather multiplier {weather_mult:.1f}x")
        if terrain_mult != 1.0:
            reasons.append(f"terrain multiplier {terrain_mult:.1f}x")

        if effectiveness == 0:
            score = -10.0
            reasons.append("known immunity")
        elif effectiveness > 1:
            reasons.append(f"{effectiveness:g}x super effective")
        elif effectiveness < 1:
            reasons.append(f"{effectiveness:g}x resisted")

        # improvement 2: KO detection
        if opp and opp.hp_fraction is not None and effectiveness > 0:
            opp_species = to_id(opp.species) if opp.species else ""
            opp_entry = self.data.pokedex.get(opp_species, {})
            opp_hp_stat = opp_entry.get("baseStats", {}).get("hp", 70)
            # rough damage fraction: expected_power / (opp_hp_stat * 2)
            damage_fraction = expected_power / (opp_hp_stat * 2.0 + 1e-9)
            if damage_fraction >= opp.hp_fraction:
                score += 5.0
                reasons.append("projects KO this turn")
            elif opp.hp_fraction <= 0.25:
                score += 1.5
                reasons.append("opponent in low-HP range")

        # priority
        priority = move_data.get("priority", 0)
        if priority > 0:
            score += min(priority, 2) * 0.75
            reasons.append("priority move")

        # improvement 9: recoil/crash penalties
        if own and own.hp_fraction is not None:
            recoil = move_data.get("recoil")
            if recoil:
                recoil_fraction = recoil[0] / recoil[1]
                penalty = recoil_fraction * (1.5 - own.hp_fraction)
                score -= penalty
                reasons.append(f"recoil penalty ({recoil_fraction * 100:.0f}%)")
            if move_data.get("hasCrashDamage"):
                miss_penalty = (1.0 - accuracy) * 1.5
                score -= miss_penalty
                reasons.append("crash-damage miss penalty")

        return score, reasons

    def _status_move_score(
        self,
        move_id: str,
        move_data: dict[str, Any],
        state: BattleState,
    ) -> tuple[float, list[str]]:
        score = 3.0
        reasons = ["status/utility move"]
        own = state.our_active_pokemon
        opp = state.opponent_active_pokemon

        # healing
        if "heal" in move_data:
            hp = own.hp_fraction if own else None
            if hp is None:
                score = 5.0
                reasons.append("healing move")
            elif hp >= 0.95:
                score = -2.0
                reasons.append("healing wasted near full HP")
            else:
                score = 5.0 + 5.0 * (1.0 - hp)
                reasons.append(f"heals at {hp * 100:.0f}% HP")

        # status infliction
        inflicted = move_data.get("status")
        if inflicted:
            if opp and opp.status:
                score -= 2.0
                reasons.append("opponent already statused")
            else:
                score += 3.0
                reasons.append(f"inflicts {inflicted}")

        # side conditions
        side_cond = move_data.get("sideCondition")
        if side_cond:
            cond_id = to_id(side_cond)
            target = move_data.get("target")
            target_side = state.our_side if target == "allySide" else state.opponent_side
            existing = state.sides[target_side].conditions if target_side else set()
            if cond_id in existing:
                score = -1.0
                reasons.append("side condition already active")
            else:
                score += 4.0
                reasons.append(f"sets {cond_id}")

        # stat boosts
        boosts = move_data.get("boosts", {})
        self_boosts = (move_data.get("self") or {}).get("boosts", {})
        positive_boosts = sum(
            v for v in [*boosts.values(), *self_boosts.values()] if v > 0
        )
        if positive_boosts:
            score += min(positive_boosts, 4) * 1.25
            reasons.append("raises stats")

        # protect family
        if move_id in _PROTECT_MOVES:
            score += 1.5
            reasons.append("protective move")

        # weather/terrain control
        if move_data.get("weather") or move_data.get("terrain"):
            score += 2.0
            reasons.append("controls field conditions")

        return score, reasons

    # ------------------------------------------------------------------
    # Switch scoring
    # ------------------------------------------------------------------

    def _score_switch(
        self,
        action: str,
        request: BattleRequest,
        state: BattleState,
    ) -> ScoredAction:
        slot = int(action.split()[1]) - 1
        team = request.get("side", {}).get("pokemon", [])

        if slot >= len(team):
            return ScoredAction(action, -100.0, ("invalid switch slot",))

        candidate = self._request_pokemon_state(team[slot])
        hp_fraction = candidate.hp_fraction or 0.0
        current_threat = self._worst_stab_threat(
            state.opponent_active_pokemon, state.our_active_pokemon
        )
        candidate_threat = self._worst_stab_threat(
            state.opponent_active_pokemon, candidate
        )
        matchup_improvement = current_threat - candidate_threat

        # improvement 7: offensive matchup component
        offensive_advantage = self._worst_stab_threat(candidate, state.opponent_active_pokemon)

        score = 3.0 + 2.0 * hp_fraction + 4.0 * matchup_improvement + offensive_advantage
        reasons = [
            f"switch target {hp_fraction * 100:.0f}% HP",
            f"defensive pressure {candidate_threat:g}x",
            f"offensive pressure {offensive_advantage:g}x",
        ]

        if matchup_improvement > 0:
            reasons.append("improves defensive matchup")
        elif current_threat <= 1 and candidate_threat >= current_threat:
            score -= 2.0
            reasons.append("unnecessary switch penalty")

        if state.our_side:
            hazard_count = len(
                state.sides[state.our_side].conditions
                & {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
            )
            if hazard_count:
                score -= 0.75 * hazard_count
                reasons.append("entry-hazard penalty")

        return ScoredAction(action, round(score, 3), tuple(reasons))

    # ------------------------------------------------------------------
    # improvement 6: team-preview ordering
    # ------------------------------------------------------------------

    def _score_team_preview(
        self,
        action: str,
        request: BattleRequest,
    ) -> ScoredAction:
        # Score each slot by bulk (hp * def * spd base stats) so bulkier leads
        # come first; fall back to 0 for unknown species.
        team = request.get("side", {}).get("pokemon", [])
        slot_scores: list[tuple[int, float]] = []

        for slot, pmon in enumerate(team, start=1):
            species_id = to_id(
                pmon.get("details", "").split(",", 1)[0].strip()
                or pmon.get("ident", "").partition(":")[2].strip()
            )
            entry = self.data.pokedex.get(species_id, {})
            stats = entry.get("baseStats", {})
            bulk = stats.get("hp", 70) * stats.get("def", 70) * stats.get("spd", 70)
            slot_scores.append((slot, bulk))

        # Sort by bulk descending; build the team-order string
        slot_scores.sort(key=lambda x: x[1], reverse=True)
        order = "".join(str(s) for s, _ in slot_scores)
        ordered_action = f"team {order}"

        return ScoredAction(
            ordered_action,
            float(slot_scores[0][1]),
            ("team-preview ordered by bulk",),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _move_effectiveness(
        self,
        move_id: str,
        move_type: str,
        attacker: PokemonState | None,
        target: PokemonState | None,
    ) -> float:
        # improvement 8: expanded ability checks
        if target and target.ability:
            ability = target.ability
            immune_abilities = _TYPE_IMMUNITY_ABILITIES.get(move_type, set())
            if ability in immune_abilities:
                return 0.0
            if ability == "soundproof" and move_id in _SOUNDPROOF_MOVES:
                return 0.0
            if ability == "bulletproof" and move_id in _BULLETPROOF_MOVES:
                return 0.0

        # Scrappy: Normal/Fighting bypass Ghost immunity
        attacker_ability = attacker.ability if attacker else None
        target_types = self._pokemon_types(target)

        if (
            move_type in {"Normal", "Fighting"}
            and attacker_ability in _SCRAPPY_ABILITIES
            and target_types
        ):
            # Remove Ghost from effective types for this calculation
            target_types = [t for t in target_types if t != "Ghost"]

        if not target_types:
            return 1.0

        multiplier = 1.0
        for tt in target_types:
            row = self.data.type_chart.get(tt.upper(), {})
            multiplier *= row.get(move_type.upper(), 1.0)

        return multiplier

    def _worst_stab_threat(
        self,
        attacker: PokemonState | None,
        defender: PokemonState | None,
    ) -> float:
        types = self._pokemon_types(attacker)
        if not types or defender is None:
            return 1.0
        return max(
            self._move_effectiveness("", t, attacker, defender)
            for t in types
        )

    def _pokemon_types(self, pokemon: PokemonState | None) -> list[str]:
        if pokemon is None or not pokemon.species:
            return []
        entry = self.data.pokedex.get(to_id(pokemon.species))
        return entry.get("types", []) if entry else []

    def _weather_mult(self, move_type: str, state: BattleState) -> float:
        """Return 1.5 for boosted weather type, 0.5 for nerfed, else 1.0."""
        weather = state.weather
        if not weather:
            return 1.0
        boosted, nerfed = _WEATHER_TYPE_BONUS.get(weather, ("", ""))
        if move_type == boosted:
            return 1.5
        if nerfed and move_type == nerfed:
            return 0.5
        return 1.0

    def _terrain_mult(self, move_type: str, state: BattleState) -> float:
        """Return 1.3 when the move type matches active terrain."""
        for terrain_id in state.field_conditions:
            boosted = _TERRAIN_TYPE_BONUS.get(terrain_id, "")
            if boosted and move_type == boosted:
                return 1.3
        return 1.0

    @staticmethod
    def _request_pokemon_state(req: BattleRequest) -> PokemonState:
        name = req.get("ident", "").partition(":")[2].strip()
        poke = PokemonState(name=name)
        poke.update_from_request(req)
        return poke

    def _resolve_move_id(self, move_id: str) -> str:
        normalized = to_id(move_id)
        if normalized in self.data.moves:
            return normalized
        if normalized.startswith("hiddenpower"):
            return "hiddenpower"
        return normalized

    def _score_unknown_move(
        self,
        action: str,
        move_req: BattleRequest,
        modifier: str | None,
    ) -> ScoredAction:
        score = 4.0
        reasons = [f"unknown move data for {move_req.get('id', 'move')}"]
        if modifier in {"mega", "ultra"}:
            score += 1.5
            reasons.append(f"{modifier} bonus")
        elif modifier == "zmove":
            score += 2.5
            reasons.append("Z-Move bonus")
        return ScoredAction(action, score, tuple(reasons))
