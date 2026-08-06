from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any


BattleRequest = dict[str, Any]

_CONDITION_PATTERN = re.compile(
    r"^(?P<current>\d+)/(?P<maximum>\d+)(?:\s+(?P<status>\S+))?"
)


def to_id(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _parse_ident(ident: str) -> tuple[str, str]:
    prefix, separator, name = ident.partition(":")
    side = prefix[:2] if prefix.startswith(("p1", "p2")) else prefix
    return side, name.strip() if separator else ident.strip()


def _species_from_details(details: str) -> str:
    return details.split(",", 1)[0].strip()


def _effect_id(effect: str) -> str:
    return to_id(effect.split(":", 1)[-1])


@dataclass
class PokemonState:
    name: str
    species: str = ""
    details: str = ""
    condition: str = ""
    hp_fraction: float | None = None
    status: str | None = None
    fainted: bool = False
    active: bool = False
    moves: set[str] = field(default_factory=set)
    item: str | None = None
    ability: str | None = None
    boosts: dict[str, int] = field(default_factory=dict)

    def update_details(self, details: str) -> None:
        if not details:
            return

        self.details = details
        self.species = _species_from_details(details)

    def update_condition(self, condition: str) -> None:
        if not condition:
            return

        self.condition = condition
        self.fainted = "fnt" in condition.split()

        match = _CONDITION_PATTERN.match(condition)

        if match:
            current = int(match.group("current"))
            maximum = int(match.group("maximum"))
            self.hp_fraction = current / maximum if maximum else 0.0
            parsed_status = match.group("status")
            self.status = (
                parsed_status
                if parsed_status and parsed_status != "fnt"
                else None
            )

        if self.fainted:
            self.hp_fraction = 0.0
            self.status = "fnt"

    def update_from_request(self, pokemon: BattleRequest) -> None:
        _, name = _parse_ident(pokemon.get("ident", self.name))
        self.name = name
        self.active = pokemon.get("active", self.active)
        self.update_details(pokemon.get("details", ""))
        self.update_condition(pokemon.get("condition", ""))

        self.moves.update(to_id(move) for move in pokemon.get("moves", []))

        if "item" in pokemon:
            self.item = to_id(pokemon["item"]) or None

        ability = pokemon.get("ability") or pokemon.get("baseAbility")

        if ability:
            self.ability = to_id(ability)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "species": self.species,
            "condition": self.condition,
            "hp_fraction": self.hp_fraction,
            "status": self.status,
            "fainted": self.fainted,
            "active": self.active,
            "moves": sorted(self.moves),
            "item": self.item,
            "ability": self.ability,
            "boosts": dict(sorted(self.boosts.items())),
        }


@dataclass
class SideState:
    side_id: str
    name: str = ""
    team_size: int | None = None
    pokemon: dict[str, PokemonState] = field(default_factory=dict)
    conditions: set[str] = field(default_factory=set)

    def get_or_create(self, ident: str) -> PokemonState:
        _, name = _parse_ident(ident)
        key = to_id(name)

        if key not in self.pokemon:
            self.pokemon[key] = PokemonState(name=name)

        return self.pokemon[key]

    def set_active(self, pokemon: PokemonState) -> None:
        for team_member in self.pokemon.values():
            team_member.active = False

        pokemon.active = True

    @property
    def active_pokemon(self) -> PokemonState | None:
        return next(
            (pokemon for pokemon in self.pokemon.values() if pokemon.active),
            None,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "team_size": self.team_size,
            "conditions": sorted(self.conditions),
            "pokemon": {
                key: pokemon.snapshot()
                for key, pokemon in sorted(self.pokemon.items())
            },
        }


@dataclass
class BattleState:
    room_id: str
    turn: int = 0
    format_name: str = ""
    our_side: str | None = None
    weather: str | None = None
    field_conditions: set[str] = field(default_factory=set)
    sides: dict[str, SideState] = field(
        default_factory=lambda: {
            "p1": SideState("p1"),
            "p2": SideState("p2"),
        }
    )
    last_request: BattleRequest | None = None
    winner: str | None = None
    tied: bool = False
    finished: bool = False

    @property
    def opponent_side(self) -> str | None:
        if self.our_side == "p1":
            return "p2"
        if self.our_side == "p2":
            return "p1"
        return None

    @property
    def our_active_pokemon(self) -> PokemonState | None:
        if self.our_side is None:
            return None
        return self.sides[self.our_side].active_pokemon

    @property
    def opponent_active_pokemon(self) -> PokemonState | None:
        if self.opponent_side is None:
            return None
        return self.sides[self.opponent_side].active_pokemon

    def update_from_request(self, request: BattleRequest) -> None:
        self.last_request = copy.deepcopy(request)
        side_request = request.get("side", {})
        team = side_request.get("pokemon", [])
        side_id = side_request.get("id")

        if side_id not in self.sides and team:
            side_id, _ = _parse_ident(team[0].get("ident", ""))

        if side_id not in self.sides:
            return

        self.our_side = side_id
        side = self.sides[side_id]
        side.name = side_request.get("name", side.name)

        for team_member in side.pokemon.values():
            team_member.active = False

        for pokemon_request in team:
            pokemon = side.get_or_create(pokemon_request.get("ident", ""))
            pokemon.update_from_request(pokemon_request)

        side.team_size = max(side.team_size or 0, len(team)) or side.team_size

    def update_from_protocol(self, line: str) -> None:
        if not line.startswith("|"):
            return

        parts = line.split("|")

        if len(parts) < 2:
            return

        event = parts[1]

        if event == "turn" and len(parts) > 2:
            self.turn = int(parts[2])
        elif event == "tier" and len(parts) > 2:
            self.format_name = parts[2]
        elif event == "player" and len(parts) > 3 and parts[2] in self.sides:
            self.sides[parts[2]].name = parts[3]
        elif event == "teamsize" and len(parts) > 3 and parts[2] in self.sides:
            self.sides[parts[2]].team_size = int(parts[3])
        elif event in {"switch", "drag", "replace"} and len(parts) > 4:
            self._update_switch(parts[2], parts[3], parts[4])
        elif event in {"detailschange", "-formechange"} and len(parts) > 3:
            self._update_species(parts[2], parts[3])
        elif event in {"-damage", "-heal"} and len(parts) > 3:
            self._pokemon_for_ident(parts[2]).update_condition(parts[3])
        elif event == "-status" and len(parts) > 3:
            pokemon = self._pokemon_for_ident(parts[2])
            pokemon.status = parts[3]
        elif event == "-curestatus" and len(parts) > 2:
            self._pokemon_for_ident(parts[2]).status = None
        elif event == "faint" and len(parts) > 2:
            pokemon = self._pokemon_for_ident(parts[2])
            pokemon.update_condition("0 fnt")
            pokemon.active = False
        elif event == "move" and len(parts) > 3:
            self._pokemon_for_ident(parts[2]).moves.add(to_id(parts[3]))
        elif event == "-item" and len(parts) > 3:
            self._pokemon_for_ident(parts[2]).item = to_id(parts[3])
        elif event == "-enditem" and len(parts) > 2:
            self._pokemon_for_ident(parts[2]).item = None
        elif event == "-ability" and len(parts) > 3:
            self._pokemon_for_ident(parts[2]).ability = to_id(parts[3])
        elif event in {"-boost", "-unboost", "-setboost"} and len(parts) > 4:
            self._update_boost(event, parts[2], parts[3], parts[4])
        elif event == "-clearboost" and len(parts) > 2:
            self._pokemon_for_ident(parts[2]).boosts.clear()
        elif event == "-clearallboost":
            for side in self.sides.values():
                for pokemon in side.pokemon.values():
                    pokemon.boosts.clear()
        elif event == "-weather" and len(parts) > 2:
            weather = _effect_id(parts[2])
            self.weather = None if weather in {"", "none"} else weather
        elif event == "-fieldstart" and len(parts) > 2:
            self.field_conditions.add(_effect_id(parts[2]))
        elif event == "-fieldend" and len(parts) > 2:
            self.field_conditions.discard(_effect_id(parts[2]))
        elif event == "-sidestart" and len(parts) > 3:
            self._update_side_condition(parts[2], parts[3], add=True)
        elif event == "-sideend" and len(parts) > 3:
            self._update_side_condition(parts[2], parts[3], add=False)
        elif event == "win" and len(parts) > 2:
            self.winner = parts[2]
            self.finished = True
        elif event == "tie":
            self.tied = True
            self.finished = True

    def _pokemon_for_ident(self, ident: str) -> PokemonState:
        side_id, _ = _parse_ident(ident)

        if side_id not in self.sides:
            self.sides[side_id] = SideState(side_id)

        return self.sides[side_id].get_or_create(ident)

    def _update_switch(
        self,
        ident: str,
        details: str,
        condition: str,
    ) -> None:
        side_id, _ = _parse_ident(ident)

        if side_id not in self.sides:
            self.sides[side_id] = SideState(side_id)

        side = self.sides[side_id]
        pokemon = side.get_or_create(ident)
        pokemon.update_details(details)
        pokemon.update_condition(condition)
        pokemon.boosts.clear()
        side.set_active(pokemon)

    def _update_species(self, ident: str, details: str) -> None:
        pokemon = self._pokemon_for_ident(ident)
        pokemon.update_details(details)

    def _update_boost(
        self,
        event: str,
        ident: str,
        stat: str,
        amount_text: str,
    ) -> None:
        pokemon = self._pokemon_for_ident(ident)
        amount = int(amount_text)

        if event == "-setboost":
            pokemon.boosts[stat] = amount
            return

        direction = -1 if event == "-unboost" else 1
        current = pokemon.boosts.get(stat, 0)
        pokemon.boosts[stat] = max(-6, min(6, current + direction * amount))

    def _update_side_condition(
        self,
        side_ident: str,
        condition: str,
        *,
        add: bool,
    ) -> None:
        side_id, _ = _parse_ident(side_ident)

        if side_id not in self.sides:
            return

        condition_id = _effect_id(condition)

        if add:
            self.sides[side_id].conditions.add(condition_id)
        else:
            self.sides[side_id].conditions.discard(condition_id)

    def result_for_our_side(self) -> int | None:
        if not self.finished or self.our_side is None:
            return None
        if self.tied:
            return 0
        our_name = self.sides[self.our_side].name
        return 1 if to_id(self.winner or "") == to_id(our_name) else -1

    def snapshot(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "turn": self.turn,
            "format": self.format_name,
            "our_side": self.our_side,
            "weather": self.weather,
            "field_conditions": sorted(self.field_conditions),
            "sides": {
                side_id: side.snapshot()
                for side_id, side in sorted(self.sides.items())
            },
        }
