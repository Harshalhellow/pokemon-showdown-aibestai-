"""
Self-contained battle simulator.

Pits ImprovedHeuristicAgent vs HeuristicAgent / HarderHeuristicAgent for N
battles and prints a win/loss/draw breakdown.  No network or Pokémon Showdown
server needed.

Usage:
    python -m tests.battle_sim [--battles 1000] [--seed 42]

    --baseline  {original|harder}   which baseline to fight (default: harder)

The simulator is deliberately simple:
  - Each Pokémon has scaled HP (base_hp * 2) and deals damage based on
    base_power * STAB * type-effectiveness.
  - No items, no abilities, no stat stages for damage (the agents can still
    score them internally).
  - Teams of 3 are selected from a pool of 15 classic Pokémon.
  - The sim generates a realistic-looking request dict and BattleState so
    the agent decision-making is realistic.
"""
from __future__ import annotations

import argparse
import random
import sys
import os

# Make sure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poke_env.data import GenData
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.harder_heuristic_agent import HarderHeuristicAgent
from src.agents.improved_heuristic_agent import ImprovedHeuristicAgent
from src.battle.state import BattleState, PokemonState, to_id


# ---------------------------------------------------------------------------
# Pokémon pool with simplified moveset definitions
# Format: (species, types, base_hp, moves)
# moves: list of (move_id, category, type, base_power, accuracy, priority)
# ---------------------------------------------------------------------------
POKEMON_POOL = [
    ("Charizard",   ["Fire","Flying"], 78,  [
        ("flamethrower","Special","Fire",90,100,0),
        ("airslash","Special","Flying",75,95,0),
        ("dragonpulse","Special","Dragon",85,100,0),
        ("roost","Status","Flying",0,100,0),
    ]),
    ("Blastoise",   ["Water"], 79, [
        ("surf","Special","Water",90,100,0),
        ("icebeam","Special","Ice",90,100,0),
        ("flashcannon","Special","Steel",80,100,0),
        ("withdraw","Status","Water",0,100,0),
    ]),
    ("Venusaur",    ["Grass","Poison"], 80, [
        ("energyball","Special","Grass",90,100,0),
        ("sludgebomb","Special","Poison",90,100,0),
        ("sleeppowder","Status","Grass",0,75,0),
        ("synthesis","Status","Grass",0,100,0),
    ]),
    ("Gengar",      ["Ghost","Poison"], 60, [
        ("shadowball","Special","Ghost",80,100,0),
        ("sludgebomb","Special","Poison",90,100,0),
        ("thunderbolt","Special","Electric",90,100,0),
        ("willowisp","Status","Fire",0,85,0),
    ]),
    ("Alakazam",    ["Psychic"], 55, [
        ("psychic","Special","Psychic",90,100,0),
        ("focusblast","Special","Fighting",120,70,0),
        ("shadowball","Special","Ghost",80,100,0),
        ("calmmind","Status","Psychic",0,100,0),
    ]),
    ("Machamp",     ["Fighting"], 90, [
        ("closecombat","Physical","Fighting",120,100,0),
        ("stoneedge","Physical","Rock",100,80,0),
        ("earthquake","Physical","Ground",100,100,0),
        ("bulkup","Status","Fighting",0,100,0),
    ]),
    ("Starmie",     ["Water","Psychic"], 60, [
        ("surf","Special","Water",90,100,0),
        ("psychic","Special","Psychic",90,100,0),
        ("icebeam","Special","Ice",90,100,0),
        ("recover","Status","Normal",0,100,0),
    ]),
    ("Lapras",      ["Water","Ice"], 130, [
        ("icebeam","Special","Ice",90,100,0),
        ("surf","Special","Water",90,100,0),
        ("thunderbolt","Special","Electric",90,100,0),
        ("rest","Status","Normal",0,100,0),
    ]),
    ("Snorlax",     ["Normal"], 160, [
        ("bodyslam","Physical","Normal",85,100,0),
        ("crunch","Physical","Dark",80,100,0),
        ("earthquake","Physical","Ground",100,100,0),
        ("rest","Status","Normal",0,100,0),
    ]),
    ("Arcanine",    ["Fire"], 90, [
        ("flareblitz","Physical","Fire",120,100,0),
        ("extremespeed","Physical","Normal",80,100,1),
        ("wildcharge","Physical","Electric",90,100,0),
        ("willowisp","Status","Fire",0,85,0),
    ]),
    ("Vaporeon",    ["Water"], 130, [
        ("surf","Special","Water",90,100,0),
        ("icebeam","Special","Ice",90,100,0),
        ("shadowball","Special","Ghost",80,100,0),
        ("wish","Status","Normal",0,100,0),
    ]),
    ("Jolteon",     ["Electric"], 65, [
        ("thunderbolt","Special","Electric",90,100,0),
        ("shadowball","Special","Ghost",80,100,0),
        ("voltswitch","Special","Electric",70,100,0),
        ("thunderwave","Status","Electric",0,90,0),
    ]),
    ("Nidoking",    ["Poison","Ground"], 81, [
        ("earthquake","Physical","Ground",100,100,0),
        ("poisonjab","Physical","Poison",80,100,0),
        ("icepunch","Physical","Ice",75,100,0),
        ("megahorn","Physical","Bug",120,85,0),
    ]),
    ("Tauros",      ["Normal"], 75, [
        ("bodyslam","Physical","Normal",85,100,0),
        ("earthquake","Physical","Ground",100,100,0),
        ("blizzard","Special","Ice",110,70,0),
        ("firespin","Special","Fire",35,85,0),
    ]),
    ("Clefable",    ["Normal","Fairy"], 95, [
        ("moonblast","Special","Fairy",95,100,0),
        ("icebeam","Special","Ice",90,100,0),
        ("flamethrower","Special","Fire",90,100,0),
        ("softboiled","Status","Normal",0,100,0),
    ]),
]

GEN_DATA = GenData.from_gen(7)


# ---------------------------------------------------------------------------
# Sim utilities
# ---------------------------------------------------------------------------

def _type_chart_mult(move_type: str, defender_types: list[str]) -> float:
    mult = 1.0
    for dt in defender_types:
        row = GEN_DATA.type_chart.get(dt.upper(), {})
        mult *= row.get(move_type.upper(), 1.0)
    return mult


def _make_team(pool_indices: list[int], side_id: str) -> list[dict]:
    """Build a list of 3 pokemon request dicts from pool_indices."""
    team = []
    for slot, idx in enumerate(pool_indices, start=1):
        species, types, base_hp, moves = POKEMON_POOL[idx]
        hp = base_hp * 2
        move_list = [
            {
                "move": m[0],
                "id": m[0],
                "pp": 24,
                "maxpp": 24,
                "target": "normal",
                "disabled": False,
            }
            for m in moves
        ]
        team.append({
            "ident": f"{side_id}: {species}",
            "details": species,
            "condition": f"{hp}/{hp}",
            "active": (slot == 1),
            "moves": move_list,
            "baseAbility": "none",
            "ability": "none",
            "item": "",
        })
    return team


def _build_request(team: list[dict], active_idx: int, side_id: str) -> dict:
    """Build a request dict with the given pokemon at active_idx active."""
    return {
        "active": [{"moves": team[active_idx]["moves"]}],
        "side": {
            "id": side_id,
            "name": side_id,
            "pokemon": team,
        },
    }


def _pool_entry(species_name: str):
    for entry in POKEMON_POOL:
        if entry[0] == species_name:
            return entry
    return None


def _damage(
    attacker_species: str,
    move_idx: int,
    defender_types: list[str],
    defender_hp: int,
) -> int:
    entry = _pool_entry(attacker_species)
    if entry is None:
        return 0
    _, atk_types, _, moves = entry
    _, category, move_type, base_power, accuracy, _ = moves[move_idx]
    if base_power == 0:
        return 0  # status move
    if random.random() * 100 >= accuracy:
        return 0  # miss
    stab = 1.5 if move_type in atk_types else 1.0
    eff = _type_chart_mult(move_type, defender_types)
    raw = base_power * stab * eff
    # scale to HP pool — a base 90 neutral move does ~30–40% damage
    dmg = max(1, int(raw / 3.5))
    return dmg


# ---------------------------------------------------------------------------
# SimPokemon: mutable sim state
# ---------------------------------------------------------------------------

class SimPokemon:
    def __init__(self, pool_idx: int, side_id: str):
        species, types, base_hp, moves = POKEMON_POOL[pool_idx]
        self.species = species
        self.types = types
        self.max_hp = base_hp * 2
        self.current_hp = self.max_hp
        self.moves = moves  # list of tuples
        self.side_id = side_id
        self.fainted = False

    @property
    def hp_fraction(self) -> float:
        return self.current_hp / self.max_hp if self.max_hp else 0.0

    def take_damage(self, dmg: int) -> None:
        self.current_hp = max(0, self.current_hp - dmg)
        if self.current_hp == 0:
            self.fainted = True

    def condition_str(self) -> str:
        if self.fainted:
            return "0 fnt"
        return f"{self.current_hp}/{self.max_hp}"

    def move_dicts(self) -> list[dict]:
        """List of move dicts suitable for the 'active' portion of a request."""
        return [
            {
                "move": m[0],
                "id": m[0],
                "pp": 24,
                "maxpp": 24,
                "target": "normal",
                "disabled": False,
            }
            for m in self.moves
        ]

    def as_request_entry(self, active: bool) -> dict:
        return {
            "ident": f"{self.side_id}: {self.species}",
            "details": self.species,
            "condition": self.condition_str(),
            "active": active,
            # state.py's update_from_request expects a list of move-id strings
            # in the side/pokemon section (it calls to_id on each element).
            "moves": [m[0] for m in self.moves],
            "baseAbility": "none",
            "ability": "none",
            "item": "",
        }


# ---------------------------------------------------------------------------
# Side
# ---------------------------------------------------------------------------

class SimSide:
    def __init__(self, side_id: str, pool_indices: list[int]):
        self.side_id = side_id
        self.pokes = [SimPokemon(idx, side_id) for idx in pool_indices]
        self.active_idx = 0

    @property
    def active(self) -> SimPokemon:
        return self.pokes[self.active_idx]

    def alive_count(self) -> int:
        return sum(1 for p in self.pokes if not p.fainted)

    def build_request(self) -> dict:
        team = [p.as_request_entry(i == self.active_idx)
                for i, p in enumerate(self.pokes)]
        return {
            "active": [{"moves": self.active.move_dicts()}],
            "side": {
                "id": self.side_id,
                "name": self.side_id,
                "pokemon": team,
            },
        }

    def build_state(self, opponent: "SimSide") -> BattleState:
        state = BattleState(room_id="sim")
        state.our_side = self.side_id
        # Populate our side
        our_side_state = state.sides[self.side_id]
        our_side_state.name = self.side_id
        for i, p in enumerate(self.pokes):
            pstate = PokemonState(name=p.species)
            pstate.species = p.species
            pstate.hp_fraction = p.hp_fraction
            pstate.fainted = p.fainted
            pstate.active = (i == self.active_idx)
            pstate.condition = p.condition_str()
            our_side_state.pokemon[to_id(p.species)] = pstate

        # Populate opponent side
        opp_side_id = opponent.side_id
        opp_side_state = state.sides[opp_side_id]
        opp_side_state.name = opp_side_id
        for i, p in enumerate(opponent.pokes):
            pstate = PokemonState(name=p.species)
            pstate.species = p.species
            pstate.hp_fraction = p.hp_fraction
            pstate.fainted = p.fainted
            pstate.active = (i == opponent.active_idx)
            pstate.condition = p.condition_str()
            opp_side_state.pokemon[to_id(p.species)] = pstate

        return state

    def apply_action(self, action: str) -> str | None:
        """
        Apply a switch action.  Returns None for moves (handled by battle loop).
        Returns the new active species on switch.
        """
        if action.startswith("switch "):
            slot = int(action.split()[1]) - 1
            if 0 <= slot < len(self.pokes) and not self.pokes[slot].fainted:
                self.active_idx = slot
                return self.pokes[slot].species
        return None

    def force_switch_if_needed(self) -> bool:
        """Switch to a non-fainted Pokémon if current active has fainted."""
        if not self.active.fainted:
            return False
        for i, p in enumerate(self.pokes):
            if not p.fainted:
                self.active_idx = i
                return True
        return False  # all fainted


# ---------------------------------------------------------------------------
# Battle
# ---------------------------------------------------------------------------

def _resolve_move_slot(action: str) -> int | None:
    """Return 0-based move slot or None if it's not a move action."""
    if action.startswith("move "):
        return int(action.split()[1]) - 1
    return None


def run_battle(
    improved_agent: ImprovedHeuristicAgent,
    baseline_agent: HeuristicAgent,
    pool_indices_p1: list[int],
    pool_indices_p2: list[int],
    room_id: str = "battle",
) -> int:
    """
    Simulate one battle.  Returns +1 if p1 (improved) wins, -1 if p2 (baseline) wins, 0 for draw.
    baseline_agent can be HeuristicAgent or HarderHeuristicAgent.
    """
    p1 = SimSide("p1", pool_indices_p1)  # improved
    p2 = SimSide("p2", pool_indices_p2)  # baseline

    MAX_TURNS = 200

    for _turn in range(MAX_TURNS):
        if p1.alive_count() == 0:
            return -1
        if p2.alive_count() == 0:
            return 1

        # Build requests and states
        req1 = p1.build_request()
        req2 = p2.build_request()
        state1 = p1.build_state(p2)
        state2 = p2.build_state(p1)

        # Get decisions
        dec1 = improved_agent.choose_action(req1, state1, room_id=room_id + "_p1")
        # HarderHeuristicAgent accepts room_id; plain HeuristicAgent does not.
        if isinstance(baseline_agent, HarderHeuristicAgent):
            dec2 = baseline_agent.choose_action(req2, state2, room_id=room_id + "_p2")
        else:
            dec2 = baseline_agent.choose_action(req2, state2)

        act1 = dec1.action or "pass"
        act2 = dec2.action or "pass"

        # Resolve switches first
        if act1.startswith("switch "):
            p1.apply_action(act1)
        if act2.startswith("switch "):
            p2.apply_action(act2)

        # Resolve moves
        slot1 = _resolve_move_slot(act1)
        slot2 = _resolve_move_slot(act2)

        # Priority determines order; ties broken by speed
        def priority(poke: SimPokemon, slot: int | None) -> int:
            if slot is None:
                return 0
            return poke.moves[slot][5] if slot < len(poke.moves) else 0

        speed1 = GEN_DATA.pokedex.get(to_id(p1.active.species), {}).get("baseStats", {}).get("spe", 50)
        speed2 = GEN_DATA.pokedex.get(to_id(p2.active.species), {}).get("baseStats", {}).get("spe", 50)

        prio1 = priority(p1.active, slot1)
        prio2 = priority(p2.active, slot2)

        if prio1 > prio2 or (prio1 == prio2 and speed1 >= speed2):
            order = [(p1, slot1, p2), (p2, slot2, p1)]
        else:
            order = [(p2, slot2, p1), (p1, slot1, p2)]

        for attacker_side, move_slot, defender_side in order:
            if attacker_side.active.fainted or defender_side.active.fainted:
                continue
            if move_slot is None:
                continue
            if move_slot >= len(attacker_side.active.moves):
                continue
            dmg = _damage(
                attacker_side.active.species,
                move_slot,
                defender_side.active.types,
                defender_side.active.current_hp,
            )
            defender_side.active.take_damage(dmg)

        # Force switches after faints
        p1.force_switch_if_needed()
        p2.force_switch_if_needed()

        if p1.alive_count() == 0:
            return -1
        if p2.alive_count() == 0:
            return 1

    return 0  # draw after max turns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Battle simulator: Improved vs Baseline")
    parser.add_argument("--battles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline",
        choices=["original", "harder", "both"],
        default="both",
        help="Which baseline agent to fight (default: both)",
    )
    args = parser.parse_args()

    matchups: list[tuple[str, HeuristicAgent]] = []
    if args.baseline in ("original", "both"):
        matchups.append(("HeuristicAgent (original)", HeuristicAgent(generation=7)))
    if args.baseline in ("harder", "both"):
        matchups.append(("HarderHeuristicAgent (patched)", HarderHeuristicAgent(generation=7)))

    for baseline_label, baseline in matchups:
        random.seed(args.seed)
        improved = ImprovedHeuristicAgent(generation=7)

        wins = draws = losses = 0
        pool_size = len(POKEMON_POOL)
        team_size = 3

        print(f"\n  >>> ImprovedHeuristicAgent  vs  {baseline_label}  <<<\n")

        for battle_num in range(1, args.battles + 1):
            p1_indices = random.sample(range(pool_size), team_size)
            p2_indices = random.sample(range(pool_size), team_size)

            result = run_battle(
                improved, baseline,
                p1_indices, p2_indices,
                room_id=f"battle_{battle_num}",
            )

            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

            if battle_num % 100 == 0:
                wp = wins / battle_num * 100
                print(
                    f"  [{battle_num:>4}/{args.battles}]  "
                    f"Improved W={wins}  {baseline_label} W={losses}  Draws={draws}  "
                    f"Win%={wp:.1f}%"
                )

        total = args.battles
        print()
        print("=" * 65)
        print(f"  FINAL RESULTS  ({total} battles, seed={args.seed})")
        print(f"  ImprovedHeuristicAgent  vs  {baseline_label}")
        print("=" * 65)
        print(f"  ImprovedHeuristicAgent wins : {wins:>5}  ({wins/total*100:.1f}%)")
        print(f"  {baseline_label:<35}: {losses:>5}  ({losses/total*100:.1f}%)")
        print(f"  Draws                       : {draws:>5}  ({draws/total*100:.1f}%)")
        decided = wins + losses
        if decided:
            print(f"  Win rate (excl. draws)      : {wins/decided*100:.1f}%")
        print("=" * 65)


if __name__ == "__main__":
    main()
