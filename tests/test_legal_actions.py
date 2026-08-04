import random
import unittest
from collections import Counter
from copy import deepcopy
from unittest.mock import patch

from src.battle.legal_actions import legal_actions, choose_uniform_action


def modifier_request() -> dict:
    return {
        "rqid": 42,
        "active": [
            {
                "moves": [
                    {
                        "move": "Flamethrower",
                        "id": "flamethrower",
                        "pp": 15,
                        "maxpp": 15,
                        "target": "normal",
                        "disabled": False,
                    },
                    {
                        "move": "Air Slash",
                        "id": "airslash",
                        "pp": 15,
                        "maxpp": 15,
                        "target": "normal",
                        "disabled": False,
                    },
                    {
                        "move": "Protect",
                        "id": "protect",
                        "pp": 10,
                        "maxpp": 10,
                        "target": "self",
                        "disabled": True,
                    },
                    {
                        "move": "Roost",
                        "id": "roost",
                        "pp": 10,
                        "maxpp": 10,
                        "target": "self",
                        "disabled": False,
                    },
                ],
                "canMegaEvo": True,
                "canZMove": [
                    {
                        "move": "Inferno Overdrive",
                        "target": "normal",
                    },
                    None,
                    {
                        "move": "Z-Protect",
                        "target": "self",
                    },
                    None,
                ],
                "trapped": False,
            }
        ],
        "side": {
            "pokemon": [
                {
                    "ident": "p1: Charizard",
                    "condition": "100/100",
                    "active": True,
                },
                {
                    "ident": "p1: Pikachu",
                    "condition": "100/100",
                    "active": False,
                },
                {
                    "ident": "p1: Gengar",
                    "condition": "0 fnt",
                    "active": False,
                },
                {
                    "ident": "p1: Snorlax",
                    "condition": "200/200",
                    "active": False,
                },
            ]
        },
    }


def expected_actions() -> set[str]:
    return {
        "move 1",
        "move 2",
        "move 4",
        "move 1 mega",
        "move 2 mega",
        "move 4 mega",
        "move 1 zmove",
        "move 3 zmove",
        "switch 2",
        "switch 4",
    }


class TestLegalActions(unittest.TestCase):
    def test_generates_every_atomic_action(self):
        actions = legal_actions(modifier_request())

        self.assertSetEqual(set(actions), expected_actions())
        self.assertEqual(len(actions), len(expected_actions()))

    def test_disabled_normal_move_can_have_legal_z_move(self):
        actions = legal_actions(modifier_request())

        self.assertNotIn("move 3", actions)
        self.assertNotIn("move 3 mega", actions)
        self.assertIn("move 3 zmove", actions)

    def test_mega_and_zmove_are_separate_choices(self):
        actions = legal_actions(modifier_request())

        self.assertNotIn("move 1 mega zmove", actions)
        self.assertNotIn("move 3 mega zmove", actions)

    def test_trapped_pokemon_cannot_switch(self):
        request = modifier_request()
        request["active"][0]["trapped"] = True

        actions = legal_actions(request)

        self.assertFalse(any(action.startswith("switch ") for action in actions))

    def test_forced_switch_returns_only_switches(self):
        request = modifier_request()
        request.pop("active")
        request["forceSwitch"] = [True]

        actions = legal_actions(request)

        self.assertSetEqual(set(actions), {"switch 2", "switch 4"})

    def test_wait_request_returns_no_action(self):
        request = modifier_request()
        request["wait"] = True

        self.assertEqual(legal_actions(request), [])

    @patch("src.battle.legal_actions.random.choice")
    def test_picker_uses_one_flat_action_pool(self, mock_choice):
        request = modifier_request()
        actions = legal_actions(request)

        mock_choice.side_effect = lambda supplied_actions: supplied_actions[0]

        chosen = choose_uniform_action(request)

        mock_choice.assert_called_once_with(actions)
        self.assertEqual(chosen, actions[0])

    def test_selection_is_approximately_uniform(self):
        request = modifier_request()
        actions = legal_actions(request)

        random.seed(20260728)

        number_of_selections = 100_000
        counts = Counter(
            choose_uniform_action(request)
            for _ in range(number_of_selections)
        )

        expected_count = number_of_selections / len(actions)

        self.assertSetEqual(set(counts), set(actions))

        for action in actions:
            relative_difference = abs(
                counts[action] - expected_count
            ) / expected_count

            self.assertLess(
                relative_difference,
                0.04,
                f"{action} appeared {counts[action]} times; "
                f"expected approximately {expected_count:.0f}",
            )


if __name__ == "__main__":
    unittest.main()