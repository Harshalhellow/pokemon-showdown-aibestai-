from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from src.agents.heuristic_agent import HeuristicAgent
from src.battle.legal_actions import legal_actions
from src.battle.state import BattleState
from src.battle.trajectory import TrajectoryRecorder

SERVER_URL = "ws://localhost:8000/showdown/websocket"
ORIGIN = "http://localhost:8000"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
RAW_LOG_FILE = DATA_DIRECTORY / "showdown_messages.log"
LATEST_REQUEST_FILE = DATA_DIRECTORY / "latest_request.json"
TRAJECTORY_FILE = DATA_DIRECTORY / "trajectories.jsonl"


def split_frame(frame: str) -> tuple[str, list[str]]:
    """
    Separate a Showdown WebSocket frame into its room ID and protocol lines.
    """

    lines = frame.splitlines()
    room_id = "global"

    if lines and lines[0].startswith(">"):
        room_id = lines[0][1:]
        lines = lines[1:]

    non_empty_lines = [line for line in lines if line]

    return room_id, non_empty_lines


def save_raw_frame(frame: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()

    with RAW_LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{timestamp}]\n")
        log_file.write(frame)
        log_file.write("\n")


def save_request(room_id: str, request: dict) -> None:
    request_record = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "room_id": room_id,
        "request": request,
    }

    with LATEST_REQUEST_FILE.open("w", encoding="utf-8") as request_file:
        json.dump(request_record, request_file, indent=2)


async def send_command(websocket, command: str, room_id: str = "") -> None:
    message = f"{room_id}|{command}"

    await websocket.send(message)

    destination = room_id or "global"
    print(f"[SENT][{destination}] {command}")


async def handle_protocol_line(
    websocket,
    room_id: str,
    line: str,
    username: str,
    battle_states: dict[str, BattleState],
    agent: HeuristicAgent,
    trajectory_recorder: TrajectoryRecorder,
) -> None:
    print(f"[RECEIVED][{room_id}] {line}")

    battle_state = None

    if room_id.startswith("battle-"):
        battle_state = battle_states.setdefault(
            room_id,
            BattleState(room_id),
        )
        battle_state.update_from_protocol(line)

        if battle_state.finished:
            saved_file = trajectory_recorder.finish_battle(battle_state)

            if saved_file:
                print(f"Battle trajectory saved to: {saved_file}")

            battle_states.pop(room_id, None)
            return

    if line.startswith("|challstr|"):
        await send_command(websocket, f"/trn {username}")
        return

    if line.startswith("|pm|"):
        parts = line.split("|", 4)

        if len(parts) < 5:
            return

        sender = parts[2].strip()
        receiver = parts[3].strip()
        message = parts[4]

        print(
            f"PM received from {sender} to {receiver}: "
            f"{message}"
        )

        if (
            receiver.lower() == username.lower()
            and message.startswith("/challenge ")
        ):
            challenge_payload = message.removeprefix(
                "/challenge "
            )
            battle_format = challenge_payload.split(
                "|",
                1,
            )[0]

            print(
                f"Challenge received from {sender}: "
                f"{battle_format}"
            )

            if battle_format == "gen7randombattle":
                await send_command(websocket, "/utm null")
                await send_command(
                    websocket,
                    f"/accept {sender}",
                )
            else:
                print(
                    f"Challenge ignored because format "
                    f"was {battle_format}."
                )

        return

    if line.startswith("|updatechallenges|"):
        payload = line.split("|", 2)[2]

        try:
            challenge_data = json.loads(payload)
        except json.JSONDecodeError:
            print("Could not decode challenge information.")
            return

        challenges = challenge_data.get(
            "challengesFrom",
            {},
        )

        for challenger, battle_format in challenges.items():
            print(
                f"Challenge received from {challenger}: "
                f"{battle_format}"
            )

            if battle_format == "gen7randombattle":
                await send_command(websocket, "/utm null")
                await send_command(
                    websocket,
                    f"/accept {challenger}",
                )
            else:
                print(
                    "Challenge ignored. For this test, "
                    "use Gen 7 Random Battle."
                )

        return

    if line.startswith("|request|"):
        payload = line.split("|", 2)[2]

        if not payload:
            return

        try:
            request = json.loads(payload)
        except json.JSONDecodeError:
            print("Could not decode the battle request JSON.")
            return

        if not isinstance(request, dict):
            print("Battle request did not contain a decision object.")
            return

        if battle_state is None:
            battle_state = battle_states.setdefault(
                room_id,
                BattleState(room_id),
            )

        battle_state.update_from_request(request)

        save_request(room_id, request)

        team = request.get("side", {}).get("pokemon", [])
        active = request.get("active", [])
        force_switch = request.get("forceSwitch")
        waiting = request.get("wait", False)
        request_id = request.get("rqid")

        print("\n=== PRIVATE BATTLE REQUEST RECEIVED ===")
        print(f"Room: {room_id}")
        print(f"Request ID: {request_id}")
        print(f"Team Pokémon: {len(team)}")
        print(f"Active positions: {len(active)}")
        print(f"Forced switch: {force_switch}")
        print(f"Waiting: {waiting}")
        print(f"Saved to: {LATEST_REQUEST_FILE}")
        print("=======================================\n")

        choices = legal_actions(request)

        if not choices:
            print("No decision is required for this request.")
            return

        decision = agent.choose_action(request, battle_state)
        chosen_action = decision.action

        if chosen_action is None:
            print("Could not select an action.")
            return

        print(f"Legal actions: {choices}")
        print("Heuristic ranking:")

        for scored_action in decision.ranked_actions:
            reason_text = "; ".join(scored_action.reasons)
            print(
                f"  {scored_action.action}: "
                f"{scored_action.score:.3f} ({reason_text})"
            )

        print(f"Selected action: {chosen_action}")

        trajectory_recorder.record_decision(
            battle_state,
            request,
            decision,
        )

        command = f"/choose {chosen_action}"

        if request_id is not None:
            command += f"|{request_id}"

        await send_command(
            websocket,
            command,
            room_id,
        )

        return


async def run_receiver(username: str) -> None:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    battle_states: dict[str, BattleState] = {}
    agent = HeuristicAgent(generation=7)
    trajectory_recorder = TrajectoryRecorder(TRAJECTORY_FILE)

    print(f"Connecting to {SERVER_URL}...")

    try:
        async with connect(
            SERVER_URL,
            origin=ORIGIN,
            max_size=None,
        ) as websocket:
            print("Connected to Pokémon Showdown.")
            print(f"Requested username: {username}")
            print("Listening for server messages...\n")

            async for frame in websocket:
                save_raw_frame(frame)

                room_id, protocol_lines = split_frame(frame)

                for line in protocol_lines:
                    await handle_protocol_line(
                        websocket,
                        room_id,
                        line,
                        username,
                        battle_states,
                        agent,
                        trajectory_recorder,
                    )

    except ConnectionClosed as error:
        print(
            f"Connection closed: code={error.code}, "
            f"reason={error.reason}"
        )
    except OSError as error:
        print(f"Could not connect to the local server: {error}")
        print("Make sure Pokémon Showdown is running on port 8000.")


def main() -> None:
    username = "ReceiverBot"

    if len(sys.argv) > 1:
        username = sys.argv[1]

    try:
        asyncio.run(run_receiver(username))
    except KeyboardInterrupt:
        print("\nReceiver stopped.")


if __name__ == "__main__":
    main()
