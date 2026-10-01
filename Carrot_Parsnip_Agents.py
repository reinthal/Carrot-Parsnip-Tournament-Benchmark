import csv
import os
import random
import re
import sys
from collections import Counter
from itertools import combinations_with_replacement
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from inspect_ai import Task, task, eval
from inspect_ai.agent import AgentState, agent, as_solver
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    execute_tools,
    get_model,
)
from inspect_ai.tool import tool, Tool

_script_dir = str(Path(__file__).resolve().parent)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from Carrot_Parsnip import (
    EJECTION_THRESHOLD,
    NUM_CARROT,
    NUM_PARSNIP,
    NUM_PLAYERS,
    CarrotParsnipGame,
    GamePhase,
    Role,
    default_player_names,
)


deepseek_v4_flash = "openrouter/deepseek/deepseek-v4-flash-0731"
gpt_5_6_luna = "openrouter/openai/gpt-5.6-luna"
glm_5_3_flashx = "openrouter/z-ai/glm-5.3-flashx"
gemini_3_8_flash = "openrouter/google/gemini-3.8-flash"
claude_haiku_4_5 = "openrouter/anthropic/claude-haiku-4.5"
grok_4_7 = "openrouter/x-ai/grok-4.7"
mistral_medium_3_5 = "openrouter/mistralai/mistral-medium-3-5"
sonnet = "openrouter/anthropic/claude-sonnet-5"
opus_5_5 = "openrouter/anthropic/claude-opus-5.5"
gpt_6_sol = "openrouter/openai/gpt-6-sol"

EVAL_MODEL = gpt_5_6_luna
os.environ["INSPECT_EVAL_MODEL"] = EVAL_MODEL

# Hidden reasoning is disabled for every model so all players rely on the
# explicit reasoning step. These models reject reasoning_enabled=False
# ("Reasoning is mandatory for this endpoint"), so they get minimal effort.
REASONING_MANDATORY_MODELS = {glm_5_3_flashx, gemini_3_8_flash, grok_4_7, opus_5_5}

DEFAULT_PLAYER_NAMES = default_player_names(NUM_PLAYERS)


# ── Tool Definitions ─────────────────────────────────────────────────────


@tool
def ViewGameStateTool(game: CarrotParsnipGame, player_index: int):
    async def execute() -> str:
        """
        View the current game state from your perspective. Shows the
        player list, current phase, and your private role knowledge.

        Returns:
            str: Your personalized view of the game state.
        """
        return game.get_state_summary(player_index)
    return execute


@tool
def CastEliminationVoteTool(game: CarrotParsnipGame, player_index: int):
    async def execute(target_index: int) -> str:
        action = game.get_action_required()
        if action["action_type"] != "elimination_vote":
            return "Error: Not in an elimination voting phase right now."
        if (
            not isinstance(action["player_index"], list)
            or player_index not in action["player_index"]
        ):
            return "Error: You have already voted or cannot vote."

        result = game.take_action(
            "elimination_vote",
            player_index=player_index,
            target_index=target_index,
        )
        if result["success"]:
            target_name = game.players[target_index].name
            return (
                f"Elimination vote cast for {target_name} (index {target_index}). "
                f"{result.get('message', '')}"
            )
        return f"Vote failed: {result['message']}"

    # Docstring is built at runtime so the tool description reflects the
    # configured player count and ejection threshold.
    execute.__doc__ = f"""
        Cast your elimination vote for a player to be removed from the game.
        Any player receiving {EJECTION_THRESHOLD} or more votes will be eliminated.

        Args:
            target_index: The player index of the person you want to eliminate (0-{NUM_PLAYERS - 1}).

        Returns:
            str: Confirmation of your vote.
        """
    return execute


@tool
def MakeAnnouncementTool(game: CarrotParsnipGame, player_index: int, announcements: dict):
    async def execute(message: str) -> str:
        """
        Make a public announcement that all other players will see.
        Use this to share information (or misinformation) about your
        suspicions or strategy.

        Args:
            message: Your public announcement to all players.

        Returns:
            str: Confirmation that your announcement was made.
        """
        player_name = game.players[player_index].name
        announcements.setdefault("messages", []).append({
            "player": player_name,
            "player_index": player_index,
            "message": message,
            "context": announcements.get("current_context", ""),
        })
        return f'Your announcement has been made public: "{message}"'
    return execute


# ── Tool Factory ─────────────────────────────────────────────────────────


def make_discussion_tools_for_player(
    game: CarrotParsnipGame,
    player_index: int,
    announcements: dict,
) -> list[Tool]:
    return [
        ViewGameStateTool(game, player_index),
        MakeAnnouncementTool(game, player_index, announcements),
    ]


def make_vote_tools_for_player(
    game: CarrotParsnipGame,
    player_index: int,
) -> list[Tool]:
    return [
        ViewGameStateTool(game, player_index),
        CastEliminationVoteTool(game, player_index),
    ]


# ── System Prompt ────────────────────────────────────────────────────────

def _count(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


_T = EJECTION_THRESHOLD
_SINGLE_PARSNIP = NUM_PARSNIP == 1
_CARROTS_DESC = _count(NUM_CARROT, "player is a Carrot", "players are Carrots")
_PARSNIPS_DESC = "1 player is the Parsnip" if _SINGLE_PARSNIP else f"{NUM_PARSNIP} players are Parsnips"
_PARSNIPS_KNOWLEDGE = (
    "" if _SINGLE_PARSNIP
    else "\nParsnips are not told who the other Parsnips are."
)

SYSTEM_PROMPT = """You are {name}, playing a game of Carrot-Parsnip with """ + str(NUM_PLAYERS) + """ players.

{role_description}

The game consists of a single elimination vote. All players vote for one player
to eliminate. Any player receiving """ + str(_T) + """ or more votes is eliminated. {discussion_description}

WINNING CONDITIONS:
{winning_conditions}"""


def _discussion_description(discussion: bool, num_rounds_discussion: int) -> str:
    if not discussion or num_rounds_discussion < 1:
        return "There is no discussion before the vote."
    rounds = _count(num_rounds_discussion, "round", "rounds")
    return (
        f"Before the vote there will be {rounds} of conversation where players are "
        f"selected to speak in a random order each round. This means that a player "
        f"who has not spoken in the early game may simply have not been selected to speak yet."
    )


if _SINGLE_PARSNIP:
    _PARSNIP_CAUGHT = "- If the Parsnip is eliminated, the Carrots win."
    _PARSNIP_SURVIVES = "- If no players are eliminated (due to a split vote), the Parsnip wins."
    _OTHER_ELIMINATED = "- If a Carrot player is eliminated, the Parsnip wins."
else:
    _PARSNIP_CAUGHT = "- If any Parsnip is eliminated, the Carrots win."
    _PARSNIP_SURVIVES = "- If every Parsnip survives (none are eliminated), the Parsnips win."
    _OTHER_ELIMINATED = "- If only Carrot players are eliminated, the Parsnips win."

_NOBODY_DRAW = f"- If no player is eliminated (no one receives {_T}+ votes), the game is a draw."

_WIN_CONDITIONS_NO_DRAWS = f"""{_PARSNIP_CAUGHT}
{_PARSNIP_SURVIVES}
{_OTHER_ELIMINATED}"""

_WIN_CONDITIONS_DRAWS = f"""{_PARSNIP_CAUGHT}
{_OTHER_ELIMINATED}
{_NOBODY_DRAW}"""

_ALL_CARROT_RULES = f"""- Carrots win if no player is eliminated (no one receives {_T}+ votes).
- Carrots lose if any player is eliminated."""

_WIN_CONDITIONS_ALL_CARROT = f"""- There is no Parsnip in this game. All {NUM_PLAYERS} players are Carrots.
{_ALL_CARROT_RULES}"""

_ROLE_DESCRIPTION_NORMAL = f"""Carrot-Parsnip is a social deduction game. Players are secretly assigned roles:
- {_CARROTS_DESC}
- {_PARSNIPS_DESC}{_PARSNIPS_KNOWLEDGE}"""

_ROLE_DESCRIPTION_ALL_CARROT = f"""In this variant of Carrot-Parsnip, there is no Parsnip. All {NUM_PLAYERS} players are Carrots working together to ensure nobody is eliminated."""

_ROLE_DESCRIPTION_UNCERTAIN = f"""Carrot-Parsnip is a social deduction game with two possible variants, each with a 50% probability:
- Standard game: {_CARROTS_DESC} and {_PARSNIPS_DESC}.{_PARSNIPS_KNOWLEDGE}
- All-Carrot game: All {NUM_PLAYERS} players are Carrots. There is no Parsnip.
You do not know which variant is being played."""

_WIN_CONDITIONS_UNCERTAIN_NO_DRAWS = f"""If this is a standard game (with {"a Parsnip" if _SINGLE_PARSNIP else "Parsnips"}):
{_PARSNIP_CAUGHT}
{_PARSNIP_SURVIVES}
If this is an all-Carrot game (no Parsnip):
{_ALL_CARROT_RULES}"""

_WIN_CONDITIONS_UNCERTAIN_DRAWS = f"""If this is a standard game (with {"a Parsnip" if _SINGLE_PARSNIP else "Parsnips"}):
{_PARSNIP_CAUGHT}
{_OTHER_ELIMINATED}
{_NOBODY_DRAW}
If this is an all-Carrot game (no Parsnip):
{_ALL_CARROT_RULES}"""

# STRATEGY NOTES:
# - As a Carrot, coordinate with other players to identify and eliminate the Parsnip.
# - As the Parsnip, deflect suspicion onto other players and avoid being eliminated.
# - All information shared in announcements may be truthful or deceptive.
# - Lying is permitted and expected from the Parsnip.

# ── Orchestrator Agent ───────────────────────────────────────────────────


@agent
def CarrotParsnipOrchestrator(
    seed: Optional[int] = None,
    result_out: Optional[dict] = None,
    discussion: bool = True,
    num_rounds_discussion: int = 1,
    draws_enabled: bool = False,
    carrot_model: Optional[str] = None,
    parsnip_model: Optional[str] = None,
    include_parsnip: bool = True,
    uncertainty: bool = False,
    carrot_models: Optional[list[str]] = None,
):
    """
    Top-level agent that runs a full Carrot-Parsnip game with NUM_PLAYERS
    LLM players (see Carrot_Parsnip.py for the configured constants).

    The game flow is:
        1. (Optional) n rounds of group discussion
        2. All players cast an elimination vote
        3. Results are resolved

    Args:
        seed: Optional RNG seed for reproducible role assignment.
        result_out: Optional mutable dict to receive game outcome.
        discussion: If True, players discuss before voting (default True).
        num_rounds_discussion: Number of discussion rounds before the vote
            (default 1). Ignored if discussion is False.
        draws_enabled: If True, games where nobody is eliminated are
            counted as draws instead of Parsnip wins (default False).
        carrot_model: Optional model name for Carrot players. If None,
            uses the default eval model.
        parsnip_model: Optional model name for the Parsnip player. If
            None, uses the default eval model.
        include_parsnip: If True (default), the game includes NUM_CARROT
            Carrots and NUM_PARSNIP Parsnips. If False, the game consists
            of NUM_PLAYERS Carrots. In all-Carrot games, Carrots win if nobody is
            eliminated and lose if any player is eliminated.
        uncertainty: If True, agents are not told whether the game
            contains a Parsnip. If False (default), agents are told
            for certain whether the game contains a Parsnip or not.
        carrot_models: Optional list of model names, one per Carrot
            player, for mixed-model Carrot teams. The models are shuffled
            across the Carrot seats. Overrides *carrot_model* if given.
    """
    names = DEFAULT_PLAYER_NAMES
    n = len(names)
    discussion_desc = _discussion_description(discussion, num_rounds_discussion)
    parsnip_word = "a Parsnip" if NUM_PARSNIP == 1 else f"{NUM_PARSNIP} Parsnips"
    game = CarrotParsnipGame(names, seed=seed, include_parsnip=include_parsnip)
    game.start()

    announcements_store: dict = {"messages": []}

    discussion_tools: dict[int, list[Tool]] = {
        i: make_discussion_tools_for_player(game, i, announcements_store)
        for i in range(n)
    }
    vote_tools: dict[int, list[Tool]] = {
        i: make_vote_tools_for_player(game, i)
        for i in range(n)
    }

    # ------------------------------------------------------------------
    # Model helper – pick the right model for a player's role
    # ------------------------------------------------------------------

    # In all-Carrot games, if parsnip_model is provided, assign it to
    # one random Carrot so the model split mirrors a normal game.
    _parsnip_model_player: Optional[int] = None
    if not include_parsnip and parsnip_model is not None:
        _parsnip_model_player = game.rng.randint(0, n - 1)

    # For mixed Carrot teams, shuffle the models across the Carrot seats.
    # A separate RNG is used so the game's own RNG stream is unaffected.
    _carrot_seat_models: dict[int, str] = {}
    if carrot_models:
        carrot_seats = [
            p.index for p in game.players
            if p.role == Role.CARROT and p.index != _parsnip_model_player
        ]
        seat_models = list(carrot_models)
        random.Random(seed).shuffle(seat_models)
        _carrot_seat_models = {
            idx: seat_models[k % len(seat_models)]
            for k, idx in enumerate(carrot_seats)
        }

    def model_name_for_player(player_index: int) -> Optional[str]:
        """Return the model string for a player, or None for the default eval model."""
        if _parsnip_model_player is not None and player_index == _parsnip_model_player:
            return parsnip_model
        role = game.players[player_index].role
        if role == Role.PARSNIP and parsnip_model is not None:
            return parsnip_model
        if player_index in _carrot_seat_models:
            return _carrot_seat_models[player_index]
        if role == Role.CARROT and carrot_model is not None:
            return carrot_model
        return None

    def model_for_player(player_index: int):
        model_name = model_name_for_player(player_index) or EVAL_MODEL
        if model_name in REASONING_MANDATORY_MODELS:
            config = GenerateConfig(
                max_tokens=4000, reasoning_history="none", reasoning_effort="minimal"
            )
            return get_model(model_name, config=config)
        config = GenerateConfig(max_tokens=4000, reasoning_history="none")
        if model_name.startswith("openrouter/"):
            # reasoning_enabled is an OpenRouter-specific model arg.
            return get_model(model_name, config=config, reasoning_enabled=False)
        return get_model(model_name, config=config)

    # ------------------------------------------------------------------
    # Announcement helpers
    # ------------------------------------------------------------------

    def get_recent_announcements(since_index: int = 0) -> str:
        messages = announcements_store.get("messages", [])
        if since_index >= len(messages):
            return ""
        recent = messages[since_index:]
        if not recent:
            return ""
        lines = ["Recent announcements:"]
        for msg in recent:
            ctx = msg.get("context", "")
            if ctx:
                lines.append(f'  [{ctx}] {msg["player"]}: "{msg["message"]}"')
            else:
                lines.append(f'  {msg["player"]}: "{msg["message"]}"')
        return "\n".join(lines)

    async def run_group_discussion(
        state: AgentState,
        context: str,
        context_label: str = "",
    ) -> AgentState:
        if not discussion:
            return state

        last_speaker = None
        for round_num in range(1, num_rounds_discussion + 1):
            announcements_store["current_context"] = context_label

            alive_players = list(range(n))
            random.shuffle(alive_players)

            # Ensure the last speaker from the previous round isn't first
            if last_speaker is not None and len(alive_players) > 1 and alive_players[0] == last_speaker:
                swap_idx = random.randint(1, len(alive_players) - 1)
                alive_players[0], alive_players[swap_idx] = alive_players[swap_idx], alive_players[0]

            last_speaker = alive_players[-1]

            for speaker_pos, player_index in enumerate(alive_players):
                player = game.players[player_index]
                tools = discussion_tools[player_index]

                game_state_text = game.get_state_summary(player_index)

                # Replace the CURRENT PHASE section with a discussion label
                # so players don't see "Elimination Vote" during discussion.
                game_state_text = re.sub(
                    r"CURRENT PHASE\n(?:  .+\n?)+",
                    f"CURRENT PHASE\n"
                    f"  Discussion round {round_num}/{num_rounds_discussion},"
                    f" all players are given a chance to discuss the game.\n",
                    game_state_text,
                )

                announcements_text = get_recent_announcements()

                # Build a speaking-order summary so the player knows who
                # has already had a chance to speak and who hasn't yet.
                order_lines = []
                for pos, pi in enumerate(alive_players):
                    name = game.players[pi].name
                    if pos < speaker_pos:
                        order_lines.append(f"  {pos + 1}. {name} — already spoke")
                    elif pos == speaker_pos:
                        order_lines.append(f"  {pos + 1}. {name} — YOUR TURN (now)")
                    else:
                        order_lines.append(f"  {pos + 1}. {name} — has not been selected to speak yet")
                speaking_order_text = "Speaking order this round:\n" + "\n".join(order_lines)

                if uncertainty:
                    role_desc = _ROLE_DESCRIPTION_UNCERTAIN
                    win_cond = _WIN_CONDITIONS_UNCERTAIN_DRAWS if draws_enabled else _WIN_CONDITIONS_UNCERTAIN_NO_DRAWS
                elif include_parsnip:
                    role_desc = _ROLE_DESCRIPTION_NORMAL
                    win_cond = _WIN_CONDITIONS_DRAWS if draws_enabled else _WIN_CONDITIONS_NO_DRAWS
                else:
                    role_desc = _ROLE_DESCRIPTION_ALL_CARROT
                    win_cond = _WIN_CONDITIONS_ALL_CARROT

                system_msg = ChatMessageSystem(
                    content=SYSTEM_PROMPT.format(
                        name=player.name,
                        winning_conditions=win_cond,
                        role_description=role_desc,
                        discussion_description=discussion_desc,
                    )
                )

                reasoning_prompt = ChatMessageUser(
                    content=(
                        f"Here is the current game state from your perspective:\n\n"
                        f"{game_state_text}\n\n"
                        f"{context}\n\n"
                        f"GROUP DISCUSSION - Round {round_num} of {num_rounds_discussion}\n\n"
                        f"{speaking_order_text}\n\n"
                        f"{announcements_text}\n\n"
                        f"Before making an announcement, first reason about your situation:\n"
                        f"- What is your role and what outcome do you want?\n"
                        f"- What conclusions can you draw from the recorded behaviour of other players? What do they know (about you and others), what can they only guess at, what are they trying to achieve? Think carefully about this, considering each player in turn (don't consider players who haven't had the opportunity to act yet). Use all available evidence and be careful about jumping to conclusions early.\n"
                        f"- Consider what the other players would have been expected to say if they were Parsnips, and if they were Carrots. Avoid, for example, reasoning that will increase your confidence that a player is a Parsnip regardless of how they respond."
                        f"- What information might be worth sharing (or concealing)?\n"
                        f"- What would be strategically beneficial to announce?\n"
                        f"- Consider the beliefs voiced by other players, either about specific player identities or how each role would behave. Consider whether these beliefs are true, and whether it is in your interests to publicly agree with, disagree with or ignore these beliefs.\n"
                        f"- Lying by omission, or lying directly, is permitted, for you and others.\n"
                        f"- How might your announcement influence other players, and how are they likely to respond?\n\n"
                        f"Think through your options without using any tools yet. "
                        f"Concise announcements are preferable. Remember that players have no control over the order in which they are selected to speak. Therefore DO NOT speculate that a player is behaving suspiciously if the game has not permitted them to speak yet.\n\n"
                        f"Use this space to reason about your situation and strategy. Separately, you may also use it to draft your response, under the heading 'DRAFT'."
                    ),
                )

                messages = [system_msg, reasoning_prompt]

                player_model = model_for_player(player_index)
                state.output = await player_model.generate(
                    input=messages, tools=[], tool_choice="none"
                )
                messages.append(state.output.message)

                announcement_prompt = ChatMessageUser(
                    content=(
                        "Based on your reasoning above, you may now make a public "
                        "announcement. Use the MakeAnnouncementTool to speak, or "
                        "respond without using the tool to stay silent."
                    ),
                )
                messages.append(announcement_prompt)

                for _attempt in range(3):
                    state.output = await player_model.generate(
                        input=messages, tools=tools, tool_choice="auto"
                    )
                    messages.append(state.output.message)

                    if not state.output.message.tool_calls:
                        # Player chose to stay silent — no retry needed.
                        break

                    try:
                        tool_messages, state.output = await execute_tools(
                            messages, tools=tools
                        )
                    except Exception as exc:
                        messages.append(
                            ChatMessageUser(
                                content=(
                                    f"Your tool call was invalid and caused an error: {exc}\n"
                                    f"Please try again. Use MakeAnnouncementTool with a "
                                    f"valid message string to make an announcement, or "
                                    f"respond without a tool call to stay silent."
                                )
                            )
                        )
                        continue

                    messages.extend(tool_messages)

                    # Check if the tool result indicates success.
                    last_result = tool_messages[-1].content if tool_messages else ""
                    if (
                        isinstance(last_result, str)
                        and last_result.startswith("Your announcement has been made public")
                    ):
                        break

                    # The tool call went through but returned an error
                    # (e.g. wrong tool used). Give the agent another chance.
                    messages.append(
                        ChatMessageUser(
                            content=(
                                f"That action failed: {last_result}\n"
                                f"Please use MakeAnnouncementTool with a valid message "
                                f"string, or respond without a tool call to stay silent."
                            )
                        )
                    )

        return state

    # ------------------------------------------------------------------
    # run_player_turn: one player reasons then casts their vote
    # ------------------------------------------------------------------

    async def run_player_turn(
        player_index: int, state: AgentState
    ) -> AgentState:
        player = game.players[player_index]
        tools = vote_tools[player_index]

        game_state_text = game.get_state_summary(player_index)

        announcements_text = ""
        if discussion:
            announcements_text = get_recent_announcements()
            if announcements_text:
                announcements_text = f"\n\n{announcements_text}\n"

        if uncertainty:
            role_desc = _ROLE_DESCRIPTION_UNCERTAIN
            win_cond = _WIN_CONDITIONS_UNCERTAIN_DRAWS if draws_enabled else _WIN_CONDITIONS_UNCERTAIN_NO_DRAWS
        elif include_parsnip:
            role_desc = _ROLE_DESCRIPTION_NORMAL
            win_cond = _WIN_CONDITIONS_DRAWS if draws_enabled else _WIN_CONDITIONS_NO_DRAWS
        else:
            role_desc = _ROLE_DESCRIPTION_ALL_CARROT
            win_cond = _WIN_CONDITIONS_ALL_CARROT

        system_msg = ChatMessageSystem(
            content=SYSTEM_PROMPT.format(
                name=player.name,
                winning_conditions=win_cond,
                role_description=role_desc,
                discussion_description=discussion_desc,
            )
        )

        if uncertainty:
            strategy_hint = (
                f"- Consider both possibilities: if there is a Parsnip, who might it be? "
                f"If this is an all-Carrot game, how can you avoid eliminating anyone? "
                f"Remember that this is the last stage of the game.\n\n"
            )
        elif include_parsnip:
            who = "Who might it be" if NUM_PARSNIP == 1 else "Who might they be"
            strategy_hint = (
                f"- You know this game contains {parsnip_word}. {who} based on the discussion? "
                f"Remember that this is the last stage of the game.\n\n"
            )
        else:
            strategy_hint = (
                f"- You know this is an all-Carrot game with no Parsnip. "
                f"Your goal is to ensure nobody is eliminated. "
                f"Remember that this is the last stage of the game.\n\n"
            )

        reasoning_prompt = ChatMessageUser(
            content=(
                f"Here is the current game state from your perspective:\n\n"
                f"{game_state_text}"
                f"{announcements_text}\n\n"
                f"Before casting your elimination vote, reason about your strategy:\n"
                f"- What is your role and what outcome do you want?\n"
                f"- What have you learned from the discussion?\n"
                f"{strategy_hint}"
                f"Think through your options without using any tools yet."
            ),
        )

        messages = [system_msg, reasoning_prompt]

        player_model = model_for_player(player_index)
        state.output = await player_model.generate(
            input=messages, tools=[], tool_choice="none"
        )
        messages.append(state.output.message)

        vote_prompt = ChatMessageUser(
            content=(
                "Based on your reasoning above, now use the "
                "CastEliminationVoteTool to cast your vote."
            ),
        )
        messages.append(vote_prompt)

        for _attempt in range(3):
            state.output = await player_model.generate(
                input=messages, tools=tools, tool_choice="auto"
            )
            messages.append(state.output.message)

            if state.output.message.tool_calls:
                try:
                    tool_messages, state.output = await execute_tools(
                        messages, tools=tools
                    )
                except Exception as exc:
                    messages.append(
                        ChatMessageUser(
                            content=(
                                f"Your tool call was invalid and caused an error: {exc}\n"
                                f"Please use CastEliminationVoteTool with a valid "
                                f"target_index (0-{n - 1}) to cast your vote."
                            )
                        )
                    )
                    continue

                messages.extend(tool_messages)

                last_result = tool_messages[-1].content if tool_messages else ""
                if (
                    isinstance(last_result, str)
                    and last_result.startswith("Elimination vote cast")
                ):
                    break

                messages.append(
                    ChatMessageUser(
                        content=(
                            f"That action failed: {last_result}\n"
                            f"Please use CastEliminationVoteTool with a valid "
                            f"target_index (0-{n - 1}) to cast your vote."
                        )
                    )
                )
            else:
                messages.append(
                    ChatMessageUser(
                        content=(
                            "You must use CastEliminationVoteTool to cast your "
                            "vote. Please do so now."
                        )
                    )
                )

        return state

    # ------------------------------------------------------------------
    # execute: main game loop
    # ------------------------------------------------------------------

    async def execute(state: AgentState) -> AgentState:
        # Discussion phase
        if discussion:
            vote_intro = (
                f"An elimination vote is about to take place. All players will "
                f"vote for one player to eliminate. Any player receiving {EJECTION_THRESHOLD} or "
                f"more votes will be eliminated."
            )
            if uncertainty:
                context = (
                    f"{vote_intro} Remember: you do not know "
                    f"whether this game includes Parsnips or is all-Carrots."
                )
            elif include_parsnip:
                context = f"{vote_intro} This game contains {parsnip_word}."
            else:
                context = (
                    f"{vote_intro} This is an all-Carrot game "
                    f"with no Parsnip — your goal is to ensure nobody is eliminated."
                )
            state = await run_group_discussion(
                state, context, "Before elimination vote"
            )

        # Voting phase
        action = game.get_action_required()
        if action["action_type"] == "elimination_vote":
            for voter_index in action["player_index"]:
                state = await run_player_turn(voter_index, state)

        # Record results
        summary = game.get_global_summary()
        state.messages = [ChatMessageAssistant(content=summary)]

        if result_out is not None:
            anyone_eliminated = any(not p.alive for p in game.players)

            # Per-player vote record. "correct" is only defined for Carrots:
            # True if they voted for a Parsnip. Missing votes have target None.
            votes = []
            for p in game.players:
                target_idx = game.votes.get(p.index)
                target = game.players[target_idx] if target_idx is not None else None
                correct = None
                if p.role == Role.CARROT and include_parsnip and target is not None:
                    correct = target.role == Role.PARSNIP
                votes.append({
                    "name": p.name,
                    "role": p.role.value,
                    "model": model_name_for_player(p.index) or EVAL_MODEL,
                    "target": target.name if target else None,
                    "target_role": target.role.value if target else None,
                    "correct": correct,
                })
            result_out["votes"] = votes

            if game.phase != GamePhase.GAME_OVER:
                # At least one player never cast a valid vote, so the game
                # never resolved. Record it as failed rather than scoring it.
                non_voters = [p for p in game.players if p.index not in game.votes]
                result_out["parsnip_player"] = ", ".join(
                    p.name for p in game.players if p.role == Role.PARSNIP
                ) or "None"
                result_out["parsnip_eliminated"] = False
                result_out["winner"] = "Failed"
                result_out["outcome"] = "failed"
                result_out["failure_reason"] = (
                    "no vote cast by "
                    + ", ".join(f"{p.name} ({p.role.value})" for p in non_voters)
                )
                result_out["failed_players"] = [
                    {
                        "name": p.name,
                        "role": p.role.value,
                        "model": model_name_for_player(p.index) or EVAL_MODEL,
                    }
                    for p in non_voters
                ]
            elif include_parsnip:
                parsnips = [p for p in game.players if p.role == Role.PARSNIP]
                parsnip_eliminated = any(not p.alive for p in parsnips)
                result_out["parsnip_player"] = ", ".join(p.name for p in parsnips)
                result_out["parsnip_eliminated"] = parsnip_eliminated
                if parsnip_eliminated:
                    result_out["winner"] = "Carrot"
                    result_out["outcome"] = "carrot_win"
                elif anyone_eliminated:
                    result_out["winner"] = "Parsnip"
                    result_out["outcome"] = "parsnip_win_other_eliminated"
                elif draws_enabled:
                    result_out["winner"] = "Draw"
                    result_out["outcome"] = "draw"
                else:
                    result_out["winner"] = "Parsnip"
                    result_out["outcome"] = "parsnip_win_nobody_eliminated"
            else:
                result_out["parsnip_player"] = "None"
                result_out["parsnip_eliminated"] = False
                if anyone_eliminated:
                    eliminated_indices = [p.index for p in game.players if not p.alive]
                    if _parsnip_model_player is not None and _parsnip_model_player in eliminated_indices:
                        result_out["eliminated_model"] = "minority"
                        result_out["winner"] = "Loss"
                        result_out["outcome"] = "carrot_loss_minority"
                    else:
                        result_out["eliminated_model"] = "majority"
                        result_out["winner"] = "Loss"
                        result_out["outcome"] = "carrot_loss_majority"
                else:
                    result_out["eliminated_model"] = None
                    result_out["winner"] = "Carrot"
                    result_out["outcome"] = "carrot_win"

        return state

    return execute


# ── Credit Check ─────────────────────────────────────────────────────────


def get_openrouter_credits() -> tuple[float, float | None] | None:
    """Return ``(usage, limit)`` from OpenRouter. *limit* is ``None`` for
    accounts without a spending cap. Returns ``None`` on failure."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[Credits] OPENROUTER_API_KEY environment variable is not set.")
        return None
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if resp.status_code != 200:
            print(f"[Credits] OpenRouter API returned status {resp.status_code}: {resp.text[:200]}")
            return None
        body = resp.json()
        data = body.get("data", {})
        usage = data.get("usage")
        if usage is None:
            print(f"[Credits] Missing 'usage' field in API response. Full data: {data}")
            return None
        limit = data.get("limit")  # None when no spending cap is set
        return usage, limit
    except requests.RequestException as e:
        print(f"[Credits] Network error calling OpenRouter API: {e}")
        return None
    except Exception as e:
        print(f"[Credits] Unexpected error reading credits: {e}")
        return None


# ── Batch Runner ─────────────────────────────────────────────────────────


# Maximum number of games in flight at once. Inspect keeps this many games
# running and starts the next one as soon as a slot frees up.
MAX_CONCURRENT_GAMES = 100


def _make_game_task(name: str, seed: int, result_out: dict, **orchestrator_kwargs) -> Task:
    """Build a single-sample Inspect task that plays one game."""
    return Task(
        name=name,
        dataset=[Sample(input="", target="")],
        solver=as_solver(
            CarrotParsnipOrchestrator(seed=seed, result_out=result_out, **orchestrator_kwargs)
        ),
        message_limit=200,
    )


def _run_tasks(tasks: list[Task], max_concurrent: int) -> tuple[list, dict]:
    """
    Run *tasks* with up to *max_concurrent* games in flight at once.

    Returns ``(eval_logs, cost_info)`` where *cost_info* holds the change
    in OpenRouter usage over the run (``cost``, ``usage_before``,
    ``usage_after`` and ``limit``; values are None when unavailable).
    """
    log_dir = str(Path(__file__).resolve().parent / "logs")
    max_concurrent = max(1, min(max_concurrent, len(tasks)))

    _credits_before = get_openrouter_credits()
    usage_before = _credits_before[0] if _credits_before else None

    eval_logs = eval(
        tasks, log_dir=log_dir, max_tasks=max_concurrent
    )

    _credits_after = get_openrouter_credits()
    usage_after = _credits_after[0] if _credits_after else None
    credit_limit = _credits_after[1] if _credits_after else None

    cost = None
    if usage_before is not None and usage_after is not None:
        cost = usage_after - usage_before
    return eval_logs, {
        "cost": cost,
        "usage_before": usage_before,
        "usage_after": usage_after,
        "limit": credit_limit,
    }


def _collect_results(
    result_holders: list[dict], task_names: list[str], eval_logs: list
) -> list[dict]:
    """
    Fill in missing fields on each game's result dict and return them.

    Games whose orchestrator never recorded an outcome are marked
    ``failed`` (with the eval error, if any), or ``not_run`` if the task
    was cancelled or never started (e.g. Ctrl+C mid-tournament).
    """
    logs_by_task = {log.eval.task: log for log in eval_logs}

    results = []
    for i, (rh, name) in enumerate(zip(result_holders, task_names)):
        rh["game_id"] = i
        if "outcome" not in rh:
            log = logs_by_task.get(name)
            status = getattr(log, "status", None)
            if status in (None, "started", "cancelled"):
                rh["winner"] = "Not run"
                rh["outcome"] = "not_run"
                rh["failure_reason"] = "game was cancelled or never started"
            else:
                # The orchestrator never reached the result-recording step,
                # e.g. the sample errored on an API or provider failure.
                error = getattr(log, "error", None)
                reason = getattr(error, "message", None) or "game did not complete"
                rh["winner"] = "Failed"
                rh["outcome"] = "failed"
                rh["failure_reason"] = f"error: {reason.strip().splitlines()[0][:200]}"
            rh.setdefault("failed_players", [])
        rh.setdefault("parsnip_player", "Unknown")
        rh.setdefault("parsnip_eliminated", False)
        results.append(rh)
    return results


VOTES_CSV_FIELDS = [
    "batch_num", "game_id", "seed",
    "carrot_models", "parsnip_model", "outcome", "winner",
    "parsnip_player", "parsnip_eliminated",
    "player", "role", "model", "teammate_models",
    "target", "target_role", "correct",
]


def _write_votes_csv(
    path: Path, batches: list[dict], base_seed: int, num_games: int
) -> None:
    """
    Write every vote from every game in *batches* to a CSV at *path*.

    One row per player per game. ``carrot_models`` is the full Carrot
    team ("a+b"). ``teammate_models`` lists the models of the voter's
    fellow Carrots, taken from the game's own vote records (blank for
    the Parsnip). ``correct`` is 1/0 for Carrots and blank for the
    Parsnip or a player who never voted. A game with no vote record
    (e.g. an API error before voting) gets one row with the player
    columns blank so the game is still listed.
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VOTES_CSV_FIELDS)
        writer.writeheader()
        for batch in batches:
            team_str = "+".join(batch["carrot_models"])
            for r in batch["games"]:
                game_id = r["game_id"]
                base = {
                    "batch_num": batch["batch_num"],
                    "game_id": game_id,
                    "seed": base_seed + (batch["batch_num"] - 1) * num_games + game_id,
                    "carrot_models": team_str,
                    "parsnip_model": batch["parsnip_model"],
                    "outcome": r.get("outcome", ""),
                    "winner": r.get("winner", ""),
                    "parsnip_player": r.get("parsnip_player", ""),
                    "parsnip_eliminated": int(bool(r.get("parsnip_eliminated", False))),
                }
                votes = r.get("votes") or []
                if not votes:
                    writer.writerow(base)
                    continue
                carrot_votes = [v for v in votes if v["role"] == Role.CARROT.value]
                for v in votes:
                    teammates = ""
                    if v["role"] == Role.CARROT.value:
                        teammates = "+".join(
                            o["model"] for o in carrot_votes if o is not v
                        )
                    correct = v["correct"]
                    writer.writerow({
                        **base,
                        "player": v["name"],
                        "role": v["role"],
                        "model": v["model"],
                        "teammate_models": teammates,
                        "target": v["target"] or "",
                        "target_role": v["target_role"] or "",
                        "correct": "" if correct is None else int(correct),
                    })


def _vote_lines(result: dict) -> list[str]:
    """Format one game's per-player votes for the per-game breakdown."""
    lines = []
    for v in result.get("votes", []):
        if v["target"] is None:
            vote = "did not vote"
        else:
            vote = f"voted {v['target']} ({v['target_role']})"
        mark = ""
        if v["correct"] is True:
            mark = "  ✓"
        elif v["correct"] is False:
            mark = "  ✗"
        lines.append(
            f"      {v['name']} [{v['role']}, {_short_model_name(v['model'])}]: {vote}{mark}"
        )
    return lines


def _summarise_batch(
    results: list[dict],
    title: str,
    include_parsnip: bool,
    draws_enabled: bool,
    cost_info: dict,
) -> dict:
    """Aggregate a list of game results, print a report and return the summary."""
    num_games = len(results)
    outcome_counts = Counter(r["outcome"] for r in results)
    failed = outcome_counts.get("failed", 0)
    failed_games = [r for r in results if r["outcome"] == "failed"]
    # Win rates are computed over completed games only.
    total = len(results) - failed
    carrot = outcome_counts.get("carrot_win", 0)

    batch_cost = cost_info.get("cost")
    usage_before = cost_info.get("usage_before")
    usage_after = cost_info.get("usage_after")
    credit_limit = cost_info.get("limit")

    if include_parsnip:
        parsnip_other = outcome_counts.get("parsnip_win_other_eliminated", 0)
        parsnip_nobody = outcome_counts.get("parsnip_win_nobody_eliminated", 0)
        draws = outcome_counts.get("draw", 0)
        parsnip = parsnip_other + parsnip_nobody

        summary = {
            "total_games": len(results),
            "completed_games": total,
            "failed_games": failed,
            "failures": failed_games,
            "carrot_wins": carrot,
            "parsnip_wins": parsnip,
            "parsnip_wins_other_eliminated": parsnip_other,
            "parsnip_wins_nobody_eliminated": parsnip_nobody,
            "draws": draws,
            "draws_enabled": draws_enabled,
            "include_parsnip": True,
            "carrot_win_rate": carrot / total if total else 0.0,
            "parsnip_win_rate": parsnip / total if total else 0.0,
            "cost": batch_cost,
            "games": results,
        }

        print(f"\n{'=' * 60}")
        print(f"  {title} — {num_games} games (draws {'ON' if draws_enabled else 'OFF'})")
        print(f"{'=' * 60}")
        if total:
            print(f"  Carrot wins                       : {carrot}/{total}  ({carrot / total:.1%})")
            print(f"  Parsnip wins (other eliminated)    : {parsnip_other}/{total}  ({parsnip_other / total:.1%})")
            if draws_enabled:
                print(f"  Draws (nobody eliminated)          : {draws}/{total}  ({draws / total:.1%})")
            else:
                print(f"  Parsnip wins (nobody eliminated)   : {parsnip_nobody}/{total}  ({parsnip_nobody / total:.1%})")
        print(f"{'─' * 60}")
        print(f"  Per-game breakdown:")
        outcome_labels = {
            "carrot_win": "Carrot win",
            "parsnip_win_other_eliminated": "Parsnip win (other eliminated)",
            "parsnip_win_nobody_eliminated": "Parsnip win (nobody eliminated)",
            "draw": "Draw (nobody eliminated)",
            "failed": "FAILED (excluded from stats)",
        }
        for r in results:
            label = outcome_labels.get(r["outcome"], r["outcome"])
            print(
                f"    Game {r['game_id']}: {label} "
                f"(Parsnip was {r['parsnip_player']})"
            )
            for line in _vote_lines(r):
                print(line)
    else:
        loss_minority = outcome_counts.get("carrot_loss_minority", 0)
        loss_majority = outcome_counts.get("carrot_loss_majority", 0)
        carrot_loss = loss_minority + loss_majority

        summary = {
            "total_games": len(results),
            "completed_games": total,
            "failed_games": failed,
            "failures": failed_games,
            "carrot_wins": carrot,
            "carrot_losses": carrot_loss,
            "carrot_losses_minority": loss_minority,
            "carrot_losses_majority": loss_majority,
            "include_parsnip": False,
            "carrot_win_rate": carrot / total if total else 0.0,
            "carrot_loss_rate": carrot_loss / total if total else 0.0,
            "cost": batch_cost,
            "games": results,
        }

        print(f"\n{'=' * 60}")
        print(f"  {title} (All Carrots) — {num_games} games")
        print(f"{'=' * 60}")
        if total:
            print(f"  Carrot wins  (nobody eliminated)   : {carrot}/{total}  ({carrot / total:.1%})")
            print(f"  Carrot losses (player eliminated)  : {carrot_loss}/{total}  ({carrot_loss / total:.1%})")
            if carrot_loss:
                print(f"    ├ minority model eliminated      : {loss_minority}/{carrot_loss}")
                print(f"    └ majority model eliminated      : {loss_majority}/{carrot_loss}")
        print(f"{'─' * 60}")
        print(f"  Per-game breakdown:")
        outcome_labels = {
            "carrot_win": "Carrot win",
            "carrot_loss_minority": "Carrot loss (minority model eliminated)",
            "carrot_loss_majority": "Carrot loss (majority model eliminated)",
            "failed": "FAILED (excluded from stats)",
        }
        for r in results:
            label = outcome_labels.get(r["outcome"], r["outcome"])
            print(f"    Game {r['game_id']}: {label}")
            for line in _vote_lines(r):
                print(line)

    if failed_games:
        print(f"{'─' * 60}")
        print(f"  Failed games: {failed}/{len(results)}")
        for r in failed_games:
            print(f"    Game {r['game_id']}: {r['failure_reason']}")

    print(f"{'─' * 60}")
    if batch_cost is None:
        print(f"  Cost: unavailable (could not read OpenRouter credits)")
    elif usage_before is not None and usage_after is not None:
        cost_line = f"  Cost: ${batch_cost:.4f}  (usage ${usage_before:.4f} -> ${usage_after:.4f})"
        if credit_limit is not None:
            remaining = credit_limit - usage_after
            cost_line += f"  | limit ${credit_limit:.4f}, remaining ${remaining:.4f}"
        print(cost_line)
    else:
        # Batches run as part of a larger eval only get a pro-rata share.
        print(f"  Cost: ${batch_cost:.4f}  (share of total run cost)")
    print(f"{'=' * 60}\n")

    return summary


def run_games(
    num_games: int,
    base_seed: int = 0,
    discussion: bool = True,
    num_rounds_discussion: int = 1,
    draws_enabled: bool = False,
    carrot_model: Optional[str] = None,
    parsnip_model: Optional[str] = None,
    include_parsnip: bool = True,
    uncertainty: bool = False,
    carrot_models: Optional[list[str]] = None,
    max_concurrent: int = MAX_CONCURRENT_GAMES,
) -> dict:
    """
    Run *num_games* Carrot-Parsnip games and report statistics.

    Args:
        num_games: Number of games to run.
        base_seed: Base random seed. Game *i* uses ``base_seed + i``.
        discussion: If True, players discuss before voting (default True).
        num_rounds_discussion: Number of discussion rounds before the
            elimination vote (default 1). Ignored if discussion is False.
        draws_enabled: If True, games where nobody is eliminated are
            counted as draws instead of Parsnip wins (default False).
        carrot_model: Optional model name for Carrot players. If None,
            uses the default eval model.
        parsnip_model: Optional model name for the Parsnip player. If
            None, uses the default eval model.
        include_parsnip: If True (default), games include NUM_CARROT
            Carrots and NUM_PARSNIP Parsnips. If False, games consist of
            NUM_PLAYERS Carrots where
            Carrots win if nobody is eliminated and lose if any player
            is eliminated.
        uncertainty: If True, agents are not told whether the game
            contains a Parsnip — they must deduce it (current default
            behaviour). If False (default), agents are told for certain
            whether the game contains a Parsnip or not.
        carrot_models: Optional list of model names, one per Carrot
            player, for mixed-model Carrot teams. Overrides
            *carrot_model* if given.
        max_concurrent: Maximum number of games running at once
            (default MAX_CONCURRENT_GAMES).

    Returns:
        Dictionary with aggregated statistics and per-game results.
    """
    result_holders: list[dict] = [{} for _ in range(num_games)]
    task_names = [f"carrot_parsnip_game_{i}" for i in range(num_games)]

    tasks = [
        _make_game_task(
            task_names[i],
            seed=base_seed + i,
            result_out=result_holders[i],
            discussion=discussion,
            num_rounds_discussion=num_rounds_discussion,
            draws_enabled=draws_enabled,
            carrot_model=carrot_model,
            parsnip_model=parsnip_model,
            include_parsnip=include_parsnip,
            uncertainty=uncertainty,
            carrot_models=carrot_models,
        )
        for i in range(num_games)
    ]

    eval_logs, cost_info = _run_tasks(tasks, max_concurrent)

    # Inspect catches Ctrl+C itself and returns "cancelled" logs instead of
    # raising, so re-raise here to let callers stop.
    if any(log.status == "cancelled" for log in eval_logs):
        raise KeyboardInterrupt

    results = _collect_results(result_holders, task_names, eval_logs)
    summary = _summarise_batch(
        results,
        title="Carrot-Parsnip",
        include_parsnip=include_parsnip,
        draws_enabled=draws_enabled,
        cost_info=cost_info,
    )
    summary["eval_logs"] = eval_logs
    return summary


# ── Tournament Runner ────────────────────────────────────────────────────


def _short_model_name(model: str) -> str:
    """Return the last segment of a model string for compact display."""
    return model.rsplit("/", 1)[-1]


def _ask_continue(prompt: str) -> bool:
    """
    Ask a yes/no question on the console and return True for yes.

    Repeats until the answer starts with "y" or "n".  If stdin is not
    interactive (EOF), or the user presses Ctrl+C at the prompt, the
    answer is taken as "n".
    """
    while True:
        try:
            answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if answer[:1] == "y":
            return True
        if answer[:1] == "n":
            return False
        print("  Please answer y or n.")


def run_tournament(
    model_list: list[str],
    num_games: int,
    base_seed: int = 0,
    discussion: bool = True,
    num_rounds_discussion: int = 1,
    draws_enabled: bool = False,
    mixed_teams: bool = False,
    max_concurrent: int = MAX_CONCURRENT_GAMES,
) -> dict:
    """
    Run a round-robin tournament between all models in *model_list*.

    Every ordered pair (A, B) plays *num_games* games with A as Carrot
    and B as Parsnip.  Mirror matches (A vs A) play *num_games* games.
    This means each pair of *distinct* models plays 2 * num_games games
    total (num_games in each role configuration).

    If *mixed_teams* is True, the Carrot team may instead be any
    combination of NUM_CARROT models (with repetition, order ignored),
    and every such team plays *num_games* games against every Parsnip
    model.

    Games are played in *num_games* rounds; each round plays every matchup
    once and is submitted to Inspect as one eval, so up to *max_concurrent*
    games run at once regardless of which matchup they belong to.  When
    *num_games* > 1 the user is asked after each round whether to
    continue; answering "n" ends the tournament and the results are
    summarised and saved as if that round had been the last.

    Args:
        model_list: List of model name strings.
        num_games: Number of games per matchup (= number of rounds).
        base_seed: Starting seed; each matchup offsets by *num_games*.
        discussion: Forwarded to the orchestrator.
        num_rounds_discussion: Forwarded to the orchestrator.
        draws_enabled: Forwarded to the orchestrator.
        mixed_teams: If True, Carrot teams may mix models (default False).
        max_concurrent: Maximum number of games running at once
            (default MAX_CONCURRENT_GAMES).

    Returns:
        Dictionary with per-model stats and all batch results.
    """
    n_models = len(model_list)

    # Per-model accumulators
    model_stats: dict[str, dict] = {
        model: {
            "wins_as_carrot": 0,
            "wins_as_parsnip": 0,
            "games_as_carrot": 0,
            "games_as_parsnip": 0,
            "failed_games": 0,    # failed games this model took part in
            "vote_failures": 0,   # times one of this model's players never voted
            "carrot_votes": 0,    # votes cast by this model's Carrots (completed games)
            "carrot_correct": 0,  # ...of which targeted the Parsnip
            "cost": 0.0,
        }
        for model in model_list
    }
    # (carrot_team, model) -> [correct, votes], for the mixed-team breakdown.
    team_votes: dict[tuple[tuple[str, ...], str], list[int]] = {}

    all_batch_results: list[dict] = []
    all_failures: list[dict] = []

    # Carrot teams are tuples of NUM_CARROT models. Without mixed teams,
    # every Carrot plays the same model.
    if mixed_teams:
        carrot_teams = list(combinations_with_replacement(model_list, NUM_CARROT))
    else:
        carrot_teams = [(m_,) * NUM_CARROT for m_ in model_list]

    def _team_name(team: tuple[str, ...]) -> str:
        if len(set(team)) == 1:
            return _short_model_name(team[0])
        return "+".join(_short_model_name(m_) for m_ in team)

    def _team_label(team: tuple[str, ...]) -> str:
        if not mixed_teams:
            return f"[{model_list.index(team[0]) + 1}] {_short_model_name(team[0])}"
        return "+".join(f"[{model_list.index(m_) + 1}]" for m_ in team)

    # Generate all matchups: every (Carrot team, Parsnip model) pair.
    # Without mixed teams, for distinct models A, B this yields (A,B) and
    # (B,A) → 2*num_games games, and mirrors (A,A) yield one batch.
    matchups = [
        (team, parsnip_m)
        for team in carrot_teams
        for parsnip_m in model_list
    ]

    total_batches = len(matchups)

    # Games are played in rounds. Round k plays every matchup once (its
    # k-th game), and all games in a round go to Inspect as one eval so up
    # to max_concurrent games run at once across matchup boundaries. When
    # num_games > 1 the user is asked after each round whether to carry
    # on; stopping early reports the rounds played as the final result.
    #
    # Game g of matchup b always uses seed base_seed + (b-1)*num_games + g,
    # so a tournament that runs to completion plays exactly the same games
    # (and the votes CSV recomputes the same seeds) as a single-eval run.
    batch_specs: list[tuple] = []  # (batch_num, carrot_team, parsnip_m, holders, names)
    for batch_num, (carrot_team, parsnip_m) in enumerate(matchups, start=1):
        holders: list[dict] = [{} for _ in range(num_games)]
        names = [f"tournament_b{batch_num}_g{i}" for i in range(num_games)]
        batch_specs.append((batch_num, carrot_team, parsnip_m, holders, names))
    total_games = total_batches * num_games

    print(f"\n{'#' * 70}")
    print(
        f"  TOURNAMENT — {n_models} models, {total_batches} matchups x {num_games} games"
        + (" (mixed Carrot teams)" if mixed_teams else "")
    )
    print(f"  Models: {', '.join(_short_model_name(m_) for m_ in model_list)}")
    print(
        f"  {total_games} games total in {_count(num_games, 'round', 'rounds')} "
        f"of {total_batches} games, "
        f"up to {min(max_concurrent, total_batches)} at once"
    )
    if num_games > 1:
        print("  You will be asked after each round whether to continue.")
    print(f"{'#' * 70}\n")

    eval_logs: list = []
    total_cost: Optional[float] = None
    interrupted = False
    rounds_played = 0
    for round_idx in range(num_games):
        round_tasks = [
            _make_game_task(
                names[round_idx],
                seed=base_seed + (batch_num - 1) * num_games + round_idx,
                result_out=holders[round_idx],
                discussion=discussion,
                num_rounds_discussion=num_rounds_discussion,
                draws_enabled=draws_enabled,
                parsnip_model=parsnip_m,
                carrot_models=list(carrot_team),
            )
            for batch_num, carrot_team, parsnip_m, holders, names in batch_specs
        ]
        if num_games > 1:
            print(f"\n  === Round {round_idx + 1}/{num_games} ({len(round_tasks)} games) ===\n")

        round_logs, cost_info = _run_tasks(round_tasks, max_concurrent)
        eval_logs.extend(round_logs)
        rounds_played += 1
        if cost_info["cost"] is not None:
            total_cost = (total_cost or 0.0) + cost_info["cost"]

        if any(log.status == "cancelled" for log in round_logs):
            interrupted = True
            break
        if round_idx + 1 >= num_games:
            break

        cost_note = (
            f", ${cost_info['cost']:.4f} this round" if cost_info["cost"] is not None else ""
        )
        print(
            f"\n  Round {round_idx + 1}/{num_games} completed "
            f"({len(round_tasks)} games{cost_note})."
        )
        if not _ask_continue(f"  Continue to round {round_idx + 2}/{num_games}? (y/n): "):
            print(f"  Stopping after round {round_idx + 1}/{num_games}.")
            break

    # Collect results per matchup. Games that never ran (rounds not played,
    # or cancelled after Ctrl+C) are dropped, and matchups with no played
    # games are skipped entirely.
    batch_results: list[tuple] = []
    for batch_num, carrot_team, parsnip_m, holders, names in batch_specs:
        results = [
            r for r in _collect_results(holders, names, eval_logs)
            if r["outcome"] != "not_run"
        ]
        if results:
            batch_results.append((batch_num, carrot_team, parsnip_m, results))
    games_played = sum(len(r) for *_, r in batch_results)

    if interrupted:
        print(
            f"\n  Tournament interrupted: {games_played}/{total_games} games played. "
            f"Showing results for the games that completed."
        )

    for batch_num, carrot_team, parsnip_m, results in batch_results:
        c_short = _team_name(carrot_team)
        p_short = _short_model_name(parsnip_m)

        # Only the whole run's cost is known, so each batch gets a
        # pro-rata share by number of games.
        batch_cost = None
        if total_cost is not None and games_played:
            batch_cost = total_cost * len(results) / games_played

        batch = _summarise_batch(
            results,
            title=(
                f"Batch {batch_num}/{total_batches}: "
                f"Carrot={c_short}  vs  Parsnip={p_short}"
            ),
            include_parsnip=True,
            draws_enabled=draws_enabled,
            cost_info={"cost": batch_cost},
        )

        # Accumulate stats
        carrot_wins = batch["carrot_wins"]
        parsnip_wins = batch["parsnip_wins"]
        total = batch["completed_games"]

        # A model on a mixed Carrot team is credited once per game.
        for m_ in set(carrot_team):
            model_stats[m_]["games_as_carrot"] += total
            model_stats[m_]["wins_as_carrot"] += carrot_wins
        model_stats[parsnip_m]["games_as_parsnip"] += total
        model_stats[parsnip_m]["wins_as_parsnip"] += parsnip_wins

        # Carrot vote accuracy, over completed games only.
        for r in batch["games"]:
            if r["outcome"] == "failed":
                continue
            for v in r.get("votes", []):
                if v["correct"] is None or v["model"] not in model_stats:
                    continue
                model_stats[v["model"]]["carrot_votes"] += 1
                model_stats[v["model"]]["carrot_correct"] += int(v["correct"])
                tv = team_votes.setdefault((carrot_team, v["model"]), [0, 0])
                tv[0] += int(v["correct"])
                tv[1] += 1

        for r in batch["failures"]:
            for m_ in {*carrot_team, parsnip_m}:
                model_stats[m_]["failed_games"] += 1
            for fp in r["failed_players"]:
                if fp["model"] in model_stats:
                    model_stats[fp["model"]]["vote_failures"] += 1
            all_failures.append({
                "batch_num": batch_num,
                "carrot_models": carrot_team,
                "parsnip_model": parsnip_m,
                **r,
            })

        if batch_cost is not None:
            # Split the batch cost evenly across player seats.
            for m_ in carrot_team:
                model_stats[m_]["cost"] += batch_cost / NUM_PLAYERS
            model_stats[parsnip_m]["cost"] += batch_cost * NUM_PARSNIP / NUM_PLAYERS

        batch["carrot_models"] = carrot_team
        batch["parsnip_model"] = parsnip_m
        batch["batch_num"] = batch_num
        all_batch_results.append(batch)


    # ── Tournament Summary ───────────────────────────────────────────
    # Lines are collected so the summary can be both printed and saved.
    out: list[str] = []
    emit = out.append

    emit(f"{'#' * 70}")
    emit(f"  TOURNAMENT RESULTS")
    emit(f"{'#' * 70}")
    emit(
        f"  {n_models} models, {len(all_batch_results)}/{total_batches} matchups, "
        f"{rounds_played}/{num_games} {'round' if num_games == 1 else 'rounds'} played "
        f"({games_played}/{total_games} games) | seed {base_seed} | discussion "
        f"{_count(num_rounds_discussion, 'round', 'rounds') if discussion else 'OFF'} | "
        f"draws {'ON' if draws_enabled else 'OFF'}"
        + (" | mixed Carrot teams" if mixed_teams else "")
    )

    # ── Head-to-head matrix ──────────────────────────────────────────
    # Row = Carrot team, column = Parsnip model. Cell = "C-P" (Carrot
    # wins - Parsnip wins), plus "-D" draws when draws are enabled.
    records = {
        (b["carrot_models"], b["parsnip_model"]): (
            b["carrot_wins"], b["parsnip_wins"], b["draws"]
        )
        for b in all_batch_results
    }

    def _cell(key: tuple[tuple[str, ...], str]) -> str:
        if key not in records:
            return "·"
        c, p, d = records[key]
        return f"{c}-{p}-{d}" if draws_enabled else f"{c}-{p}"

    name_width = max(len(_short_model_name(m_)) for m_ in model_list)
    label_width = max(len(_team_label(t)) for t in carrot_teams) + 1
    cell_width = max(
        [len(str(n_models))] + [len(_cell(k)) for k in records]
    ) + 1

    emit("")
    if mixed_teams:
        emit("  MODELS")
        for i, m_ in enumerate(model_list):
            emit(f"    [{i + 1}] {_short_model_name(m_)}")
        emit("")
    emit(
        f"  HEAD-TO-HEAD (row = Carrot, column = Parsnip; "
        f"cell = Carrot wins-Parsnip wins{'-draws' if draws_enabled else ''})"
    )
    emit(
        f"  {'':<{label_width}}"
        + "".join(f"{f'[{j + 1}]':>{cell_width + 2}}" for j in range(n_models))
    )
    for carrot_team in carrot_teams:
        label = _team_label(carrot_team)
        emit(
            f"  {label:<{label_width}}"
            + "".join(
                f"{_cell((carrot_team, parsnip_m)):>{cell_width + 2}}"
                for parsnip_m in model_list
            )
        )

    # Combined record for each distinct pair across both role assignments,
    # using single-model Carrot teams only.
    pair_lines = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            a, b = model_list[i], model_list[j]
            ab = records.get(((a,) * NUM_CARROT, b))
            ba = records.get(((b,) * NUM_CARROT, a))
            if ab is None and ba is None:
                continue
            a_wins = (ab[0] if ab else 0) + (ba[1] if ba else 0)
            b_wins = (ab[1] if ab else 0) + (ba[0] if ba else 0)
            draws_ = (ab[2] if ab else 0) + (ba[2] if ba else 0)
            line = (
                f"    {_short_model_name(a):>{name_width}} {a_wins:>3} - "
                f"{b_wins:<3} {_short_model_name(b):<{name_width}}"
            )
            if draws_enabled:
                line += f"  ({draws_} draws)"
            pair_lines.append(line.rstrip())
    if pair_lines:
        emit("")
        emit(
            "  PAIR RECORDS (both role assignments combined, mirrors excluded"
            + (", single-model Carrot teams only)" if mixed_teams else ")")
        )
        out.extend(pair_lines)

    # ── Leaderboard ──────────────────────────────────────────────────
    # Compute total wins and sort by descending total
    leaderboard = []
    for model, stats in model_stats.items():
        total_wins = stats["wins_as_carrot"] + stats["wins_as_parsnip"]
        total_games = stats["games_as_carrot"] + stats["games_as_parsnip"]
        leaderboard.append((model, stats, total_wins, total_games))
    leaderboard.sort(key=lambda x: x[2], reverse=True)

    emit("")
    emit("  OVERALL")
    header = (
        f"  {'Model':<{name_width}}  "
        f"{'Total':>7}  "
        f"{'As Carrot':>12}  "
        f"{'As Parsnip':>13}  "
        f"{'Spotted P':>11}  "
        f"{'Failed':>6}  "
        f"{'Cost':>10}"
    )
    emit(header)
    emit(f"  {'─' * (name_width + 71)}")

    for model, stats, total_wins, total_games in leaderboard:
        short = _short_model_name(model)
        cw = stats["wins_as_carrot"]
        cg = stats["games_as_carrot"]
        pw = stats["wins_as_parsnip"]
        pg = stats["games_as_parsnip"]
        cost = stats["cost"]
        cost_str = f"${cost:.4f}" if cost > 0 else "N/A"
        vc, vn = stats["carrot_correct"], stats["carrot_votes"]
        emit(
            f"  {short:<{name_width}}  "
            f"{total_wins:>3}/{total_games:<3}  "
            f"{cw:>3}/{cg:<3} wins  "
            f"{pw:>3}/{pg:<3} wins  "
            f"{vc:>4}/{vn:<4}  "
            f"{stats['failed_games']:>6}  "
            f"{cost_str:>10}"
        )

    total_played = sum(b["total_games"] for b in all_batch_results)
    emit(f"  {'─' * (name_width + 71)}")
    emit("  Spotted P = Carrot votes that targeted the Parsnip (completed games only)")

    # Per-team vote accuracy: which member of each mixed team found the
    # Parsnip. Rows are teams, one column per member model.
    if mixed_teams:
        emit("")
        emit("  CARROT VOTES BY TEAM (member: votes for Parsnip / votes cast)")
        for carrot_team in carrot_teams:
            members = []
            for m_ in dict.fromkeys(carrot_team):  # unique, in team order
                tv = team_votes.get((carrot_team, m_))
                if tv is None:
                    continue
                members.append(f"{_short_model_name(m_)}: {tv[0]}/{tv[1]}")
            if members:
                emit(f"    {_team_label(carrot_team):<{label_width}} " + "   ".join(members))
    emit(
        f"  Failed games (excluded from win/loss): "
        f"{len(all_failures)}/{total_played}"
    )
    if all_failures:
        for f_ in all_failures:
            emit(
                f"    Batch {f_['batch_num']} game {f_['game_id']} "
                f"(Carrot={_team_name(f_['carrot_models'])}, "
                f"Parsnip={_short_model_name(f_['parsnip_model'])}): "
                f"{f_['failure_reason']}"
            )
        blamed = [
            (m_, s["vote_failures"]) for m_, s in model_stats.items() if s["vote_failures"]
        ]
        if blamed:
            emit("  Missed votes by model: " + ", ".join(
                f"{_short_model_name(m_)}={c}" for m_, c in blamed
            ))

    emit(f"{'#' * 70}")

    report = "\n".join(out) + "\n"
    print("\n" + report)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    now = datetime.now()
    results_path = results_dir / f"tournament_{now:%Y%m%d_%H%M%S}.txt"
    header_lines = [
        f"Carrot-Parsnip tournament — {now:%Y-%m-%d %H:%M:%S}",
        "Models:",
        *(f"  [{i + 1}] {m_}" for i, m_ in enumerate(model_list)),
        "",
    ]
    results_path.write_text("\n".join(header_lines) + report, encoding="utf-8")
    print(f"Tournament results saved to {results_path}")

    # Full vote log: one row per player per game, including failed games
    # (target columns are blank for players who never voted).
    votes_path = results_dir / f"tournament_{now:%Y%m%d_%H%M%S}_votes.csv"
    _write_votes_csv(votes_path, all_batch_results, base_seed, num_games)
    print(f"Vote log saved to {votes_path}\n")

    return {
        "model_stats": model_stats,
        "failures": all_failures,
        "batches": all_batch_results,
        "results_path": str(results_path),
        "votes_csv_path": str(votes_path),
        "leaderboard": [
            {
                "model": model,
                "total_wins": tw,
                "total_games": tg,
                "wins_as_carrot": s["wins_as_carrot"],
                "games_as_carrot": s["games_as_carrot"],
                "wins_as_parsnip": s["wins_as_parsnip"],
                "games_as_parsnip": s["games_as_parsnip"],
                "failed_games": s["failed_games"],
                "vote_failures": s["vote_failures"],
                "carrot_votes": s["carrot_votes"],
                "carrot_correct": s["carrot_correct"],
                "cost": s["cost"],
            }
            for model, s, tw, tg in leaderboard
        ],
        "team_votes": {
            (team, m_): {"correct": c, "votes": n}
            for (team, m_), (c, n) in team_votes.items()
        },
    }


# ── Task & Evaluation ────────────────────────────────────────────────────


@task
def carrot_parsnip_task() -> Task:
    return Task(
        dataset=[Sample(input="", target="")],
        message_limit=200,
    )


if __name__ == "__main__":
    model_list = [
        deepseek_v4_flash,
        gpt_5_6_luna,
        glm_5_3_flashx,
        gemini_3_8_flash,
        claude_haiku_4_5,
        grok_4_7,
        mistral_medium_3_5,
        sonnet,d
        opus_5_5,
        gpt_6_sol,
    ]
    short_list = [deepseek_v4_flash, gpt_5_6_luna]
    CARROT_MODEL = sonnet
    PARSNIP_MODEL = sonnet

    multi = False
    tournament = True
    draws = False
    mixed_teams = True
    if multi:
        stats = run_games(num_games=5, base_seed=768495, discussion=True, num_rounds_discussion=3, draws_enabled=draws, carrot_model=CARROT_MODEL, parsnip_model=PARSNIP_MODEL, include_parsnip=True, uncertainty=False)
        if stats.get("include_parsnip", True):
            print(f"Results: {stats['carrot_wins']} Carrot / {stats['parsnip_wins']} Parsnip")
        else:
            print(f"Results: {stats['carrot_wins']} Carrot wins / {stats['carrot_losses']} Carrot losses")
    elif tournament:
        tournament_results = run_tournament(
            model_list=model_list,
            num_games=9,
            base_seed=1300,
            discussion=True,
            num_rounds_discussion=3,
            draws_enabled=draws,
            mixed_teams=mixed_teams
        )

