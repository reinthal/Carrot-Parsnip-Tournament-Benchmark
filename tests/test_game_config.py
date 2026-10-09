"""Tests for runtime game-configuration injection (game_config.py).

No API calls — covers YAML round-trips, strict validation, prompt rendering,
and the config -> engine bridge.
"""

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Carrot_Parsnip import CarrotParsnipGame, Role, Team
from game_config import (
    GameConfig,
    PromptConfig,
    RoleConfig,
    make_game,
    render_strategy_hint,
    render_system_prompt,
    render_vote_context,
)

CONFIGS = REPO / "configs"


# ── YAML round-trip ──────────────────────────────────────────────────────


def test_default_round_trip(tmp_path):
    cfg = GameConfig()
    loaded = GameConfig.from_yaml(cfg.to_yaml(tmp_path / "cfg.yaml"))
    assert loaded == cfg
    assert isinstance(loaded.carrot, RoleConfig)
    assert isinstance(loaded.parsnip, RoleConfig)
    assert isinstance(loaded.prompts, PromptConfig)
    assert loaded.carrot.role is Role.CARROT
    assert loaded.parsnip.role is Role.PARSNIP


def test_modified_round_trip(tmp_path):
    cfg = GameConfig.from_dict({
        "carrot": {"count": 4, "models": ["m1", "m2", "m3", "m4"]},
        "parsnip": {"count": 1, "model": "m5"},
        "ejection_threshold": 3,
        "uncertainty": True,
        "player_names": ["A", "B", "C", "D", "E"],
        "prompts": {"system_prompt": "custom {name}"},
    })
    assert GameConfig.from_yaml(cfg.to_yaml(tmp_path / "cfg.yaml")) == cfg


def test_repo_example_configs_load():
    default = GameConfig.from_yaml(CONFIGS / "default.yaml")
    assert default == GameConfig()

    five = GameConfig.from_yaml(CONFIGS / "five_player.yaml")
    assert (five.carrot.count, five.parsnip.count) == (4, 1)
    assert five.ejection_threshold == 3
    assert five.num_rounds_discussion == 3
    # everything not in the delta file keeps its default
    assert five.prompts == PromptConfig()
    assert five.eval_model == GameConfig().eval_model

    uncertain = GameConfig.from_yaml(CONFIGS / "uncertain_two_parsnips.yaml")
    assert uncertain.uncertainty
    assert uncertain.num_players == 7
    assert uncertain.parsnip.model == "openrouter/anthropic/claude-sonnet-5"


# ── Strict validation ────────────────────────────────────────────────────


def test_unknown_top_level_key():
    with pytest.raises(ValueError, match="num_rounds"):
        GameConfig.from_dict({"num_rounds": 3})


def test_unknown_role_key():
    with pytest.raises(ValueError, match="colour"):
        GameConfig.from_dict({"carrot": {"count": 2, "colour": "orange"}})


def test_unknown_prompt_key():
    with pytest.raises(ValueError, match="greeting"):
        GameConfig.from_dict({"prompts": {"greeting": "hi"}})


def test_bad_counts():
    with pytest.raises(ValueError, match="at least 1"):
        GameConfig.from_dict({"parsnip": {"count": 0}})


def test_bad_threshold():
    with pytest.raises(ValueError, match="ejection_threshold"):
        GameConfig.from_dict({"ejection_threshold": 9})


def test_player_names_length_mismatch():
    with pytest.raises(ValueError, match="player_names"):
        GameConfig.from_dict({"player_names": ["A", "B"]})


def test_wrong_role_enum():
    with pytest.raises(ValueError, match="carrot.role"):
        GameConfig.from_dict({"carrot": {"role": "Parsnip", "count": 2}})


# ── Prompt rendering ─────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are Alice, playing a game of Carrot-Parsnip with 3 players.

Carrot-Parsnip is a social deduction game. Players are secretly assigned roles:
- 2 players are Carrots
- 1 player is the Parsnip

The game consists of a single elimination vote. All players vote for one player
to eliminate. Any player receiving 2 or more votes is eliminated. Before the vote there will be 1 round of conversation where players are selected to speak in a random order each round. This means that a player who has not spoken in the early game may simply have not been selected to speak yet.

WINNING CONDITIONS:
- If the Parsnip is eliminated, the Carrots win.
- If no players are eliminated (due to a split vote), the Parsnip wins.
- If a Carrot player is eliminated, the Parsnip wins."""


def test_default_system_prompt_golden():
    assert render_system_prompt(GameConfig(), "Alice") == DEFAULT_SYSTEM_PROMPT


def test_shape_change_propagates_into_prompts():
    cfg = GameConfig.from_yaml(CONFIGS / "five_player.yaml")
    text = render_system_prompt(cfg, "Bob")
    assert "with 5 players" in text
    assert "- 4 players are Carrots" in text
    assert "- 1 player is the Parsnip" in text
    assert "Any player receiving 3 or more votes is eliminated." in text
    assert "3 rounds of conversation" in text


def test_multi_parsnip_wording():
    cfg = GameConfig.from_dict({"carrot": {"count": 5}, "parsnip": {"count": 2},
                                "ejection_threshold": 4})
    text = render_system_prompt(cfg, "Eve")
    assert "- 2 players are Parsnips" in text
    assert "Parsnips are not told who the other Parsnips are." in text
    assert "- If any Parsnip is eliminated, the Carrots win." in text
    hint = render_strategy_hint(cfg)
    assert "contains 2 Parsnips" in hint and "Who might they be" in hint


def test_variant_selection():
    all_carrot = GameConfig.from_dict({"include_parsnip": False})
    text = render_system_prompt(all_carrot, "Alice")
    assert "there is no Parsnip" in text
    assert "- Carrots win if no player is eliminated (no one receives 2+ votes)." in text

    uncertain = GameConfig.from_dict({"uncertainty": True, "draws_enabled": True})
    text = render_system_prompt(uncertain, "Alice")
    assert "two possible variants" in text
    assert "the game is a draw" in text
    assert render_vote_context(uncertain).endswith(
        "you do not know whether this game includes Parsnips or is all-Carrots."
    )


def test_injected_template_used_verbatim():
    cfg = GameConfig.from_dict({"prompts": {
        "system_prompt": "{name} plays with {num_players} players; threshold {ejection_threshold}."
    }})
    assert render_system_prompt(cfg, "Zoe") == "Zoe plays with 3 players; threshold 2."


# ── Engine bridge ────────────────────────────────────────────────────────


def test_make_game_deals_configured_roles():
    cfg = GameConfig.from_dict({"carrot": {"count": 5}, "parsnip": {"count": 2},
                                "ejection_threshold": 4})
    game = make_game(cfg, seed=7)
    game.start()
    roles = [p.role for p in game.players]
    assert roles.count(Role.CARROT) == 5
    assert roles.count(Role.PARSNIP) == 2
    assert game.ejection_threshold == 4
    # multi-parsnip private knowledge comes from the role config
    parsnip_idx = roles.index(Role.PARSNIP)
    assert game.private_knowledge[parsnip_idx] == [
        "You are a Parsnip player (one of 2 Parsnips)."
    ]
    # same seed -> same deal
    again = make_game(cfg, seed=7)
    assert [p.role for p in again.players] == roles


def test_make_game_all_carrot_control():
    cfg = GameConfig.from_dict({"include_parsnip": False})
    game = make_game(cfg, seed=0)
    assert all(p.role == Role.CARROT for p in game.players)


def test_seven_player_game_runs_to_completion():
    cfg = GameConfig.from_dict({"carrot": {"count": 5}, "parsnip": {"count": 2},
                                "ejection_threshold": 4})
    game = make_game(cfg, seed=3)
    game.start()
    rng = random.Random(3)
    action = game.get_action_required()
    for pi in action["player_index"]:
        game.take_action("elimination_vote", player_index=pi,
                         target_index=rng.choice(action["options"]))
    assert game.winner in (Team.CARROT, Team.PARSNIP)


def test_engine_legacy_path_unchanged():
    with pytest.raises(ValueError, match="exactly 3 players"):
        CarrotParsnipGame(["A", "B", "C", "D"])
    with pytest.raises(ValueError, match="must equal the number of players"):
        CarrotParsnipGame(["A", "B", "C", "D"], num_carrot=2, num_parsnip=1)
    with pytest.raises(ValueError, match="or neither"):
        CarrotParsnipGame(["A", "B", "C"], num_carrot=2)
