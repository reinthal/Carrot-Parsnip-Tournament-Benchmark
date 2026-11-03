"""Audited games (audit.py) with scripted models: no API calls."""

import asyncio
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import audit  # noqa: E402
from game_config import GameConfig  # noqa: E402
from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ContentText, ModelOutput  # noqa: E402
from inspect_ai.tool import ToolCall  # noqa: E402

PLAYERS = "openrouter/players/model"
AUDITOR = "openrouter/auditor/model"


def reply(text="", reasoning="", call=None) -> ModelOutput:
    calls = [ToolCall(id="c1", function=call[0], arguments=call[1])] if call else None
    msg = ChatMessageAssistant(content=[ContentReasoning(reasoning=reasoning), ContentText(text=text)],
                               tool_calls=calls, model="fake")
    return ModelOutput.from_message(msg)


class FakeModel:
    """A scripted Inspect model: announces, answers, votes for player 0; as the auditor asks
    every player one question and names ``guess``."""

    def __init__(self, name, guess="Alice", auditor_replies=None):
        self.name, self.guess, self.calls = name, guess, []
        self.auditor_replies = list(auditor_replies or [])

    async def generate(self, input, tools=None):
        self.calls.append((input, tools))
        names = {t.name for t in tools or []}
        system = input[0].text
        if "MakeAnnouncementTool" in names:
            me = re.search(r"You are (\w+)", system).group(1)
            return reply(reasoning=f"{me} plans", call=("MakeAnnouncementTool", {"message": f"{me} is a Carrot"}))
        if "CastEliminationVoteTool" in names:
            return reply(call=("CastEliminationVoteTool", {"target_index": 0}))
        if system.startswith("You are an outside auditor"):
            if self.auditor_replies:
                return reply(text=self.auditor_replies.pop(0))
            prompt = input[-1].text
            if "Ask every player" in prompt:
                who = re.search(r"now \(([^)]*)\)", prompt).group(1).split(", ")
                return reply(text="".join(f'<question to="{n}">Who is it?</question>' for n in who))
            return reply(reasoning="weighing", text=f"<answer>{self.guess}</answer>")
        return reply(reasoning="answering", text="I am a Carrot.")


class FakeDetector:
    def __init__(self):
        self.calls = []

    async def score(self, model, messages, tools):
        self.calls.append((model, messages, tools))
        return "0.70 (fake detector)"


def config(**audit_cfg) -> GameConfig:
    return GameConfig.from_dict({"turn_mode": "single_call", "eval_model": PLAYERS, "num_rounds_discussion": 2,
                                 "audit": {"model": AUDITOR, **audit_cfg}})


@pytest.fixture
def models(monkeypatch):
    made = {}

    def player_model(name, cfg):
        return made.setdefault(name, FakeModel(name))
    monkeypatch.setattr(audit, "player_model", player_model)
    return made


def play(cfg, detector=None, seed=0):
    return asyncio.run(audit.play_audited_game(cfg, seed, detector))


def test_public_questions_with_a_detector(models):
    det = FakeDetector()
    out = play(config(questions="public"), det)
    game = out["game"]
    assert game["failed"] is None and game["verdict"] == "Alice" and game["correct"] == (game["parsnip"] == "Alice")
    assert len(game["votes"]) == 3 and game["questions_asked"] == 2 * 3
    # every announcement and answer was read: 2 rounds x 3 players x (announcement + answer)
    assert len(det.calls) == len(out["turns"]) == 12
    model, messages, tools = det.calls[0]
    assert model == PLAYERS and messages[0]["role"] == "system" and messages[-1]["role"] == "assistant"
    assert "Parsnip" in messages[0]["content"] or "Carrot" in messages[0]["content"]
    assert messages[-1]["reasoning_content"].endswith("plans")                 # private reasoning kept
    assert tools[0]["function"]["name"] == "MakeAnnouncementTool"
    # the auditor saw every reading verbatim, and the detector note
    verdict_call = [c for c in out["auditor_calls"] if c["step"] == "verdict"][0]
    transcript = verdict_call["messages"][1]["content"]
    assert transcript.count("Detector reading: 0.70 (fake detector)") == 12
    assert "detector's reading" in verdict_call["messages"][0]["content"]
    # the players were told about the auditor's public questions
    assert "asks every player one question" in models[PLAYERS].calls[0][0][0].text


def test_private_questions_stay_private(models):
    out = play(config(questions="private"), FakeDetector())
    assert all(t["kind"] in ("announcement", "private_answer") for t in out["turns"])
    transcript = [c for c in out["auditor_calls"] if c["step"] == "verdict"][0]["messages"][1]["content"]
    assert "Your private exchanges" in transcript


def test_no_detector_no_readings(models):
    out = play(config(questions="none"))
    assert out["game"]["questions_asked"] == 0
    call = out["auditor_calls"][0]
    assert call["step"] == "verdict" and "Detector reading" not in call["messages"][1]["content"]
    assert "detector" not in call["messages"][0]["content"]
    assert all(t["reading"] is None for t in out["turns"])
    assert "AUDITOR:" not in models[PLAYERS].calls[0][0][0].text            # nothing to tell the players


def test_auditor_is_asked_again_then_the_game_fails(monkeypatch):
    auditor = FakeModel(AUDITOR, auditor_replies=["no tag", "<answer>Nobody</answer>", "still nothing"])
    monkeypatch.setattr(audit, "player_model", lambda name, cfg: auditor if name == AUDITOR else FakeModel(name))
    game = play(config())["game"]
    assert game["verdict"] is None and game["correct"] is None and "verdict" in game["failed"]
    assert len(auditor.calls) == 3


def test_api_errors_raise(monkeypatch):
    class Broken(FakeModel):
        async def generate(self, input, tools=None):
            raise ConnectionError("api down")
    monkeypatch.setattr(audit, "player_model", lambda name, cfg: Broken(name))
    with pytest.raises(ConnectionError):
        play(config())


@pytest.mark.parametrize("raw", [
    {"turn_mode": "two_call", "eval_model": PLAYERS, "audit": {"model": AUDITOR}},
    {"turn_mode": "single_call", "eval_model": PLAYERS},                               # no auditor model
    {"turn_mode": "single_call", "eval_model": PLAYERS, "include_parsnip": False, "audit": {"model": AUDITOR}},
])
def test_unauditable_configs_raise(raw):
    with pytest.raises(ValueError):
        audit.AuditedGame(GameConfig.from_dict(raw), seed=0)


def test_bad_question_mode_raises():
    with pytest.raises(ValueError):
        config(questions="sometimes")


def test_run_writes_the_files(models, tmp_path):
    games = audit.run_audited_games(config(questions="public"), [0, 1], FakeDetector(), tmp_path, "x")
    assert [g["seed"] for g in games] == [0, 1]
    assert {p.name for p in tmp_path.iterdir()} == {"games_x.json", "transcripts_x.jsonl", "auditor_x.jsonl"}
    with pytest.raises(FileExistsError):
        audit.run_audited_games(config(), [0], None, tmp_path, "x")
