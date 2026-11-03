"""The gifted Carrot: a Carrot that sees a detector's readings, otherwise the same as every
other Carrot. Whole games through the orchestrator with a scripted model: no API calls."""

import asyncio
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import Carrot_Parsnip_Agents as cpa  # noqa: E402
from game_config import GameConfig  # noqa: E402
from inspect_ai.agent import AgentState  # noqa: E402
from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ContentText, ModelOutput  # noqa: E402
from inspect_ai.tool import ToolCall, ToolDef  # noqa: E402


class FakeModel:
    """Announces "<name> is a Carrot" with private reasoning; votes for player 0."""

    def __init__(self):
        self.calls = []

    async def generate(self, input, tools=None, tool_choice=None):
        self.calls.append(input)
        names = {ToolDef(t).name for t in tools or []}
        me = re.search(r"You are (\w+)", input[0].text).group(1)
        if "MakeAnnouncementTool" in names:
            call = ToolCall(id="a", function="MakeAnnouncementTool", arguments={"message": f"{me} is a Carrot"})
        else:
            call = ToolCall(id="v", function="CastEliminationVoteTool", arguments={"target_index": 0})
        msg = ChatMessageAssistant(content=[ContentReasoning(reasoning=f"{me} thinks"), ContentText(text="")],
                                   tool_calls=[call], model="fake")
        return ModelOutput.from_message(msg)


class FakeDetector:
    def __init__(self):
        self.calls = []

    async def score(self, model, messages, tools):
        self.calls.append((model, messages, tools))
        return f"0.{len(self.calls)} (fake detector)"


@pytest.fixture
def model(monkeypatch):
    m = FakeModel()
    monkeypatch.setattr(cpa, "player_model", lambda name, config: m)
    return m


def play(detector=None, seed=0, **raw):
    cfg = GameConfig.from_dict({"turn_mode": "single_call", "eval_model": "openrouter/x/y",
                                "num_rounds_discussion": 2, **raw})
    out: dict = {}
    asyncio.run(cpa.CarrotParsnipOrchestrator(seed=seed, result_out=out, config=cfg, detector=detector)(
        AgentState(messages=[])))
    return out


def prompts_of(model, name):
    return [c for c in model.calls if re.search(rf"You are {name}\b", c[0].text)]


def test_the_gifted_carrot_sees_readings_of_every_other_announcement(model):
    det = FakeDetector()
    out = play(det)
    gifted = out["gifted_player"]
    assert out["detector"] and next(v for v in out["votes"] if v["name"] == gifted)["role"] == "Carrot"
    # 2 rounds x 2 other players: every other player's announcement was read, its own was not
    assert len(det.calls) == len(out["readings"]) == 4
    assert all(r["player"] != gifted for r in out["readings"])
    _, messages, tools = det.calls[0]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[-1]["reasoning_content"].endswith("thinks")              # private reasoning kept
    assert "MakeAnnouncementTool" in {t["function"]["name"] for t in tools}
    # only the gifted Carrot is told, and sees the readings (in its vote turn: all four)
    mine = prompts_of(model, gifted)
    assert all("lie detector" in c[0].text for c in mine)
    vote_prompt = mine[-1][1].text
    assert vote_prompt.count("Detector reading: ") == 4
    for other in ({"Alice", "Bob", "Charlie"} - {gifted}):
        assert all("lie detector" not in c[0].text and "Detector reading" not in c[1].text
                   for c in prompts_of(model, other))


def test_without_a_detector_the_gifted_carrot_is_a_plain_carrot(model):
    out = play(None)
    assert not out["detector"] and out["readings"] == []
    assert all("lie detector" not in c[0].text and "Detector reading" not in c[1].text for c in model.calls)


def test_the_gifted_seat_depends_on_the_seed_only(model):
    seats = [play(FakeDetector(), seed=s)["gifted_player"] for s in range(6)]
    assert seats == [play(None, seed=s)["gifted_player"] for s in range(6)]
    assert len(set(seats)) > 1
