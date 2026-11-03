# Carrot-Parsnip Tournament Benchmark

A single-round social-deduction elimination game for LLM players. Carrots try
to vote out the hidden Parsnip; the Parsnip tries to survive (lying permitted).

- `Carrot_Parsnip.py` — the pure game engine (state, votes, win resolution).
- `Carrot_Parsnip_Agents.py` — the Inspect-based orchestrator: LLM players,
  discussion rounds, batch runner and round-robin tournament.
- `game_config.py` — runtime game configuration: dataclasses + YAML.
- `configs/` — example game configurations.

## Configuring games at runtime

Every game parameter is injected through a `GameConfig`, loaded from YAML —
no code edits needed to change the game:

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...

python Carrot_Parsnip_Agents.py --config configs/five_player.yaml --num-games 5
python Carrot_Parsnip_Agents.py --config configs/default.yaml --mode tournament \
    --models openrouter/openai/gpt-5.6-luna openrouter/anthropic/claude-sonnet-5
```

A YAML file lists only the fields it changes; everything else keeps the
default. `configs/default.yaml` shows the commonly-changed fields (all at
their default values); dump every injectable field — including all prompt
templates as `|` block scalars — with
`python -c "from game_config import GameConfig; GameConfig().to_yaml('full.yaml')"`.

```yaml
# configs/five_player.yaml
carrot:
  count: 4
parsnip:
  count: 1
ejection_threshold: 3
num_rounds_discussion: 3
```

The injectable surface:

- **Role configs** (`carrot:` / `parsnip:`, one per `Role` enum member):
  `count`, `model` (per-role model), `models` (mixed-model seats),
  role-description fragments, and the private role knowledge each player
  receives from the engine.
- **Shape and variants**: `ejection_threshold`, `player_names`,
  `include_parsnip` (all-Carrot control game), `uncertainty` (players are
  not told which variant they are in), `draws_enabled`, `discussion`,
  `num_rounds_discussion`.
- **Run parameters**: `eval_model` (default model), `max_tokens`,
  `message_limit`, `max_concurrent_games`.
- **Prompts** (`prompts:`): every template — system prompt, win-condition
  blocks, discussion/vote reasoning prompts, strategy hints — with
  placeholders like `{num_players}` and `{ejection_threshold}` filled at
  render time, so a shape change propagates into the rules text
  automatically.

Validation is strict: unknown keys, role counts that don't sum to the player
count, or an out-of-range threshold raise immediately.

From Python:

```python
from game_config import GameConfig
from Carrot_Parsnip_Agents import run_games, run_tournament

cfg = GameConfig.from_yaml("configs/uncertain_two_parsnips.yaml")
stats = run_games(num_games=10, base_seed=0, config=cfg)
```

## Audited games (`audit.py`)

An outside auditor watches the game and names the Parsnip before the players' vote. After
each discussion round it can question every player (`audit.questions`: `none`, `public`, or
`private`). A detector can be injected: any object with
`async score(model, messages, tools) -> str`. It reads every player message together with that
player's own conversation (system prompt, reasoning, the message), and the auditor sees its
reading verbatim after the message. Every prompt and setting is in the `audit:` section of the
game config.

Each model's call settings (endpoint, sampling, thinking, `max_tokens`, reasoning history) are
injected per model string through `model_profiles`. The experiment that runs the game sets
them, and the game adds nothing to them. Profiles need `turn_mode: single_call`: one call that
reasons in the model's own thinking and then acts.

```python
from audit import run_audited_games
cfg = GameConfig.from_dict({"turn_mode": "single_call", "eval_model": M, "audit": {"model": M, "questions": "public"},
                            "model_profiles": {M: {"base_url": URL, "generate": {...}}}})
run_audited_games(cfg, seeds=[0, 1], detector=my_detector, out_dir="out", arm="public_god")
```

This writes `games_{arm}.json`, `transcripts_{arm}.jsonl` (every player message, its reasoning,
the detector's input and reading) and `auditor_{arm}.jsonl` (every auditor call).

## Tests

```bash
python -m pytest tests/          # config round-trips, validation, rendering
python Carrot_Parsnip.py         # seeded 50-game engine smoke test
```
