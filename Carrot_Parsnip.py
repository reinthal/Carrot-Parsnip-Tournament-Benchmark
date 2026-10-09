"""
Carrot-Parsnip – A single-round elimination game.

Roles (the constants below are the defaults; pass num_carrot/num_parsnip to
the constructor — or use game_config.GameConfig — to override per game):
    NUM_CARROT  x Carrot players
    NUM_PARSNIP x Parsnip players

Rules:
    All NUM_PLAYERS players vote to eliminate one player.
    A player receiving EJECTION_THRESHOLD+ votes is eliminated.
    If any Parsnip is eliminated, Carrots win.
    If every Parsnip survives, the Parsnip(s) win.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────


class Team(Enum):
    CARROT = "Carrot"
    PARSNIP = "Parsnip"


class Role(Enum):
    CARROT = "Carrot"
    PARSNIP = "Parsnip"


class GamePhase(Enum):
    ELIMINATION_VOTE = auto()
    GAME_OVER = auto()


# ── Data Classes ─────────────────────────────────────────────────────────


@dataclass
class Player:
    name: str
    index: int
    role: Role
    alive: bool = True

    @property
    def team(self) -> Team:
        return Team.CARROT if self.role == Role.CARROT else Team.PARSNIP


@dataclass
class LogEntry:
    text: str


# ── Constants ────────────────────────────────────────────────────────────

NUM_PLAYERS = 3
NUM_CARROT = 2
NUM_PARSNIP = 1
EJECTION_THRESHOLD = 2

if NUM_CARROT + NUM_PARSNIP != NUM_PLAYERS:
    raise ValueError(
        f"NUM_CARROT ({NUM_CARROT}) + NUM_PARSNIP ({NUM_PARSNIP}) "
        f"must equal NUM_PLAYERS ({NUM_PLAYERS})"
    )
if NUM_CARROT < 1 or NUM_PARSNIP < 1:
    raise ValueError("NUM_CARROT and NUM_PARSNIP must each be at least 1")
if not 1 <= EJECTION_THRESHOLD <= NUM_PLAYERS:
    raise ValueError(f"EJECTION_THRESHOLD must be between 1 and NUM_PLAYERS ({NUM_PLAYERS})")

_NAME_POOL = [
    "Alice", "Bob", "Charlie", "Dave", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj", "Olivia", "Peggy",
    "Quentin", "Rupert", "Sybil", "Trent", "Uma", "Victor", "Walter",
    "Xena", "Yusuf", "Zoe",
]


def default_player_names(n: int = NUM_PLAYERS) -> list[str]:
    """Return *n* distinct player names (falls back to 'Player N' beyond the pool)."""
    return [_NAME_POOL[i] if i < len(_NAME_POOL) else f"Player {i + 1}" for i in range(n)]


# ── Game Engine ──────────────────────────────────────────────────────────


class CarrotParsnipGame:
    """Single-round elimination game. The role split, ejection threshold and
    private role knowledge are per-instance; omitting the counts uses the
    module-constant defaults (which require exactly NUM_PLAYERS names)."""

    def __init__(
        self,
        player_names: list[str],
        seed: Optional[int] = None,
        include_parsnip: bool = True,
        *,
        num_carrot: Optional[int] = None,
        num_parsnip: Optional[int] = None,
        ejection_threshold: int = EJECTION_THRESHOLD,
        role_knowledge: Optional[dict[Role, str]] = None,
    ):
        n = len(player_names)
        if (num_carrot is None) != (num_parsnip is None):
            raise ValueError("pass both num_carrot and num_parsnip, or neither")
        if num_carrot is None:
            if n != NUM_PLAYERS:
                raise ValueError(f"Carrot-Parsnip requires exactly {NUM_PLAYERS} players, got {n}")
            num_carrot, num_parsnip = NUM_CARROT, NUM_PARSNIP
        if num_carrot + num_parsnip != n:
            raise ValueError(
                f"num_carrot ({num_carrot}) + num_parsnip ({num_parsnip}) "
                f"must equal the number of players ({n})"
            )
        if num_carrot < 1 or num_parsnip < 1:
            raise ValueError("num_carrot and num_parsnip must each be at least 1")
        if not 1 <= ejection_threshold <= n:
            raise ValueError(f"ejection_threshold must be between 1 and {n}, got {ejection_threshold}")

        self.num_players = n
        self.num_carrot = num_carrot
        self.num_parsnip = num_parsnip
        self.ejection_threshold = ejection_threshold
        self.role_knowledge = role_knowledge

        self.rng = random.Random(seed)
        self.include_parsnip = include_parsnip

        # Assign roles
        if include_parsnip:
            roles: list[Role] = [Role.CARROT] * num_carrot + [Role.PARSNIP] * num_parsnip
        else:
            roles: list[Role] = [Role.CARROT] * n
        self.rng.shuffle(roles)

        self.players: list[Player] = [
            Player(name=name, index=i, role=roles[i])
            for i, name in enumerate(player_names)
        ]

        self.phase: GamePhase = GamePhase.ELIMINATION_VOTE
        self.votes: dict[int, int] = {}  # voter_index -> target_index
        self.winner: Optional[Team] = None
        self.log: list[LogEntry] = []
        self.private_knowledge: dict[int, list[str]] = {i: [] for i in range(n)}

        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Game already started")
        self._started = True

        for p in self.players:
            if self.role_knowledge is not None:
                self.private_knowledge[p.index].append(self.role_knowledge[p.role])
            elif p.role == Role.CARROT:
                self.private_knowledge[p.index].append("You are a Carrot player.")
            elif self.num_parsnip == 1:
                self.private_knowledge[p.index].append("You are the Parsnip player.")
            else:
                self.private_knowledge[p.index].append(
                    f"You are a Parsnip player (one of {self.num_parsnip} Parsnips)."
                )

        self._log("Game started with %d players. Vote to eliminate!", self.num_players)

    # ── Agent Interface ──────────────────────────────────────────────

    def get_action_required(self) -> dict:
        if self.phase == GamePhase.GAME_OVER:
            return {
                "phase": self.phase,
                "action_type": "game_over",
                "player_index": None,
                "options": [],
                "description": f"Game over. {self.winner.value} wins!",
            }

        pending = [p.index for p in self.players if p.index not in self.votes]
        targets = [p.index for p in self.players]
        return {
            "phase": self.phase,
            "action_type": "elimination_vote",
            "player_index": pending,
            "options": targets,
            "description": (
                f"Elimination Vote: all players vote for a player to eliminate. "
                f"A player receiving {self.ejection_threshold}+ votes is eliminated."
            ),
        }

    def take_action(self, action_type: str, **kwargs) -> dict:
        if action_type != "elimination_vote":
            return {"success": False, "message": f"Unknown action: {action_type}"}
        return self._act_elimination_vote(**kwargs)

    # ── State Summaries ──────────────────────────────────────────────

    def get_state_summary(self, player_index: int) -> str:
        p = self.players[player_index]
        lines: list[str] = []

        lines.append("=" * 50)
        lines.append(f"  CARROT-PARSNIP — State for: {p.name}")
        lines.append("=" * 50)

        lines.append("")
        lines.append("PLAYERS")
        for pl in self.players:
            lines.append(f"  [{pl.index}] {pl.name}")

        lines.append("")
        lines.append("CURRENT PHASE")
        action = self.get_action_required()
        lines.append(f"  {action['description']}")

        acting = action.get("player_index")
        is_actor = (
            acting == player_index
            or (isinstance(acting, list) and player_index in acting)
        )
        if is_actor and action["options"]:
            lines.append(f"  Valid options: {action['options']}")

        lines.append("")
        lines.append("YOUR PRIVATE KNOWLEDGE")
        for info in self.private_knowledge[player_index]:
            lines.append(f"  * {info}")

        lines.append("")
        lines.append("LOG")
        for entry in self.log:
            lines.append(f"  {entry.text}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def get_global_summary(self) -> str:
        lines: list[str] = []
        lines.append("=" * 50)
        lines.append("  CARROT-PARSNIP — Omniscient View")
        lines.append("=" * 50)

        lines.append("")
        lines.append("PLAYERS (roles revealed)")
        for pl in self.players:
            status = "ALIVE" if pl.alive else "ELIMINATED"
            lines.append(f"  [{pl.index}] {pl.name} — {pl.role.value} [{status}]")

        if self.winner:
            lines.append(f"\nWINNER: {self.winner.value}")

        lines.append("")
        lines.append("LOG")
        for entry in self.log:
            lines.append(f"  {entry.text}")

        lines.append("=" * 50)
        return "\n".join(lines)

    # ── Internal ─────────────────────────────────────────────────────

    def _log(self, fmt: str, *args) -> None:
        text = fmt % args if args else fmt
        self.log.append(LogEntry(text))

    def _act_elimination_vote(self, *, player_index: int, target_index: int) -> dict:
        if self.phase != GamePhase.ELIMINATION_VOTE:
            return {"success": False, "message": "Not in ELIMINATION_VOTE phase."}
        if player_index in self.votes:
            return {"success": False, "message": f"{self.players[player_index].name} already voted."}
        if target_index < 0 or target_index >= self.num_players:
            return {"success": False, "message": f"Invalid target index {target_index}."}

        self.votes[player_index] = target_index

        if len(self.votes) == self.num_players:
            return self._resolve_elimination()
        return {"success": True, "message": f"{self.players[player_index].name} voted."}

    def _resolve_elimination(self) -> dict:
        # Tally votes
        tally: dict[int, int] = {}
        for target_idx in self.votes.values():
            tally[target_idx] = tally.get(target_idx, 0) + 1

        ballots = ", ".join(
            f"{self.players[i].name}->{self.players[t].name}"
            for i, t in sorted(self.votes.items())
        )

        eliminated: list[int] = [idx for idx, count in tally.items() if count >= self.ejection_threshold]

        for idx in eliminated:
            self.players[idx].alive = False

        if self.include_parsnip:
            # Carrots win if any Parsnip was eliminated
            parsnips = [p for p in self.players if p.role == Role.PARSNIP]
            parsnip_eliminated = any(not p.alive for p in parsnips)
            names = ", ".join(self.players[i].name for i in eliminated)

            if parsnip_eliminated:
                self.winner = Team.CARROT
                self._log("Elimination: %s eliminated! [%s]", names, ballots)
                if self.num_parsnip == 1:
                    self._log("Carrots win! The Parsnip has been found.")
                else:
                    self._log("Carrots win! A Parsnip has been found.")
            else:
                if eliminated:
                    no_parsnip = "the Parsnip" if self.num_parsnip == 1 else "a Parsnip"
                    self._log("Elimination: %s eliminated, but not %s! [%s]", names, no_parsnip, ballots)
                else:
                    self._log("Elimination: No one received %d+ votes. [%s]", self.ejection_threshold, ballots)
                self.winner = Team.PARSNIP
                if self.num_parsnip == 1:
                    self._log("Parsnip wins! The Parsnip survived.")
                else:
                    self._log("Parsnips win! All Parsnips survived.")
        else:
            # All-Carrot mode: win if nobody eliminated, lose if anyone eliminated
            if eliminated:
                names = ", ".join(self.players[i].name for i in eliminated)
                self._log("Elimination: %s eliminated! [%s]", names, ballots)
                self.winner = Team.PARSNIP  # Carrots lose
                self._log("Carrots lose! A player was eliminated.")
            else:
                self._log("Elimination: No one received %d+ votes. [%s]", self.ejection_threshold, ballots)
                self.winner = Team.CARROT
                self._log("Carrots win! No one was eliminated.")

        self.phase = GamePhase.GAME_OVER
        return {"success": True, "message": f"Game over. {self.winner.value} wins!"}


# ── Demo ─────────────────────────────────────────────────────────────────


def play_random_game(
    player_names: list[str] | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> CarrotParsnipGame:
    if player_names is None:
        player_names = default_player_names()
    game = CarrotParsnipGame(player_names, seed=seed)
    game.start()
    rng = random.Random(seed)

    action = game.get_action_required()
    for pi in action["player_index"]:
        game.take_action("elimination_vote", player_index=pi, target_index=rng.choice(action["options"]))

    if verbose:
        print(game.get_global_summary())
    return game


if __name__ == "__main__":
    print("Running 50 random games as a smoke test...\n")
    results = {"Carrot": 0, "Parsnip": 0}
    for i in range(50):
        g = play_random_game(seed=i)
        if g.winner:
            results[g.winner.value] += 1
        else:
            print(f"  Game {i}: no winner (bug)")
    print(f"Results over 50 games:  Carrot {results['Carrot']}  |  Parsnip {results['Parsnip']}")

    print("\n--- Example: Player 0 state summary (seed=42) ---\n")
    demo = play_random_game(seed=42)
    print(demo.get_state_summary(0))
    print("\n--- Omniscient summary ---\n")
    print(demo.get_global_summary())
