"""Searches over the holes of a `DrillholePlan`."""

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from boitata.errors import InvalidInput

__all__ = ["Annealing", "DrillholeSearch", "Greedy", "ModifiedRandomSearch", "Swap"]


@runtime_checkable
class DrillholeSearch(Protocol):
    """What `DrillholePlan.optimize` calls: any object with a ``search(problem, n)`` method, no subclassing needed.

    ``search`` receives the `DrillholePlan` and the most holes to choose, and returns distinct candidate indices
    that `DrillholePlan.feasible` allows together. It reads the problem through ``score``, ``gains``, ``loss``,
    ``feasible``, ``neighbors``, ``cost`` and ``budget``.

    Examples
    --------
    >>> class Ranking:
    ...     def search(self, problem, n):
    ...         gains = problem.gains([])
    ...         return [int(i) for i in np.argsort(-gains, kind="stable")[:n] if gains[i] > 0]
    >>> table = plan.optimize(5, search=Ranking())
    """

    def search(self, problem, n: int) -> list[int]: ...


def _best(values: np.ndarray) -> int | None:
    """Index of the largest finite value, the first among ties; None when none is finite."""
    if not values.size or not np.isfinite(values).any():
        return None
    return int(np.argmax(np.where(np.isfinite(values), values, -np.inf)))


def _better(new: float, old: float) -> bool:
    return new > old + 1e-9 * max(1.0, abs(old))


@dataclass(frozen=True, kw_only=True)
class Greedy:
    """Adds the hole of largest gain, one at a time, until `n` holes or no feasible hole adds to the objective.

    When the problem has a budget, the hole of largest gain per unit cost goes in instead, and the search returns
    the better of that plan and the best single hole: ranking by gain per cost alone can fill the budget with cheap
    holes of little value.
    """

    def search(self, problem, n: int) -> list[int]:
        chosen: list[int] = []
        budget = problem.budget is not None
        cost = problem.cost
        while len(chosen) < n:
            gains = problem.gains(chosen)
            useful = np.isfinite(gains) & (gains > 0)
            if not useful.any():
                break
            if budget:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rank = np.where(cost > 0, gains / cost, np.inf)
            else:
                rank = gains
            chosen.append(_best(np.where(useful, rank, -np.inf)))
        if budget and chosen:
            single = problem.gains([])
            best = _best(single)
            if best is not None and _better(single[best], problem.score(chosen)):
                return [best]
        return chosen


@dataclass(frozen=True, kw_only=True)
class Swap:
    """Best-improvement exchange: from the plan of `start`, applies the one-for-one exchange (or, below `n`
    holes, the addition) that raises the objective most, until none raises it.

    Parameters
    ----------
    start : DrillholeSearch
        The search giving the first plan.
    candidates : float, optional
        Exchange a hole only for candidates whose collars lie within this distance of its collar in plan; any
        feasible candidate when None.

    Notes
    -----
    Each round reads ``gains`` once per hole of the plan, with that hole removed; the problem re-kriges only the
    blocks the change reaches, so a round costs about as much as a few greedy steps. The result is a local
    optimum: no single exchange improves it.
    """

    start: DrillholeSearch = field(default_factory=Greedy)
    candidates: float | None = None

    def __post_init__(self):
        if self.candidates is not None and not (math.isfinite(self.candidates) and self.candidates >= 0):
            raise InvalidInput("candidates must be a finite distance >= 0 or None")

    def search(self, problem, n: int) -> list[int]:
        current = [int(i) for i in self.start.search(problem, n)]
        score = problem.score(current)
        near = {}
        while True:
            best, plan = score, None
            if len(current) < n:
                gains = problem.gains(current)
                c = _best(gains)
                if c is not None and _better(score + gains[c], best):
                    best, plan = score + gains[c], [*current, c]
            loss = problem.loss(current) if current else []
            for k, hole in enumerate(current):
                rest = current[:k] + current[k + 1 :]
                gains = problem.gains(rest)
                gains[hole] = -np.inf
                if self.candidates is not None:
                    if hole not in near:
                        near[hole] = problem.neighbors(hole, self.candidates)
                    keep = np.full(len(gains), -np.inf)
                    keep[near[hole]] = gains[near[hole]]
                    gains = keep
                c = _best(gains)
                if c is not None and _better(score - loss[k] + gains[c], best):
                    best, plan = score - loss[k] + gains[c], [*current[:k], c, *current[k + 1 :]]
            if plan is None:
                return current
            current = plan
            score = problem.score(current)


class _Moves:
    """Moves shared by the random searches: drop one hole of the plan, picked with probability rising as its
    contribution falls, and drill a feasible candidate instead, within `radius` of it or, with probability
    `jump`, anywhere."""

    def __post_init__(self):
        if not (isinstance(self.iterations, (int, np.integer)) and self.iterations >= 0):
            raise InvalidInput("iterations must be an integer >= 0")
        if not (math.isfinite(self.radius) and self.radius >= 0):
            raise InvalidInput("radius must be finite and >= 0")
        if not 0 <= self.jump <= 1:
            raise InvalidInput("jump must lie in [0, 1]")

    def _move(self, problem, current, contribution, rng, near):
        """`current` with one hole moved, or None when the picked hole has nowhere to go."""
        weights = contribution.max() + contribution.min() - contribution + 1e-9
        k = int(rng.choice(len(current), p=weights / weights.sum()))
        rest = current[:k] + current[k + 1 :]
        allowed = problem.feasible(rest)
        allowed[current[k]] = False
        anywhere = np.flatnonzero(allowed)
        if current[k] not in near:
            near[current[k]] = problem.neighbors(current[k], self.radius)
        local = near[current[k]][allowed[near[current[k]]]]
        pool = anywhere if rng.random() < self.jump or not len(local) else local
        if not len(pool):
            return None
        return [*rest[:k], int(pool[rng.integers(len(pool))]), *rest[k:]]


@dataclass(frozen=True, kw_only=True)
class ModifiedRandomSearch(_Moves):
    """Random search of local moves that keeps a move only when it raises the objective.

    Each iteration drops hole ``i`` of the plan with probability proportional to
    ``max(C) + min(C) - C_i``, ``C`` the contributions (`DrillholePlan.loss`) of the plan's holes, so weak holes
    move most often, and puts a feasible candidate in its place: one within `radius` of the dropped collar, or
    with probability `jump` any feasible candidate.

    Parameters
    ----------
    iterations : int
        Moves tried.
    radius : float
        Reach of a local move, in plan.
    jump : float
        Probability of a move to any feasible candidate.
    seed : int
        Seeds the moves; the result depends on it only, not on the thread count.
    start : DrillholeSearch
        The search giving the first plan.
    """

    iterations: int = 4000
    radius: float = 25.0
    jump: float = 0.2
    seed: int = 0
    start: DrillholeSearch = field(default_factory=Greedy)

    def search(self, problem, n: int) -> list[int]:
        current = [int(i) for i in self.start.search(problem, n)]
        if not current:
            return current
        rng = np.random.default_rng(self.seed)
        score, contribution, near = problem.score(current), problem.loss(current), {}
        for _ in range(self.iterations):
            new = self._move(problem, current, contribution, rng, near)
            if new is None:
                continue
            value = problem.score(new)
            if value > score:
                current, score = new, value
                contribution = problem.loss(current)
        return current


@dataclass(frozen=True, kw_only=True)
class Annealing(_Moves):
    """Simulated annealing with the moves of `ModifiedRandomSearch`, returning the best plan it visits.

    A move that lowers the objective by ``d`` is kept with probability ``exp(-d / T)`` (Metropolis); the
    temperature ``T`` falls geometrically from the first to the second value of `temperature` over the
    iterations. ``T`` is in units of the objective: with the built-in objectives, weighted blocks.

    Parameters
    ----------
    iterations : int
        Moves tried.
    temperature : (float, float)
        Initial and final temperature, both > 0.
    radius : float
        Reach of a local move, in plan.
    jump : float
        Probability of a move to any feasible candidate.
    seed : int
        Seeds the moves; the result depends on it only, not on the thread count.
    start : DrillholeSearch
        The search giving the first plan.
    """

    iterations: int = 4000
    temperature: tuple[float, float] = (0.3, 0.01)
    radius: float = 25.0
    jump: float = 0.2
    seed: int = 0
    start: DrillholeSearch = field(default_factory=Greedy)

    def __post_init__(self):
        super().__post_init__()
        first, last = (float(t) for t in self.temperature)
        if not (math.isfinite(first) and math.isfinite(last) and first > 0 and last > 0):
            raise InvalidInput("temperature must be two finite values > 0")

    def search(self, problem, n: int) -> list[int]:
        current = [int(i) for i in self.start.search(problem, n)]
        if not current:
            return current
        rng = np.random.default_rng(self.seed)
        first, last = (float(t) for t in self.temperature)
        cooling = (last / first) ** (1 / max(self.iterations, 1))
        score, contribution, near = problem.score(current), problem.loss(current), {}
        best, top, t = current, score, first
        for _ in range(self.iterations):
            new = self._move(problem, current, contribution, rng, near)
            if new is not None:
                value = problem.score(new)
                if value >= score or rng.random() < math.exp((value - score) / t):
                    if value != score:
                        contribution = problem.loss(new)
                    current, score = new, value
                    if score > top:
                        best, top = current, score
            t *= cooling
        return best
