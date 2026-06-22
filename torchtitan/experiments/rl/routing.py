# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Transport-agnostic request routing primitives.

These policy objects are shared by both routing layers in the RL generator:

- Layer 1: ``GeneratorRouter`` (controller side) routes a call across generator
  *meshes* (replicas). See ``generator_router.py``.
- Layer 2: ``DPRequestRouter`` (in-mesh, rank-0 side) routes a request across
  the *data-parallel groups* within one generator mesh. See
  ``actors/generator.py``.

A ``RoutingStrategy`` chooses one candidate from a set. It depends only on the
``RoutingCandidate`` protocol -- a ``reserved_load`` field plus object identity --
so the same strategy classes serve both layers; each layer supplies its own
candidate type (``_GeneratorHandle`` for meshes, ``_DPGroupHandle`` for DP
groups).
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from torchtitan.config import Configurable


@runtime_checkable
class RoutingCandidate(Protocol):
    """A routable target. Strategies key on ``reserved_load`` and on identity.

    ``reserved_load`` is a generic in-flight / reserved-work counter; what one
    unit means is the candidate's concern (estimated request cost for meshes,
    in-flight request count for DP groups)."""

    reserved_load: int


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingContext:
    """Routing metadata for one generation request."""

    estimated_cost: int = 1
    """Estimated request cost used by load-aware routing strategies."""

    session_id: str | None = None
    """Stable session key consumed only by sticky routing strategies; other
    strategies ignore it. ``None`` means the request is unpinned and uses fallback
    routing without session affinity."""


class RoutingStrategy(Configurable, ABC):
    """Policy object that chooses one candidate for a request.

    Add a new strategy by subclassing this, defining a nested ``Config``, and
    selecting it explicitly in config, e.g.
    ``GeneratorRouter.Config(strategy=MyRoutingStrategy.Config())``.
    """

    def __init__(self, config: Configurable.Config):
        # Stateless by default; stateful strategies override __init__.
        del config

    @abstractmethod
    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Choose one candidate from the (non-empty) candidates."""


class RoundRobinRoutingStrategy(RoutingStrategy):
    """Cycle over the candidates in order, ignoring load."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        del config
        self._counter = itertools.count()

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Return the next candidate in round-robin order."""

        del routing_ctx
        return candidates[next(self._counter) % len(candidates)]


class LeastLoadedRoutingStrategy(RoutingStrategy):
    """Pick the candidate with the least reserved load."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Return the candidate with the lowest reserved load."""

        del routing_ctx
        return min(candidates, key=lambda h: h.reserved_load)


class StickySessionRoutingStrategy(RoutingStrategy):
    """Keep requests from the same session on the same candidate."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        max_sessions: int = 4096
        """Maximum number of session-to-candidate assignments to retain,
        evicting least-recently-used sessions first."""

        fallback_strategy: RoutingStrategy.Config = field(
            default_factory=LeastLoadedRoutingStrategy.Config
        )
        """Routing strategy used for new sessions and requests without a session."""

        def __post_init__(self):
            if self.max_sessions <= 0:
                raise ValueError(
                    f"max_sessions must be positive, got {self.max_sessions}"
                )

    def __init__(self, config: Config):
        self._max_sessions = config.max_sessions
        self._fallback_strategy = config.fallback_strategy.build()
        self._sessions: OrderedDict[str, RoutingCandidate] = OrderedDict()

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[RoutingCandidate],
    ) -> RoutingCandidate:
        """Return the session's assigned candidate, or assign a new one.

        Unpinned requests (no ``session_id``) and first-seen sessions defer to
        the fallback strategy; a session's first assignment is then remembered so
        every later request with that key reuses the same candidate. If a
        session's pinned candidate is no longer available (e.g. a mesh draining
        for a weight sync), the request falls back and the session is re-pinned to
        the newly chosen candidate. The map is bounded by ``max_sessions`` and
        evicts the least-recently-used session.
        """

        # Unpinned request: no affinity, defer entirely to the fallback.
        if routing_ctx.session_id is None:
            return self._fallback_strategy.choose(routing_ctx, candidates)

        # Reuse the pinned candidate, but only while it is still available
        # (e.g. not draining for a weight sync).
        sticky_candidate = self._sessions.get(routing_ctx.session_id)
        if sticky_candidate is not None:
            if any(h is sticky_candidate for h in candidates):
                # End of the dict means it's the most-recently-used session.
                self._sessions.move_to_end(routing_ctx.session_id)
                return sticky_candidate

        # New session, or the pinned candidate is unavailable: choose via the
        # fallback and (re)pin the session to that candidate.
        chosen = self._fallback_strategy.choose(routing_ctx, candidates)
        self._sessions[routing_ctx.session_id] = chosen
        # End of the dict means it's the most-recently-used session.
        self._sessions.move_to_end(routing_ctx.session_id)
        # Evict the least-recently-used session if the map is full. We assume
        # max_sessions is large enough that active sessions are never the LRU
        # victim (only stale, finished sessions get evicted).
        # TODO: relying solely on max_sessions to avoid premature eviction is
        # easy to implement, but not robust for all scenarios. Revisit with an
        # more robust approach.
        if len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return chosen


@dataclass(kw_only=True, slots=True)
class _DPGroupHandle:
    """One data-parallel group as a routing candidate (satisfies ``RoutingCandidate``)."""

    group_idx: int
    """Index of this DP group within the mesh (== vLLM data_parallel_rank)."""

    reserved_load: int = 0
    """In-flight routed requests on this DP group (count, not token cost)."""


class DPRequestRouter:
    """In-mesh, rank-0-only router that partitions requests across DP groups.

    Reuses the ``RoutingStrategy`` family over ``_DPGroupHandle`` candidates. Load
    is measured in in-flight request count -- ``choose`` reserves one unit and
    ``release`` frees it when the completion comes back -- because the relevant
    cost for balancing the MoE expert-parallel all-to-all is the number of active
    sequences per group, not their token length.

    Not thread-safe: like ``GeneratorRouter`` it is driven from a single event
    loop (rank 0's engine loop + result-drain task).
    """

    def __init__(self, config: RoutingStrategy.Config, *, dp_size: int):
        if dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")
        self._strategy = config.build()
        self._handles = [_DPGroupHandle(group_idx=g) for g in range(dp_size)]

    def choose(self, *, session_id: str | None, estimated_cost: int) -> int:
        """Pick a DP group for one request, reserving one unit of load on it."""

        ctx = RoutingContext(estimated_cost=estimated_cost, session_id=session_id)
        handle = self._strategy.choose(ctx, self._handles)
        handle.reserved_load += 1
        return handle.group_idx

    def release(self, group_idx: int) -> None:
        """Free the unit of load reserved by ``choose`` when a request finishes."""

        handle = self._handles[group_idx]
        handle.reserved_load -= 1
        assert (
            handle.reserved_load >= 0
        ), f"dp group {group_idx} reserved_load went negative: {handle.reserved_load}"

    @property
    def any_inflight(self) -> bool:
        """True iff any DP group has in-flight routed work.

        Rank 0 uses this in ``_decide_next_action`` instead of the engine's local
        ``has_unfinished_requests`` (which would be a DP all-reduce, illegal on a
        rank-0-only decision, and would also miss a busy peer group when rank 0's
        own group is idle -- stalling the peer)."""

        return any(handle.reserved_load for handle in self._handles)
