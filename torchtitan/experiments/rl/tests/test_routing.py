# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the in-mesh DP request router (`DPRequestRouter`).

The routing *strategies* are shared with the Layer-1 ``GeneratorRouter`` and are
tested over that router in ``test_generator_router.py``; here we cover the
DP-group candidate accounting (choose / release / any_inflight) and the strategy
reuse over ``_DPGroupHandle``.
"""

from __future__ import annotations

import pytest

from torchtitan.experiments.rl.routing import (
    DPRequestRouter,
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
    StickySessionRoutingStrategy,
)


def _loads(router: DPRequestRouter) -> list[int]:
    return [h.reserved_load for h in router._handles]


def test_round_robin_cycles_across_groups():
    router = DPRequestRouter(RoundRobinRoutingStrategy.Config(), dp_size=3)
    chosen = [router.choose(session_id=None, estimated_cost=1) for _ in range(7)]
    assert chosen == [0, 1, 2, 0, 1, 2, 0]


def test_least_loaded_picks_lowest_then_balances():
    router = DPRequestRouter(LeastLoadedRoutingStrategy.Config(), dp_size=2)
    # First two ties resolve to the lowest index, then load steers the choice.
    assert router.choose(session_id=None, estimated_cost=1) == 0
    assert router.choose(session_id=None, estimated_cost=1) == 1
    assert router.choose(session_id=None, estimated_cost=1) in (0, 1)
    assert _loads(router) == [2, 1] or _loads(router) == [1, 2]


def test_release_frees_load():
    router = DPRequestRouter(LeastLoadedRoutingStrategy.Config(), dp_size=2)
    g0 = router.choose(session_id=None, estimated_cost=1)
    g1 = router.choose(session_id=None, estimated_cost=1)
    assert router.any_inflight
    router.release(g0)
    router.release(g1)
    assert not router.any_inflight
    assert _loads(router) == [0, 0]


def test_release_below_zero_asserts():
    router = DPRequestRouter(LeastLoadedRoutingStrategy.Config(), dp_size=1)
    with pytest.raises(AssertionError, match="went negative"):
        router.release(0)


def test_sticky_pins_session_to_group():
    router = DPRequestRouter(StickySessionRoutingStrategy.Config(), dp_size=3)
    first = router.choose(session_id="s0", estimated_cost=1)
    # Even though s0's group now has load, the same session sticks to it.
    assert router.choose(session_id="s0", estimated_cost=1) == first
    # A different session uses the (least-loaded) fallback -> a different group.
    other = router.choose(session_id="s1", estimated_cost=1)
    assert other != first


def test_dp_size_one_always_group_zero():
    router = DPRequestRouter(LeastLoadedRoutingStrategy.Config(), dp_size=1)
    assert [router.choose(session_id=None, estimated_cost=1) for _ in range(3)] == [
        0,
        0,
        0,
    ]


def test_rejects_non_positive_dp_size():
    with pytest.raises(ValueError, match="dp_size must be >= 1"):
        DPRequestRouter(RoundRobinRoutingStrategy.Config(), dp_size=0)
