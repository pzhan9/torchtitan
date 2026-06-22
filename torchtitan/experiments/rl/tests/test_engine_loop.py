# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the generator engine loop's decision logic (`_decide_next_action`).

Built on a bare `VLLMGenerator` (no vLLM engine) + a fake engine, so the loop's
admit/pull/shutdown branching is tested without a GPU.
"""

from __future__ import annotations

import asyncio

from torchtitan.experiments.rl.actors.generator import (
    CloseRequest,
    GenerationRequest,
    LoopAction,
    ModelStateDictPullRequest,
    SamplingConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.routing import DPRequestRouter, RoundRobinRoutingStrategy


def _bare_generator(
    *,
    close_requested: bool = False,
    model_state_dict_pull_request: ModelStateDictPullRequest | None = None,
    pending: list[GenerationRequest] | None = None,
    unfinished: bool = False,
    dp_size: int = 1,
) -> VLLMGenerator:
    # Bypass __init__ (which builds the vLLM engine); set only the loop's state.
    # _decide_next_action gates on rank 0's own DP-router bookkeeping (any_inflight),
    # NOT the engine, so no fake engine is needed here.
    generator = object.__new__(VLLMGenerator)
    generator._engine_loop_condition = asyncio.Condition()
    generator._close_request = CloseRequest() if close_requested else None
    generator._model_state_dict_pull_request = model_state_dict_pull_request
    generator._queued_generation_requests = pending or []
    generator._dp_size = dp_size
    generator._dp_router = DPRequestRouter(
        RoundRobinRoutingStrategy.Config(), dp_size=dp_size
    )
    generator._request_dp_group = {}
    if unfinished:
        # Simulate an in-flight routed request on group 0 (no completion yet).
        generator._dp_router.choose(session_id=None, estimated_cost=1)
    return generator


def _request(request_id: str = "r0") -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        sampling=SamplingConfig(),
    )


def test_closing_returns_close() -> None:
    decision = asyncio.run(_bare_generator(close_requested=True)._decide_next_action())
    assert decision.action is LoopAction.CLOSE


def test_pull_takes_precedence_over_queued_requests() -> None:
    request = _request()
    pull = ModelStateDictPullRequest(version=5)
    generator = _bare_generator(model_state_dict_pull_request=pull, pending=[request])
    decision = asyncio.run(generator._decide_next_action())
    assert (
        decision.action is LoopAction.PULL_MODEL_STATE_DICT
        and decision.pull_version == 5
    )
    # `_model_state_dict_pull_request` is NOT cleared at decide — the PULL_MODEL_STATE_DICT branch clears it after
    # applying; the single-threaded loop can't re-decide before then, so the predicate won't re-fire.
    assert generator._model_state_dict_pull_request is pull
    assert generator._queued_generation_requests == [
        request
    ]  # NOT consumed — pull runs first


def test_step_drains_the_queue() -> None:
    request = _request()
    generator = _bare_generator(pending=[request])
    decision = asyncio.run(generator._decide_next_action())
    # DP=1: a single group holds the whole batch.
    assert decision.action is LoopAction.STEP and decision.requests_by_dp == [[request]]
    assert generator._queued_generation_requests == []  # drained into the decision
    assert generator._request_dp_group == {"r0": 0}  # routed group recorded


def test_step_with_empty_queue_when_only_in_flight_work_remains() -> None:
    # No queue, no pull, but a routed request is still in flight (per the DP router).
    decision = asyncio.run(_bare_generator(unfinished=True)._decide_next_action())
    assert decision.action is LoopAction.STEP and decision.requests_by_dp == [[]]


def test_step_partitions_requests_across_dp_groups() -> None:
    # Round-robin over 2 groups: r0 -> group 0, r1 -> group 1.
    requests = [_request("r0"), _request("r1")]
    generator = _bare_generator(pending=requests, dp_size=2)
    decision = asyncio.run(generator._decide_next_action())
    assert decision.action is LoopAction.STEP
    assert decision.requests_by_dp == [[requests[0]], [requests[1]]]
    assert generator._request_dp_group == {"r0": 0, "r1": 1}


def test_busy_peer_group_keeps_issuing_step_when_group_0_idle() -> None:
    # Stall-avoidance: group 1 is busy while rank 0's own group (0) is idle. The
    # predicate gates on any_inflight (global), so it must still return STEP rather
    # than block -- otherwise the busy peer group would stall.
    generator = _bare_generator(dp_size=2)
    generator._dp_router._handles[1].reserved_load = 1  # group 1 in flight
    decision = asyncio.run(generator._decide_next_action())
    assert decision.action is LoopAction.STEP and decision.requests_by_dp == [[], []]
