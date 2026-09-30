# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the generator engine loop: its decision logic (`_decide_next_action`)
and the engine thread's hand-offs with the event loop.

Built on a bare `VLLMGenerator` (no vLLM engine) + a fake engine, so the loop's
admit/pull/shutdown branching is tested without a GPU.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from types import SimpleNamespace

import pytest

import torchtitan.rl.generator as generator_module

from torchtitan.rl.distributed.routing.intra_generator import IntraGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoutingStrategy,
    StickySessionRoutingStrategy,
)

from torchtitan.rl.generator import (
    CloseRequest,
    GenerationFuture,
    GenerationRequest,
    LoopAction,
    ModelStateDictPullRequest,
    RequestDispatcher,
    SamplingConfig,
    VLLMGenerator,
)

_TIMEOUT_S = 5


def _bare_generator(
    *,
    close_requested: bool = False,
    model_state_dict_pull_request: ModelStateDictPullRequest | None = None,
    pending: list[GenerationRequest] | None = None,
    inflight: bool = False,
    dp_size: int = 1,
    dp_routing_strategy: RoutingStrategy.Config | None = None,
) -> VLLMGenerator:
    # Bypass __init__ (which builds the vLLM engine); set only the loop's state.
    # _decide_next_action delegates the in-flight check and routing to the
    # dispatcher, so wire up a bare one (no engine / GPU needed here).
    generator = object.__new__(VLLMGenerator)
    generator._engine_loop_condition = asyncio.Condition()
    generator._close_request = CloseRequest() if close_requested else None
    generator._model_state_dict_pull_request = model_state_dict_pull_request
    generator.policy_version = 0
    generator._request_dispatcher = RequestDispatcher(
        rank=0,
        dp_rank=0,
        tp_rank=0,
        dp_degree=dp_size,
        broadcast_group=None,
        open_result_channel=None,
        intra_generator_router=IntraGeneratorRouter.Config(
            strategy=dp_routing_strategy or LeastLoadedRoutingStrategy.Config()
        ),
    )
    # A registered-but-unresolved future models in-flight work (possibly in a peer DP rank).
    if inflight:
        generator._request_dispatcher._rank0_generation_futures = {"inflight": object()}
    _queue(generator, pending or [])
    return generator


def _queue(generator: VLLMGenerator, requests: list[GenerationRequest]) -> None:
    """Queue requests the way `generate` does: each with a registered future."""
    generator._queued_generation_requests = requests
    for request in requests:
        generator._request_dispatcher._rank0_generation_futures[
            request.request_id
        ] = GenerationFuture(future=None, metrics_prefix="generator")


def _request(
    request_id: str = "r0",
    *,
    routing_session_id: str | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        sampling=SamplingConfig(),
        routing_session_id=routing_session_id or request_id,
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
    # `_model_state_dict_pull_request` is NOT cleared at decide -- `_rank0_finish_pull` clears it after the
    # engine thread applies the pull, before that thread asks for its next decision.
    assert generator._model_state_dict_pull_request is pull
    assert generator._queued_generation_requests == [
        request
    ]  # NOT consumed — pull runs first


def test_step_drains_the_queue() -> None:
    request = _request()
    generator = _bare_generator(pending=[request])
    generator.policy_version = 3
    decision = asyncio.run(generator._decide_next_action())
    # DP=1: a single DP rank holds the whole batch.
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [
        [request]
    ]
    assert generator._queued_generation_requests == []  # drained into the decision
    assert generator._request_dispatcher._rank0_dp_router is None
    # Admission stamps the version the request is sampled under.
    futures = generator._request_dispatcher._rank0_generation_futures
    assert futures["r0"].min_policy_version == 3


def test_step_with_empty_queue_when_only_in_flight_work_remains() -> None:
    # No queue, no pull, but a registered future means a request is still in flight
    # (possibly in a peer DP rank), so rank 0 must keep issuing STEP.
    decision = asyncio.run(_bare_generator(inflight=True)._decide_next_action())
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [[]]


def test_step_routes_requests_across_dp_ranks() -> None:
    # Least-loaded over 3 idle DP ranks: r0 -> rank 0, r1 -> rank 1 (rank 0 now loaded).
    requests = [_request("r0"), _request("r1")]
    generator = _bare_generator(pending=requests, dp_size=3)
    decision = asyncio.run(generator._decide_next_action())
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[requests[0]], [requests[1]], []]
    # Each request reserves one load unit on its chosen DP rank.
    dp_router = generator._request_dispatcher._rank0_dp_router
    assert dp_router._reservations == {"r0": 0, "r1": 1}
    assert [h.reserved_load for h in dp_router._handles] == [1, 1, 0]


def test_step_sticky_session_reuses_dp_rank() -> None:
    first = _request("r0", routing_session_id="s0")
    generator = _bare_generator(
        pending=[first],
        dp_size=3,
        dp_routing_strategy=StickySessionRoutingStrategy.Config(),
    )

    first_decision = asyncio.run(generator._decide_next_action())
    assert first_decision.action is LoopAction.STEP
    assert first_decision.requests_per_dp_rank == [[first], [], []]

    same_session = _request("r1", routing_session_id="s0")
    new_session = _request("r2", routing_session_id="s1")
    _queue(generator, [same_session, new_session])

    second_decision = asyncio.run(generator._decide_next_action())
    assert second_decision.action is LoopAction.STEP
    assert second_decision.requests_per_dp_rank == [
        [same_session],
        [new_session],
        [],
    ]
    # r0 and r1 share session s0 -> same DP rank; r2's new session falls back.
    assert generator._request_dispatcher._rank0_dp_router._reservations == {
        "r0": 0,
        "r1": 0,
        "r2": 1,
    }


# --- engine thread ---


class _FakeEngine:
    """Every step finishes all admitted requests. `step_hook` runs inside each step."""

    def __init__(self, step_hook=lambda: None):
        self.renderer = SimpleNamespace(
            render_cmpl=lambda prompts: prompts, shutdown=lambda: None
        )
        self.step_hook = step_hook
        self.running: list[str] = []

    def add_request(self, *, request_id, prompt, params):
        self.running.append(request_id)

    def has_unfinished_requests(self) -> bool:
        return bool(self.running)

    def step(self):
        self.step_hook()
        finished, self.running = self.running, []
        return [
            SimpleNamespace(
                request_id=request_id,
                num_cached_tokens=0,
                metrics=SimpleNamespace(
                    first_token_latency=0.01,
                    queued_ts=1.0,
                    scheduled_ts=1.0,
                    first_token_ts=1.01,
                    last_token_ts=1.02,
                    num_generation_tokens=1,
                ),
                outputs=[
                    SimpleNamespace(
                        token_ids=[7],
                        logprobs=[{7: SimpleNamespace(logprob=-0.5)}],
                        finish_reason="stop",
                    )
                ],
            )
            for request_id in finished
        ]


@pytest.fixture
def engine_thread(monkeypatch):
    """Returns a function that builds a rank-0 generator around `engine`, with its
    engine thread running and (unless `start_loop=False`) the engine loop started;
    joins the thread afterwards."""
    # Rank 0 is the only rank, so the decision broadcast is a no-op.
    monkeypatch.setattr(
        generator_module.dist, "broadcast_object_list", lambda *a, **k: None
    )

    async def start(engine: _FakeEngine, *, start_loop: bool = True) -> VLLMGenerator:
        monkeypatch.setattr(
            generator_module,
            "LLMEngine",
            SimpleNamespace(from_engine_args=lambda *a, **k: engine),
        )
        generator = _bare_generator()
        generator._rank = 0
        generator._broadcast_group = None
        generator.config = SimpleNamespace(
            sampling=SamplingConfig(),
            max_engine_steps_between_decisions=16,
            reset_prefix_cache_on_weight_sync=False,
        )
        generator._pull_model_state_dict_future = None
        generator._start_engine_thread(None, None, name="test-engine")
        if start_loop:
            await generator.start_engine_loop()
        return generator

    yield start
    for thread in threading.enumerate():
        if thread.name.startswith("test-engine"):
            thread.join(timeout=_TIMEOUT_S)
            assert not thread.is_alive()


def _pulling_engine(monkeypatch, get_state_dict) -> _FakeEngine:
    """A fake engine whose weight pull reads TorchStore through `get_state_dict`."""
    monkeypatch.setattr(generator_module.ts, "get_state_dict", get_state_dict)
    monkeypatch.setattr(
        generator_module, "plain_tensor_to_dtensor_state_dict", lambda sd, **k: sd
    )
    monkeypatch.setattr(generator_module, "dtensor_to_plain_tensor_state_dict", dict)
    model = SimpleNamespace(
        model=SimpleNamespace(state_dict=dict, load_state_dict=lambda sd, strict: None),
        get_state_dict_layouts=dict,
        parallelism_context=None,
    )
    engine = _FakeEngine()
    engine.model_executor = SimpleNamespace(
        driver_worker=SimpleNamespace(get_model=lambda: model)
    )
    return engine


def _generate(generator: VLLMGenerator, request_id: str) -> asyncio.Task:
    return asyncio.create_task(
        generator.generate([1, 2], request_id=request_id, routing_session_id=request_id)
    )


def test_event_loop_admits_requests_while_engine_steps(engine_thread) -> None:
    in_step, release_step = threading.Event(), threading.Event()

    def block_first_step():
        in_step.set()
        assert release_step.wait(timeout=_TIMEOUT_S)

    async def run() -> None:
        generator = await engine_thread(_FakeEngine(block_first_step))
        first = _generate(generator, "r0")
        assert await asyncio.to_thread(in_step.wait, _TIMEOUT_S)

        # engine.step() is blocked on the engine thread; the event loop still takes requests.
        second = _generate(generator, "r1")
        await asyncio.sleep(0)
        assert [r.request_id for r in generator._queued_generation_requests] == ["r1"]

        release_step.set()
        completions = await asyncio.wait_for(asyncio.gather(first, second), _TIMEOUT_S)
        assert [c.request_id for c in completions] == ["r0", "r1"]
        assert generator._request_dispatcher._rank0_generation_futures == {}

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)
        assert generator._engine is None

    asyncio.run(run())


def test_cancelled_generate_does_not_strand_its_batch(engine_thread) -> None:
    in_step, release_step = threading.Event(), threading.Event()

    def block_first_step():
        in_step.set()
        assert release_step.wait(timeout=_TIMEOUT_S)

    async def run() -> None:
        generator = await engine_thread(_FakeEngine(block_first_step))
        first = _generate(generator, "r0")
        assert await asyncio.to_thread(in_step.wait, _TIMEOUT_S)

        # r1 and r2 finish in the same step, after r1's caller is gone.
        cancelled, survivor = _generate(generator, "r1"), _generate(generator, "r2")
        await asyncio.sleep(0)
        cancelled.cancel()
        release_step.set()

        completions = await asyncio.wait_for(
            asyncio.gather(first, survivor), _TIMEOUT_S
        )
        assert [c.request_id for c in completions] == ["r0", "r2"]
        assert cancelled.cancelled()
        assert generator._request_dispatcher._rank0_generation_futures == {}

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_engine_thread_crash_fails_outstanding_futures(engine_thread) -> None:
    def crash():
        raise RuntimeError("step failed")

    async def run() -> None:
        generator = await engine_thread(_FakeEngine(crash))
        with pytest.raises(RuntimeError, match="step failed"):
            await asyncio.wait_for(_generate(generator, "r0"), _TIMEOUT_S)
        with pytest.raises(RuntimeError, match="engine loop has stopped"):
            await _generate(generator, "r1")
        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_engine_build_failure_is_reported(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise SystemExit("build failed")

    monkeypatch.setattr(
        generator_module, "LLMEngine", SimpleNamespace(from_engine_args=fail)
    )
    generator = _bare_generator()
    with pytest.raises(SystemExit, match="build failed"):
        generator._start_engine_thread(None, None, name="test-engine")
    generator._engine_executor.shutdown()


def test_close_before_start_releases_engine_thread(engine_thread) -> None:
    async def run() -> None:
        generator = await engine_thread(_FakeEngine(), start_loop=False)
        await asyncio.wait_for(generator.close(), _TIMEOUT_S)
        assert generator._engine is None

    asyncio.run(run())


def test_pull_reads_torchstore_on_engine_thread(engine_thread, monkeypatch) -> None:
    endpoint_context = contextvars.ContextVar("endpoint_context", default=None)
    reads: list[tuple] = []

    async def get_state_dict(*args, **kwargs):
        reads.append(
            (
                threading.current_thread().name,
                asyncio.get_running_loop(),
                endpoint_context.get(),
            )
        )

    async def run() -> None:
        generator = await engine_thread(
            _pulling_engine(monkeypatch, get_state_dict), start_loop=False
        )
        # Set after the engine thread started, so only `start_engine_loop` can hand it over.
        endpoint_context.set("endpoint")
        await generator.start_engine_loop()

        for version in (4, 5):
            await asyncio.wait_for(generator.pull_model_state_dict(version), _TIMEOUT_S)
        assert generator.policy_version == 5
        assert generator._model_state_dict_pull_request is None

        # The loop moves on to STEP instead of pulling again.
        completion = await asyncio.wait_for(_generate(generator, "r0"), _TIMEOUT_S)
        assert completion.min_policy_version == completion.max_policy_version == 5
        # Both reads ran on the engine thread, on one reused loop, in the endpoint's context.
        assert reads == [("test-engine_0", generator._torchstore_loop, "endpoint")] * 2

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_cancelled_pull_still_clears_the_request(engine_thread, monkeypatch) -> None:
    read_started, release_read = threading.Event(), threading.Event()

    async def get_state_dict(*args, **kwargs):
        read_started.set()
        assert release_read.wait(timeout=_TIMEOUT_S)

    async def run() -> None:
        generator = await engine_thread(_pulling_engine(monkeypatch, get_state_dict))
        pull = asyncio.create_task(generator.pull_model_state_dict(4))
        assert await asyncio.to_thread(read_started.wait, _TIMEOUT_S)
        pull.cancel()
        release_read.set()

        completion = await asyncio.wait_for(_generate(generator, "r0"), _TIMEOUT_S)
        assert completion.max_policy_version == 4
        assert generator._model_state_dict_pull_request is None

        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())


def test_torchstore_read_failure_fails_the_pull(engine_thread, monkeypatch) -> None:
    async def get_state_dict(*args, **kwargs):
        raise RuntimeError("read failed")

    async def run() -> None:
        generator = await engine_thread(_pulling_engine(monkeypatch, get_state_dict))
        with pytest.raises(RuntimeError, match="read failed"):
            await asyncio.wait_for(generator.pull_model_state_dict(4), _TIMEOUT_S)
        with pytest.raises(RuntimeError, match="engine loop has stopped"):
            await _generate(generator, "r0")
        await asyncio.wait_for(generator.close(), _TIMEOUT_S)

    asyncio.run(run())
