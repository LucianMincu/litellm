from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Final, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.litellm_core_utils.internal_call_metadata import EvaluationBillingOwner
from litellm.proxy._types import Litellm_EntityType, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.hooks.model_max_budget_limiter import (
    ResolvedModelBudget,
    model_budget_spend_cache_key,
    model_budget_start_time_cache_key,
    resolve_model_budget,
)
from litellm.proxy.spend_tracking import budget_reservation
from litellm.proxy.spend_tracking.budget_reservation import (
    estimate_request_input_cost,
    estimate_request_max_cost,
    reserve_budget_for_request,
)
from litellm.router import Router
from litellm.router_utils.common_utils import resolve_model_group_alias
from litellm.types.utils import API_ROUTE_TO_CALL_TYPES, CallTypes

_NUMBER: Final = TypeAdapter(float)
_METADATA: Final = TypeAdapter(Mapping[str, object])
_REQUEST: Final = TypeAdapter(dict[str, object])
_HOLDS: Final = TypeAdapter(Mapping[str, float])
_RECONCILE: Final = TypeAdapter[Callable[[dict[str, object] | None, float | None], Awaitable[object]]](
    Callable[[dict[str, object] | None, float | None], Awaitable[object]]
).validate_python(
    budget_reservation.reconcile_budget_reservation  # pyright: ignore[reportUnknownMemberType]  # legacy reservation parameter is untyped
)
_LEASE_SECONDS: Final = 60
_LOCAL_HOLDS_LOCK: Final = Lock()
_LEASE_TASKS: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: asyncio only weakly references pending tasks
_Result: Final = TypeVar("_Result")

_HOLD_SCRIPT: Final = """
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if ARGV[1] == 'reserve' then
    redis.call('ZADD', KEYS[1], now + tonumber(ARGV[3]), ARGV[2])
elseif ARGV[1] == 'renew' then
    if not redis.call('ZSCORE', KEYS[1], ARGV[2]) then
        return redis.error_reply('Evaluation budget reservation expired')
    end
    redis.call('ZADD', KEYS[1], 'XX', now + tonumber(ARGV[3]), ARGV[2])
elseif ARGV[1] == 'release' then
    redis.call('ZREM', KEYS[1], ARGV[2])
end
if ARGV[1] == 'reserve' or ARGV[1] == 'renew' then
    redis.call('EXPIRE', KEYS[1], ARGV[3])
end
local total = 0
for _, member in ipairs(redis.call('ZRANGE', KEYS[1], 0, -1)) do
    total = total + tonumber(string.match(member, ':([^:]+)$'))
end
return tostring(total)
"""


@runtime_checkable
class _NumericCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object, *, ttl: int) -> object: ...

    def delete_cache(self, key: str) -> object: ...

    async def async_increment(self, key: str, value: float, *, ttl: int) -> float: ...


@runtime_checkable
class _Script(Protocol):
    async def __call__(self, *, keys: Sequence[str], args: Sequence[str | int]) -> object: ...


_NUMERIC_CACHE: Final = TypeAdapter(_NumericCache, config=ConfigDict(arbitrary_types_allowed=True))
_SCRIPT: Final = TypeAdapter(_Script, config=ConfigDict(arbitrary_types_allowed=True))


async def model_budget_settled_spend(cache: DualCache, spend_key: str) -> float:
    if cache.redis_cache is None:
        local: Final = _NUMERIC_CACHE.validate_python(cache.in_memory_cache)
        return _NUMBER.validate_python(local.get_cache(spend_key) or 0.0)
    script: Final = _SCRIPT.validate_python(
        cache.redis_cache.async_register_script("return redis.call('GET', KEYS[1])")
    )
    return _NUMBER.validate_python((await script(keys=(spend_key,), args=())) or 0.0)


async def model_budget_outstanding_spend(
    cache: DualCache,
    spend_key: str,
    *,
    operation: Literal["read", "reserve", "renew", "release"] = "read",
    member: str = "",
) -> float:
    key: Final = f"evaluation_holds:{spend_key}"
    if cache.redis_cache is not None:
        script: Final = _SCRIPT.validate_python(cache.redis_cache.async_register_script(_HOLD_SCRIPT))
        return _NUMBER.validate_python(await script(keys=(key,), args=(operation, member, _LEASE_SECONDS)))
    local: Final = _NUMERIC_CACHE.validate_python(cache.in_memory_cache)
    with _LOCAL_HOLDS_LOCK:
        now: Final = cache.in_memory_cache._clock()  # pyright: ignore[reportPrivateUsage]  # use the cache's injected clock for lease expiry
        current: Final = _HOLDS.validate_python(local.get_cache(key) or {})
        active: Final = {token: expiry for token, expiry in current.items() if expiry > now}
        if operation == "renew" and member not in active:
            raise RuntimeError("Evaluation budget reservation expired")
        added: Final = (
            {member: now + _LEASE_SECONDS}
            if operation == "reserve" or (operation == "renew" and member in active)
            else {}
        )
        kept: Final = {token: expiry for token, expiry in active.items() if token != member}
        updated: Final = {**kept, **added}
        local.delete_cache(key)
        if updated:
            local.set_cache(key, updated, ttl=_LEASE_SECONDS)
        return sum(float(token.rsplit(":", 1)[1]) for token in updated)


async def _drain(task: asyncio.Future[_Result]) -> _Result:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _complete(operation: Awaitable[_Result]) -> _Result:
    task: Final = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _drain(task)
        raise


async def _model_window(cache: DualCache, start_key: str, duration: int) -> int:
    if cache.redis_cache is not None:
        script: Final = _SCRIPT.validate_python(
            cache.redis_cache.async_register_script(
                "local now = redis.call('TIME'); "
                "redis.call('SET', KEYS[1], now[1], 'EX', ARGV[1], 'NX'); "
                "return math.max(1, redis.call('TTL', KEYS[1]))"
            )
        )
        return math.ceil(_NUMBER.validate_python(await script(keys=(start_key,), args=(duration,))))
    local: Final = _NUMERIC_CACHE.validate_python(cache.in_memory_cache)
    now: Final = time.time()
    cached: Final = local.get_cache(start_key)
    if cached is None:
        local.set_cache(key=start_key, value=now, ttl=duration)
    start: Final = _NUMBER.validate_python(now if cached is None else cached)
    return max(1, math.ceil(duration - (now - start)))


@dataclass(slots=True)
class EvaluationModelReservation:
    cache: DualCache
    spend_key: str
    start_key: str
    duration: int
    reserved_cost: float
    member: str = field(init=False)
    settled_cost: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.member = f"{uuid.uuid4()}:{self.reserved_cost}"

    async def settle(self, actual_cost: float) -> None:
        await _complete(self._settle(actual_cost))

    async def _settle(self, actual_cost: float) -> None:
        async with self.lock:
            adjustment: Final = max(actual_cost - (self.settled_cost or 0.0), 0.0)
            if adjustment:
                ttl: Final = await _model_window(self.cache, self.start_key, self.duration)
                backend: Final = _NUMERIC_CACHE.validate_python(self.cache.redis_cache or self.cache.in_memory_cache)
                await backend.async_increment(key=self.spend_key, value=adjustment, ttl=ttl)
            self.settled_cost = max(actual_cost, self.settled_cost or 0.0)
            await model_budget_outstanding_spend(self.cache, self.spend_key, operation="release", member=self.member)

    async def renew(self, request_task: asyncio.Task[object] | None) -> None:
        # Allow the request timeout again for queued success callbacks to settle the hold.
        deadline: Final = time.monotonic() + 2 * litellm.request_timeout
        while self.settled_cost is None:
            await asyncio.sleep(_LEASE_SECONDS / 2)
            async with self.lock:
                if self.settled_cost is not None:
                    return
                if time.monotonic() >= deadline:
                    if request_task is not None and not request_task.done():
                        request_task.cancel()
                    return
                try:
                    await model_budget_outstanding_spend(
                        self.cache, self.spend_key, operation="renew", member=self.member
                    )
                except Exception:  # noqa: BLE001  # any lease backend failure must stop an unreserved provider call
                    verbose_proxy_logger.exception("Unable to renew evaluation budget reservation")
                    if request_task is not None and not request_task.done():
                        request_task.cancel()
                    return


@dataclass(slots=True)
class EvaluationBudgetReservation:
    total: dict[str, object] | None = None  # mutable-ok: shared reservation is finalized by the spend writer
    model: EvaluationModelReservation | None = None
    input_cost: float = 0.0
    settled_failure_cost: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def settle_failure(self, actual_cost: float) -> None:
        await _complete(self._settle_failure(actual_cost))

    async def _settle_failure(self, actual_cost: float) -> None:
        async with self.lock:
            cost: Final = max(actual_cost, self.settled_failure_cost)
            total: Final = {**self.total, "finalized": False} if self.total is not None else None
            try:
                await _RECONCILE(total, cost)
                if self.total is not None:
                    self.total["finalized"] = True
            finally:
                if self.model is not None:
                    await self.model.settle(cost)
            self.settled_failure_cost = cost


def _pricing_model(request: Mapping[str, object], model: str, router: Router | None) -> str:
    if router is None:
        return model
    model_info: Final = _METADATA.validate_python(request.get("model_info") or {})
    model_id: Final = model_info.get("id")
    deployment: Final = router.get_deployment(model_id) if isinstance(model_id, str) else None
    if deployment is not None:
        return deployment.model_name
    return resolve_model_group_alias(router.model_group_alias, model) or model


async def reserve_evaluation_budget(
    owner: EvaluationBillingOwner, request: Mapping[str, object], call_type: str
) -> EvaluationBudgetReservation | None:
    task: Final = asyncio.create_task(_reserve_evaluation_budget(owner, request, call_type, asyncio.current_task()))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            reservation: Final = await _drain(task)
        except Exception:  # noqa: BLE001  # acquisition already rolled back; preserve the caller's cancellation
            raise asyncio.CancelledError from None
        await _complete(release_evaluation_budget(reservation))
        raise


async def _reserve_evaluation_budget(
    owner: EvaluationBillingOwner,
    request: Mapping[str, object],
    call_type: str,
    request_task: asyncio.Task[object] | None,
) -> EvaluationBudgetReservation | None:
    from litellm.proxy.proxy_server import (
        get_current_spend,
        llm_router,
        model_max_budget_limiter,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    metadata: Final = _METADATA.validate_python(request.get("litellm_metadata") or request.get("metadata") or {})
    model: Final = str(metadata.get("model_group") or request["model"])
    resolved: Final = resolve_model_budget(model, owner.user_model_max_budget or {})
    model_budget: Final = (
        resolved
        if resolved is not None
        and resolved.budget_config.max_budget is not None
        and math.isfinite(resolved.budget_config.max_budget)
        and resolved.budget_config.max_budget >= 0
        else None
    )
    total_budget: Final = owner.max_budget is not None and math.isfinite(owner.max_budget)
    if not total_budget and model_budget is None:
        return None
    if total_budget:
        current: Final = await get_current_spend(
            counter_key=f"spend:user:{owner.user_id}", fallback_spend=owner.spend, max_budget=owner.max_budget
        )
        if current >= _NUMBER.validate_python(owner.max_budget):
            raise litellm.BudgetExceededError(
                current_cost=current,
                max_budget=_NUMBER.validate_python(owner.max_budget),
                entity_type=Litellm_EntityType.USER.value,
                entity_id=owner.user_id,
            )
    route: Final = next(route for route, types in API_ROUTE_TO_CALL_TYPES.items() if CallTypes(call_type) in types)
    body: Final = _REQUEST.validate_python(
        {
            **request,
            "model": _pricing_model(request, model, llm_router),
            "metadata": {},
            "litellm_metadata": {},
            "tags": [],
        }
    )
    reservation: Final = EvaluationBudgetReservation()
    try:
        reservation.total = await reserve_budget_for_request(
            request_body=body,
            route=route,
            llm_router=llm_router,
            valid_token=UserAPIKeyAuth(user_id=owner.user_id),
            team_object=None,
            user_object=LiteLLM_UserTable(user_id=owner.user_id, max_budget=owner.max_budget, spend=owner.spend),
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            fail_closed_budget_enforcement=True,
            request_task=request_task,
        )
        estimate: Final = (
            _NUMBER.validate_python(reservation.total["reserved_cost"])
            if reservation.total is not None
            else estimate_request_max_cost(body, route, llm_router)
        )
        if estimate is None or not math.isfinite(estimate) or estimate < 0:
            raise ValueError("Evaluation budget cannot be checked for an unpriced model")
        if model_budget is not None:
            reservation.model = _model_reservation(owner, model_budget, estimate, model_max_budget_limiter.dual_cache)
            held: Final = await model_budget_outstanding_spend(
                reservation.model.cache,
                reservation.model.spend_key,
                operation="reserve",
                member=reservation.model.member,
            )
            spend: Final = await model_max_budget_limiter.get_spend_for_model_budget(
                Litellm_EntityType.USER, owner.user_id, model, model_budget, outstanding_spend=held
            )
            limit: Final = _NUMBER.validate_python(model_budget.budget_config.max_budget)
            if spend > limit or spend - estimate >= limit:
                raise litellm.BudgetExceededError(
                    current_cost=spend - estimate,
                    max_budget=limit,
                    entity_type=Litellm_EntityType.USER.value,
                    entity_id=owner.user_id,
                )
            lease: Final = asyncio.create_task(reservation.model.renew(request_task))
            _LEASE_TASKS.add(lease)
            lease.add_done_callback(_LEASE_TASKS.discard)
        reservation.input_cost = (
            _NUMBER.validate_python(reservation.total["input_cost"])
            if reservation.total is not None
            else estimate_request_input_cost(body, route, llm_router) or 0.0
        )
    except Exception:
        await reservation.settle_failure(0.0)
        raise
    return reservation


def _model_reservation(
    owner: EvaluationBillingOwner, resolved: ResolvedModelBudget, estimate: float, cache: DualCache
) -> EvaluationModelReservation:
    duration: Final = str(resolved.budget_config.budget_duration)
    return EvaluationModelReservation(
        cache=cache,
        spend_key=model_budget_spend_cache_key(Litellm_EntityType.USER, owner.user_id, resolved.budget_model, duration),
        start_key=model_budget_start_time_cache_key(
            Litellm_EntityType.USER, owner.user_id, resolved.budget_model, duration
        ),
        duration=duration_in_seconds(duration),
        reserved_cost=estimate,
    )


async def release_evaluation_budget(
    reservation: EvaluationBudgetReservation | None, *, cancelled: bool = False, actual_cost: float = 0.0
) -> None:
    if reservation is not None:
        await reservation.settle_failure(max(reservation.input_cost if cancelled else 0.0, actual_cost))
