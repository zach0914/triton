from __future__ import annotations

import functools
import inspect
import itertools
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, TypeVar, overload


F = TypeVar("F", bound=Callable[..., Any])
_ACTIVE_BUILDER: ContextVar[Any | None] = ContextVar("timely_builder", default=None)


class ScheduleValue:
    """Base class for statically evaluable scheduling expressions."""

    def evaluate(self, bindings: Mapping[str, int] | None = None) -> int:
        raise NotImplementedError

    def __add__(self, other: int | ScheduleValue) -> ScheduleExpr:
        return ScheduleExpr("add", self, _as_schedule_value(other))

    def __radd__(self, other: int | ScheduleValue) -> ScheduleExpr:
        return ScheduleExpr("add", _as_schedule_value(other), self)

    def __sub__(self, other: int | ScheduleValue) -> ScheduleExpr:
        return ScheduleExpr("sub", self, _as_schedule_value(other))

    def __rsub__(self, other: int | ScheduleValue) -> ScheduleExpr:
        return ScheduleExpr("sub", _as_schedule_value(other), self)

    def __mul__(self, other: int | ScheduleValue) -> ScheduleExpr:
        return ScheduleExpr("mul", self, _as_schedule_value(other))

    def __rmul__(self, other: int | ScheduleValue) -> ScheduleExpr:
        return ScheduleExpr("mul", _as_schedule_value(other), self)

    def __floordiv__(self, other: int | ScheduleValue) -> ScheduleExpr:
        return ScheduleExpr("floordiv", self, _as_schedule_value(other))

    def __rfloordiv__(self, other: int | ScheduleValue) -> ScheduleExpr:
        return ScheduleExpr("floordiv", _as_schedule_value(other), self)


@dataclass(frozen=True)
class Const(ScheduleValue):
    """A compile-time constant used by a logical-time formula."""

    value: int
    name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("Timely Const values must be integers")

    def evaluate(self, bindings: Mapping[str, int] | None = None) -> int:
        if self.name is not None and bindings is not None and self.name in bindings:
            return bindings[self.name]
        return self.value

    def __int__(self) -> int:
        return self.value


@dataclass(frozen=True)
class ScheduleExpr(ScheduleValue):
    op: str
    lhs: ScheduleValue
    rhs: ScheduleValue

    def evaluate(self, bindings: Mapping[str, int] | None = None) -> int:
        lhs = self.lhs.evaluate(bindings)
        rhs = self.rhs.evaluate(bindings)
        if self.op == "add":
            return lhs + rhs
        if self.op == "sub":
            return lhs - rhs
        if self.op == "mul":
            return lhs * rhs
        if self.op == "floordiv":
            if rhs == 0:
                raise ValueError("logical-time formula divides by zero")
            return lhs // rhs
        raise ValueError(f"unsupported scheduling operation: {self.op}")


@dataclass(frozen=True)
class Time:
    """A logical issue-time expression, not a physical duration."""

    expression: ScheduleValue = Const(0)

    def __init__(self, value: int | ScheduleValue = 0):
        object.__setattr__(self, "expression", _as_schedule_value(value))

    def evaluate(self, bindings: Mapping[str, int] | None = None) -> int:
        return self.expression.evaluate(bindings)

    def __add__(self, other: int | ScheduleValue) -> Time:
        return Time(self.expression + other)

    def __sub__(self, other: int | ScheduleValue) -> Time:
        return Time(self.expression - other)


def _as_schedule_value(value: int | ScheduleValue) -> ScheduleValue:
    if isinstance(value, ScheduleValue):
        return value
    return Const(value)


@dataclass(frozen=True)
class Domain:
    """A finite unordered domain whose iteration order has no schedule meaning."""

    bounds: tuple[tuple[int, int], ...]

    def __iter__(self) -> Iterator[int | tuple[int, ...]]:
        ranges = [range(lower, upper) for lower, upper in self.bounds]
        points = itertools.product(*ranges)
        if len(ranges) == 1:
            return (point[0] for point in points)
        return points

    def __len__(self) -> int:
        result = 1
        for lower, upper in self.bounds:
            result *= upper - lower
        return result


def domain(*bounds: int | Const | Sequence[int | Const]) -> Domain:
    if not bounds:
        raise ValueError("tm.domain requires at least one bound")
    normalized: list[tuple[int, int]] = []
    for bound in bounds:
        if isinstance(bound, Const):
            bound = bound.evaluate()
        if isinstance(bound, bool):
            raise TypeError("tm.domain bounds must be compile-time integers")
        if isinstance(bound, int):
            lower, upper = 0, bound
        elif isinstance(bound, Sequence) and len(bound) == 2:
            lower, upper = bound
        else:
            raise TypeError("each tm.domain bound must be upper or (lower, upper)")
        if isinstance(lower, Const):
            lower = lower.evaluate()
        if isinstance(upper, Const):
            upper = upper.evaluate()
        if not isinstance(lower, int) or not isinstance(upper, int):
            raise TypeError("tm.domain bounds must be compile-time integers")
        if lower >= upper:
            raise ValueError("tm.domain lower bounds must be smaller than upper bounds")
        normalized.append((lower, upper))
    result = Domain(tuple(normalized))
    builder = _ACTIVE_BUILDER.get()
    if builder is not None:
        builder.record_domain(result)
    return result


@dataclass(frozen=True)
class ResourceSpec:
    resource_class: str
    sms: int = 1
    threads: int = 32
    warps: int = 1
    shared_memory_bytes: int = 0

    def __post_init__(self) -> None:
        if self.resource_class not in ("communication", "compute"):
            raise ValueError("resource_class must be 'communication' or 'compute'")
        if self.sms <= 0 or self.threads <= 0 or self.warps <= 0:
            raise ValueError("sms, threads, and warps must be positive")
        if self.shared_memory_bytes < 0:
            raise ValueError("shared_memory_bytes must be non-negative")


class _DecoratedFunction:
    def __init__(self, fn: F):
        if not inspect.isfunction(fn):
            raise TypeError("Timely decorators require a Python function")
        self.fn = fn
        self.signature = inspect.signature(fn)
        functools.update_wrapper(self, fn)


class Kernel(_DecoratedFunction):
    kind = "kernel"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("a Timely kernel must be compiled into a task plan before execution")

    def _capture(self, *args: Any, signature: Mapping[str, Any] | None = None,
                 target: Any = None, options: Mapping[str, Any] | None = None, **kwargs: Any):
        from .compiler import GraphBuilder, prepare_kernel_arguments

        call_args, call_kwargs, constants = prepare_kernel_arguments(
            self.signature, args, kwargs, compile_signature=signature
        )
        builder = GraphBuilder(self.__name__, constants, target=target, options=options)
        token = _ACTIVE_BUILDER.set(builder)
        try:
            self.fn(*call_args, **call_kwargs)
        finally:
            _ACTIVE_BUILDER.reset(token)
        return builder

    def lower(self, *args: Any, signature: Mapping[str, Any] | None = None,
              target: Any = None, options: Mapping[str, Any] | None = None, **kwargs: Any) -> str:
        """Capture a static task graph and return verified typed TTIR/TM MLIR."""
        return self._capture(
            *args, signature=signature, target=target, options=options, **kwargs
        ).to_mlir()

    def _lower_native_plan(self, builder: Any, target_capacity: Any = None):
        """Run the sole TM planning pipeline and return its native module."""
        from triton._C.libtriton import ir, passes
        from .runtime import TargetCapacity

        capacity = target_capacity or TargetCapacity()
        module = builder.build_analysis_module()
        context = module.context
        manager = ir.pass_manager(context)
        passes.timely.add_normalize_issue_time(manager)
        passes.timely.add_build_dependency_graph(manager)
        passes.timely.add_plan_resources(
            manager,
            capacity.sms,
            capacity.threads,
            capacity.warps,
            capacity.shared_memory_bytes,
        )
        passes.timely.add_materialize_synchronization(manager)
        manager.run(module, "timely_plan")
        return module

    def lower_plan(self, *args: Any, signature: Mapping[str, Any] | None = None,
                   target: Any = None, options: Mapping[str, Any] | None = None,
                   target_capacity: Any = None, **kwargs: Any) -> str:
        """Run the native TM analysis pipeline and return inspectable plan IR."""
        builder = self._capture(
            *args, signature=signature, target=target, options=options, **kwargs
        )
        return self._lower_native_plan(builder, target_capacity).str_nodebug()

    def plan(self, *args: Any, signature: Mapping[str, Any] | None = None,
             target: Any = None, options: Mapping[str, Any] | None = None,
             target_capacity: Any = None, **kwargs: Any):
        from .runtime import execution_plan_from_descriptor

        builder = self._capture(
            *args, signature=signature, target=target, options=options, **kwargs
        )
        module = self._lower_native_plan(builder, target_capacity)
        descriptor = module.get_timely_plan_descriptor()
        return execution_plan_from_descriptor(descriptor, builder)


class Task(_DecoratedFunction):
    kind = "task"

    def __init__(self, fn: F, *, reads: Iterable[str] = (), writes: Iterable[str] = (),
                 depends_on: Iterable[str] = (), resources: ResourceSpec | None = None,
                 reference: Callable[..., Any] | None = None,
                 grid: Any = None, launch_options: Mapping[str, Any] | None = None):
        super().__init__(fn)
        self.reads = _normalize_annotations(reads, "reads")
        self.writes = _normalize_annotations(writes, "writes")
        self.depends_on = _normalize_annotations(depends_on, "depends_on")
        self.resources = resources or ResourceSpec("compute")
        self.reference = reference
        self.grid = grid
        self.launch_options = dict(launch_options or {})
        if self.resources.resource_class != "compute":
            raise ValueError("@tm.task requires compute resources")
        self._triton_jit = None

    @property
    def triton_jit(self):
        if self._triton_jit is None:
            from triton import jit
            self._triton_jit = jit(self.fn)
        return self._triton_jit

    def compile(self, signature: Mapping[str, str], *, target: Any,
                constants: Mapping[str, Any] | None = None,
                options: Mapping[str, Any] | None = None):
        """Outline this task through Triton's ordinary compilation pipeline."""
        from triton import compile
        from triton.compiler import ASTSource

        source = ASTSource(
            fn=self.triton_jit,
            signature=dict(signature),
            constexprs=dict(constants or {}),
        )
        return compile(source, target=target, options=dict(options or {}))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        builder = _ACTIVE_BUILDER.get()
        if builder is None:
            raise RuntimeError("a Timely task may only be called while building a Timely kernel")
        at = kwargs.pop("at", None)
        reads = kwargs.pop("reads", self.reads)
        writes = kwargs.pop("writes", self.writes)
        depends_on = kwargs.pop("depends_on", self.depends_on)
        resources = kwargs.pop("resources", self.resources)
        task_id = kwargs.pop("id", None)
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return builder.record_task(
            self,
            bound.arguments,
            at=at,
            reads=reads,
            writes=writes,
            depends_on=depends_on,
            resources=resources,
            task_id=task_id,
        )


def _normalize_annotations(values: Iterable[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise TypeError(f"{name} annotations must be non-empty strings")
    return result


@overload
def kernel(fn: F) -> Kernel: ...


@overload
def kernel(fn: None = None) -> Callable[[F], Kernel]: ...


def kernel(fn: F | None = None):
    def decorate(inner: F) -> Kernel:
        return Kernel(inner)
    return decorate(fn) if fn is not None else decorate


@overload
def task(fn: F) -> Task: ...


@overload
def task(*, reads: Iterable[str] = (), writes: Iterable[str] = (),
         depends_on: Iterable[str] = (), resources: ResourceSpec | None = None,
         reference: Callable[..., Any] | None = None, grid: Any = None,
         launch_options: Mapping[str, Any] | None = None) -> Callable[[F], Task]: ...


def task(fn: F | None = None, *, reads: Iterable[str] = (), writes: Iterable[str] = (),
         depends_on: Iterable[str] = (), resources: ResourceSpec | None = None,
         reference: Callable[..., Any] | None = None, grid: Any = None,
         launch_options: Mapping[str, Any] | None = None):
    def decorate(inner: F) -> Task:
        return Task(inner, reads=reads, writes=writes, depends_on=depends_on,
                    resources=resources, reference=reference, grid=grid,
                    launch_options=launch_options)
    return decorate(fn) if fn is not None else decorate


def load(source: Any) -> Any:
    """Declare a task input without assigning completion semantics to the load."""
    return source


def store(target: Any, value: Any) -> None:
    """Reject the legacy compute-return store form.

    Device writes belong in the Triton task body and use an explicit output
    pointer.  This helper remains temporarily to provide a focused migration
    diagnostic to existing Timely programs.
    """
    builder = _ACTIVE_BUILDER.get()
    if builder is None:
        raise RuntimeError("tm.store may only be used while building a Timely kernel")
    builder.record_store(target, value)


def allgather_shard(source: Any, *, at: Time, reads: Iterable[str] = (),
                    writes: Iterable[str] = (), depends_on: Iterable[Any] = (),
                    resources: ResourceSpec | None = None, id: str | None = None):
    builder = _ACTIVE_BUILDER.get()
    if builder is None:
        raise RuntimeError(
            "tm.allgather_shard may only be used while building a Timely kernel"
        )
    return builder.record_allgather(
        source,
        at=at,
        reads=reads,
        writes=writes,
        depends_on=depends_on,
        resources=resources or ResourceSpec("communication"),
        task_id=id,
    )
