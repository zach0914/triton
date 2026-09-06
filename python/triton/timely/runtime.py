from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .compiler import (
    AsyncValue,
    Buffer,
    BufferRef,
    Completion,
    GraphBuilder,
    ScalarBinding,
    ScalarLiteral,
    TaskNode,
    TaskSpecialization,
    _freeze,
    _specialize_task_argument,
)
from .core import Const


@dataclass(frozen=True, order=True)
class Edge:
    source: str
    target: str
    kind: str


@dataclass(frozen=True)
class Synchronization:
    source: str
    target: str
    kind: str
    scope: str


@dataclass(frozen=True)
class PlanNodeDescriptor:
    id: str
    kind: str
    raw_time: int
    issue_rank: int
    callee: str = ""
    specialization: str = ""


@dataclass(frozen=True)
class PlanDescriptor:
    """Immutable structured binding of the sole native tm.plan."""

    nodes: tuple[PlanNodeDescriptor, ...]
    issue_layers: tuple[tuple[str, ...], ...]
    time_order: tuple[Edge, ...]
    data_deps: tuple[Edge, ...]
    resource_order: tuple[Edge, ...]
    synchronizations: tuple[Synchronization, ...]

    @classmethod
    def from_binding(cls, value: Mapping[str, Any]) -> PlanDescriptor:
        def edges(name: str) -> tuple[Edge, ...]:
            return tuple(
                Edge(str(edge["source"]), str(edge["target"]), str(edge["kind"]))
                for edge in value[name]
            )

        nodes = tuple(
            PlanNodeDescriptor(
                id=str(node["id"]),
                kind=str(node["kind"]),
                raw_time=int(node["raw_time"]),
                issue_rank=int(node["issue_rank"]),
                callee=str(node["callee"]),
                specialization=str(node["specialization"]),
            )
            for node in value["nodes"]
        )
        layer_records = tuple(value["issue_layers"])
        ranks = tuple(int(layer["rank"]) for layer in layer_records)
        if ranks != tuple(range(len(layer_records))):
            raise ValueError(
                "native tm.plan issue-layer ranks must be dense and ordered"
            )
        issue_layers = tuple(
            tuple(str(task) for task in layer["tasks"])
            for layer in layer_records
        )
        descriptor = cls(
            nodes,
            issue_layers,
            edges("time_order"),
            edges("data_deps"),
            edges("resource_order"),
            tuple(
                Synchronization(
                    str(sync["source"]), str(sync["target"]),
                    str(sync["kind"]), str(sync["scope"]),
                )
                for sync in value["synchronizations"]
            ),
        )
        descriptor._verify()
        return descriptor

    def _verify(self) -> None:
        node_ids = tuple(node.id for node in self.nodes)
        known = set(node_ids)
        if not node_ids or len(known) != len(node_ids):
            raise ValueError("native tm.plan must contain unique nodes")
        flattened = tuple(node for layer in self.issue_layers for node in layer)
        if len(flattened) != len(known) or set(flattened) != known:
            raise ValueError(
                "native tm.plan issue layers must contain every node exactly once"
            )
        ranks = {node.id: node.issue_rank for node in self.nodes}
        for rank, layer in enumerate(self.issue_layers):
            if any(ranks[node] != rank for node in layer):
                raise ValueError("native tm.plan node rank disagrees with issue layers")
        for node in self.nodes:
            if node.kind == "compute":
                if not node.callee or not node.specialization:
                    raise ValueError(
                        f"native tm.plan compute node '{node.id}' lacks callee specialization identity"
                    )
            elif node.kind == "communication":
                if node.callee or node.specialization:
                    raise ValueError(
                        f"native tm.plan communication node '{node.id}' has a compute specialization"
                    )
            else:
                raise ValueError(f"native tm.plan node '{node.id}' has unknown kind '{node.kind}'")
        for edge in (*self.time_order, *self.data_deps, *self.resource_order):
            if edge.source not in known or edge.target not in known:
                raise ValueError("native tm.plan edge refers to an unknown node")
        for sync in self.synchronizations:
            if sync.source not in known or sync.target not in known:
                raise ValueError("native tm.plan synchronization refers to an unknown node")
        data_pairs = {(edge.source, edge.target) for edge in self.data_deps}
        sync_pairs = {(sync.source, sync.target) for sync in self.synchronizations}
        if data_pairs != sync_pairs:
            raise ValueError("native tm.plan synchronizations must cover exactly its DataDep pairs")


@dataclass(frozen=True)
class TargetCapacity:
    sms: int = 128
    threads: int = 2048
    warps: int = 64
    shared_memory_bytes: int = 233472

    def __post_init__(self) -> None:
        if self.sms <= 0 or self.threads <= 0 or self.warps <= 0:
            raise ValueError("target SM, thread, and warp capacities must be positive")
        if self.shared_memory_bytes < 0:
            raise ValueError("target shared memory capacity must be non-negative")


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    node: str
    detail: str | None
    timestamp: float


@dataclass(frozen=True)
class ExecutionResult:
    trace: tuple[TraceEvent, ...]
    values: Mapping[str, Any]


@dataclass(frozen=True)
class CudaNodeTiming:
    node: str
    resource_class: str
    start_ms: float
    end_ms: float


@dataclass(frozen=True)
class CudaExecutionResult:
    issue_order: tuple[str, ...]
    timings: tuple[CudaNodeTiming, ...]
    elapsed_ms: float
    values: Mapping[str, Any]


@dataclass
class SpecializationArtifact:
    record: TaskSpecialization
    task: Any
    kernel: Any
    compiled: Any = None
    compile_lock: Any = field(default_factory=threading.Lock, repr=False)


@dataclass(frozen=True)
class ExecutionPlan:
    descriptor: PlanDescriptor
    node_registry: Mapping[str, TaskNode]
    specialization_registry: Mapping[str, TaskSpecialization]
    node_specializations: Mapping[str, str]
    task_registry: Mapping[str, Any]
    kernel_registry: Mapping[str, Any]
    artifact_registry: Mapping[str, SpecializationArtifact]

    @property
    def nodes(self) -> tuple[TaskNode, ...]:
        return tuple(self.node_registry[node.id] for node in self.descriptor.nodes)

    @property
    def issue_layers(self) -> tuple[tuple[str, ...], ...]:
        return self.descriptor.issue_layers

    @property
    def issue_ranks(self) -> Mapping[str, int]:
        return MappingProxyType({
            node.id: node.issue_rank for node in self.descriptor.nodes
        })

    @property
    def time_order(self) -> tuple[Edge, ...]:
        return self.descriptor.time_order

    @property
    def data_deps(self) -> tuple[Edge, ...]:
        return self.descriptor.data_deps

    @property
    def resource_order(self) -> tuple[Edge, ...]:
        return self.descriptor.resource_order

    @property
    def synchronizations(self) -> tuple[Synchronization, ...]:
        return self.descriptor.synchronizations

    def specialization_for(self, node_id: str) -> TaskSpecialization:
        try:
            identity = self.node_specializations[node_id]
            return self.specialization_registry[identity]
        except KeyError as exc:
            raise RuntimeError(
                f"compute node '{node_id}' has no registered Timely specialization"
            ) from exc

    def execute(self, bindings: Mapping[str, Any], *,
                communication_backend: AsyncShardCopyBackend | None = None) -> ExecutionResult:
        executor = ReferenceExecutor(
            self,
            bindings,
            communication_backend or AsyncShardCopyBackend(),
        )
        return executor.run()

    def execute_cuda(self, bindings: Mapping[str, Any], *,
                     overlap: bool = True) -> CudaExecutionResult:
        return CudaPlanExecutor(self, bindings, overlap=overlap).run()


class AsyncShardCopyBackend:
    """Controllable asynchronous copy used to test issue/dependency semantics."""

    def __init__(self, delays: Mapping[str, float] | None = None):
        self.delays = dict(delays or {})

    def copy(self, node_id: str, value: Any) -> Any:
        delay = self.delays.get(node_id, 0.0)
        if delay < 0:
            raise ValueError("communication delay must be non-negative")
        if delay:
            time.sleep(delay)
        if hasattr(value, "copy"):
            return value.copy()
        return copy.deepcopy(value)


def execution_plan_from_descriptor(value: Mapping[str, Any],
                                   builder: GraphBuilder) -> ExecutionPlan:
    descriptor = PlanDescriptor.from_binding(value)
    node_registry = {node.id: node for node in builder.nodes}
    descriptor_ids = {node.id for node in descriptor.nodes}
    if descriptor_ids != set(node_registry):
        raise ValueError("native tm.plan nodes disagree with captured launch payloads")
    for node in descriptor.nodes:
        captured_kind = (
            "communication"
            if node_registry[node.id].kind == "allgather"
            else "compute"
        )
        if node.kind != captured_kind:
            raise ValueError(f"native tm.plan kind disagrees for task '{node.id}'")
        if node.kind != "compute":
            if node.id in builder.node_specializations:
                raise ValueError(
                    f"native tm.plan communication task '{node.id}' has a captured specialization"
                )
            continue
        captured = node_registry[node.id]
        identity = builder.node_specializations.get(node.id)
        if identity is None or captured.specialization is None:
            raise ValueError(f"native tm.plan compute task '{node.id}' is not specialized")
        if node.specialization != identity or captured.specialization.identity != identity:
            raise ValueError(
                f"native tm.plan specialization disagrees for task '{node.id}': "
                f"descriptor={node.specialization!r}, captured={identity!r}"
            )
        record = builder.specialization_registry.get(identity)
        if record is None:
            raise ValueError(
                f"native tm.plan specialization '{identity}' for task '{node.id}' is not registered"
            )
        if node.callee != record.callee or captured.callee != record.callee:
            raise ValueError(
                f"native tm.plan callee disagrees for task '{node.id}': "
                f"descriptor=@{node.callee}, captured=@{record.callee}"
            )
        if identity not in builder.task_registry:
            raise ValueError(
                f"native tm.plan compute task specialization '{identity}' is not registered"
            )
        if identity not in builder.kernel_registry:
            raise ValueError(
                f"native tm.plan Triton kernel specialization '{identity}' is not registered"
            )
    specialization_registry = dict(builder.specialization_registry)
    task_registry = dict(builder.task_registry)
    kernel_registry = dict(builder.kernel_registry)
    artifacts = {
        identity: SpecializationArtifact(
            record,
            task_registry[identity],
            kernel_registry[identity],
        )
        for identity, record in specialization_registry.items()
    }
    return ExecutionPlan(
        descriptor,
        MappingProxyType(node_registry),
        MappingProxyType(specialization_registry),
        MappingProxyType(dict(builder.node_specializations)),
        MappingProxyType(task_registry),
        MappingProxyType(kernel_registry),
        MappingProxyType(artifacts),
    )


def _resolve_binding(value: Any, bindings: Mapping[str, Any]) -> Any:
    if isinstance(value, Buffer):
        if value.name in bindings:
            return bindings[value.name]
        if value.runtime_value is not None:
            return value.runtime_value
        raise KeyError(f"missing Timely runtime binding '{value.name}'")
    if isinstance(value, BufferRef):
        root = bindings.get(value.buffer, value.runtime_value)
        if root is None:
            raise KeyError(f"missing Timely runtime binding '{value.buffer}'")
        return root[value.indices]
    if isinstance(value, ScalarBinding):
        if value.name in bindings:
            return bindings[value.name]
        if value.runtime_value is not None:
            return value.runtime_value
        raise KeyError(f"missing Timely runtime binding '{value.name}'")
    if isinstance(value, ScalarLiteral):
        return value.value
    if isinstance(value, Completion):
        raise TypeError(
            f"compute completion from '{value.producer}' cannot be resolved as task data"
        )
    return value


def _ordered_task_arguments(record: TaskSpecialization,
                            dynamic_values: list[Any]) -> tuple[Any, ...]:
    if len(dynamic_values) != len(record.dynamic_signature):
        raise RuntimeError(
            f"specialization '{record.identity}' expects {len(record.dynamic_signature)} "
            f"dynamic arguments, got {len(dynamic_values)}"
        )
    dynamic = dict(zip(record.launch_operand_order, dynamic_values))
    constexprs = dict(record.constexprs)
    arguments: list[Any] = []
    for name in record.parameter_order:
        if name in dynamic:
            arguments.append(dynamic[name])
        elif name in constexprs:
            arguments.append(constexprs[name])
        else:
            raise RuntimeError(
                f"specialization '{record.identity}' has no value for parameter '{name}'"
            )
    return tuple(arguments)


def _validate_launch_signature(record: TaskSpecialization,
                               dynamic_values: list[Any]) -> None:
    if len(dynamic_values) != len(record.dynamic_signature):
        raise RuntimeError(
            f"callee '@{record.callee}' expects {len(record.dynamic_signature)} dynamic "
            f"arguments, got {len(dynamic_values)}"
        )
    from triton.compiler import make_backend
    from triton.runtime.jit import mangle_type

    backend = make_backend(record.target_object)
    parameters = {parameter.name: parameter for parameter in record.task.triton_jit.params}
    planned_specializations = dict(record.runtime_specialization)

    for index, ((name, expected), value) in enumerate(
            zip(record.dynamic_signature, dynamic_values)):
        try:
            actual = str(mangle_type(value))
        except Exception as exc:
            raise TypeError(
                f"cannot determine runtime type for '@{record.callee}' argument "
                f"{index} ('{name}')"
            ) from exc
        if actual != expected:
            raise TypeError(
                f"callee '@{record.callee}' argument {index} ('{name}') type mismatch: "
                f"expected {expected}, got {actual}"
            )
        parameter = parameters.get(name)
        planned = planned_specializations.get(name)
        if parameter is None or planned is None:
            raise RuntimeError(
                f"specialization '{record.identity}' has no binder metadata for "
                f"argument {index} ('{name}')"
            )
        # An explicit signature or a derived view may deliberately select the
        # conservative, attribute-free variant. Extra runtime alignment does
        # not invalidate that artifact. A planned binder attribute, however,
        # must still hold for the value that is about to be launched.
        if len(planned) < 2 or planned[1] is None:
            continue
        specialized = tuple(
            _freeze(item)
            for item in _specialize_task_argument(
                backend, parameter, value, actual
            )
        )
        if specialized != planned:
            raise TypeError(
                f"callee '@{record.callee}' argument {index} ('{name}') "
                f"specialization mismatch: expected {planned!r}, got {specialized!r}"
            )


def _registered_specialization(plan: ExecutionPlan, node_id: str):
    record = plan.specialization_for(node_id)
    identity = record.identity
    try:
        task = plan.task_registry[identity]
        kernel = plan.kernel_registry[identity]
        artifact = plan.artifact_registry[identity]
    except KeyError as exc:
        raise RuntimeError(
            f"specialization '{identity}' for compute node '{node_id}' has no launch artifact"
        ) from exc
    if (task is not record.task or kernel is not task.triton_jit or
            artifact.record is not record or artifact.task is not task or
            artifact.kernel is not kernel):
        raise RuntimeError(
            f"specialization '{identity}' for compute node '{node_id}' has a stale launch artifact"
        )
    return artifact


def _compile_specialization(artifact: SpecializationArtifact):
    if artifact.compiled is not None:
        return artifact.compiled
    with artifact.compile_lock:
        if artifact.compiled is not None:
            return artifact.compiled
        from triton import compile
        from triton.compiler import ASTSource

        record = artifact.record
        source = ASTSource(
            fn=artifact.kernel,
            signature=record.signature,
            constexprs=record.constants,
            attrs=record.attrs,
        )
        artifact.compiled = compile(
            source,
            target=record.target_object,
            options=dict(record.codegen_options),
        )
        return artifact.compiled


def _launch_grid(task: Any, record: TaskSpecialization,
                 arguments: tuple[Any, ...]) -> tuple[int, int, int]:
    grid = task.grid
    if callable(grid):
        grid = grid(dict(zip(record.parameter_order, arguments)))
    if isinstance(grid, int):
        grid = (grid,)
    if not isinstance(grid, (tuple, list)) or not 1 <= len(grid) <= 3:
        raise ValueError(
            f"task specialization '{record.identity}' requires a one-to-three dimensional launch grid"
        )
    values = tuple(int(value) for value in grid)
    if any(value <= 0 for value in values):
        raise ValueError(
            f"task specialization '{record.identity}' launch grid must be positive"
        )
    return values + (1,) * (3 - len(values))


def _execution_dependencies(plan: ExecutionPlan) -> dict[str, set[str]]:
    dependencies = {node.id: set() for node in plan.descriptor.nodes}
    data_pairs = {(edge.source, edge.target) for edge in plan.data_deps}
    sync_pairs = {(sync.source, sync.target) for sync in plan.synchronizations}
    if data_pairs != sync_pairs:
        raise RuntimeError("native tm.plan has inconsistent DataDep synchronization")
    if any(sync.kind != "completion_event" or sync.scope != "cross_task"
           for sync in plan.synchronizations):
        raise RuntimeError("executor does not support a native synchronization record")
    for source, target in sync_pairs:
        dependencies[target].add(source)
    for edge in plan.resource_order:
        dependencies[edge.target].add(edge.source)
    return dependencies


class ReferenceExecutor:
    def __init__(self, plan: ExecutionPlan, bindings: Mapping[str, Any],
                 communication_backend: AsyncShardCopyBackend):
        self.plan = plan
        self.bindings = dict(bindings)
        self.communication_backend = communication_backend
        self._trace: list[TraceEvent] = []
        self._trace_lock = threading.Lock()

    def _record(self, kind: str, node: str, detail: str | None = None) -> None:
        with self._trace_lock:
            self._trace.append(TraceEvent(kind, node, detail, time.monotonic()))

    def _resolve(self, value: Any, promises: Mapping[str, Future]) -> Any:
        if isinstance(value, AsyncValue):
            return promises[value.producer].result()
        return _resolve_binding(value, self.bindings)

    def run(self) -> ExecutionResult:
        nodes = self.plan.node_registry
        promises = {node.id: Future() for node in self.plan.nodes}
        dependencies = _execution_dependencies(self.plan)

        def run_node(node: TaskNode) -> None:
            try:
                for dependency in sorted(dependencies[node.id]):
                    if not promises[dependency].done():
                        self._record("wait", node.id, dependency)
                    promises[dependency].result()
                self._record("start", node.id)
                inputs = [self._resolve(value, promises) for value in node.inputs]
                if node.kind == "allgather":
                    result = self.communication_backend.copy(node.id, inputs[0])
                else:
                    artifact = _registered_specialization(self.plan, node.id)
                    record, task = artifact.record, artifact.task
                    if task.reference is None:
                        raise RuntimeError(
                            f"task '{node.id}' has no reference implementation"
                        )
                    result = task.reference(*_ordered_task_arguments(record, inputs))
                self._record("finish", node.id)
                promises[node.id].set_result(result)
            except BaseException as error:
                self._record("error", node.id, str(error))
                promises[node.id].set_exception(error)

        communication_count = sum(
            node.kind == "allgather" for node in self.plan.nodes
        )
        compute_count = len(self.plan.nodes) - communication_count
        with ThreadPoolExecutor(max_workers=max(1, communication_count)) as communication_pool, \
                ThreadPoolExecutor(max_workers=max(1, compute_count)) as compute_pool:
            previous_layer: tuple[str, ...] = ()
            for rank, layer in enumerate(self.plan.issue_layers):
                if previous_layer:
                    completed, _ = wait(
                        tuple(promises[node_id] for node_id in previous_layer),
                        return_when=FIRST_COMPLETED,
                    )
                    completed_ids = sorted(
                        node_id
                        for node_id in previous_layer
                        if promises[node_id] in completed
                    )
                    promises[completed_ids[0]].result()
                    self._record("open_layer", str(rank), completed_ids[0])
                for node_id in layer:
                    node = nodes[node_id]
                    self._record("issue", node.id, str(self.plan.issue_ranks[node.id]))
                    pool = (
                        communication_pool
                        if node.kind == "allgather"
                        else compute_pool
                    )
                    pool.submit(run_node, node)
                previous_layer = layer
            for node in self.plan.nodes:
                promises[node.id].result()

        values = {node.id: promises[node.id].result() for node in self.plan.nodes}
        return ExecutionResult(tuple(self._trace), values)


class CudaPlanExecutor:
    """CUDA stream/event executor for the local shard-copy backend."""

    def __init__(self, plan: ExecutionPlan, bindings: Mapping[str, Any], *,
                 overlap: bool = True):
        self.plan = plan
        self.bindings = dict(bindings)
        self.overlap = overlap

    def _resolve(self, value: Any, values: Mapping[str, Any]) -> Any:
        if isinstance(value, AsyncValue):
            return values[value.producer]
        if isinstance(value, Const):
            return value.evaluate()
        return _resolve_binding(value, self.bindings)

    def run(self) -> CudaExecutionResult:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for Timely plan execution")
        nodes = self.plan.node_registry
        dependencies = _execution_dependencies(self.plan)

        control_stream = torch.cuda.current_stream()
        communication_stream = torch.cuda.Stream()
        compute_stream = torch.cuda.Stream() if self.overlap else communication_stream
        global_start = torch.cuda.Event(enable_timing=True)
        global_end = torch.cuda.Event(enable_timing=True)
        global_start.record(control_stream)
        communication_stream.wait_event(global_start)
        if compute_stream != communication_stream:
            compute_stream.wait_event(global_start)

        values: dict[str, Any] = {}
        completion_events: dict[str, Any] = {}
        timing_events: dict[str, tuple[Any, Any]] = {}
        issue_order: list[str] = []

        previous_layer: tuple[str, ...] = ()
        for layer in self.plan.issue_layers:
            if previous_layer:
                while not any(completion_events[node_id].query()
                              for node_id in previous_layer):
                    time.sleep(0.0001)
            pending = list(layer)
            while pending:
                progressed = False
                for node_id in tuple(pending):
                    if not dependencies[node_id].issubset(completion_events):
                        continue
                    node = nodes[node_id]
                    stream = (communication_stream if node.kind == "allgather"
                              else compute_stream)
                    for dependency in sorted(dependencies[node_id]):
                        stream.wait_event(completion_events[dependency])

                    start = torch.cuda.Event(enable_timing=True)
                    done = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        start.record(stream)
                        inputs = [
                            self._resolve(value, values) for value in node.inputs
                        ]
                        if node.kind == "allgather":
                            result = torch.empty_like(inputs[0])
                            result.copy_(inputs[0], non_blocking=True)
                        else:
                            artifact = _registered_specialization(self.plan, node.id)
                            record, task = artifact.record, artifact.task
                            if task.grid is None:
                                raise RuntimeError(
                                    f"task '{node.id}' requires a CUDA launch grid"
                                )
                            _validate_launch_signature(record, inputs)
                            try:
                                from triton.runtime.driver import driver
                                active_target = driver.active.get_current_target()
                            except Exception as exc:
                                raise RuntimeError(
                                    f"cannot resolve the active target for specialization '{record.identity}'"
                                ) from exc
                            if active_target != record.target_object:
                                raise RuntimeError(
                                    f"specialization '{record.identity}' target mismatch: "
                                    f"planned for {record.target_object}, executing on {active_target}"
                                )
                            arguments = _ordered_task_arguments(record, inputs)
                            compiled = _compile_specialization(artifact)
                            compiled[_launch_grid(task, record, arguments)](*arguments)
                            result = None
                        done.record(stream)

                    values[node.id] = result
                    completion_events[node.id] = done
                    timing_events[node.id] = (start, done)
                    issue_order.append(node.id)
                    pending.remove(node_id)
                    progressed = True
                if not progressed:
                    raise RuntimeError(
                        "same-layer dependencies could not be materialized as CUDA events"
                    )
            previous_layer = layer

        for event in completion_events.values():
            control_stream.wait_event(event)
        global_end.record(control_stream)
        global_end.synchronize()
        timings = tuple(
            CudaNodeTiming(
                node.id,
                node.resources.resource_class,
                global_start.elapsed_time(timing_events[node.id][0]),
                global_start.elapsed_time(timing_events[node.id][1]),
            )
            for node in self.plan.nodes
        )
        return CudaExecutionResult(
            tuple(issue_order), timings, global_start.elapsed_time(global_end), values
        )
