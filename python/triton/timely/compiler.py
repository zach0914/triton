from __future__ import annotations

import functools
import hashlib
import inspect
import json
import re
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core import Const, Domain, ResourceSpec, Time


def _is_annotation(annotation: Any, expected: type) -> bool:
    return annotation is expected or annotation in (expected.__name__, f"tm.{expected.__name__}")


@dataclass(frozen=True)
class ValueProvenance:
    root: str
    region: str
    lookup_key: str


@dataclass(frozen=True)
class Buffer:
    name: str
    type_name: str | None = None
    runtime_value: Any = field(default=None, repr=False, compare=False, hash=False)

    @property
    def provenance(self) -> ValueProvenance:
        return ValueProvenance(self.name, "*", self.name)

    def __getitem__(self, index: Any) -> BufferRef:
        if not isinstance(index, tuple):
            index = (index,)
        return BufferRef(self.name, index, self.type_name, self.runtime_value)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class BufferRef:
    buffer: str
    indices: tuple[Any, ...]
    type_name: str | None = None
    runtime_value: Any = field(default=None, repr=False, compare=False, hash=False)

    @property
    def provenance(self) -> ValueProvenance:
        region = "[" + ",".join(str(index) for index in self.indices) + "]"
        return ValueProvenance(self.buffer, region, str(self))

    def __str__(self) -> str:
        indices = ",".join(str(index) for index in self.indices)
        return f"{self.buffer}[{indices}]"


@dataclass(frozen=True)
class ScalarBinding:
    name: str
    type_name: str
    runtime_value: Any = field(default=None, repr=False, compare=False, hash=False)

    @property
    def provenance(self) -> ValueProvenance:
        return ValueProvenance(self.name, "scalar", self.name)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ScalarLiteral:
    value: bool | int | float
    type_name: str


@dataclass(frozen=True)
class AsyncValue:
    """A typed data value produced by an asynchronous communication op."""

    producer: str
    value_name: str
    event_name: str
    type_name: str
    provenance: ValueProvenance

    @property
    def storage(self) -> str:
        return self.provenance.lookup_key


@dataclass(frozen=True)
class Completion:
    """A completion token; unlike AsyncValue this is never a data operand."""

    producer: str
    event_name: str


@dataclass(frozen=True)
class TaskSpecialization:
    identity: str
    callee: str
    body_identity: str
    dynamic_signature: tuple[tuple[str, str], ...]
    constexprs: tuple[tuple[str, Any], ...]
    argument_attrs: tuple[tuple[tuple[int, ...], tuple[tuple[Any, ...], ...]], ...]
    runtime_specialization: tuple[tuple[str, tuple[Any, ...]], ...]
    target: tuple[tuple[str, Any], ...]
    options: tuple[tuple[str, Any], ...]
    codegen_options: tuple[tuple[str, Any], ...]
    parameter_order: tuple[str, ...]
    launch_operand_order: tuple[str, ...]
    artifact_key: str
    task: Any = field(repr=False, compare=False, hash=False)
    target_object: Any = field(repr=False, compare=False, hash=False)
    backend_options: Any = field(repr=False, compare=False, hash=False)

    @property
    def signature(self) -> dict[str, str]:
        return dict(self.dynamic_signature)

    @property
    def constants(self) -> dict[str, Any]:
        return dict(self.constexprs)

    @property
    def attrs(self) -> dict[tuple[int, ...], list[list[Any]]]:
        return {
            path: [list(attribute) for attribute in attributes]
            for path, attributes in self.argument_attrs
        }


@dataclass
class TaskNode:
    id: str
    kind: str
    issue_time: int
    inputs: tuple[Any, ...]
    reads: list[str]
    writes: list[str]
    depends_on: list[str]
    resources: ResourceSpec
    callee: str | None = None
    specialization: TaskSpecialization | None = None
    result_type: str | None = None
    provenance: ValueProvenance | None = None
    value_name: str = ""
    event_name: str = ""
    task: Any = None
    stores: list[BufferRef] = field(default_factory=list)


def _canonical_signature(signature: Mapping[str, Any] | None) -> dict[str, str]:
    if signature is None:
        return {}
    if not isinstance(signature, Mapping):
        raise TypeError("Timely signature must be a mapping from parameter names to Triton types")
    result: dict[str, str] = {}
    for name, type_name in signature.items():
        if not isinstance(name, str):
            raise TypeError("Timely signature keys must be parameter names")
        canonical = str(type_name)
        if not canonical or canonical == "constexpr":
            raise TypeError(f"invalid runtime type for Timely parameter '{name}': {canonical!r}")
        _mlir_type(canonical)
        result[name] = canonical
    return result


def _infer_type(value: Any, name: str) -> str:
    if isinstance(value, (Buffer, BufferRef, ScalarBinding, ScalarLiteral, AsyncValue)):
        type_name = value.type_name
    else:
        try:
            from triton.runtime.jit import mangle_type
            type_name = mangle_type(value)
        except Exception as exc:
            raise TypeError(
                f"cannot determine a Triton type for Timely parameter '{name}'; "
                "pass an explicit signature or a supported concrete input"
            ) from exc
    if not type_name:
        raise TypeError(
            f"cannot determine a Triton type for Timely parameter '{name}'; "
            "pass an explicit signature or a supported concrete input"
        )
    return str(type_name)


def _typed_runtime_argument(name: str, value: Any, type_name: str | None) -> Any:
    if isinstance(value, Buffer):
        if type_name is not None and value.type_name is not None and value.type_name != type_name:
            raise TypeError(f"conflicting types for Timely parameter '{name}': {value.type_name} and {type_name}")
        return replace(value, type_name=type_name or value.type_name)
    if isinstance(value, BufferRef):
        if type_name is not None and value.type_name is not None and value.type_name != type_name:
            raise TypeError(f"conflicting types for Timely parameter '{name}': {value.type_name} and {type_name}")
        return replace(value, type_name=type_name or value.type_name)
    if isinstance(value, ScalarBinding):
        if type_name is not None and value.type_name != type_name:
            raise TypeError(f"conflicting types for Timely parameter '{name}': {value.type_name} and {type_name}")
        return value
    if type_name is None and value is not _MISSING:
        type_name = _infer_type(value, name)
    if type_name is not None and type_name.startswith("*"):
        runtime_value = None if value is _MISSING else value
        return Buffer(name, type_name, runtime_value)
    if type_name is not None:
        runtime_value = None if value is _MISSING else value
        return ScalarBinding(name, type_name, runtime_value)
    return Buffer(name)


_MISSING = object()


def prepare_kernel_arguments(python_signature: inspect.Signature, args: tuple[Any, ...], kwargs: Mapping[str, Any],
                             *, compile_signature: Mapping[str, Any] | None = None):
    provided = python_signature.bind_partial(*args, **kwargs)
    canonical_signature = _canonical_signature(compile_signature)
    unknown = set(canonical_signature).difference(python_signature.parameters)
    if unknown:
        raise TypeError(f"Timely signature contains unknown parameters: {', '.join(sorted(unknown))}")

    values: dict[str, Any] = {}
    constants: dict[str, int] = {}
    for name, parameter in python_signature.parameters.items():
        if name in provided.arguments:
            value = provided.arguments[name]
        elif parameter.default is not inspect.Parameter.empty:
            value = parameter.default
        elif _is_annotation(parameter.annotation, Const):
            raise TypeError(f"missing compile-time Timely Const argument: {name}")
        elif _is_annotation(parameter.annotation, Time):
            value = Time()
        else:
            value = _MISSING

        if _is_annotation(parameter.annotation, Const):
            value = value if isinstance(value, Const) else Const(value, name=name)
            constants[name] = value.evaluate()
        elif _is_annotation(parameter.annotation, Time):
            value = value if isinstance(value, Time) else Time(value)
        else:
            value = _typed_runtime_argument(name, value, canonical_signature.get(name))
        values[name] = value

    bound = python_signature.bind(**values)
    return bound.args, bound.kwargs, constants


def _freeze(value: Any) -> Any:
    if is_dataclass(value):
        return _freeze(asdict(value))
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze(item)) for key, item in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if hasattr(value, "__dict__"):
        return _freeze(vars(value))
    return repr(value)


def _json_identity(value: Any) -> str:
    return json.dumps(_freeze(value), sort_keys=True, separators=(",", ":"), default=repr)


def _resolve_target(target: Any) -> Any:
    if target is not None:
        return target
    try:
        from triton.runtime.driver import driver
        return driver.active.get_current_target()
    except Exception:
        return None


def _storage_provenance(storage: str) -> ValueProvenance:
    match = re.fullmatch(r"([^\[]+)\[(.*)\]", storage)
    if match is None:
        return ValueProvenance(storage, "*", storage)
    return ValueProvenance(match.group(1), f"[{match.group(2)}]", storage)


def _value_type(value: Any, *, task: str | None = None, parameter: str | None = None) -> str:
    if isinstance(value, Completion):
        suffix = f" for task '{task}'" if task else ""
        raise TypeError(
            "a compute completion cannot be used as a data operand"
            f"{suffix}; pass an explicit output buffer to the Triton kernel"
        )
    if isinstance(value, (Buffer, BufferRef, ScalarBinding, ScalarLiteral, AsyncValue)) and value.type_name:
        return value.type_name
    location = f" parameter '{parameter}' of task '{task}'" if task and parameter else " Timely value"
    raise TypeError(
        f"cannot determine a Triton type for{location}; pass an explicit signature or concrete inputs"
    )


def _dynamic_value(value: Any, *, task: str, parameter: str) -> Any:
    if isinstance(value, Completion):
        _value_type(value, task=task, parameter=parameter)
    if isinstance(value, (Buffer, BufferRef, ScalarBinding, ScalarLiteral, AsyncValue)):
        _value_type(value, task=task, parameter=parameter)
        return value
    if isinstance(value, (bool, int, float)):
        return ScalarLiteral(value, _infer_type(value, parameter))
    raise TypeError(
        f"task '{task}' parameter '{parameter}' is not a captured Timely binding; "
        "pass it through the Timely kernel signature"
    )


def _concrete_value(value: Any) -> Any:
    if isinstance(value, BufferRef):
        # Resolving a view would execute user tensor indexing and can change
        # alignment.  Keep it conservatively unspecialized during capture.
        return _MISSING
    if isinstance(value, (Buffer, ScalarBinding)):
        return _MISSING if value.runtime_value is None else value.runtime_value
    if isinstance(value, ScalarLiteral):
        return value.value
    return _MISSING


def _specialize_task_argument(backend: Any, parameter: Any, concrete: Any,
                              actual_type: str) -> tuple[Any, ...]:
    """Apply the ordinary JIT binder's per-parameter specialization rules."""
    from triton.runtime.jit import native_specialize_impl

    inferred = native_specialize_impl(
        backend,
        concrete,
        parameter.is_const,
        not parameter.do_not_specialize,
        not parameter.do_not_specialize_on_alignment,
    )
    annotated_type = parameter.annotation_type
    if not annotated_type:
        return tuple(inferred)
    annotate_specialization = not parameter.do_not_specialize
    if annotated_type == "u1" or annotated_type[:2] in ("fp", "bf"):
        annotate_specialization = False
    if annotate_specialization:
        return (annotated_type,) + tuple(inferred[1:])
    return (annotated_type, None)


class GraphBuilder:
    def __init__(self, name: str, constants: Mapping[str, int], *, target: Any = None,
                 options: Mapping[str, Any] | None = None):
        self.name = name
        self.constants = dict(constants)
        self.target = _resolve_target(target)
        self.options = dict(options or {})
        self.domains: list[Domain] = []
        self.nodes: list[TaskNode] = []
        self.task_registry: dict[str, Any] = {}
        self.kernel_registry: dict[str, Any] = {}
        self.specialization_registry: dict[str, TaskSpecialization] = {}
        self.node_specializations: dict[str, str] = {}
        self._kind_counts: dict[str, int] = {}

    def record_domain(self, domain: Domain) -> None:
        if domain not in self.domains:
            self.domains.append(domain)

    def _next_id(self, prefix: str, requested: str | None) -> str:
        if requested is not None:
            if any(node.id == requested for node in self.nodes):
                raise ValueError(f"duplicate Timely task id: {requested}")
            return requested
        index = self._kind_counts.get(prefix, 0)
        self._kind_counts[prefix] = index + 1
        return f"{prefix}{index}"

    def _time(self, value: Time | None) -> int:
        if not isinstance(value, Time):
            raise TypeError("Timely operations require an explicit 'at=Time(...)'")
        return value.evaluate(self.constants)

    def _dependencies(self, inputs: Iterable[Any], explicit: Iterable[Any]) -> list[str]:
        result: list[str] = []
        for value in inputs:
            if not isinstance(value, (AsyncValue, Completion)):
                continue
            if value.producer not in result:
                result.append(value.producer)
        for value in explicit:
            dependency = value.producer if isinstance(value, (AsyncValue, Completion)) else value
            if not isinstance(dependency, str):
                raise TypeError("depends_on entries must be task ids or asynchronous values")
            if dependency not in result:
                result.append(dependency)
        return result

    def _accesses(self, values: Iterable[str], inputs: Iterable[Any]) -> list[str]:
        result = list(values)
        for value in inputs:
            access = None
            if isinstance(value, AsyncValue):
                access = value.provenance.lookup_key
            elif result:
                continue
            elif isinstance(value, (Buffer, BufferRef, ScalarBinding)):
                access = value.provenance.lookup_key
            if access is not None and access not in result:
                result.append(access)
        return result

    def _append(self, node: TaskNode) -> AsyncValue | Completion:
        index = len(self.nodes)
        node.event_name = f"event{index}"
        if node.kind != "compute":
            node.value_name = f"value{index}"
        self.nodes.append(node)
        if node.specialization is not None:
            specialization = node.specialization
            self.node_specializations[node.id] = specialization.identity
            self.specialization_registry.setdefault(specialization.identity, specialization)
            self.task_registry[specialization.identity] = node.task
            self.kernel_registry[specialization.identity] = node.task.triton_jit
            return Completion(node.id, node.event_name)
        if node.result_type is None or node.provenance is None:
            raise AssertionError(f"communication node '{node.id}' is missing typed result provenance")
        return AsyncValue(node.id, node.value_name, node.event_name, node.result_type, node.provenance)

    def record_allgather(self, source: Any, *, at: Time, reads: Iterable[str], writes: Iterable[str],
                         depends_on: Iterable[Any], resources: ResourceSpec, task_id: str | None) -> AsyncValue:
        if resources.resource_class != "communication":
            raise ValueError("tm.allgather_shard requires communication resources")
        source_type = _value_type(source)
        if not isinstance(source, (Buffer, BufferRef, AsyncValue)):
            raise TypeError("tm.allgather_shard source must be a typed buffer binding or communication value")
        node_id = self._next_id("ag", task_id)
        inferred_reads = self._accesses(reads, (source,))
        inferred_writes = list(writes) or [f"gathered::{inferred_reads[0]}"]
        provenance = _storage_provenance(inferred_writes[0])
        return self._append(TaskNode(
            node_id, "allgather", self._time(at), (source,), inferred_reads,
            inferred_writes, self._dependencies((), depends_on), resources,
            result_type=source_type, provenance=provenance,
        ))

    def _make_specialization(self, task: Any, arguments: Mapping[str, Any],
                             resources: ResourceSpec) -> tuple[TaskSpecialization, tuple[Any, ...]]:
        if self.target is None:
            raise RuntimeError(
                f"cannot lower Timely task '{task.__name__}' without a GPU target; "
                "pass target=GPUTarget(...) or initialize an active Triton driver"
            )
        from triton.compiler import make_backend

        jit_fn = task.triton_jit
        backend = make_backend(self.target)
        params = {parameter.name: parameter for parameter in jit_fn.params}
        dynamic: list[tuple[str, Any]] = []
        constexprs: list[tuple[str, Any]] = []
        argument_attrs: dict[tuple[int, ...], list[list[Any]]] = {}
        runtime_specialization: list[tuple[str, tuple[Any, ...]]] = []
        for name in task.signature.parameters:
            value = arguments[name]
            parameter = params[name]
            if parameter.is_constexpr:
                if isinstance(value, Const):
                    value = value.evaluate()
                constexprs.append((name, value))
                runtime_specialization.append((name, ("constexpr", _freeze(value))))
            else:
                value = _dynamic_value(value, task=task.__name__, parameter=name)
                actual_type = _value_type(value, task=task.__name__, parameter=name)
                annotated_type = parameter.annotation_type
                if annotated_type:
                    expected_type = annotated_type.replace("*k", "*", 1)
                    if actual_type != expected_type:
                        raise TypeError(
                            f"task '{task.__name__}' parameter '{name}' type mismatch: "
                            f"expected {expected_type}, got {actual_type}"
                        )

                concrete = _concrete_value(value)
                specialized: tuple[Any, ...] = (actual_type, None)
                if concrete is not _MISSING:
                    specialized = _specialize_task_argument(
                        backend, parameter, concrete, actual_type
                    )
                    inferred_type = specialized[0]
                    if inferred_type not in (actual_type, "constexpr"):
                        raise TypeError(
                            f"task '{task.__name__}' parameter '{name}' concrete specialization "
                            f"has type {inferred_type}, but graph value has type {actual_type}"
                        )

                runtime_specialization.append((name, tuple(_freeze(item) for item in specialized)))
                if specialized[0] == "constexpr":
                    constexprs.append((name, concrete))
                    continue
                descriptor = specialized[1] if len(specialized) > 1 else None
                if isinstance(descriptor, str) and descriptor:
                    parsed_attrs = backend.parse_attr(descriptor)
                    if parsed_attrs:
                        argument_attrs[(parameter.num,)] = parsed_attrs
                dynamic.append((name, value))

        signature = tuple((name, _value_type(value, task=task.__name__, parameter=name)) for name, value in dynamic)
        codegen_options = dict(self.options)
        codegen_options.update(task.launch_options)
        codegen_options.setdefault("num_warps", resources.warps)
        parsed_options = backend.parse_options(codegen_options)
        body_identity = f"{jit_fn.__module__}.{jit_fn.__qualname__}:{jit_fn.cache_key}"
        target_key = _freeze(self.target)
        options_key = _freeze(parsed_options)
        payload = {
            "body": body_identity,
            "signature": signature,
            "constexprs": constexprs,
            "argument_attrs": argument_attrs,
            "runtime_specialization": runtime_specialization,
            "target": target_key,
            "options": options_key,
        }
        digest = hashlib.sha256(_json_identity(payload).encode("utf-8")).hexdigest()
        readable = _identifier(f"{jit_fn.__module__}_{jit_fn.__qualname__}")[-64:]
        callee = f"__tm_{readable}_{digest[:16]}"
        identity = f"tm-specialization-v1:{digest}"
        record = TaskSpecialization(
            identity=identity,
            callee=callee,
            body_identity=body_identity,
            dynamic_signature=signature,
            constexprs=tuple(constexprs),
            argument_attrs=tuple(
                (path, tuple(tuple(attribute) for attribute in attributes))
                for path, attributes in sorted(argument_attrs.items())
            ),
            runtime_specialization=tuple(runtime_specialization),
            target=tuple(target_key),
            options=tuple(options_key),
            codegen_options=tuple((name, _freeze(value)) for name, value in sorted(codegen_options.items())),
            parameter_order=tuple(task.signature.parameters),
            launch_operand_order=tuple(name for name, _ in dynamic),
            artifact_key=digest,
            task=task,
            target_object=self.target,
            backend_options=parsed_options,
        )
        existing = self.specialization_registry.get(identity)
        if existing is not None and existing != record:
            raise RuntimeError(f"Timely specialization identity collision for '{callee}'")
        return existing or record, tuple(value for _, value in dynamic)

    def record_task(self, task: Any, arguments: Mapping[str, Any], *, at: Time, reads: Iterable[str],
                    writes: Iterable[str], depends_on: Iterable[Any], resources: ResourceSpec,
                    task_id: str | None) -> Completion:
        if not isinstance(resources, ResourceSpec) or resources.resource_class != "compute":
            raise ValueError("@tm.task calls require compute resources")
        specialization, inputs = self._make_specialization(task, arguments, resources)
        node_id = self._next_id(task.__name__, task_id)
        return self._append(TaskNode(
            node_id, "compute", self._time(at), inputs,
            list(reads), list(writes),
            self._dependencies(inputs, depends_on), resources,
            callee=specialization.callee,
            specialization=specialization,
            task=task,
        ))

    def record_store(self, target: Any, value: Any) -> None:
        if isinstance(value, Completion):
            raise TypeError(
                "tm.store(target, task_return) is no longer supported: a compute task returns only a completion "
                "event; pass the output buffer explicitly to the @tm.task kernel and call tl.store there"
            )
        raise TypeError(
            "tm.store does not represent a device write; pass the output buffer explicitly to the @tm.task kernel"
        )

    def _scheduling_mlir(self) -> str:
        if not self.nodes:
            raise ValueError("a Timely kernel must contain at least one task")
        lines = ["module {", "  %root = tm.event.none"]
        handles: dict[str, tuple[str, str, ValueProvenance]] = {}
        literal_handles: dict[tuple[Any, str], str] = {}

        def handle(value: Any) -> tuple[str, str]:
            if isinstance(value, Completion):
                _value_type(value)
            if isinstance(value, AsyncValue):
                return f"%{value.value_name}", _mlir_type(value.type_name)
            if isinstance(value, ScalarLiteral):
                key = (value.value, value.type_name)
                if key not in literal_handles:
                    name = f"literal{len(literal_handles)}"
                    literal_handles[key] = name
                    lines.append(
                        f"  %{name} = arith.constant {_literal(value.value, value.type_name)} "
                        f": {_mlir_type(value.type_name)}"
                    )
                return f"%{literal_handles[key]}", _mlir_type(value.type_name)
            if not isinstance(value, (Buffer, BufferRef, ScalarBinding)):
                raise TypeError(f"unsupported Timely data operand: {value!r}")
            type_name = _value_type(value)
            provenance = value.provenance
            existing = handles.get(provenance.lookup_key)
            if existing is not None:
                name, existing_type, existing_provenance = existing
                if existing_type != type_name or existing_provenance != provenance:
                    raise TypeError(f"conflicting Timely binding for lookup key '{provenance.lookup_key}'")
                return f"%{name}", _mlir_type(type_name)
            name = f"binding{len(handles)}"
            handles[provenance.lookup_key] = (name, type_name, provenance)
            lines.append(
                f'  %{name} = "tm.binding"() {{root = {json.dumps(provenance.root)}, '
                f'region = {json.dumps(provenance.region)}, lookup_key = {json.dumps(provenance.lookup_key)}}} '
                f': () -> {_mlir_type(type_name)}'
            )
            return f"%{name}", _mlir_type(type_name)

        if self.domains:
            for index, domain in enumerate(self.domains):
                lower = ", ".join(str(bound[0]) for bound in domain.bounds)
                upper = ", ".join(str(bound[1]) for bound in domain.bounds)
                parameters = ", ".join(
                    f"{{name = {json.dumps(name)}, value = {value} : i64}}"
                    for name, value in sorted(self.constants.items())
                )
                lines.extend([
                    '  "tm.domain"() {',
                    f'    sym_name = "domain{index}", lower = array<i64: {lower}>,',
                    f'    upper = array<i64: {upper}>, parameters = [{parameters}]',
                    "  } : () -> ()",
                ])

        time_names: dict[int, str] = {}
        for time in sorted({node.issue_time for node in self.nodes}):
            time_name = f"time{len(time_names)}"
            time_names[time] = time_name
            lines.append(f"  %{time_name} = tm.time.constant {time}")

        known_events = {node.id: node.event_name for node in self.nodes}
        emitted_events: set[str] = set()
        for node in self.nodes:
            input_values = [handle(value) for value in node.inputs]
            operands = [value for value, _ in input_values]
            operand_types = [type_name for _, type_name in input_values]
            dependency_event = "%root"
            for dependency in node.depends_on:
                event = known_events.get(dependency) if dependency in emitted_events else None
                if event is not None:
                    dependency_event = f"%{event}"
                    break
            operands.extend((f"%{time_names[node.issue_time]}", dependency_event))
            operand_types.extend(("!tm.time", "!tm.event"))
            attrs = [
                f"id = {json.dumps(node.id)}",
                f"reads = {_string_array(node.reads)}",
                f"writes = {_string_array(node.writes)}",
                f"depends_on = {_string_array(node.depends_on)}",
                f"resource_class = {json.dumps(node.resources.resource_class)}",
                f"sms = {node.resources.sms} : i64",
                f"threads = {node.resources.threads} : i64",
                f"warps = {node.resources.warps} : i64",
                f"shared_memory_bytes = {node.resources.shared_memory_bytes} : i64",
            ]
            if node.kind == "allgather":
                if node.provenance is None or node.result_type is None:
                    raise AssertionError(f"allgather node '{node.id}' lacks typed provenance")
                attrs.extend((
                    f"root = {json.dumps(node.provenance.root)}",
                    f"region = {json.dumps(node.provenance.region)}",
                    f"lookup_key = {json.dumps(node.provenance.lookup_key)}",
                ))
                lines.extend([
                    f'  %{node.value_name}, %{node.event_name} = "tm.allgather_shard"({", ".join(operands)}) {{',
                    f"    {', '.join(attrs)}",
                    f"  }} : ({', '.join(operand_types)}) -> ({_mlir_type(node.result_type)}, !tm.event)",
                ])
            else:
                if node.specialization is None or node.callee is None:
                    raise AssertionError(f"compute node '{node.id}' lacks a specialization")
                attrs.insert(0, f"specialization = {json.dumps(node.specialization.identity)}")
                attrs.insert(0, f"callee = @{node.callee}")
                lines.extend([
                    f'  %{node.event_name} = "tm.task"({", ".join(operands)}) {{',
                    f"    {', '.join(attrs)}",
                    f"  }} : ({', '.join(operand_types)}) -> !tm.event",
                ])
            emitted_events.add(node.id)
        lines.append("}")
        return "\n".join(lines) + "\n"

    def build_analysis_module(self):
        from triton._C.libtriton import ir
        from triton.compiler import ASTSource, ASTSourceBatch

        if self.specialization_registry:
            batch = ASTSourceBatch(self.target)
            for record in self.specialization_registry.values():
                source = ASTSource(
                    fn=record.task.triton_jit,
                    signature=record.signature,
                    constexprs=record.constants,
                    attrs=record.attrs,
                )
                batch.add(
                    source,
                    entry_name=record.callee,
                    visibility="private",
                    parsed_options=record.backend_options,
                )
            context = batch.context
        else:
            context = ir.context()
            ir.load_dialects(context)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{_identifier(self.name)}.mlir"
            path.write_text(self._scheduling_mlir())
            module = ir.parse_mlir_module(
                str(path), context, not self.specialization_registry
            )
        if self.specialization_registry:
            module.merge_functions_from(batch.module)
        module.context = context
        module.timely_analysis_only = True
        if not module.verify():
            raise RuntimeError("Timely combined TTIR/TM module verification failed")
        return module

    def to_mlir(self) -> str:
        return self.build_analysis_module().str_nodebug()

@functools.lru_cache(maxsize=None)
def _mlir_type(type_name: str) -> str:
    try:
        from triton._C.libtriton import ir
        from triton.language import str_to_ty
        context = ir.context()
        ir.load_dialects(context)
        return str(str_to_ty(type_name, None).to_ir(ir.builder(context)))
    except Exception as exc:
        raise TypeError(f"invalid Triton type in Timely signature: {type_name!r}") from exc


def _literal(value: Any, type_name: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == float("inf"):
            return "0x7F800000" if type_name == "fp32" else "0x7FF0000000000000"
        return repr(value)
    raise TypeError(f"unsupported Timely scalar literal: {value!r}")


def _string_array(values: Iterable[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def _identifier(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not result or result[0].isdigit():
        result = "_" + result
    return result
