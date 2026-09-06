from dataclasses import replace

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton import timely as tm
from triton.timely import runtime as timely_runtime


TARGET = GPUTarget("cuda", 80, 32)


def _reference_copy(source, output, block):
    output[:block] = source[:block]


@tm.task(reference=_reference_copy, grid=(1,))
def _copy_task(source, output, BLOCK: tl.constexpr):
    pass


@tm.kernel
def _copy_pipeline(source, output):
    _copy_task(source, output, 4, at=tm.Time(0), id="copy")


def _captured_builder():
    return _copy_pipeline._capture(
        signature={"source": "*fp32", "output": "*fp32"},
        target=TARGET,
    )


def _native_descriptor(builder, *, callee=None, specialization=None):
    record = next(iter(builder.specialization_registry.values()))
    return {
        "nodes": [{
            "id": "copy",
            "kind": "compute",
            "raw_time": 0,
            "issue_rank": 0,
            "callee": record.callee if callee is None else callee,
            "specialization": (
                record.identity if specialization is None else specialization
            ),
        }],
        "issue_layers": [{"rank": 0, "tasks": ["copy"]}],
        "time_order": [],
        "data_deps": [],
        "resource_order": [],
        "synchronizations": [],
    }


def _execution_plan():
    builder = _captured_builder()
    return timely_runtime.execution_plan_from_descriptor(
        _native_descriptor(builder), builder
    )


def test_descriptor_preserves_invocation_and_specialization_identity():
    plan = _execution_plan()
    descriptor_node = plan.descriptor.nodes[0]
    identity = plan.node_specializations["copy"]
    record = plan.specialization_registry[identity]

    assert descriptor_node.id == "copy"
    assert descriptor_node.callee == record.callee
    assert descriptor_node.specialization == identity
    assert set(plan.task_registry) == {identity}
    assert set(plan.kernel_registry) == {identity}
    assert set(plan.artifact_registry) == {identity}


@pytest.mark.parametrize("field", ["callee", "specialization"])
def test_descriptor_rejects_stale_callee_or_specialization(field):
    builder = _captured_builder()
    overrides = {field: "stale-value"}

    with pytest.raises(ValueError, match=f"native tm.plan {field} disagrees"):
        timely_runtime.execution_plan_from_descriptor(
            _native_descriptor(builder, **overrides), builder
        )


def test_reference_executor_uses_explicit_output_and_reinserts_constexpr():
    np = pytest.importorskip("numpy")
    plan = _execution_plan()
    source = np.arange(4, dtype=np.float32)
    output = np.zeros_like(source)

    result = plan.execute({"source": source, "output": output})

    np.testing.assert_array_equal(output, source)
    assert result.values["copy"] is None


def test_missing_or_stale_artifact_fails_before_reference_call():
    np = pytest.importorskip("numpy")
    source = np.arange(4, dtype=np.float32)
    output = np.zeros_like(source)

    plan = _execution_plan()
    missing = replace(plan, artifact_registry={})
    with pytest.raises(RuntimeError, match="has no launch artifact"):
        missing.execute({"source": source, "output": output})

    plan = _execution_plan()
    artifact = next(iter(plan.artifact_registry.values()))
    artifact.kernel = object()
    with pytest.raises(RuntimeError, match="stale launch artifact"):
        plan.execute({"source": source, "output": output})


def test_runtime_signature_validation_names_bad_parameter():
    torch = pytest.importorskip("torch")
    plan = _execution_plan()
    record = plan.specialization_for("copy")

    timely_runtime._validate_launch_signature(
        record,
        [torch.empty(4, dtype=torch.float32), torch.empty(4, dtype=torch.float32)],
    )
    with pytest.raises(TypeError, match=r"argument 1 \('output'\).*expected \*fp32, got \*fp16"):
        timely_runtime._validate_launch_signature(
            record,
            [torch.empty(4, dtype=torch.float32), torch.empty(4, dtype=torch.float16)],
        )


def test_runtime_signature_validation_rejects_stale_binder_attributes():
    torch = pytest.importorskip("torch")
    source = torch.empty(8, dtype=torch.float32)
    output = torch.empty(8, dtype=torch.float32)
    builder = _copy_pipeline._capture(source, output, target=TARGET)
    record = next(iter(builder.specialization_registry.values()))

    timely_runtime._validate_launch_signature(record, [source, output])
    misaligned = torch.empty(9, dtype=torch.float32)[1:]
    with pytest.raises(TypeError, match=r"argument 0 \('source'\).*specialization mismatch"):
        timely_runtime._validate_launch_signature(record, [misaligned, output])


def test_artifact_compilation_uses_the_planned_specialization(monkeypatch):
    plan = _execution_plan()
    record = plan.specialization_for("copy")
    artifact = plan.artifact_registry[record.identity]
    calls = []
    compiled = object()

    def fake_compile(source, *, target, options):
        calls.append((source, target, options))
        return compiled

    monkeypatch.setattr(triton, "compile", fake_compile)

    assert timely_runtime._compile_specialization(artifact) is compiled
    assert timely_runtime._compile_specialization(artifact) is compiled
    assert len(calls) == 1
    source, target, options = calls[0]
    assert source.fn is artifact.kernel
    assert source.signature == record.signature
    assert source.constants == {(2,): 4}
    assert source.attrs == record.attrs
    assert target == record.target_object
    assert options == dict(record.codegen_options)


def test_launch_arguments_and_grid_follow_original_parameter_order():
    plan = _execution_plan()
    record = plan.specialization_for("copy")
    dynamic = [object(), object()]

    arguments = timely_runtime._ordered_task_arguments(record, dynamic)

    assert arguments == (dynamic[0], dynamic[1], 4)
    assert timely_runtime._launch_grid(record.task, record, arguments) == (1, 1, 1)
