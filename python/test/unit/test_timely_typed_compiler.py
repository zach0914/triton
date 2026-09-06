import types

import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, ASTSourceBatch, make_backend
from triton.runtime.jit import MockTensor, create_function_from_signature
from triton import timely as tm
from triton.timely.compiler import Completion


TARGET = GPUTarget("cuda", 80, 32)


@triton.jit
def _shared_increment(value):
    return value + 1


@triton.jit
def _batch_entry_a(output):
    tl.store(output, _shared_increment(tl.load(output)))


@triton.jit
def _batch_entry_b(output):
    tl.store(output, _shared_increment(tl.load(output)))


@tm.task
def _typed_copy(source, output, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(output + offsets, tl.load(source + offsets))


@tm.kernel
def _copy_pipeline(source, output):
    _typed_copy(source, output, 16, at=tm.Time(0), id="copy")


@tm.task
def _typed_scale(source, output, count, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < count
    tl.store(output + offsets, tl.load(source + offsets, mask=mask), mask=mask)


@tm.task
def _expects_fp32(source: "*fp32", output: "*fp32"):
    tl.store(output, tl.load(source))


def _same_name_template(source, output):
    tl.store(output, tl.load(source))


def test_ast_source_batch_uses_private_unique_entries_and_shared_helpers():
    batch = ASTSourceBatch(TARGET)
    batch.add(ASTSource(_batch_entry_a, {"output": "*i32"}), entry_name="__tm_entry_a")
    batch.add(ASTSource(_batch_entry_b, {"output": "*i32"}), entry_name="__tm_entry_b")

    text = batch.verify().str()

    assert "tt.func private @__tm_entry_a" in text
    assert "tt.func private @__tm_entry_b" in text
    assert text.count("tt.func private @") == 3
    assert text.count("_shared_increment__i32") >= 2


def test_capture_separates_completion_and_reuses_specialization():
    @tm.kernel
    def twice(source, output):
        first = _typed_copy(source, output, 16, at=tm.Time(0), id="first")
        second = _typed_copy(source, output, 16, at=tm.Time(1), id="second", depends_on=(first,))
        assert isinstance(second, Completion)

    builder = twice._capture(
        signature={"source": "*fp16", "output": "*fp16"}, target=TARGET
    )

    assert len(builder.specialization_registry) == 1
    assert builder.node_specializations["first"] == builder.node_specializations["second"]
    record = next(iter(builder.specialization_registry.values()))
    assert record.dynamic_signature == (("source", "*fp16"), ("output", "*fp16"))
    assert record.constexprs == (("BLOCK", 16),)
    assert record.parameter_order == ("source", "output", "BLOCK")
    assert record.launch_operand_order == ("source", "output")
    assert record.callee.startswith("__tm_")
    assert builder.nodes[0].specialization is record
    assert builder.nodes[1].specialization is record
    assert builder.nodes[0].callee == builder.nodes[1].callee == record.callee


def test_capture_separates_different_task_specializations():
    @tm.kernel
    def mixed(source16, output16, source32, output32):
        _typed_copy(source16, output16, 16, at=tm.Time(0), id="fp16")
        _typed_copy(source32, output32, 16, at=tm.Time(0), id="fp32")

    builder = mixed._capture(
        signature={
            "source16": "*fp16",
            "output16": "*fp16",
            "source32": "*fp32",
            "output32": "*fp32",
        },
        target=TARGET,
    )

    assert len(builder.specialization_registry) == 2
    records = tuple(builder.specialization_registry.values())
    assert records[0].identity != records[1].identity
    assert records[0].callee != records[1].callee


def test_specialization_identity_tracks_constexpr_target_and_codegen_options():
    baseline = _copy_pipeline._capture(
        signature={"source": "*fp16", "output": "*fp16"}, target=TARGET,
        options={"num_stages": 2},
    )
    equivalent = _copy_pipeline._capture(
        signature={"output": "*fp16", "source": "*fp16"}, target=TARGET,
        options={"num_stages": 2},
    )
    changed_options = _copy_pipeline._capture(
        signature={"source": "*fp16", "output": "*fp16"}, target=TARGET,
        options={"num_stages": 3},
    )
    changed_target = _copy_pipeline._capture(
        signature={"source": "*fp16", "output": "*fp16"},
        target=GPUTarget("cuda", 90, 32), options={"num_stages": 2},
    )

    identities = [next(iter(item.specialization_registry)) for item in (
        baseline, equivalent, changed_options, changed_target,
    )]
    assert identities[0] == identities[1]
    assert len(set(identities)) == 3

    @tm.kernel
    def variable_block(source, output, BLOCK: tm.Const):
        _typed_copy(source, output, BLOCK, at=tm.Time(0))

    block16 = variable_block._capture(
        BLOCK=16, signature={"source": "*fp16", "output": "*fp16"}, target=TARGET
    )
    block32 = variable_block._capture(
        BLOCK=32, signature={"source": "*fp16", "output": "*fp16"}, target=TARGET
    )
    assert next(iter(block16.specialization_registry)) != next(iter(block32.specialization_registry))


def test_same_named_tasks_from_different_modules_do_not_collide():
    def clone(module):
        fn = types.FunctionType(
            _same_name_template.__code__, _same_name_template.__globals__,
            "same_task", _same_name_template.__defaults__, _same_name_template.__closure__,
        )
        fn.__qualname__ = "same_task"
        fn.__module__ = module
        fn.__annotations__ = dict(_same_name_template.__annotations__)
        return tm.Task(fn)

    first_task = clone("timely_test_module_a")
    second_task = clone("timely_test_module_b")

    @tm.kernel
    def pipeline(source, output):
        first_task(source, output, at=tm.Time(0), id="first")
        second_task(source, output, at=tm.Time(0), id="second")

    builder = pipeline._capture(
        signature={"source": "*fp32", "output": "*fp32"}, target=TARGET
    )
    records = tuple(builder.specialization_registry.values())
    assert len(records) == 2
    assert records[0].body_identity != records[1].body_identity
    assert records[0].callee != records[1].callee


def test_scalar_and_constexpr_have_distinct_runtime_abi_roles():
    @tm.kernel
    def scale(source, output, count):
        _typed_scale(source, output, count, 32, at=tm.Time(0))

    builder = scale._capture(
        signature={"source": "*fp32", "output": "*fp32", "count": "i32"},
        target=TARGET,
    )
    record = next(iter(builder.specialization_registry.values()))

    assert record.dynamic_signature == (
        ("source", "*fp32"), ("output", "*fp32"), ("count", "i32"),
    )
    assert record.constexprs == (("BLOCK", 32),)
    assert record.launch_operand_order == ("source", "output", "count")


def test_buffer_refs_and_collectives_preserve_type_and_region_provenance():
    @tm.kernel
    def gather(source):
        first = tm.allgather_shard(source[0], at=tm.Time(0), writes=("G[0]",))
        second = tm.allgather_shard(source[1], at=tm.Time(0), writes=("G[1]",))
        assert first.type_name == second.type_name == "*fp16"
        assert first.provenance.root == second.provenance.root == "G"
        assert first.provenance.region == "[0]"
        assert second.provenance.region == "[1]"

    builder = gather._capture(signature={"source": "*fp16"})
    assert builder.nodes[0].inputs[0].provenance.root == "source"
    assert builder.nodes[0].inputs[0].provenance.region == "[0]"
    assert builder.nodes[1].inputs[0].provenance.region == "[1]"


def test_concrete_inputs_use_triton_mangled_types_without_tensor_operations():
    source = MockTensor(tl.float16, shape=(16,))
    output = MockTensor(tl.float16, shape=(16,))

    builder = _copy_pipeline._capture(source, output, target=TARGET)
    record = next(iter(builder.specialization_registry.values()))
    jit_fn = _typed_copy.triton_jit
    binder = create_function_from_signature(jit_fn.signature, jit_fn.params, make_backend(TARGET))
    bound, ordinary_specialization, raw_options = binder(source, output, 16)
    ordinary_signature = tuple(
        (parameter.name, specialized[0])
        for parameter, specialized in zip(jit_fn.params, ordinary_specialization)
        if specialized[0] != "constexpr"
    )
    _, _, _, ordinary_attrs = jit_fn._pack_args(
        make_backend(TARGET), {}, bound, ordinary_specialization, raw_options
    )

    assert record.dynamic_signature == ordinary_signature
    assert record.attrs == {
        path: attributes for path, attributes in ordinary_attrs.items() if attributes
    }
    assert record.runtime_specialization == tuple(
        (parameter.name, tuple(specialized))
        for parameter, specialized in zip(jit_fn.params, ordinary_specialization)
    )


def test_unknown_input_type_and_legacy_compute_store_fail_before_ir_emission():
    with pytest.raises(TypeError, match="cannot determine a Triton type.*source"):
        _copy_pipeline._capture(target=TARGET)

    @tm.kernel
    def legacy(source, output):
        result = _typed_copy(source, output, 16, at=tm.Time(0))
        tm.store(output[0], result)

    with pytest.raises(TypeError, match="returns only a completion event"):
        legacy._capture(
            signature={"source": "*fp32", "output": "*fp32"}, target=TARGET
        )

    @tm.kernel
    def completion_as_data(source, output):
        done = _typed_copy(source, output, 16, at=tm.Time(0))
        _typed_copy(done, output, 16, at=tm.Time(1))

    with pytest.raises(TypeError, match="completion cannot be used as a data operand"):
        completion_as_data._capture(
            signature={"source": "*fp32", "output": "*fp32"}, target=TARGET
        )

    @tm.kernel
    def mismatched_annotation(source, output):
        _expects_fp32(source, output, at=tm.Time(0))

    with pytest.raises(TypeError, match=r"expected \*fp32, got \*fp16"):
        mismatched_annotation._capture(
            signature={"source": "*fp16", "output": "*fp16"}, target=TARGET
        )


def test_combined_module_has_bindings_real_callee_and_event_only_task():
    text = _copy_pipeline.lower(
        signature={"source": "*fp32", "output": "*fp32"}, target=TARGET
    )

    assert text.count('"tm.binding"') == 2
    assert "arith.constant 1.0 : f32" not in text
    assert "tt.func private @__tm_" in text
    assert "tt.load" in text
    assert "tt.store" in text
    assert '"tm.task"' in text
    assert ") -> !tm.event" in text
    assert ") -> (f32, !tm.event)" not in text
