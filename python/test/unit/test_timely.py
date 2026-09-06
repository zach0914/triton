import pytest

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton import timely as tm
from triton.timely import runtime as timely_runtime


TARGET = GPUTarget("cuda", 80, 32)


def test_timely_public_api_and_metadata():
    @tm.kernel
    def pipeline(A, B, C, LAG: tm.Const):
        pass

    @tm.task(
        reads=("gathered_A[q]", "B"),
        writes=("C[q]",),
        depends_on=("ag[q]",),
        resources=tm.ResourceSpec("compute", sms=2, threads=128, warps=4),
    )
    def gemm(a, b, c):
        pass

    assert triton.timely is tm
    assert pipeline.kind == "kernel"
    assert pipeline.__name__ == "pipeline"
    assert tuple(pipeline.signature.parameters) == ("A", "B", "C", "LAG")
    assert gemm.kind == "task"
    assert gemm.reads == ("gathered_A[q]", "B")
    assert gemm.writes == ("C[q]",)
    assert gemm.depends_on == ("ag[q]",)
    assert gemm.resources.threads == 128


def test_time_algebra_and_static_domain():
    lag = tm.Const(2, name="LAG")
    issue_time = tm.Time(10) + lag * 3

    assert issue_time.evaluate() == 16
    assert issue_time.evaluate({"LAG": 4}) == 22
    assert (5 // tm.Const(2)).evaluate() == 2
    assert list(tm.domain(3)) == [0, 1, 2]
    assert list(tm.domain((1, 3), (2, 4))) == [
        (1, 2), (1, 3), (2, 2), (2, 3)
    ]


def test_rejects_dynamic_or_invalid_scheduling_values():
    with pytest.raises(TypeError, match="Const values must be integers"):
        tm.Const(1.5)
    with pytest.raises(TypeError, match="compile-time integers"):
        tm.domain((0, "runtime"))
    with pytest.raises(ValueError, match="smaller"):
        tm.domain((2, 2))


def test_pipeline_lowers_to_tm_ir_without_explicit_synchronization(tmp_path):
    @tm.task(reads=("B",))
    def gemm_shard(gathered, B, C):
        tl.store(C, tl.load(gathered) + tl.load(B))

    @tm.kernel
    def pipeline(A, B, C, Q: tm.Const, LAG: tm.Const, II: tm.Const):
        t0 = tm.Time()
        for q in tm.domain(Q):
            issue_comm = t0 + q * II
            gathered = tm.allgather_shard(
                tm.load(A[q]), at=issue_comm, writes=(f"gathered_A[{q}]",)
            )
            gemm_shard(
                gathered, B, C[q], at=issue_comm + LAG * II,
                writes=(f"C[{q}]",),
            )

    compile_args = {
        "Q": 2,
        "LAG": 1,
        "II": 1,
        "signature": {"A": "*fp32", "B": "*fp32", "C": "*fp32"},
        "target": TARGET,
    }
    ir = pipeline.lower(**compile_args)

    assert ir.count('"tm.allgather_shard"') == 2
    assert ir.count('"tm.task"') == 2
    assert "tm.domain" in ir
    assert 'depends_on = ["ag0"]' in ir
    assert 'depends_on = ["ag1"]' in ir
    assert "wait" not in ir
    assert "stream" not in ir

    path = tmp_path / "pipeline.mlir"
    path.write_text(ir)
    context = triton._C.libtriton.ir.context()
    triton._C.libtriton.ir.load_dialects(context)
    assert triton._C.libtriton.ir.parse_mlir_module(str(path), context) is not None

    plan_ir = pipeline.lower_plan(**compile_args)
    assert '"tm.graph"' in plan_ir
    assert '"tm.plan"' in plan_ir
    assert "completion_event" in plan_ir


def test_native_plan_descriptor_and_task_registry_are_execution_source():
    def copy_reference(value, output):
        output[0] = value

    @tm.task(reference=copy_reference)
    def consume(value, output):
        tl.store(output, tl.load(value))

    @tm.kernel
    def pipeline(A, C):
        gathered = tm.allgather_shard(
            A[0], at=tm.Time(0), id="communication", writes=("G",)
        )
        consume(
            gathered, C, at=tm.Time(1), id="compute",
            reads=("G",), writes=("C",),
        )

    plan = pipeline.plan(
        signature={"A": "*fp32", "C": "*fp32"}, target=TARGET
    )

    assert isinstance(plan.descriptor, tm.PlanDescriptor)
    assert tuple(node.id for node in plan.descriptor.nodes) == (
        "communication", "compute",
    )
    assert plan.issue_layers == (("communication",), ("compute",))
    assert {(edge.source, edge.target) for edge in plan.data_deps} == {
        ("communication", "compute"),
    }
    assert plan.synchronizations == (
        tm.Synchronization(
            "communication", "compute", "completion_event", "cross_task"
        ),
    )
    identity = plan.node_specializations["compute"]
    assert plan.task_registry[identity] is consume
    assert plan.kernel_registry[identity] is consume.triton_jit
    assert plan.descriptor.nodes[1].callee == plan.specialization_registry[identity].callee
    assert plan.descriptor.nodes[1].specialization == identity
    assert not hasattr(timely_runtime, "build_plan")


def test_task_body_uses_standard_triton_dot_compilation(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    @tm.task(reads=("A", "B"), writes=("C",))
    def dot_task(A, B, C):
        row = tl.arange(0, 16)
        col = tl.arange(0, 16)
        a = tl.load(A + row[:, None] * 16 + col[None, :])
        b = tl.load(B + row[:, None] * 16 + col[None, :])
        result = tl.dot(a, b)
        tl.store(C + row[:, None] * 16 + col[None, :], result)

    compiled = dot_task.compile(
        {"A": "*fp16", "B": "*fp16", "C": "*fp32"},
        target=GPUTarget("cuda", 80, 32),
    )

    assert "tt.dot" in compiled.asm["ttir"]
    assert "ttg.convert_layout" in compiled.asm["ttgir"]


def test_reference_plan_exposes_overlap_and_preserves_results():
    np = pytest.importorskip("numpy")

    def gemm_reference(gathered, B, C, shard):
        C[shard] = gathered @ B

    @tm.task(reference=gemm_reference, reads=("B",))
    def gemm_shard(gathered, B, C, shard: tl.constexpr):
        tl.store(C + shard, tl.load(gathered) + tl.load(B))

    @tm.kernel
    def pipeline(A, B, C, Q: tm.Const, LAG: tm.Const):
        t0 = tm.Time()
        for q in tm.domain(Q):
            issue_comm = t0 + q
            gathered = tm.allgather_shard(
                A[q], at=issue_comm, writes=(f"G[{q}]",)
            )
            gemm_shard(
                gathered, B, C, q, at=issue_comm + LAG,
                writes=(f"C[{q}]",),
            )

    plan = pipeline.plan(
        Q=2, LAG=1,
        signature={"A": "*fp32", "B": "*fp32", "C": "*fp32"},
        target=TARGET,
    )
    assert plan.issue_layers == (("ag0",), ("ag1", "gemm_shard0"), ("gemm_shard1",))
    assert {(edge.source, edge.target) for edge in plan.data_deps} == {
        ("ag0", "gemm_shard0"),
        ("ag1", "gemm_shard1"),
    }

    A = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    B = np.arange(16, dtype=np.float32).reshape(4, 4)
    C = np.empty((2, 4, 4), dtype=np.float32)
    result = plan.execute(
        {"A": A, "B": B, "C": C},
        communication_backend=tm.AsyncShardCopyBackend(
            {"ag0": 0.005, "ag1": 0.08}
        ),
    )

    np.testing.assert_allclose(C, A @ B)
    events = [(event.kind, event.node, event.detail) for event in result.trace]
    assert ("issue", "ag1", "1") in events
    assert ("issue", "gemm_shard0", "1") in events
    assert events.index(("finish", "ag0", None)) < events.index(
        ("issue", "ag1", "1")
    )
    assert events.index(("start", "gemm_shard0", None)) < events.index(
        ("finish", "ag1", None)
    )


def test_next_issue_layer_opens_after_any_previous_task_finishes():
    np = pytest.importorskip("numpy")

    def consume_reference(value, output):
        output[0] = value + 1

    @tm.task(reference=consume_reference)
    def consume(value, output):
        tl.store(output, tl.load(value) + 1)

    @tm.kernel
    def pipeline(A, C):
        fast = tm.allgather_shard(
            A[0], at=tm.Time(0), id="fast", writes=("fast_buffer",)
        )
        tm.allgather_shard(
            A[1], at=tm.Time(0), id="slow", writes=("slow_buffer",)
        )
        consume(
            fast, C, at=tm.Time(1), id="consumer",
            reads=("fast_buffer",), writes=("C[0]",),
        )

    plan = pipeline.plan(
        signature={"A": "*fp32", "C": "*fp32"}, target=TARGET
    )
    assert plan.resource_order == (tm.Edge("fast", "slow", "MVP_SERIAL"),)

    C = np.empty(1, dtype=np.float32)
    result = plan.execute(
        {"A": np.array([3, 7], dtype=np.float32), "C": C},
        communication_backend=tm.AsyncShardCopyBackend(
            {"fast": 0.005, "slow": 0.08}
        ),
    )
    trace = {(event.kind, event.node): event for event in result.trace}

    assert trace[("finish", "fast")].timestamp <= trace[("open_layer", "1")].timestamp
    assert trace[("issue", "consumer")].timestamp < trace[("finish", "slow")].timestamp
    assert trace[("finish", "fast")].timestamp <= trace[("start", "consumer")].timestamp
    np.testing.assert_array_equal(C, np.array([4], dtype=np.float32))


@pytest.mark.parametrize(
    "shards,m,k,n,lag",
    [(2, 4, 4, 4, 0), (3, 8, 4, 6, 1), (4, 3, 8, 5, 2)],
)
def test_reference_allgather_gemm_matches_sequential(shards, m, k, n, lag):
    np = pytest.importorskip("numpy")

    def gemm_reference(gathered, B, C, shard):
        C[shard] = gathered @ B

    @tm.task(reference=gemm_reference, reads=("B",))
    def gemm_shard(gathered, B, C, shard: tl.constexpr):
        tl.store(C + shard, tl.load(gathered) + tl.load(B))

    @tm.kernel
    def pipeline(A, B, C, Q: tm.Const, LAG: tm.Const):
        for q in tm.domain(Q):
            issue_comm = tm.Time(q)
            gathered = tm.allgather_shard(
                A[q], at=issue_comm, writes=(f"G[{q}]",)
            )
            gemm_shard(
                gathered, B, C, q, at=issue_comm + LAG,
                writes=(f"C[{q}]",),
            )

    rng = np.random.default_rng(17)
    A = rng.standard_normal((shards, m, k), dtype=np.float32)
    B = rng.standard_normal((k, n), dtype=np.float32)
    C = np.empty((shards, m, n), dtype=np.float32)
    plan = pipeline.plan(
        Q=shards, LAG=lag,
        signature={"A": "*fp32", "B": "*fp32", "C": "*fp32"},
        target=TARGET,
    )
    plan.execute({"A": A, "B": B, "C": C})

    np.testing.assert_allclose(C, np.stack([A[q] @ B for q in range(shards)]),
                               rtol=1e-5, atol=1e-5)


def test_reference_executor_obeys_resource_order():
    np = pytest.importorskip("numpy")
    resources = tm.ResourceSpec("compute", sms=1, threads=32, warps=1)

    def increment_reference(value, output, index):
        output[index] = value + 1

    @tm.task(reference=increment_reference, resources=resources)
    def increment(value, output, index: tl.constexpr):
        tl.store(output + index, tl.load(value) + 1)

    @tm.kernel
    def two_tasks(A, C):
        increment(A[0], C, 0, at=tm.Time(), writes=("C[0]",))
        increment(A[1], C, 1, at=tm.Time(), writes=("C[1]",))

    plan = two_tasks.plan(
        signature={"A": "*fp32", "C": "*fp32"},
        target=TARGET,
        target_capacity=tm.TargetCapacity(
            sms=1, threads=2048, warps=64, shared_memory_bytes=233472
        )
    )
    assert plan.data_deps == ()
    assert plan.resource_order == (
        tm.Edge("increment0", "increment1", "MVP_SERIAL"),
    )

    C = np.empty(2, dtype=np.float32)
    result = plan.execute({"A": np.array([2, 5], dtype=np.float32), "C": C})
    events = [(event.kind, event.node) for event in result.trace]
    assert events.index(("finish", "increment0")) < events.index(
        ("start", "increment1")
    )
    np.testing.assert_array_equal(C, np.array([3, 6], dtype=np.float32))


def test_parallelism_const_participates_in_time_formula_and_densifies():
    @tm.kernel
    def folded(A, Q: tm.Const, P: tm.Const, II: tm.Const):
        for q in tm.domain(Q):
            issue_time = tm.Time() + (q // P) * II
            tm.allgather_shard(A[q], at=issue_time, writes=(f"G[{q}]",))

    plan_ir = folded.lower_plan(
        Q=4, P=2, II=100, signature={"A": "*fp32"}
    )

    assert plan_ir.count("tm.raw_issue_time = 0 : i64") == 2
    assert plan_ir.count("tm.raw_issue_time = 100 : i64") == 2
    assert plan_ir.count("tm.issue_rank = 0 : i64") >= 2
    assert plan_ir.count("tm.issue_rank = 1 : i64") >= 2


def test_same_time_forward_annotation_does_not_create_invalid_ssa_order():
    np = pytest.importorskip("numpy")

    def identity_reference(value, output, index):
        output[index] = value

    @tm.task(reference=identity_reference)
    def identity(value, output, index: tl.constexpr):
        tl.store(output + index, tl.load(value))

    @tm.kernel
    def annotated(A, C):
        identity(
            A[0], C, 0, at=tm.Time(), id="consumer", depends_on=("producer",),
            reads=("A[0]",), writes=("C[0]",),
        )
        identity(
            A[1], C, 1, at=tm.Time(), id="producer",
            reads=("A[1]",), writes=("C[1]",),
        )

    compile_args = {
        "signature": {"A": "*fp32", "C": "*fp32"},
        "target": TARGET,
    }
    plan_ir = annotated.lower_plan(**compile_args)
    assert 'from = "producer"' in plan_ir
    assert 'to = "consumer"' in plan_ir
    assert "completion_event" in plan_ir

    plan = annotated.plan(**compile_args)
    assert plan.resource_order == (
        tm.Edge("producer", "consumer", "MVP_SERIAL"),
    )
    result = plan.execute({
        "A": np.array([1, 2], dtype=np.float32),
        "C": np.empty(2, dtype=np.float32),
    })
    events = [(event.kind, event.node) for event in result.trace]
    assert events.index(("finish", "producer")) < events.index(
        ("start", "consumer")
    )


def test_timely_cuda_allgather_gemm_correctness_and_overlap():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    block = 64
    resources = tm.ResourceSpec("compute", sms=1, threads=256, warps=8)

    @tm.task(
        grid=lambda meta: (
            triton.cdiv(meta["M"], meta["BLOCK_M"]),
            triton.cdiv(meta["N"], meta["BLOCK_N"]),
        ),
        resources=resources,
    )
    def gemm_shard(A, B, C, M: tl.constexpr, N: tl.constexpr,
                   K: tl.constexpr, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_start in range(0, K, BLOCK_K):
            a_ptrs = A + offsets_m[:, None] * K + k_start + offsets_k[None, :]
            b_ptrs = B + (k_start + offsets_k[:, None]) * N + offsets_n[None, :]
            a = tl.load(a_ptrs, mask=(offsets_m[:, None] < M) &
                        (k_start + offsets_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(k_start + offsets_k[:, None] < K) &
                        (offsets_n[None, :] < N), other=0.0)
            accumulator += tl.dot(a, b)
        c_ptrs = C + offsets_m[:, None] * N + offsets_n[None, :]
        tl.store(c_ptrs, accumulator,
                 mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N))

    @tm.kernel
    def pipeline(A, B, C, Q: tm.Const, M: tm.Const, N: tm.Const,
                 K: tm.Const, LAG: tm.Const):
        for q in tm.domain(Q):
            issue_comm = tm.Time(q)
            gathered = tm.allgather_shard(
                A[q], at=issue_comm, writes=(f"G[{q}]",)
            )
            gemm_shard(
                gathered, B, C[q], M, N, K, block, block, block,
                at=issue_comm + LAG,
                reads=(f"G[{q}]", "B"), writes=(f"C[{q}]",),
            )

    shards, size = 2, 1024
    shard_elements = 64 * 1024 * 1024
    torch.manual_seed(17)
    A = torch.empty((shards, shard_elements), device="cuda", dtype=torch.float16)
    A_matrix = torch.randn((shards, size, size), device="cuda", dtype=torch.float16)
    A[:, :size * size].copy_(A_matrix.reshape(shards, -1))
    B = torch.randn((size, size), device="cuda", dtype=torch.float16)
    C = torch.empty((shards, size, size), device="cuda", dtype=torch.float32)
    plan = pipeline.plan(
        Q=shards, M=size, N=size, K=size, LAG=1,
        signature={"A": "*fp16", "B": "*fp16", "C": "*fp32"},
        target=triton.runtime.driver.active.get_current_target(),
    )

    plan.execute_cuda({"A": A, "B": B, "C": C})
    C.zero_()
    result = plan.execute_cuda({"A": A, "B": B, "C": C})
    expected = torch.stack([
        torch.matmul(A[q, :size * size].reshape(size, size), B).float()
        for q in range(shards)
    ])
    torch.testing.assert_close(C, expected, rtol=2e-2, atol=2e-1)

    assert result.issue_order == ("ag0", "ag1", "gemm_shard0", "gemm_shard1")
    timings = {timing.node: timing for timing in result.timings}
    assert timings["ag1"].start_ms >= timings["ag0"].end_ms
    assert timings["gemm_shard0"].start_ms >= timings["ag0"].end_ms
    overlap_ms = min(timings["ag1"].end_ms, timings["gemm_shard0"].end_ms) - max(
        timings["ag1"].start_ms, timings["gemm_shard0"].start_ms
    )
    print(f"Timely CUDA AG1/GEMM0 overlap window: {overlap_ms:.3f} ms")
    assert overlap_ms > 0
    assert result.elapsed_ms > 0


def test_timely_cuda_overlap_benchmark_smoke():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    @tm.task(
        grid=lambda meta: (triton.cdiv(meta["N"], 1024),),
        resources=tm.ResourceSpec("compute", warps=4),
    )
    def bandwidth_compute(source, output, N: tl.constexpr):
        offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
        value = tl.load(source + offsets, mask=offsets < N, other=0.0)
        for _ in range(32):
            value = value * 1.0001 + 0.0001
        tl.store(output + offsets, value, mask=offsets < N)

    @tm.kernel
    def pipeline(A, output, Q: tm.Const, N: tm.Const, LAG: tm.Const):
        for q in tm.domain(Q):
            issue_comm = tm.Time(q)
            gathered = tm.allgather_shard(
                A[q], at=issue_comm, writes=(f"G[{q}]",)
            )
            bandwidth_compute(
                gathered, output[q], N, at=issue_comm + LAG,
                reads=(f"G[{q}]",), writes=(f"output[{q}]",),
            )

    shards, elements = 4, 4 * 1024 * 1024
    A = torch.randn((shards, elements), device="cuda", dtype=torch.float32)
    output = torch.empty_like(A)
    compile_args = {
        "Q": shards,
        "N": elements,
        "signature": {"A": "*fp32", "output": "*fp32"},
        "target": triton.runtime.driver.active.get_current_target(),
    }
    overlap_plan = pipeline.plan(LAG=1, **compile_args)
    serial_plan = pipeline.plan(LAG=0, **compile_args)

    overlap_plan.execute_cuda({"A": A, "output": output}, overlap=True)
    serial_plan.execute_cuda({"A": A, "output": output}, overlap=False)
    overlap_times = [
        overlap_plan.execute_cuda({"A": A, "output": output}, overlap=True).elapsed_ms
        for _ in range(3)
    ]
    serial_times = [
        serial_plan.execute_cuda({"A": A, "output": output}, overlap=False).elapsed_ms
        for _ in range(3)
    ]
    overlap_ms = sorted(overlap_times)[1]
    serial_ms = sorted(serial_times)[1]

    print(f"Timely CUDA median: overlap={overlap_ms:.3f} ms serial={serial_ms:.3f} ms")
    assert overlap_ms > 0
    assert serial_ms > 0
