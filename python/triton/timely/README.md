# Timely DSL Prototype

Timely is an experimental orchestration DSL inside Triton. It captures a
finite task graph, while ordinary `@tm.task` bodies continue to compile through
Triton's existing TTIR, TTGIR, and target backends.

```text
Timely Python
  -> TM time/domain/task IR
  -> issue-time normalization
  -> DataDep validation
  -> conservative MVP resource planning
  -> native tm.plan
  -> structured PlanDescriptor binding
       |-> reference executor
       `-> CUDA communication stream + Triton compute stream
```

The compiler keeps three relations separate:

- `TimeOrder` controls issue layers only.
- `DataDep` represents correctness dependencies and creates completion waits.
- `ResourceOrder` currently represents conservative same-class serialization.

The native `tm.plan` is the sole scheduling source. Python executors consume its
issue layers, dependency edges, resource edges, and synchronization records;
they do not repeat dependency or resource analysis.

## Scenario 1: Allgather and GEMM pipeline

```python
from triton import timely as tm

@tm.task(reference=lambda gathered, b: gathered @ b,
         reads=("B",))
def gemm_shard(gathered, b):
    # A production task body uses triton.language operations here.
    pass

@tm.kernel
def pipeline(A, B, C, Q: tm.Const, LAG: tm.Const):
    for q in tm.domain(Q):
        comm_time = tm.Time(q)
        gathered = tm.allgather_shard(
            A[q], at=comm_time, writes=(f"G[{q}]",)
        )
        result = gemm_shard(
            gathered, B, at=comm_time + LAG, writes=(f"C[{q}]",)
        )
        tm.store(C[q], result)

tm_ir = pipeline.lower(Q=4, LAG=1)
plan_ir = pipeline.lower_plan(Q=4, LAG=1)
plan = pipeline.plan(Q=4, LAG=1)
```

`pipeline.lower()` returns task IR. `pipeline.lower_plan()` runs the native TM
passes and returns inspectable `tm.graph` and `tm.plan` IR. `pipeline.plan()`
binds that native `tm.plan` to an immutable `PlanDescriptor`, then attaches only
the captured launch payloads and Triton task registry needed by the executors.

For CUDA execution, the compute task writes its output explicitly and supplies
the same grid metadata as a normal Triton kernel:

```python
import triton
import triton.language as tl
from triton import timely as tm

@tm.task(
    grid=lambda meta: (triton.cdiv(meta["N"], 256),),
    resources=tm.ResourceSpec("compute", warps=4),
)
def consume_shard(gathered, output, N: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    values = tl.load(gathered + offsets, mask=offsets < N)
    tl.store(output + offsets, values, mask=offsets < N)

@tm.kernel
def cuda_pipeline(A, C, Q: tm.Const, N: tm.Const, LAG: tm.Const):
    for q in tm.domain(Q):
        communication_time = tm.Time(q)
        gathered = tm.allgather_shard(
            A[q], at=communication_time, writes=(f"G[{q}]",)
        )
        consume_shard(
            gathered, C[q], N, at=communication_time + LAG,
            reads=(f"G[{q}]",), writes=(f"C[{q}]",),
        )

plan = cuda_pipeline.plan(Q=4, N=1 << 20, LAG=1)
result = plan.execute_cuda({"A": A, "C": C}, overlap=True)
print(result.elapsed_ms, result.timings)
```

The CUDA backend currently implements `allgather_shard` as an asynchronous
same-device shard copy. It validates stream/event scheduling, but it is not an
NCCL or multi-GPU collective.

## Scenario 2: Same-time issue with a dependency

```python
communication_time = tm.Time(q)
gathered = tm.allgather_shard(A[q], at=communication_time)
result = consume(gathered, at=communication_time)  # LAG = 0
```

Both nodes are in one issue layer. If this is not the first layer, it opens only
after at least one task in the preceding layer completes. The consumer still
receives a `DataDep` and waits for its communication completion event. Equal
logical time does not mean that the dependency disappears.

## Scenario 3: Execution folding through an independent parameter

```python
@tm.kernel
def folded(A, Q: tm.Const, P: tm.Const, II: tm.Const):
    for q in tm.domain(Q):
        issue_time = tm.Time() + (q // P) * II
        tm.allgather_shard(A[q], at=issue_time, writes=(f"G[{q}]",))
```

`P` is an ordinary compile-time constant. It changes the mapping from shard to
issue layer without introducing a special parallel or pipeline loop. `II=1`
and `II=100` produce the same dense ranks when their relative ordering is the
same.

## Scenario 4: Non-SSA dependencies and resource constraints

```python
resources = tm.ResourceSpec(
    "compute", sms=2, threads=256, warps=8, shared_memory_bytes=32768
)

@tm.task(
    reads=("gathered_A[q]", "B"),
    writes=("C[q]",),
    depends_on=("producer_task",),
    resources=resources,
)
def annotated_compute(...):
    ...
```

`reads` and `writes` describe data facts. `depends_on` names an explicit
producer when SSA is insufficient. They do not request a particular barrier or
stream. The MVP resource planner validates each task against coarse target
limits, then conservatively chains tasks in the same issue layer and resource
class. Precise occupancy and placement are deferred to a target-specific stage.

## Semantic Boundary

- `Const` values are compile-time scheduling parameters. They enter ordinary
  logical-time formulas and are not encoded as pipeline or serial constructs.
- Logical time defines issue order only. Its distinct values are densified, so
  gaps have no physical duration meaning.
- `TimeOrder`, `DataDep`, and `ResourceOrder` remain separate. Only `DataDep`
  creates completion synchronization; capacity conflicts create
  `ResourceOrder`.
- The first issue layer opens immediately. Every later layer opens after any one
  task in the preceding layer completes; it does not wait for the whole layer.
- Native `tm.plan` is the only source of issue, dependency, resource, and
  synchronization relations. Python carries launch values but derives no edges.
- SSA results, completion events, `reads`, `writes`, and `depends_on` establish
  data dependencies. Missing producers, cycles, unknown accesses, and a
  producer issued after its consumer are compile errors.
- Same-time dependencies are legal: both nodes may be submitted, while the
  consumer waits for producer completion.
- Legal but suboptimal schedules are preserved.

The first reference communication backend is an asynchronous local shard copy,
not a production collective. Dynamic domains, runtime scheduling parameters,
NCCL or multi-node communication, a persistent fused graph, and conventional
loop fallback are outside this prototype.

All implementation, build integration, passes, and tests are self-contained in
the Triton repository. No external Timely prototype is imported or linked.

## Testing

Build and run all host-side compiler and DSL tests:

```bash
make PYTHON=python3.12
cd build/cmake.linux-x86_64-cpython-3.12
/home/zailin/.conda/envs/zailin_triton/bin/lit -v test/Timely
cd ../../
PYTHONPATH=python python3.12 -m pytest -s --tb=short \
  python/test/unit/test_timely.py -k "not cuda"
```

Run device correctness and overlap tests on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python3.12 -m pytest -s --tb=short \
  python/test/unit/test_timely.py -k "timely_cuda"
```

Correctness testing has four layers:

1. Lit tests verify parser/verifier behavior, dense issue ranks, edge kinds,
   legality diagnostics, resource planning, and synchronization materialization.
2. Python tests compare generated TM IR and native `lower_plan()` output, verify
   lossless `PlanDescriptor` binding, and confirm no Python planner remains.
3. The reference executor compares several shard shapes and `LAG` values with a
   sequential NumPy result.
4. CUDA tests compare Triton `tl.dot` output with PyTorch and use CUDA events to
   verify both the preceding-layer gate and an actual `AG1/GEMM0` overlap interval.

For performance work, warm up compilation and allocation first, run several
iterations, and report the median rather than asserting a fixed speedup. Compare
the same workload using `execute_cuda(overlap=True)` and a one-stream baseline
using `execute_cuda(overlap=False)`. Also report per-node CUDA event intervals;
wall-clock improvement alone cannot prove that the intended operations
overlapped. Sweep `LAG`, `P`, shard size, compute size, and resource summaries.
Use Nsight Systems once the runtime interface is replaced with a real collective.
