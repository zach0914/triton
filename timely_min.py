import numpy as np
from triton import timely as tm

@tm.task(reference=lambda gathered, b: gathered @ b, reads=("B",))
def matmul(gathered, b):
    pass

@tm.kernel
def pipeline(A, B, C, Q: tm.Const, LAG: tm.Const):
    for q in tm.domain(Q):
        t = tm.Time(q)
        gathered = tm.allgather_shard(A[q], at=t, writes=(f"G[{q}]",))
        result = matmul(gathered, B, at=t + LAG, writes=(f"C[{q}]",))
        tm.store(C[q], result)

native_ir = pipeline.lower_plan(Q=2, LAG=1)
plan = pipeline.plan(Q=2, LAG=1)

A = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
B = np.diag([1.0, 2.0]).astype(np.float32)
C = np.empty_like(A)
plan.execute({"A": A, "B": B, "C": C})

print('"tm.plan"' in native_ir)
print(plan.issue_layers)
print(C.tolist())