## Why

当前 Timely Python lowering 为所有数据句柄生成伪 `f32` operand/result，并只在 `tm.task` 上写入未解析的 `callee` 名称；真实 Triton task 则通过 Python registry 独立编译和启动。这使 TM Pass 无法验证 kernel ABI，也无法对 DSL 生成的任务复用 TTIR memory-effect、alias 和 buffer-region 分析，并阻断后续基于 task body 的融合。

## What Changes

- 让每个 `tm.task` 通过 symbol ref 指向同一 MLIR context 中可解析的真实 `tt.func`，由普通 Triton Python AST lowering 生成该 TTIR 计算体。
- 为 Timely orchestration 参数、通信结果和 task operand 建立真实类型传播，以 `!tt.ptr<T>`、标量及必要的 Timely 专用类型替换伪 `arith.constant ... : f32` handle。
- 为 `tm.task` 建立严格的调用契约：校验 callee 存在、参数数量与类型匹配，并统一设备侧输出 buffer 与 completion event 的结果语义。
- 将找不到 task body 时静默退回 annotation 的行为改为显式诊断；opaque host compute kernel 不再冒充普通 `tm.task`。
- 使 Python task/kernel registry 使用与 MLIR callee 相同的稳定 specialization identity，防止执行器启动与 native plan 所分析函数不一致的 kernel。
- 增加从 Python DSL 到同模块 typed TTIR、访问摘要、native plan 和 CUDA launch 的端到端测试，以及 dangling symbol、ABI 不匹配和无类型输入的负向测试。
- **BREAKING**：包含 compute task 的 `lower()`/`plan()` 必须能从显式 Timely signature 或具体编译输入获得完整类型；无法推导类型时不再生成伪 `f32` IR。
- **BREAKING**：普通 Triton kernel 的设备侧结果采用显式输出 buffer；`tm.task` 不再无条件产生 `f32` result。reference executor 的 Python 返回值仅作为 host-side 执行细节，不构成 TM/TTIR ABI。

## Capabilities

### New Capabilities

- `timely-typed-task-linkage`: 定义 Timely compute task 与真实 typed TTIR callee 的链接、验证、访问分析和执行一致性契约。

### Modified Capabilities

无。本 change 补充尚未归档的 `timely-comm-compute-pipeline` 能力，不修改已发布的主规格。

## Impact

- Python 前端：`python/triton/timely/core.py`、`compiler.py` 的 signature、捕获、TTIR 生成和 registry 构建路径。
- TM Dialect 与分析：`TimelyOps.td`、`Ops.cpp`、`TaskAccess` 及 dependency graph pass 的 typed callee 验证与访问摘要逻辑。
- Python/C++ IR 边界：需要在共享 MLIR context 中生成或导入多个 outlined `tt.func`，并暴露稳定的符号/特殊化标识。
- Runtime：reference/CUDA executor 继续消费 native `PlanDescriptor`，但必须验证 descriptor 节点、MLIR callee 与实际 launch artifact 的一一对应。
- 测试与示例：现有依赖隐式 Python 返回值或无类型 `lower(Q=...)` 的 Timely 示例需要迁移到 typed signature 和显式输出 buffer。
- 本轮不实现跨 task persistent-kernel 融合；只建立其所需的真实 task body、类型和数据流前置条件。
- 本轮不新增无 TTIR body 的 external compute task；若后续需要，应以独立 op 和独立 ABI/访问摘要契约提出。
