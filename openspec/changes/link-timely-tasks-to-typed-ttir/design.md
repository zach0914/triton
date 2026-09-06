## Context

参见 `proposal.md`。当前 `GraphBuilder.to_mlir()` 只知道 Python `Buffer` 名称，为所有运行期值生成 `arith.constant ... : f32`，并让每个 `tm.task` 固定返回 `(f32, !tm.event)`。`callee` 虽然是 `FlatSymbolRefAttr`，但 Python 生成的 module 中没有对应 `tt.func`；`TaskAccess` 找不到函数时静默返回 annotation summary。真实 kernel 由 `Task.compile()` 或 CUDA executor 中的 JIT registry 在另一条路径独立 specialization，因此 native plan 分析的 IR 与实际 launch artifact 没有同一性保证。

Triton 前端已有可复用能力：`ASTSource` 最终调用 `ast_to_ttir`，后者能够在调用方提供的 `ModuleOp` 和 `ir.context` 中直接生成 `tt.func`。但是当前公开入口未完整暴露 existing-module、显式符号名、批量延迟验证和共享 helper return-type 状态；默认 kernel 名也不包含完整 specialization，直接多次插入会造成符号碰撞或 helper 复用错误。

TM ops 与多个 task body 组成的 combined module 是分析和未来融合的中间表示。当前 CUDA backend 的若干阶段要求单一 public entry，因此该 module 不能原样进入完整 TTIR -> TTGIR -> PTX pipeline；host-stream executor 仍需逐 task specialization 独立编译和 launch。

## Goals / Non-Goals

**Goals:**

- 让 Python DSL 生成的每个 `tm.task` 在 native planning 时都能解析到真实、typed、可验证的 `tt.func`。
- 建立唯一的 specialization record，使分析用 TTIR、`tm.task.callee`、plan descriptor 和运行 artifact 使用相同 ABI 与代码生成输入。
- 为外部 binding、shard view、collective result 和 task operand 保留真实类型与 provenance。
- 让 task body 的 effect/alias/region 信息能够映射回实际 operand，并与跨 task annotation 共同驱动依赖检查。
- 保持现有 `TimeOrder`、`DataDep`、`ResourceOrder`、同步计划和 host-stream executor 边界不变。

**Non-Goals:**

- 本轮不实现跨 task persistent-kernel fusion，也不把 combined module 直接 lowering 到 TTGIR/PTX。
- 本轮不支持无 TTIR body 的 opaque external compute task；通信 operation 继续使用独立 runtime contract。
- 本轮不根据参数名称或默认 `f32` 猜测未知 dtype，也不引入完整 tensor shape/type system。
- 本轮不承诺完整的跨 task 精确 alias；无法证明的 region 继续要求 annotation 或拒绝规划。
- 本轮不重写逻辑时间、资源规划或同步策略。

## Decisions

### 1. Keep `tm.task` outlined through a symbol reference

保留 `tm.task callee=@symbol` 结构，不给 `tm.task` 增加内嵌 TTIR region。每个 callee 在 native planning 可见的符号作用域内解析为 outlined `tt.func`；Python DSL 生成的 combined module 将这些函数放在同一 module 中，并使用 private visibility 表示它们是分析/融合候选而非该 module 的多个最终设备入口。

该方案与现有 ODS、`SymbolTable::lookupNearestSymbolFrom<triton::FuncOp>` 和普通 Triton function body 兼容，也允许手写或结构化导入 TTIR。内嵌 region 会复制 Triton function 的参数、可见性、helper symbol 和递归调用语义，因此不采用。只有 registry、没有 TTIR symbol 的普通 `tm.task` 将被拒绝；未来若需要 opaque compute，应另设 operation 和完整外部 ABI contract。

### 2. Introduce one canonical task specialization record

捕获后为每种 task specialization 构造不可变 record，至少包含：

- task 定义/body identity；
- 规范化动态 signature；
- constexpr 路径和值；
- target、backend 和影响代码生成的 options/attrs；
- 稳定 callee symbol；
- runtime launch 参数顺序；
- 编译 artifact cache key。

callee symbol 使用可读 task 前缀加上述 codegen identity 的 hash，例如 `__tm_matmul_<hash>`。普通 invocation id、逻辑时间、issue rank、资源边和 launch grid 不进入该 identity；多个 shard 节点可以引用同一 specialization。identity 只在对应编译产物和 plan 生命周期内稳定，不作为跨版本 ABI。

同一 callable 出现不同类型、constexpr、target 或 codegen options 时必须产生不同 record/symbol。实现初期如果某个组合无法安全共存，可以在生成 IR 前明确拒绝，而不能复用同名函数。

### 3. Derive types from explicit signatures or normal Triton specialization

为 `Kernel.lower()`、`lower_plan()` 和 `plan()` 增加显式 `signature`、`target` 和 codegen `options` 输入；离线 lowering 可使用类似 `{"A": "*fp16", "B": "*fp16", "C": "*fp32"}` 的 Triton signature。用户提供具体 tensor/scalar 时，前端复用普通 Triton binder、`mangle_type` 和 backend option normalization 得到相同的 canonical types，而不另写 dtype 推断规则。无 GPU 的 compile-only 测试可显式传入 `GPUTarget`。

GraphBuilder 为每个 graph value 保存 type constraint 和 provenance。约束来源包括外部 kernel signature、具体输入 specialization、Python literal、task formal 参数和已知 operation contract；`allgather_shard` 强制 source/result 同型。Triton `tl.constexpr` 参数进入 record 的 constants，但不成为 `tm.task` 动态 operand。所有约束在发射 MLIR 前统一求解；冲突或未解析值直接诊断。

备选方案是在每个 `@tm.task` 上重复声明完整 signature。它可以作为独立 `Task.compile()` 的兼容入口，但不作为 graph lowering 的唯一事实来源，因为 call-site typed values 已经决定实际 ABI，重复声明容易漂移。

### 4. Model runtime inputs with typed TM bindings

新增 typed `tm.binding`（最终命名可在实现时遵循 Dialect 命名约定）来表示 executor 在运行期提供的输入或 view。operation 结果使用真实 Triton pointer/scalar type，并保存 root binding、索引/region 和稳定的 runtime lookup key，例如 `A[0]`。它是 host orchestration ABI 的符号化输入，不是数值常量、null pointer 或可执行的设备 load。

直接 Buffer、`BufferRef` 和 shard view 分别保留 root 与 region provenance；不同 shard 可以有同一 root、不同 region。`allgather_shard` 返回与 source 同型的数据 SSA value以及独立的 `!tm.event`。`tm.binding` 和 TM scheduling ops 在计划生成后留在 host plan 边界，不进入单个 Triton kernel 的后端 lowering。

仅把伪 `f32` 换成无来源的 `!tt.ptr<T>` 仍然无法保证 runtime 参数绑定和跨 task region，因此不采用。

### 5. Make compute completion event-only

普通 top-level Triton kernel 返回 `void`，设备结果通过显式 output pointer 写入。相应地，`tm.task` 改为只产生 `!tm.event`；其 data operands 与 outlined `tt.func` 的动态参数一一对应，time 和 dependency event 是 scheduling operands，不属于 callee ABI。

Python DSL 中 compute task 必须显式接收输出 Buffer/BufferRef，例如：

```python
@tm.task(reads=("G", "B"), writes=("C",), grid=(1,))
def matmul(gathered, b, c):
    value = ...
    tl.store(c, value)

done = matmul(gathered, B, C[q], at=t)
```

旧的 `result = task(...); tm.store(C[q], result)` 不再表示合法设备 ABI，前端应给出迁移诊断。reference callable 同样应以显式输出参数为主要契约；其 Python 返回值可以保存在 host trace/side table 中，但不能成为后续 TM dataflow 或改变 CUDA ABI。

### 6. Build TTIR directly in one shared context and module

新的 lowering 顺序为：

```text
capture graph and type constraints
  -> build TaskSpecialization records
  -> create target/backend/options and one MLIR context
  -> emit/parse typed TM scheduling IR in that context
  -> batch-lower every unique task AST into the same ModuleOp
  -> verify symbols and ABI once all callees exist
  -> run TM analysis/planning passes
  -> emit PlanDescriptor plus specialization registry
```

实现提供受支持的 compiler helper，在 existing `ModuleOp` 上复用 `ast_to_ttir` 能力，并显式支持：

- 调用方指定 entry symbol 与 visibility，避免 `JITFunction.repr` 和短函数名碰撞，并把分析用 task body 标记为 private；
- 一批 task 共享 helper function/return-type cache；
- 同名 helper 的签名检查与安全复用；
- batch 期间延迟 module verification，所有 callee 插入后统一验证；
- 一个 context 中加载 core、Triton、TM 和 target backend dialect。

不从 `compiled.asm["ttir"]` 拼接字符串，也不把另一个 context 的 `FuncOp` 直接 `push_back`。前者丢失结构化 symbol ownership，后者不是跨-context linker，并且都难以正确迁移 private helper closure。

TM 文本在所有 callee 插入前可能暂时包含 dangling symbol，因此 batch helper 不在中间态运行完整 module verification；最终 `module.verify()` 和 TM pass pipeline 必须严格失败。直接由 `triton-opt` 或其他调用方验证的孤立 dangling `tm.task` 仍然非法。

### 7. Verify the call contract before planning

`TaskOp::verify()` 或等价 symbol-user verifier 解析 callee，并检查：

- symbol 是 `triton::FuncOp` 且在当前阶段具有 body；
- callee 返回 `void`；
- `inputs` 与 callee runtime arguments 数量、顺序、类型一致；
- constexpr 已从动态 ABI 中移除；
- time/dependency operands 和唯一 completion result 满足 TM 类型约束。

诊断包含 task id、callee、参数索引及 expected/actual type。任何失败发生在 `tm.graph`/`tm.plan` 生成前。ODS 不再为 compute task 声明任意 data result；`AnyType` 仅在确实需要承载 typed data 的 operation 上使用。

### 8. Map body accesses back to actual bindings

`TaskAccessSummary` 从只有计数扩展为按 formal argument 记录 read/write effect、alias root 和可用 footprint。分析随后把 formal argument 映射到对应 `tm.task` actual operand，并沿 `tm.binding`、`allgather_shard` 和已知 producer 传播 storage provenance。

能够由 typed SSA、known operation contract 和 body footprint 证明的关系直接进入 dependency graph，不要求用户重复 annotation。只有跨 task region 重叠仍不确定时才消费 `reads`/`writes`；annotation 与可证明 body effect 或 binding 矛盾时失败。callee 缺失不再生成全零 summary。

精确 region 映射无法完成时保持保守：要求补充 annotation 或拒绝，而不是假定独立。

### 9. Bind the native plan to the launched artifact

`PlanNodeDescriptor` 为 compute node 增加 callee symbol 和 specialization identity；invocation id 继续用于图节点和 tracing。specialization registry 以 identity 为 key，value 保存 Task/JIT callable、canonical signature、constexpr、target/options 及可选已编译 artifact。node registry 只负责把 invocation 映射到 specialization，不再用 node id 充当 kernel ABI identity。

analysis TTIR 和实际 artifact 均从同一个 record 构造。CUDA executor 在 launch 前检查 resolved inputs 的 Triton mangled types、参数顺序和 artifact metadata；缺失、过期或不匹配直接失败。reference executor 不需要设备 artifact，但使用相同的 typed operand 顺序和显式 output binding。

### 10. Keep combined analysis IR out of the ordinary multi-entry backend path

combined module 保留 TM ops、多个 private task body 和 helper，仅运行 TTIR-level task access analysis 与 TM planning。host-stream CUDA 路径仍对每个 specialization 使用普通 Triton pipeline 独立产生可 launch artifact。

未来 persistent fusion 可以消费这些真实 private bodies，生成单一 public persistent entry，删除或转换 host-only bindings/TM ops，再进入 TTGIR/backend。未完成这一步前，MUST NOT 将 combined module 直接送入要求单一 public entry 的 PTX/instrumentation pipeline。

## Risks / Trade-offs

- [显式类型使旧的 `lower(Q=...)` 失败] -> 提供明确 signature/concrete-input 两种入口和参数级诊断，迁移所有示例后再删除伪 handle。
- [不同 specialization 或 Python module 的同名 task 发生符号碰撞] -> 使用包含 body、signature、constexpr、target/options 的 identity 命名，并在插入前检查 symbol/type。
- [多个 task 共享 private helper 时 return-type cache 漂移] -> 一轮 batch lowering 共享 function type state，并增加两个 entry 共用 helper 的回归测试。
- [分析 TTIR 与实际 JIT specialization 不一致] -> 两者只接受同一个 immutable record，descriptor 和 executor 在 launch 前验证 identity。
- [typed pointer 有类型但缺少真实 runtime provenance] -> 所有外部值必须来自 `tm.binding` 或已知 producer，禁止无 binding pointer placeholder。
- [body region 映射不能覆盖所有 Triton pointer 算术] -> 已证明部分自动使用，未知部分要求 annotation 或拒绝，保持现有严格合法性原则。
- [combined module 的多个 task body违反后端单入口假设] -> combined module 只用于分析/planning；每个 task 独立 codegen，persistent fusion 后才生成一个 backend entry。
- [batch 中间态暂时包含 dangling symbol] -> 只在内部受控 builder 中延迟验证，最终 verification 和所有公共 IR 工具保持严格。

## Migration Plan

1. 先加入 specialization record、typed graph value 和 `tm.binding`，同时保留旧 emitter 仅供对照测试。
2. 接入 shared-context batch TTIR lowering和严格 callee/ABI verifier，迁移 lit 测试中的 dangling `@gemm` 与错配 `f32` result。
3. 将 compute task 改为 event-only，并把 Python/reference/CUDA 示例迁移为显式输出 buffer。
4. 扩展 access summary、PlanDescriptor 和 specialization registry，开启 launch identity 检查。
5. 在 typed DSL、negative verifier、shared helper、multi-specialization、reference 和 CUDA 测试全部通过后删除旧 `handle()->f32` 路径。

Timely 仍是实验能力，回滚时可以禁用新的 typed lowering 入口并恢复上一个 change；不得保留“验证失败后静默使用伪 f32”的运行期回退。
