## Purpose

确保 Timely compute task 在计划分析与实际执行中绑定同一份具有真实类型和函数体的 Triton IR，使 kernel ABI、内存访问、依赖关系和执行 artifact 能够被一致地验证，并彻底消除以 `f32` 常量冒充数据句柄的临时表示。

## ADDED Requirements

### Requirement: Typed Timely orchestration values
系统 SHALL 在生成 TM IR 前确定每个运行期 orchestration 输入、通信结果和 compute task operand 的规范 Triton 类型，并 SHALL 保留其外部 binding、producer 和 buffer-region provenance。类型 SHALL 来自显式编译 signature 或具体输入的 Triton specialization，系统 MUST NOT 为未知 buffer、pointer 或异步数据值生成伪数值类型、伪 pointer 或无 binding 的占位 handle。

#### Scenario: Lower with an explicit signature
- **WHEN** 用户为 Timely kernel 提供 pointer、scalar 和 constexpr 的完整编译 signature
- **THEN** 生成的 TM IR SHALL 使用对应的真实 pointer 和 scalar 类型表示运行期输入，并 MUST NOT 以 `f32` 常量替代这些输入

#### Scenario: Specialize from concrete inputs
- **WHEN** 用户以具体 tensor 和 scalar 输入请求 Timely plan
- **THEN** 系统 SHALL 使用与普通 Triton launch 相同的类型 specialization 规则生成规范 signature，并在同一 plan 中一致复用这些类型

#### Scenario: Reject an unknown data type
- **WHEN** compute task 的任一运行期 operand 既没有显式 signature，也无法从具体编译输入推导类型
- **THEN** lowering SHALL 失败并指出无法确定类型的参数和 task，且 MUST NOT 回退为 `f32` 或其他占位类型

#### Scenario: Propagate a typed collective result
- **WHEN** typed pointer 输入经过 `tm.allgather_shard` 后被 compute task 消费
- **THEN** collective 数据结果 SHALL 保留与输入一致的真实类型，completion event SHALL 作为独立 operand 传递，且 consumer 数据 operand SHALL 与 callee 参数类型匹配

#### Scenario: Exclude constexpr from the runtime ABI
- **WHEN** task 参数被标记为 Triton constexpr 并参与 specialization
- **THEN** 该值 SHALL 进入 specialization identity 和 TTIR 常量环境，但 MUST NOT 出现在 `tm.task` 的动态数据 operands 或 runtime launch ABI 中

#### Scenario: Preserve a sliced buffer binding
- **WHEN** 不同 task 消费同一 pipeline 输入的不同 `BufferRef` 或 shard view
- **THEN** 每个 typed SSA value SHALL 保留共同 root binding 和各自 region provenance，plan 记录的 operand 顺序 SHALL 与 executor 解析出的 runtime launch 实参顺序一致

### Requirement: Resolvable outlined TTIR callee
每个 `tm.task` SHALL 通过 symbol reference 关联在 native planning 时可见且唯一、可解析的 `tt.func`。Python DSL lowering SHALL 将对应 `@tm.task` 的普通 Triton 计算体 outline 为该函数；手写或导入的合法 TTIR MAY 提供等价 callee。系统 MUST NOT 接受只有 Python registry 条目而没有可解析 TTIR callee 的 `tm.task`。

#### Scenario: Outline a Triton task body
- **WHEN** `@tm.task` 计算体包含 `tl.load`、`tl.dot` 和 `tl.store`
- **THEN** lowering SHALL 在 TM module 中生成包含对应 TTIR operations 的 `tt.func`，并使 `tm.task` 的 `callee` 精确引用该函数

#### Scenario: Reject a dangling callee
- **WHEN** `tm.task` 的 callee symbol 在所属 module 中无法解析为 `tt.func`
- **THEN** IR verification SHALL 失败并报告 task id 和缺失的 symbol

#### Scenario: Reuse one identical specialization
- **WHEN** 同一 `@tm.task` 以相同 runtime signature、constexpr 和编译选项被多个任务节点调用
- **THEN** 它们 SHALL 获得同一个稳定 specialization identity，并 MAY 复用同一个 outlined function 和执行 artifact

#### Scenario: Separate distinct specializations
- **WHEN** 同一 `@tm.task` 以不同 runtime signature、constexpr 或影响代码生成的编译选项被调用
- **THEN** 系统 SHALL 生成不同的 callee symbol 和 specialization identity，且 MUST NOT 误用另一 specialization 的函数体或执行 artifact

### Requirement: Verified task ABI and completion semantics
`tm.task` 的数据 operands SHALL 与其 `tt.func` 的运行期参数在数量、顺序和类型上完全匹配；逻辑时间、dependency event 和 constexpr MUST NOT 被视为动态 callee 参数。普通 Triton compute kernel SHALL 通过显式输出 buffer 传递设备侧结果，`tm.task` SHALL 只产生 completion event，而 MUST NOT 无条件产生伪 `f32` 结果。

#### Scenario: Verify a valid output-buffer kernel
- **WHEN** compute task 以 typed input/output pointers 调用返回 `void` 的 Triton kernel
- **THEN** verifier SHALL 接受该调用，`tm.task` SHALL 只返回 `!tm.event`，并由显式 output pointer 表达写入目标

#### Scenario: Reject an ABI mismatch
- **WHEN** `tm.task` operand 与 callee runtime 参数的数量、顺序或类型不一致
- **THEN** verification SHALL 失败且不得生成 `tm.plan`，诊断 SHALL 包含 task id、callee、出错参数索引、期望类型和实际类型

#### Scenario: Reject a non-kernel callee
- **WHEN** callee symbol 无法解析为返回 `void` 的 `tt.func`
- **THEN** verification SHALL 失败且不得生成 `tm.plan`，并报告实际 symbol 类型或不支持的 result signature

#### Scenario: Reject a compute result as data
- **WHEN** 用户继续把 compute completion 当作数据 operand，或调用 `tm.store(target, task_return)` 表达设备输出
- **THEN** Python lowering SHALL 失败并提示把目标 buffer 显式传给 Triton kernel

#### Scenario: Keep reference values outside the device ABI
- **WHEN** reference executor 使用 Python callable 计算或返回 host-side 值
- **THEN** 该值 MAY 保存在 runtime side table 中，但 MUST NOT 改变 TM IR、TTIR callee 或 CUDA launch 的设备 ABI

### Requirement: Body-backed task access validation
系统 SHALL 对每个 `tm.task` 的真实 TTIR callee 执行 memory-effect、alias 和可用的 buffer-region 分析，将 formal 参数上的访问映射回 typed actual operand，并使用结果建立或校验跨 task 数据关系。`reads`、`writes` 和依赖 annotation SHALL 只补充无法由 typed SSA、body 或已知异步操作契约证明的跨 task region；找不到 callee body 时 MUST 失败，而 MUST NOT 静默产生全零 body summary。

#### Scenario: Derive effects from a generated task body
- **WHEN** outlined task body 从一个 pointer 读取并向另一个 pointer 写入
- **THEN** task access summary SHALL 记录对应 body read/write effects，并将可证明的 alias 与 buffer-region 信息交给依赖分析

#### Scenario: Reject an annotation that contradicts the body
- **WHEN** task/call-site annotation 与可证明的 body effect 或 actual-to-formal 映射矛盾
- **THEN** dependency construction SHALL 失败并指出 body、operand 与 annotation 的冲突

#### Scenario: Require annotation only for an unresolved region
- **WHEN** 系统能够从 typed operand 和 task body 证明访问方向，但无法确定跨 task storage region 是否重叠
- **THEN** 系统 SHALL 要求足以消除歧义的 region annotation 或拒绝规划，而 MUST NOT 要求重复声明已经可证明的事实

#### Scenario: Reject analysis without a body
- **WHEN** 普通 `tm.task` 没有可解析的 TTIR body
- **THEN** task access analysis SHALL 失败，且 MUST NOT 仅依赖手写 annotation 继续规划

### Requirement: Plan and launch specialization identity
native plan descriptor SHALL 为每个 compute 节点保留其 callee symbol 和稳定 specialization identity。executor SHALL 只启动与该 identity 对应的已编译 Triton artifact，并 MUST NOT 仅按不含类型信息的 task id 选择 kernel。

specialization identity SHALL 包含影响 ABI、TTIR body 或执行 artifact 的 task 定义/body identity、动态 signature、constexpr、target 和 codegen options，并 MUST NOT 混入普通 invocation id、逻辑时间、发射层或不影响代码生成的调度信息。该 identity 只需在一次编译产物及其 plan/registry 生命周期内稳定，MUST NOT 被当作跨版本或跨进程的公共符号 ABI。

#### Scenario: Launch the analyzed specialization
- **WHEN** executor 执行一个包含 typed compute task 的 native plan
- **THEN** registry 解析出的 artifact SHALL 与该节点的 callee signature、constexpr 和编译选项一致

#### Scenario: Separate invocation and specialization identity
- **WHEN** 多个不同 task node id 调用同一个 compute specialization
- **THEN** `PlanDescriptor` SHALL 分别保留各 invocation id，同时让这些节点引用同一个 callee specialization identity

#### Scenario: Reject a stale or missing artifact
- **WHEN** registry 中缺少 plan 指定的 specialization，或已有 artifact 的 identity 与 plan 不同
- **THEN** executor SHALL 在 launch 前失败并报告节点、callee 和 specialization identity

### Requirement: Preserve Timely scheduling semantics
引入 typed TTIR linkage MUST NOT 合并或重新解释 `TimeOrder`、`DataDep` 和 `ResourceOrder`，也 MUST NOT 让 TTIR 函数在 native plan 中成为额外任务节点。由真实 body 新发现的数据访问 SHALL 参与既有合法性检查和同步规划。

#### Scenario: Preserve a legal communication-compute pipeline
- **WHEN** 原流水的类型、callee ABI 和 annotation 均合法
- **THEN** typed lowering 前后的逻辑发射层及既有显式依赖 SHALL 保持等价，同时访问分析 MAY 增加由真实 body 证明的必要数据依赖

#### Scenario: Reject a newly proven time conflict
- **WHEN** 真实 task body 证明了一条此前被伪 handle 隐藏、且生产者时间晚于消费者的数据依赖
- **THEN** 既有时间合法性检查 SHALL 拒绝该计划，而 MUST NOT 为保持旧行为而忽略该依赖
