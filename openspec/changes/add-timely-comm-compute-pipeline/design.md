## Context

参见 `proposal.md`。正式实现位于 Triton 仓库；现有 `timely/` 是独立 MLIR 正确性原型，尚未接入 Triton Python 前端、TTIR 编译链或 Triton runtime。首个闭环只需证明逻辑时间 DSL 能够表达、验证并执行一个 shard 通信与 GEMM 重叠流水。

## Goals / Non-Goals

**Goals:**

- 建立可运行的 `Timely Python -> TM IR -> 时间/依赖/资源分析 -> 执行计划 -> Triton kernel/runtime` 路径。
- 让 native `tm.plan` 成为执行计划的唯一事实来源，通过结构化 Python binding 驱动所有 executor。
- 让调度参数独立进入逻辑时间公式，同时通过 SSA、通信契约和前端 annotation 建立固有依赖。
- 根据固有依赖生成同步计划，保留资源需求并用可替换的保守策略支撑 MVP 执行。
- 保持 TM 语义与 Triton 的具体流水和后端实现解耦。

**Non-Goals:**

- 首版不支持动态 domain、运行期调度参数或完整的跨 task alias 推理；无法证明的关系要求 annotation 或拒绝编译。
- 首版不实现生产级 NCCL collective、跨节点执行或 device-side collective。
- 首版不保证时间约束给出的计划性能最优，也不实现忽略时间公式并回退为普通循环的可选机制。
- 首版不将整个任务图融合为 persistent kernel，也不重写 TritonGPU 软件流水算法。

## Decisions

### 1. Formal implementation lives in Triton

新增 DSL、Dialect、passes、runtime plan 和测试全部位于 Triton 仓库并进入其构建系统。旧 `timely/` 保留为只读参考，可人工迁移其显式时间 operand、非法调度诊断和正反例测试思想，但新实现不链接或导入旧工程。

备选方案是在 `timely/` 中继续开发并链接 Triton。该方案会长期维护两套驱动、dialect registry 和测试基础设施，因此不采用。

### 2. Use a minimal Python orchestration DSL and ordinary Triton compute tasks

表层只提供 `@tm.kernel`、`@tm.task`、`tm.domain`、`Time/Const`、`tm.allgather_shard(..., at=t)` 和定时 task 调用。`P`、`LAG`、`ROW/COL` 一类调度参数是普通 `Const`，只参与时间表达式，不进入专用 pipeline/serial/persistent 控制结构。首个端到端程序只使用一维 domain 和 `LAG`，但时间表达式 IR 不对常量名称做特殊处理。

`@tm.task` 的计算体使用普通 `triton.language` 运算，例如 `tl.load`、`tl.dot` 和 `tl.store`。异步值的生产者信息通过 SSA 自动传播；非 SSA 可见的 buffer 关系由 task/call-site 的 `reads`、`writes` 或 `depends_on` annotation 补充。annotation 描述数据事实，不直接指定 wait/signal 指令或物理资源顺序。

首版要求 domain 大小和 `LAG` 编译期已知。Timely 前端复用 Triton Python AST/type/codegen 基础设施，但由独立入口生成 TM IR，不把调度结构提前改写为 Python 串行循环。

### 3. Keep time order, data dependence, and resource order separate

最小 IR 包含逻辑时间、completion event、访问/依赖 annotation、资源需求、domain、异步通信、定时 task 调用和 graph/plan 表示。通信产生数据句柄与 completion event；计算消费该数据时，编译器自动建立 event edge。

编译器维护三类不同关系：

- `TimeOrder`：由用户公式产生，只约束操作的逻辑发射层。
- `DataDep`：由数据生产/消费、读写关系和 completion event 产生，决定正确性与同步。
- `ResourceOrder`：由目标资源容量和 placement 规划产生，只决定物理执行时的串并行。

三类边必须在 IR 和检查输出中可区分。时间先后不能证明数据完成；资源顺序也不能替代数据依赖。

### 4. Logical time is normalized issue order

用户公式产生发射键 `T(v)=t0+f(index,consts)`。编译器只保留所有任务之间的 `<`、`=`、`>` 关系，并将不同键保序压缩为连续 rank；因此只有两个时间层时，间隔 `1` 与 `100` 等价。`LAG` 只有在改变任务相对交错关系时才改变调度，不表示 GPU 周期。

相同 rank 的任务不获得彼此之间的额外时间顺序。第一层立即开放；后续逻辑层只有在前一层至少一个任务执行完成后才开放。层开放后，每个任务仍必须等待自己的全部 `DataDep` 和 `ResourceOrder` 前驱完成。例如 `AG1` 与 `GEMM0` 同层时，前一层完成门槛满足后两者均可提交，且各自的依赖条件仍独立生效。

### 5. Build dependencies before planning and reject time conflicts

首版对静态 shard domain 实例化有限任务图。直接 SSA/async value 产生显式边；`reads/writes/depends_on` annotation 产生访问摘要或显式边。分析为可证明的重叠访问建立 RAW/WAR/WAW 关系，对未知关系保守拒绝。

task 内部优先复用 Triton 的 `MemoryEffectOpInterface`、alias、buffer-region、MemoryFrontier 和 Membar 分析。跨 task/domain 新增 TM dependency analysis，消费 task 摘要与 annotation；只有当现有 Triton 分析缺少可复用接口时才扩展它们，不将 Syncopate 的 Python descriptor 直接作为 IR。

依赖图先检查缺失生产者和环。随后对每条 `DataDep(u,v)` 验证 `T(u) <= T(v)`；逆序直接报错，不自动拉长或重写时间。相同时间合法，由同步保证执行正确。合法但低并行度的时间公式保持原样。

### 6. Generate a target-aware issue and resource plan

合法图先按稠密化 `TimeOrder` 生成发射层。同层节点的稳定 tie-break 只用于 IR 和测试复现，不构成语义边。

每个 task 保留资源需求摘要，包括执行资源类别、线程/warp 数和共享内存等。当前 TM 高层只检查单任务请求不超过目标可表示上限，并在同层、同资源类别内生成确定性的保守 `ResourceOrder` 链，作为足以驱动 MVP 的占位策略。精确资源竞争、occupancy 和 placement 留给后续目标相关阶段；替换资源策略不得改变逻辑时间层或删除 `DataDep`。

Pass 边界为：

```text
Timely Python
  -> TM domain/time/task IR
  -> constexpr specialization
  -> issue-time order normalization
  -> SSA/annotation dependency construction
  -> dependency and time-order legality check
  -> target resource requirement analysis
  -> issue-layer and resource planning
  -> synchronization materialization
  -> Triton compute kernels + communication runtime plan
```

### 7. Insert synchronization after resource planning

同步插入 pass 消费 `DataDep`、执行 scope 和资源 placement，选择具体 completion token、stream event、wait/signal、CTA barrier 或目标 mbarrier。`TimeOrder` 本身不触发内存可见性假设；`ResourceOrder` 只有在目标内存模型要求时才带同步效果。

首个 reference backend 使用 communication stream event 和 compute-stream wait。task 内部的共享内存 hazard 继续交给 Triton Membar；跨 task 依赖由 TM pass 在 lowering 边界显式物化。

### 8. Split compute lowering from communication lowering

计算 task outline 为普通 Triton kernel，并完整复用现有 TTIR、TTGIR、pipeline 和目标后端。TM pass 不重新实现 dot lowering、layout 或软件流水。

通信节点通过小型 runtime interface lowering。首个 reference backend 在 communication stream 上把 shard 异步搬运到 gathered buffer 并记录 event；compute stream 按发射层与资源计划提交 Triton kernel，并在消费点依据 `DataDep` 等待 event。调度允许先发射后续 shard 通信，从而与先前 shard 计算重叠。

备选方案是将 `allgather` 直接 lowering 为 TTIR。通用 Triton IR 没有跨 GPU collective 语义，这会过早绑定 NVSHMEM、特定远端内存或目标指令，因此不采用。后续可为同一通信接口增加 NCCL host backend 或 device-side backend。

### 9. Native tm.plan is the sole plan source

`GraphBuilder` 只负责生成带 task id 的 TM MLIR，并保留 `task id -> @tm.task/Triton kernel` 注册表。native pass pipeline 产生的 `tm.plan` 包含 `issue_layers`、`DataDep`、`ResourceOrder` 和 `synchronizations`，随后由结构化 Python binding 转换为 `PlanDescriptor`。

reference executor 和 CUDA executor 只消费 `PlanDescriptor` 与 task registry，不重新扫描 task 的 reads/writes、推导依赖或规划资源。Python 可以负责将 descriptor 中已确定的关系物化为 Future、CUDA stream/event 和 launch，但不得产生新的语义边。

目标链路为：

```text
GraphBuilder -> TM MLIR -> native passes -> tm.plan
             -> PlanDescriptor binding -> Python executor
```

### 10. Gate each issue layer on progress of the previous layer

逻辑层 `L0` 无时间门槛。对于后续层 `Li`，executor 必须先观察到 `Li-1` 中至少一个任务完成，才开放 `Li`。这是 `issue_layers` 的执行语义，不表现为任意选定任务之间的 `DataDep` 或 `ResourceOrder`，也不要求等待前一层全部完成。

### 11. Preserve a future conventional-loop fallback boundary

本次不实现普通循环回退，但 TM task graph 保留与时间公式分离的 domain、计算体、依赖和资源摘要。未来可增加另一种 planning policy，忽略用户时间映射并从依赖图生成常规循环；该策略不得成为首版合法性检查失败后的静默回退。

### 12. Verify overlap through plan structure and an observable reference runtime

编译测试分别检查稠密发射层、三类 graph edges、资源可行性和必要同步；运行测试比较串行 reference 数值结果。reference 通信后端支持可控延迟和执行轨迹，以确定性地证明 `comm(q+1)` 在 `compute(q)` 完成前已发射，而不依赖短 kernel 的偶然计时结果。

## Risks / Trade-offs

- [Reference shard copy 不代表真实 collective 性能] -> 首版只验证 DSL 与调度语义，通信接口保留替换后端的边界。
- [静态实例化任务图无法扩展到很大 domain] -> 将其限定为最小闭环，后续再改为符号化/循环计划。
- [逻辑时间被误解为物理延迟] -> IR 与诊断明确区分 TimeOrder、DataDep、ResourceOrder 和 completion event。
- [用户 annotation 不完整或错误] -> 与可推导 SSA/effect 事实交叉验证，无法证明时保守报错。
- [首版资源模型过粗导致计划次优] -> 允许合法但非最优计划，保持资源 planner 接口可替换。
- [前层任一完成门槛需要运行期观察完成状态] -> reference runtime 等待任一 Future；CUDA MVP 在 host 端查询 completion event，只开放下一层而不添加伪数据依赖。
- [native 与 Python 计划语义漂移] -> 删除 Python 侧分析，测试 `PlanDescriptor` 与打印出的 `tm.plan` 字段一致。
- [过早侵入 TritonGPU 导致维护成本上升] -> TM 在 TTIR/运行时边界前完全消解，首版不改现有 pipeline 算法。

## Migration Plan

该能力是新增且默认关闭，不影响现有 Triton DSL。先接入 dialect 与 host-only IR 测试，再接入 Python DSL、计划 runtime 和 GPU 端到端测试；任一后端失败时可禁用 Timely 入口，不改变现有 Triton 编译路径。旧 `timely/` 在新闭环验收前保持不变。
