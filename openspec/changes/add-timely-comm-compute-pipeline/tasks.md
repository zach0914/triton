## 1. TM Compiler Foundation

- [x] 1.1 在 Triton 构建与 dialect registry 中接入独立 TM Dialect，并用 `triton-opt` parser/printer round-trip 测试验证该 dialect 可独立加载。
- [x] 1.2 建立 Timely Python 入口及 `@tm.kernel`、`@tm.task`、`Const`、`Time` 的最小对象模型，并验证导入和装饰器元数据测试通过。

## 2. Minimal DSL And IR

- [x] 2.1 定义逻辑发射时间、completion event、访问/依赖 annotation、资源需求、静态 domain、异步 shard 通信、定时 task 调用及 graph/plan 的最小 ops/types，并用合法与非法 IR verifier 测试覆盖其结构约束。
- [x] 2.2 扩展/复用 Triton Python AST lowering，使示例中的 `tm.domain`、一等公民 `Const` 时间参数、`allgather_shard` 和定时 `@tm.task` 调用生成预期 TM IR，并用 golden IR 测试确认用户源码不含专用调度控制流或显式 wait/stream。
- [x] 2.3 让 `@tm.task` 计算体接受普通 `triton.language` 运算并可 outline 为标准 Triton kernel，验证一个最小 `tl.dot` task 能通过既有 TTIR/TTGIR 编译链。
- [x] 2.4 为 task/call-site 增加最小 `reads`、`writes`、`depends_on` annotation，并验证合法 annotation 进入 IR、缺失或矛盾 annotation 产生明确诊断。

## 3. Time And Dependence Analysis

- [x] 3.1 实现静态 domain 和调度 `Const` 的 constexpr 特化及逻辑时间保序稠密化，并验证仅有两个时间层时 `{t,t+1}` 与 `{t,t+100}` 结果相同、运行期调度参数被拒绝。
- [x] 3.2 定义跨 task 依赖摘要接口，复用 Triton `MemoryEffectOpInterface`、alias 和 buffer-region 分析收集 task 内访问，并验证可推导的 RAW/WAR/WAW 摘要正确。
- [x] 3.3 合并 SSA/async completion、访问 annotation 和显式 `depends_on`，建立 `DataDep` 图；验证无法证明的别名关系被拒绝且通信到计算依赖无需源级 wait。
- [x] 3.4 实现缺失生产者、依赖环及 `T(producer) > T(consumer)` 检查，并验证全部直接报错；同时验证同时间依赖和合法但次优的时间映射被接受。
- [x] 3.5 生成可检查的 `TimeOrder` 与 `DataDep` 图，验证改变 `LAG` 只改变时间交错、同层稳定输出顺序不增加语义边。

## 4. Resource And Synchronization Planning

- [x] 4.1 实现 task 资源需求摘要和目标容量查询，至少覆盖执行资源类别、线程/warp 数与共享内存，并验证超出单任务上限时产生诊断。
- [x] 4.2 实现发射层与资源规划 pass，为容量冲突生成独立 `ResourceOrder`，并验证该 pass 不修改 `TimeOrder` 或删除 `DataDep`。
- [x] 4.3 实现同步插入 pass，根据 `DataDep`、执行 scope 和 placement 物化 completion token/event/wait，复用 Triton Membar 处理 task 内共享内存 hazard，并验证不同时间但无依赖时不会虚构同步。

## 5. Executable Reference Plan

- [x] 5.1 实现最小通信 runtime interface 和异步 shard-copy reference backend，验证 communication stream 能产生可供其他执行资源等待的 completion event。
- [x] 5.2 实现 plan executor，按稠密发射层提交通信节点和 outline 后的 Triton kernel，并遵守 `DataDep` 与 `ResourceOrder`，验证 launch/event 序列与三类计划关系一致。
- [x] 5.3 增加可控通信延迟与执行轨迹，验证前一层至少一个任务完成后 `AG1` 与 `GEMM0` 同层开放，并分别遵守自身依赖。

## 6. End-To-End Acceptance

- [x] 6.1 实现串行 `allgather -> GEMM` reference，并验证多个静态 shard 形状及 LAG 配置下 Timely 流水结果与 reference 一致。
- [x] 6.2 增加时间间隔等价、同时间发射、依赖 annotation、时间逆序、依赖环、资源容量、同步插入、Triton kernel 编译和 overlap 的组合测试，并运行相关 lit/pytest/GPU 测试确认全部通过。
- [x] 6.3 在不引用或构建外部 `timely/` 原型的环境中验证 Timely 能力，并记录调度参数独立性、三类计划关系、严格合法性规则、资源/同步边界及普通循环回退不属于首版。

## 7. Native Plan Contract

- [x] 7.1 为 native `tm.plan` 增加结构化 `PlanDescriptor` Python binding，完整导出节点、发射层、`DataDep`、`ResourceOrder` 和同步记录。
- [x] 7.2 让 `GraphBuilder` 保留稳定的 `task id -> @tm.task` 注册表，并验证 descriptor 中每个计算节点都能解析到对应 Triton kernel。

## 8. Single-Source Execution

- [x] 8.1 重构 `@tm.kernel.plan()`，使其运行 native pass pipeline并从 `tm.plan` 构造执行计划；删除 Python `build_plan()` 中重复的依赖分析、合法性检查和资源分析。
- [x] 8.2 重构 reference executor，只消费 native descriptor 中的关系，并实现“前一逻辑层至少一个任务完成后开放下一层”的门槛。
- [x] 8.3 重构 CUDA executor，直接消费 descriptor 的发射层、依赖边、资源边和同步记录，并以 CUDA event 实现相同的层开放语义。

## 9. MVP Resource Policy And Acceptance

- [x] 9.1 将 TM 高层资源 Pass 收敛为单任务上限检查与同层、同资源类别保守串行化，明确精确资源竞争留给后续目标相关阶段。
- [x] 9.2 增加 descriptor 无损转换、唯一计划来源、任一前层任务完成门槛及保守资源边测试，并运行 lit/host pytest。
- [x] 9.3 运行 GPU 正确性与 overlap 回归，更新文档并通过 OpenSpec strict validation。
