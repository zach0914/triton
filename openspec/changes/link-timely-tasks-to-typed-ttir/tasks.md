## 1. IR Contract And Regression Baseline

- [x] 1.1 为当前缺陷增加回归基线：覆盖 dangling `tm.task` callee、void `tt.func` 配伪 `f32` result、无 binding 的 pointer handle，并验证这些 case 在新 verifier 下均失败且不会生成 `tm.plan`。
- [x] 1.2 在 TM ODS 中增加 typed runtime binding operation，定义 root、索引/region、lookup key 和任意真实 result type，并用 parser/printer、重复/非法 binding verifier 测试确认 provenance 可检查。
- [x] 1.3 将 `tm.task` 改为 event-only result，保留 variadic typed data inputs、time、dependency 和 callee，并更新合法 hand-written MLIR 使 compute 输出全部通过显式 pointer operand 表达；运行 `test/Timely` lit round-trip 测试验证新语法。
- [x] 1.4 为 `tm.task` 实现严格 symbol/ABI verifier，检查 callee 是有 body、返回 void 的 `tt.func`，并检查动态参数数量、顺序和类型；用 missing symbol、wrong symbol kind、arity、参数类型和 result signature 负测逐项验证诊断内容。

## 2. Typed Python Graph Model

- [ ] 2.1 为 Buffer、BufferRef、collective AsyncValue 和 compute completion 建立分离的 typed/provenance 数据模型，并验证 shard views 保留共同 root、不同 region，compute completion 不能当作数据输入。
- [ ] 2.2 为 `Kernel.lower()`、`lower_plan()` 和 `plan()` 接入显式 signature、target 与 codegen options，复用 Triton canonical type/backend option 解析；用无 GPU `GPUTarget` 测试验证 pointer、scalar 和 constexpr 被正确分类。
- [ ] 2.3 接入具体 tensor/scalar 参数的普通 Triton specialization 路径，捕获时将实际参数转换为 typed runtime bindings 而不执行用户 tensor 运算；用 mock/concrete 参数测试验证其 signature 与普通 Triton binder 一致。
- [ ] 2.4 实现 graph type constraint 求解与传播，至少覆盖 external binding、BufferRef、Python scalar、allgather source/result 同型及 task actual/formal；用类型冲突和未解析参数测试验证 lowering 直接失败且不会猜测 dtype。
- [ ] 2.5 重写 TM emitter，使运行期输入来自 typed binding、collective 产生 typed data 加 event、compute task 只产生 event，并验证生成 IR 不再包含用于 Buffer/AsyncValue 的 `arith.constant ... : f32`。
- [ ] 2.6 为旧 `result = task(...); tm.store(target, result)` 增加明确迁移诊断，并将 reference-side Python 返回值隔离在 host side table；用负测确认返回值不能进入后续 TM dataflow。

## 3. Canonical Task Specialization

- [ ] 3.1 实现 immutable task specialization record，规范化 body identity、dynamic signature、constexpr、target、codegen options、launch operand order 和 artifact key；用等价输入测试 identity 稳定，用任一 codegen 输入变化测试 identity 分离。
- [ ] 3.2 生成带可读前缀和 specialization hash 的唯一 callee symbol，显式区分 invocation id 与 specialization identity；用多 shard 同 specialization、同名不同 Python module 及 fp16/fp32 specialization 测试验证无碰撞或误复用。
- [ ] 3.3 将 GraphBuilder 的 node、callee attr、specialization registry 和 runtime operand binding 全部连接到同一 record，并用断言测试确认没有仅以 task name/node id 代替 specialization 的旁路。

## 4. Shared-Context TTIR Lowering

- [ ] 4.1 为 Triton AST lowering 增加受支持的 existing-module batch helper，支持调用方指定 entry symbol/visibility、共享 MLIR context、延迟最终 verification 和共享 helper function type state；用两个 private task body 的同模块生成测试验证 module 合法。
- [ ] 4.2 处理多个 task 共用 private `@triton.jit` helper 的插入与 return-type 复用，并用两个 entry 调用同一 helper 的回归测试确认无 `KeyError`、重复 ownership 或错误函数类型。
- [ ] 4.3 在 Timely lowering 中按 target 初始化 backend/dialects/options，在同一 context/module 中把每个唯一 specialization 的 Python AST 直接 lower 为真实 private `tt.func`；验证 `tl.load`/`tl.dot`/`tl.store` 出现在 `tm.task.callee` 指向的函数体内。
- [ ] 4.4 所有 task body 插入后统一执行 module verification，再运行现有 normalize/dependency/resource/synchronization passes；用 intentionally dangling 和 ABI mismatch 的 Python DSL case 验证在产生 graph/plan 前失败。
- [ ] 4.5 增加 compile-only 测试覆盖相同 task 的多 specialization、同名跨模块 task、多个 private helper 和显式/当前 target，并确认 combined analysis module 不进入普通多入口 TTGIR/PTX pipeline。

## 5. Body-Aware Access And Dependency Analysis

- [x] 5.1 扩展 TaskAccessSummary，按 callee formal 参数记录 read/write effect、alias root 和可用 buffer footprint，并用真实 typed `tt.load`/`tt.store` body 验证 summary 不再是全零占位。
- [x] 5.2 将 formal access 映射到 `tm.task` actual operand，并沿 typed binding、BufferRef 和 allgather result 传播 root/region provenance；用同 root 相交与不相交 shard case 验证映射结果。
- [x] 5.3 调整 dependency construction，使可证明的 body/SSA/collective 关系自动形成依赖，仅在 region 无法证明时要求 reads/writes annotation；用自动推导、annotation 补充、annotation 矛盾和 unresolved region 四类测试验证严格行为。
- [x] 5.4 删除普通 compute task 找不到 callee 时的静默 annotation fallback，并验证缺失 body 必然失败；同时确认真实 body 新发现的时间逆序依赖仍由既有 legality check 拒绝。

## 6. Plan Descriptor And Executor Identity

- [x] 6.1 为 native node/PlanDescriptor binding 增加 callee symbol 和 specialization identity，同时保留 invocation id；用 descriptor round-trip 测试验证多个 node 可以引用同一 specialization 且三类计划关系不变。
- [ ] 6.2 将 kernel registry 改为 specialization-keyed record/artifact registry，并保留 node-to-specialization 映射；用 missing、stale 和 signature-mismatched artifact 测试验证 executor 在 launch 前失败。
- [ ] 6.3 让 CUDA artifact 编译和 analysis TTIR 使用同一 specialization record，并在 launch 前校验 resolved operand 的类型与顺序；用 mock launch trace 验证实际 kernel、callee、constexpr 和 options 完全一致。
- [ ] 6.4 将 reference executor 和示例迁移到显式 output buffer，确保 Python reference 返回值不改变设备 ABI；运行多 shard 数值测试确认结果与串行 reference 一致且 issue-layer 开放语义不变。
- [ ] 6.5 在可用 CUDA 环境运行 typed allgather -> compute 流水，验证真实 kernel 写入输出、event 同步和 overlap；无 CUDA 环境时测试 SHALL 明确 skip 而不回退到伪 handle。

## 7. Migration And End-To-End Verification

- [ ] 7.1 迁移 `python/test/unit/test_timely.py`、`test/Timely` 和 `python/triton/timely/README.md` 中所有 dangling callee、无类型 `lower/plan` 和 compute-return-plus-store 示例，并验证仓库搜索不再发现 Timely emitter 固定 `(f32, !tm.event)` 的路径。
- [ ] 7.2 增加一个端到端 golden 测试，验证 Python DSL 同时生成 typed bindings、真实 `tt.func`、可解析 `tm.task.callee`、非零 body access summary、native `tm.plan` 和一致 specialization descriptor。
- [ ] 7.3 运行全部 Timely lit 与 Python unit tests，验证现有 TimeOrder、DataDep、ResourceOrder、同步计划和“前层至少一个任务完成”语义没有回归。
- [ ] 7.4 运行相关 Triton frontend/compiler 测试，验证 existing-module batch helper、helper symbol handling 和普通单-kernel编译路径均通过。
- [ ] 7.5 记录 CPU-only、compile-only CUDA target 和真实 CUDA 三种验证结果及未运行原因，确认没有测试依赖仓库外 `timely/` 原型后再将 change 标记完成。
