E0 运行手册
===========

范围和责任
----------
本实现用于 π0.5 上的 RECAP 适配。科学协议见 e0/protocol.json。
开发、环境配置与调试在 cyy_dev 的 /mnt/code/recap-flow 进行。
用户固化镜像，并手动创建、启动、恢复全部 TIONE 任务。本仓库不调用
TIONE 提交接口。训练长度、critic 冻结范围和正式 checkpoint 窗口须在
seed 42 预实验后冻结，当前不把调试结果当作算法效果或正式基线结果。

环境与路径
----------
训练：/mnt/code/recap-flow/.venv-train/bin/python，Python 3.11.9；
其解释器位于同仓库 .uv-python。仿真：
/mnt/code/recap-flow/.venv-libero/bin/python，链接到镜像内
/usr/bin/python3.10（3.10.12）。两套环境均由 uv 维护。

依赖锁位于 e0/environment/{train,libero}.lock。重新构建使用::

    cd /mnt/code/recap-flow
    bash scripts/e0.sh env

运行任务不安装依赖，不调用 uv sync。安装采用 copy 模式，运行不依赖
uv 下载缓存。镜像必须保留系统 Python 和动态库，挂载必须保留代码、
两套 venv、.uv-python 的绝对路径关系。

cyy_dev 使用四张 Blackwell GPU；其已验证的 NCCL 补充运行库及 EGL
用户态库在 /mnt/recap-flow-runtime/assets 下。入口按 E0_PLATFORM
应用配置；正式 A800 环境必须重新验证，不能直接把开发机结果视为
A800 验证。固定 GPU 编译选项记录于 runtime-lock.json。

数据审计
--------
原始 RLDS：/mnt/datasets/libero-rlds；
原始转换数据：/mnt/datasets/libero_demo_pistar。保持只读。
采用的派生 SFT 快照：/mnt/recap-flow-runtime/datasets/sft-v1；
冻结 norm stats：/mnt/recap-flow-runtime/assets/libero；
报告：/mnt/recap-flow-runtime/reports/data-audit.json。

审计完整读取四套 no_noops RLDS，逐轨迹匹配全部 state/action/语言，
解码全部转换图像，并对每条轨迹首、中、末帧的双相机像素作对照。
不再次过滤 no-op、截取 demo 或按未经核对的成功字段筛选。
norm stats 在全部 demo 的 H=10 帧锚定动作块（含末动作补齐）上计算一次。
后续种子、轮次和方法复用同一统计文件。

初始审计入口如下；已存在的正式快照不得覆盖::

    bash scripts/e0.sh audit

原始集合共 1,693 episodes / 273,465 frames。图像 schema、来源、
episode UID、文件哈希、数据版本均记录于快照 manifest。
相同 demo 可以在只读快照之间硬链接；重标只写新目录。

开发机验收
----------
完整调试入口（需选择新的 output 名称）::

    cd /mnt/code/recap-flow
    E0_PLATFORM=cyy_dev bash scripts/e0.sh smoke \
      --platform cyy_dev \
      --output /mnt/recap-flow-runtime/smoke/e0-release-check \
      --phase all

依次执行四卡真实运算、四个 EGL 设备、标签与采样测试、Actor 20 次更新、
两种 critic 冻结模式各 20 次更新、连续训练对照中断恢复、小闭环。
smoke 的小 batch、短 warmup、数据和 checkpoint 均带 smoke 标记，
不允许进入预实验或正式结果。

也可以用 --phase preflight / training / recovery / collect / closed-loop
运行单阶段。--jobs-index 接收三个显式训练 manifest 路径组成的 JSON 数组。
恢复验收包含初始状态、保存/恢复状态、逐批输入、模型 RNG、
逐步指标和最终参数/optimizer/EMA 的一致性。

训练与恢复
----------
每次训练使用一个显式 manifest。新训练与断点恢复是不同入口::

    E0_PLATFORM=cyy_dev bash scripts/e0.sh run /absolute/job.json
    E0_PLATFORM=cyy_dev bash scripts/e0.sh resume /absolute/job.json

cyy_dev 只运行显式 smoke。非 smoke 训练要求单节点 8 张 A800。
正式 Actor global batch 256、fsdp_devices=1；critic batch 32、
显式 fsdp_devices。单进程使用节点内八卡。16 卡用于并行两个八卡任务。

run_dir/resolved-manifest.json 是解析后任务配置；effective-config.json 保存
完整 model/data/optimizer 设置；provenance.json 记录代码、
包版本、硬件和关键环境；source.tar.gz 保存运行时源码。
recovery 保存 optimizer、参数、EMA、完成更新数、RNG 与下一采样位置；
candidates/<完成更新数> 单独保留 EMA 及 assets。
metrics.jsonl 逐更新记录 loss、梯度、batch 哈希、RNG、数据曝光与时间。
status.json 的 complete=true 表示完成既定更新数。

同一任务恢复要求代码和 manifest 不变，不能重新 warmup。
采样由 seed/draw ID 决定；数据随机变换由每个 draw 的固定种子决定，
DataLoader 预取不推进持久化数据游标。新一轮只从指定模型参数初始化，
重新创建 optimizer、EMA 和学习率进度。

SFT 为全帧 shuffle。RECAP actor/critic 使用同一任务采样器：
75% Long / 25% 其他任务，组内均匀选任务、任务内均匀选帧。
Long demo 与累计 rollout 合池，不额外平衡来源或成功/失败。

采集与评测
----------
episodes 生成固定清单。namespace 同时决定环境和动作种子；同一比较
必须复用同一份清单，不能把方法名或 checkpoint 编号加入配对评测 namespace。
训练采集、开发选模、最终评测分别使用独立 namespace 和 purpose。

例：开发种子 42 的共享 R1 采集清单::

    bash scripts/e0.sh episodes \
      --namespace e0/pilot/s42/r1/shared/rollout \
      --purpose rollout --training-seed 42 --suites libero_10 \
      --attempts 50 --groups 0 \
      --output /mnt/recap-flow-runtime/manifests/pilot-s42-r1-rollout.jsonl

evaluate 管理每张 GPU 的策略服务和仿真客户端，同时适用于 rollout、
开发选模和最终评测；由 episode manifest 的 purpose 决定用途。
--devices 为实际可见设备编号，--workers-per-device 为每卡仿真进程数。
cyy_dev 使用 --platform cyy_dev；TIONE 使用 --platform a800。
所有 worker 共用统一分片和统计协议。对应完整命令在具体任务卡中生成。

底层 serve / rollout 仍可单独使用。服务在训练环境运行，仿真在
libero 环境运行，通过本机 WebSocket 通信。服务开端口前执行固定输入、
不同请求次序下的 seeded replay。采样种子由 episode/决策序号导出。
H=10、K=5、flow=10、positive、beta=1。Long 520 个策略动作；
10 个 dummy 动作不计数、不入训练轨迹。

每次尝试保留独立 ledger。策略超时/非法输出计失败；非法动作不入库，
合法前缀可保存。基础设施异常在相同 ID/seed 下重试一次，重启进程
不会重置重试额度。未知异常中止。缺失/基础设施失败不缩减评测分母。

标签与闭环
----------
convert-rollouts 将 ledger/NPZ 转成 LeRobot，显式指定 purpose、branch、
round-index。merge 合并四套 demo 与本分支累计 rollout。
predict --autonomous-only 使用当前选定 critic 重算全部累计 rollout，
label 输出独立重标快照；不原地修改旧标签。

reward 是稀疏成功审计；reward_label 用于 advantage；value_label 用于
critic 监督（batch 中重命名为 value）；value_pred/vlm_value 是预测；
adv_ind 是 actor 条件。轨迹 T 归一化、失败末帧 -1、50 环境步 lookahead、
终点后 bootstrap=0 均由统一协议实现。

每任务对全部自主帧取 70 百分位，A>=threshold 为 positive；记录并列
导致的实际 positive 比例。demo/干预帧固定 positive，不进百分位池。
E0 不新增人工干预。

critic 验证使用独立 Long 开发轨迹，尾 batch 的 padding 不进入 CE。
R1 使用选定 SFT 的 Long 开发轨迹；R2 合并该部分及本分支选定 R1
开发轨迹。验证数据不进入训练池或 advantage 百分位池。
select-critic 要求全部声明候选、相同验证集合和全部 Long 任务的 CE。
select-actor 要求全部候选、完整配对开发评测，任务宏平均成功率选模；
精确平局取较早完成更新数。

预实验与正式冻结
----------------
pilot-sft 生成 seed 42 的 50k SFT 初始预实验配置，候选 30k/40k/50k::

    bash scripts/e0.sh pilot-sft \
      --output /mnt/recap-flow-runtime/jobs/pilot-s42-sft-50k.json

先在 A800 验证镜像/挂载/完整 batch，再由用户启动该预实验。
先选 SFT 再采 R1，先选 R1 再采各分支 R2。
critic 比较 all_backbones 与 none，10k/20k 不同 cosine 终点分别重训。
R1/R2 actor 初始预算和 warm-start 较早候选按计划做预实验。
不得看到末端改善就自动扩展正式矩阵。

freeze 接收实际预实验的证据文件、共同训练预算、候选位置及 critic
冻结方式。正式种子 0/1/2 的 manifest 必须引用其内容哈希。
critic 更新数和候选可设为共同值，也可分别指定 r1/r2；同轮两分支共享。
未校准前不生成假装完整的正式训练参数。
matrix 只生成依赖图，不能直接执行；training-config 在前置数据、
checkpoint 选模产物均已存在后解析下一阶段任务。

同一种子共享 SFT、R1 rollout、critic 和标签；R2 各分支仅累计共享 R1
与自身 R2。base_reset 每轮回到 pi05_base；warm_start 只继承上阶段
选定 EMA 参数。critic 每轮从固定 backbone 与同种子新增层重新开始。

报告
----
最终 Long 每任务 50 初始状态 x 3 组环境/动作种子。每个训练种子先平均
三组最终评测，再对三个训练种子报告均值/样本标准差。final-report
拒绝缺少种子、非配对清单、未完整评测或未按开发集选定的 checkpoint。
逐任务结果、同种子配对差值及曲线与累计训练环境动作/计算时间一起保存。
官方初始状态在训练和评测中可重复，不声称未见初始状态泛化。
调试成功率只用于验证链路，不作为正式算法结论。

镜像与任务卡
------------
代码、虚拟环境、Python、数据、模型均位于当前挂载的 /mnt 文件系统；
保存根镜像不会自动带走它们。使用版本化代码/环境发布快照，并挂载到
相同 /mnt/code/recap-flow。数据、模型、norm stats 只读，运行输出读写。
保留 /usr/bin/python3.10 及系统 EGL/GL/CUDA 相关库。

task-card 要求真实镜像 URI/digest、资源、挂载源、发布快照、clean commit。
支持 train / evaluate / validation。输出包括全部参数、启动和恢复命令、
产物位置、验收标准。镜像尚未固化时保留“待用户固化”状态，不编造地址。
首次 A800 任务为完整 validation；通过后才开始长训练。

曲线输入补充
------------
curves 的 index 包含 points；每点明确 method、stage、training_seed、
evaluation_summary、cumulative_training_runs、cumulative_training_collections。
同一方法/阶段必须有训练种子 0/1/2；图中的点和误差条分别为训练种子
均值与样本标准差，环境动作及 GPU 时间也按训练种子聚合。formal=false
仅用于显式预实验图。训练循环 GPU 小时不含启动/加载与最终落盘，
评测/采集的整个进程分配时间另列，不把这部分估算成 FLOPs。

任务卡生成后，启动命令引用 jobs/sealed 下带内容 ID 的解析后 manifest。
数据 ID、norm stats 哈希或训练参数改变会拒绝启动，不能悄悄沿用旧卡。
