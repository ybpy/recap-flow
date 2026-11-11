Long-only R1 actor 数据配方实验（seed 42）
==========================================

研究口径
--------
* 两组都复用 B-clip / all_backbones / 20k EMA 的已封存条件标签。
* Long 全部十个任务联合训练。每次先均匀选任务。
* long-pooled：在该任务的 demo + rollout 帧中均匀抽帧。期望 demo 42.71%，rollout 57.29%。
* long-demo70：在该任务内先以 70% / 30% 选 demo / rollout，再在来源内均匀抽帧。
* 原 40 任务数据快照完整保留；两个新采样器只允许抽到 Long。有效池为 379 demo + 500 rollout，239,858 帧。
* 这些比例是数据来源比例，与每任务 rollout 第 70 百分位的 positive 标签规则不同。
* 不重新训练 critic、不改 reward/value、adv_ind 或 norm stats，不新增仿真训练数据。

训练
----
* 两组均为 base_reset，从同一 pi05_base 开始；seed 42。
* 60,000 更新，global batch 256，单节点 8 x PRO5000-72G，FSDP=4，单 JAX 进程。
* AdamW、梯度裁剪 1.0；warmup 10,000 后 LR 恒定 5e-5；EMA 0.999；condition dropout 30%。
* H=10；保留原精度和 flow loss。候选为 30k / 45k / 60k EMA。
* 两组任务和随机增强的抽样随机数按 draw 编号固定；来源和实际训练帧因配方不同而不同。
* 两卡调试采用 batch 16 / FSDP=2，仅作 smoke，不用于算法比较。

选模和确认
----------
* 两组分别在原 B/C 实验的 Long 开发清单上评估三个候选：各 1,500 次，共 3,000 次。
* 选模为任务宏平均成功率，平局选较早 checkpoint；保留三个共同预算点的曲线。
* 两组选模后，用新的 review namespace，对选定的两组、原 B 参照、SFT 40k 各评估 1,500 次，共 6,000 次。
* 新 review 清单为十任务 x 五十初始状态 x 三组环境/动作种子。共计新增 9,000 次评测。
* 保持 K=5、flow steps=10、positive、CFG beta=1、Long 上限520策略动作、初始10 dummy动作。
* 结果是一个训练种子的筛选，不把三组评测种子当三个训练重复。
* 统计按初始状态在任务内配对簇 bootstrap 10,000 次，保留各模型及三组种子的配对关系。

排期
----
1. 本开发机完成两卡预检、真实更新、完整状态恢复、EGL与评测入口调试。
2. 用户手动启动两个八卡 TIONE 训练/选模任务；有16卡时两组可并行。
3. 每个训练任务先按本配方做 batch256/FSDP4 的20步训练与10+10步恢复验收，失败则退出。
4. 验收通过后同一任务训练60k并完成三个checkpoint的开发评测、封存选择。
5. 两组完成后用户手动启动一次八卡确认评测任务，评估四个方法。
6. 回到数据配方研判节点；不自动启动R2或扩展正式多种子矩阵。

FSDP选择依据
------------
2026-10-29：八卡PRO5000、batch256、FSDP1在第一次更新时OOM；直接改FSDP4后20步通过。
pistar的pi05_star_libero本身未覆盖fsdp_devices，继承TrainConfig默认1。
未完成八卡batch256的FSDP2/4/8系统比较，因此不能称4为最优。
已有SFT和B/C长训练采用FSDP4，本轮继续固定，避免同时改变实现变量。
两卡FSDP2、batch16的通过不能证明八卡FSDP2、batch256可用。

镜像与挂载
----------
镜像：pokebot.tencentcloudcr.com/cyy/cyy_dev_pro5000rdma:v20261111174335。
镜像tag由用户提供；私有registry拒绝匿名读取manifest digest，不虚构digest。
环境、代码、解释器来自本轮独立发布快照，必须挂载到/mnt/code/recap-flow。
数据、norm stats、初始模型、LIBERO资源、运行日志和checkpoint按任务卡中的固定路径使用。
启动时不安装依赖、不升级包，训练和评测离线运行；不依赖本机SSH代理。

完整命令、实际路径、验收证据和代码提交见同目录任务卡与发布记录。
