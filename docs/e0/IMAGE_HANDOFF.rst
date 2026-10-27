镜像固化与首个 A800 验证任务
=============================

固化时机
--------
仅在交接报告明确列出 cyy_dev 完整调试通过后固化。
代码提交、环境锁、资产清单、数据 ID 和 smoke 报告共同组成验收证据。
不要仅依据“依赖安装成功”开始长任务。

已查明的目录归属
----------------
开发机上 /mnt/code/recap-flow、/mnt/recap-flow-runtime、/mnt/models、
/mnt/datasets 均位于同一 /mnt 挂载，文件系统类型 lustre。
完整挂载来源保存在运行目录 reports/mount-inventory.json，随私有交接
材料提供。TIONE 控制台资源 ID、资源组和可挂载子目录须在用户账号下
确认，不能把系统挂载字符串猜作控制台资源 ID。

保存根镜像应保留 /usr/bin/python3.10、系统库和工具。挂载另行覆盖：

* 版本化代码、.venv-train、.venv-libero、.uv-python：
  挂载到 /mnt/code/recap-flow，发布快照只读。
* 固定 actor 和 tokenizer：/mnt/models，只读。
* 冻结 SFT 数据：/mnt/recap-flow-runtime/datasets/sft-v1，只读。
* Critic、norm stats 和验证过的 GPU 用户态库：
  /mnt/recap-flow-runtime/assets，只读。
* 当前任务输出、日志、运行缓存：/mnt/recap-flow-runtime，读写。
  若平台支持子挂载，用只读子挂载覆盖上述 data/assets；若仅允许单个
  根挂载，必须用任务隔离的输出目录和不可修改的数据发布目录。
* 原始 /mnt/datasets 仅重新审计时需要，正常训练不依赖它。

本次代码/环境发布源目录：
/mnt/recap-flow-runtime/releases/e0-20261028-v1/recap-flow。
发布清单位于其上一层的 release.json；环境文件校验清单为
environment-files.json。正式任务将该发布源挂到 /mnt/code/recap-flow。

所有目录关系必须在新容器内验证。虚拟环境不是可随意移动的目录：
editable import、shebang、Python symlink 均按上述容器绝对路径创建。
不能把开发中的可写 repo 挂给正在运行的正式实验。

首个任务的实验参数
------------------
目的：验证新镜像、挂载、A800 八卡和 EGL，不产生正式实验结果。
单节点 8 x A800-80G，一个训练进程使用全部八卡。
建议起始配额：64 CPU cores、512 GiB RAM；实际可选规格以资源组为准。
初次预留 120 分钟，这是待实测的保守调试预算，不是吞吐承诺。
训练 seed=42，Actor global batch=256、fsdp_devices=1；
critic global batch=32、fsdp_devices=1，两种冻结模式各测。
Actor / 两种 critic 各 20 次更新，smoke warmup=2。
恢复比较连续 20 步与 10+10 步。Long 每任务 2 次尝试完成小闭环。
正式 batch 与短 smoke warmup 的差异会显式写入 manifest。

入口命令（镜像和挂载验收后由用户手动填入任务）::

    cd /mnt/code/recap-flow
    E0_PLATFORM=a800 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    RECAP_RUNTIME=/mnt/recap-flow-runtime \
    MUJOCO_GL=egl \
    OMP_NUM_THREADS=4 \
    WANDB_MODE=disabled \
    bash scripts/e0.sh smoke --platform a800 \
      --output /mnt/recap-flow-runtime/smoke/a800-release-v1 --phase all

仅在前一个任务进程已退出后，使用同一命令恢复验收流程。
训练子任务从已提交的 checkpoint 恢复；采集复用已完成的 episode ledger，
不额外重置每条 episode 的基础设施重试额度。未知异常应先分析日志。

验收产物必须同时存在：
preflight-passed.json、training-passed.json、recovery-passed.json、
closed-loop-passed.json。各文件下的详细报告、日志与模型记录必须可读。
八卡硬件/软件版本与真实 EGL vendor 必须与实际容器一致。
任何失败都应修复后重新验证，不跳过失败阶段进入长训练。

完整任务卡的剩余绑定信息
------------------------
交付可执行的正式任务卡前，需要用户固化后的 TCR 镜像 URI + digest、
TIONE 资源组/可用规格，以及该账号下的实际挂载源目录。
代码提交、环境 ID、验证配置、数据/模型路径、启动/恢复命令由实现方填写。
这些平台信息尚未产生时，文档只是验证任务规格，不称为已完成绑定的任务卡。

绑定后使用 task-card 命令生成 JSON，operation=validation。
训练/批量评测卡分别为 operation=train / evaluate。
本入口不会创建、启动或重试任何平台任务。

后续顺序
--------
1. A800 完整验证通过，记录实测耗时和吞吐。
2. 用户手动启动 seed 42 SFT 预实验：50k，观察 30k/40k/50k。
3. 用四 suite 完整开发清单选择 SFT，采共享 R1。
4. 分批校准 critic 与 R1/R2 训练长度；每批另交付具体任务卡。
5. 冻结正式参数后生成三种子正式任务，按依赖逐批手动启动。
