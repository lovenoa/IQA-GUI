# IQA-GUI 工程移交

更新：2026-10-25。修复记录参考基线 472ac8931e259989cc177e26bd3c04bee8cbedc7。操作、安装和变更摘要见 [README](../README.md)；本文集中记录工程接口和验证范围。

## 工程与运行

Windows / Python / PySide6 / pyiqa 桌面应用。五个页面为单图多图、多轮比较、文件夹逐张、gMAD、CSV 数据集基准、算法管理（单图多轮属于同一页）。程序导入已安装的 pyiqa；IQA-PyTorch/ 是忽略的独立参考克隆，本轮未修改，不是测试数据来源。

本轮实测 Python 3.12.10、PySide6 6.12.0、pyiqa 0.1.16、torch 2.14.1+cu126、torchvision 0.29.1+cu126，CUDA 可用。python 当前指向 Python312；py -3.12 注册不可用，历史 Python314 路径不复存在，不能继续声称当前双环境验证通过。启动器已改为本地 .venv / 当前用户 Python312 / PATH python 依次选择。

直接依赖固定于 requirements.txt，开发测试另见 requirements-dev.txt；Torch 构建按硬件先安装，未锁定全部传递依赖，也未实测干净环境重新安装。

## 关键实现合同

- config.py：load_settings 返回独立深拷贝，权限/坏 JSON 回退；save_settings 临时文件+fsync+原子替换，失败抛 SettingsSaveError；update_settings 仅合并变更字段，主窗口不再把启动快照整体覆盖。
- evaluator.py：统一 RLock 保护创建、推理、卸载、设备切换；task_guard 跨整个任务和清理。CLIPScore 范围 0—2.5，仅为图文相似度，不套画质等级；其他等级是经验提示，无统一“合格”结论。
- worker_threads.py：业务数据为 result_ready；原生 QThread.finished 无参数，表示线程实际结束；另有 error/cancelled。清理结束才发业务结果，恢复开始按钮必须等原生 finished。
- launch_worker(worker, owner)：经 MainWindow.begin_task 注册全局互斥任务，返回 bool；拒绝启动时调用方恢复 UI。关窗取消计算并等待完成；不 terminate 推理线程，不在 GUI 阻塞 wait。
- MainWindow 的 finished 连接使用 @Slot + sender，避免捕获父窗口/子线程的闭包在 DeferredDelete 清理时造成析构环。线程结束后才清引用/deleteLater。新增窗口回归覆盖连续创建与关窗。
- 当前窗口 auto_release_vram 是本次运行的权威值，保存失败时仍一致生效；独立 view 才读取磁盘回退。
- benchmark_engine：显式 MOS/DMOS 方向、参考列和独立根目录；拒绝缺图/缺列/NaN/常量标签/少于两样本，单算法错误写 metric_errors，其他算法继续。相关系数保留真正负相关。RMSE 为同一样本内非负斜率线性校准；raw_rmse 另存，均非独立泛化验收。
- 表格按稳定 BatchItem ID 映射，数值角色排序；比较报表按指标并集，真实轮次固定列，缺项 N/A，只有一个有效分数不判胜者。
- model_manager 提供 workflow_metrics/supports_workflow 过滤集合及缺少输入能力的指标；get_cache_summary 对同一去重资源列表计数计大小。缓存仅核实已知文件存在，不验证文件内容；完整资源未知时返回 None。
- dependency_installer 冻结所有现有分发版本作为 pip constraints，冲突停止；导入名 clip 映射到 openai-clip。安装正在写入时安全等待，完成后重启验证，不能用同一已导入 torch 进程的查询冒充新环境 CUDA 验收。
- 镜像在内存中适配 pyiqa 直接 HF URL；已有 HF 客户端需重启，不覆盖所有第三方下载源。不得修改上游安装文件。
- 单图热力图按目标/参考路径、mtime_ns、文件大小缓存，排序不重复重读重算；首次生成仍是当前实现的同步操作。

## 验证

默认命令 python -m pytest iqa_gui/tests/ -q；已有缓存集成 python -m pytest iqa_gui/tests/ --run-integration -q。

本轮已完成包含集成的 114 passed / 0 skipped（21.28s），覆盖已有缓存 BRISQUE/NIQE 的 CPU 运算及真实 CUDA PSNR 单项检查。空个人配置/模型缓存的默认回归为 107 passed / 7 skipped（16.43s）。初始修复前 41 passed；不能把测试数增长解读为全部模型已通过。

测试图片按固定随机种子在临时目录生成；每个测试使用临时配置与日志，集成测试不允许主动下载。缺缓存/CUDA 时明确 skip。Qt 测试需要及时关闭窗口并 flush DeferredDelete，避免用测试夹具生命周期错误代替产品故障。

仍未实测所有模型、完整数据集、真实 pip 安装/新进程 CUDA、全部下载源、所有原生 DPI/长内容/无障碍场景。一次性审计与验证材料已移到仓库外，保留历史快照。

## 维护要求

修改业务行为时补能重现问题的回归，区分真实模型运行、数学测试、替身及静态检查。验证后再汇报完成。保持中文文案正式、业务中立，不用表情符号，不以读者缺少技术背景为由降低专业判断或必要验证。

根目录仅放启动、依赖和项目入口说明；长期维护文档放 docs/。临时日志、带日期的执行计划、审计报告、截图和验证压缩包放仓库外的临时或归档目录，不作为源文件留在根目录。保留可复用的自动化测试，不重复保存过程记录。
