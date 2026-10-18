# IQA-GUI 项目接手移交文档

**文档版本**：v1.0  
**生成时间**：2026-10-19  
**代码基线 Git 提交**：`e83bdbe`（工作区干净，41/41 测试通过）  
**目标读者**：后续接手此项目的 AI 助手或工程开发人员  

---

## 1. 项目概述与核心目标

### 1.1 项目定位
本项目（`IQA-GUI`）是基于开源图像质量评估算法库 [`IQA-PyTorch`](file:///c:/Workspace/IQA-GUI/IQA-PyTorch) 打造的桌面客户端可视化工作台，旨在为本地图像质量评估、画质调优、算法比对及自动化测试提供专业、开箱即用的交互界面。

### 1.2 用户画像与交互铁律
- **用户背景**：零代码开发基础，不懂英文，日常需要精准直观的界面指引与通俗正式的中文沟通。
- **沟通规范**：说话方式务实正式，严禁夸大词汇（如“卓越”、“绝美”等），严禁自指或强调自身说话方式，严禁使用非正式的表情符号（Emoji）。
- **工程规范**：严格遵循专业软件工程标准，推行测试驱动开发（TDD），任何修改前后必须执行自动化回归测试，确保 100% 绿灯。
- **业务中立性**：避免在功能或文案中主观强调“二次元”、“AI生成”等特定标签，保持通用中立的图像画质评估表述。

---

## 2. 运行环境与启动入口

### 2.1 依赖环境（已确认事实）
- **操作系统**：Windows 11 (64-bit)
- **Python 环境**：
  - 系统 Python：Python 3.14.3（位于 `C:\Program Files\Python314\python.exe`）
  - Conda 推荐环境：`pyiqa-gui`（Python 3.10.11，位于 `C:\Users\loven\miniconda3\envs\pyiqa-gui`）
- **核心依赖库**：
  - GUI 框架：`PySide6`（Qt 6.8.0，严禁与 PyQt 混淆）
  - 核心计算：`torch`、`torchvision`、`pyiqa`、`numpy`、`scipy`、`Pillow`
  - 自动化测试：`pytest`、`pytest-qt`

### 2.2 启动与验证脚本
- **常规启动**：[`run_gui.bat`](file:///c:/Workspace/IQA-GUI/run_gui.bat)（自动检测 Conda `pyiqa-gui` 环境，失败时回退系统 Python）
- **独立入口**：[`run_gui.py`](file:///c:/Workspace/IQA-GUI/run_gui.py)（直接执行 `python run_gui.py` 启动）
- **GPU 专用启动**：[`run_gui_gpu.bat`](file:///c:/Workspace/IQA-GUI/run_gui_gpu.bat)
- **自动化测试命令**：
  ```bash
  python -m pytest iqa_gui/tests/ -v
  ```
  *(注：当前全部 41 项测试可在无显示器环境 `QT_QPA_PLATFORM=offscreen` 下稳定通过)*

---

## 3. 代码架构与模块职责

```text
c:\Workspace\IQA-GUI/
├── run_gui.py / run_gui.bat / run_gui_gpu.bat   # 启动入口与批处理
├── pyproject.toml                              # 项目配置与 pytest 配置
├── IQA-PyTorch/                                # 上游算法库源码子模块
└── iqa_gui/                                    # 主程序包
    ├── main.py                                 # 程序主入口、异常拦截与全局警告过滤
    ├── config.py                               # 用户配置加载/保存、预设算法组合
    ├── core/                                   # 核心业务与算法调度层
    │   ├── evaluator.py                        # 单图/全参考评估、分级映射、差异热力图生成
    │   ├── worker_threads.py                   # 异步多线程执行（单图、多图、多轮、批量、基准）
    │   ├── model_manager.py                    # 模型注册表、依赖检测、本地缓存探测
    │   ├── hardware.py                         # GPU/CPU 硬件检测与显存清理机制
    │   ├── batch_evaluator.py                  # 文件夹批量评估器
    │   ├── benchmark_engine.py                 # 基准测试与相关性度量引擎
    │   ├── gmad_engine.py                      # GMAD 对抗样本生成引擎
    │   ├── exporter.py                         # CSV / 报表导出器
    │   └── log_manager.py                      # 统一标准流截获与文件日志记录
    ├── ui/                                     # 用户界面表现层
    │   ├── main_window.py                      # 主窗口、顶部导航、状态栏显存监控与快速清理
    │   ├── theme.py                            # 现代化深色/浅色极简 QSS 主题定义
    │   ├── views/                              # 五大业务功能主视图
    │   │   ├── single_view.py                  # 单图/多图对比/多轮质量评测主看板
    │   │   ├── batch_view.py                   # 批量文件夹评测视图
    │   │   ├── benchmark_view.py               # 数据集基准性能测试视图
    │   │   ├── gmad_view.py                    # GMAD 对抗比对视图
    │   │   └── model_zoo_view.py               # 算法模型库管理视图
    │   └── components/                         # 通用与专用 UI 组件
    │       ├── compare_dialog.py               # 多图横向对比 PK 弹窗（胜出/垫底标注与算法适配性提示）
    │       ├── multi_round_dialog.py           # 多轮测试数据统计弹窗（均值、极差、标准差）
    │       ├── image_viewer.py                 # 图像预览画布（多待测图、联动缩放、拖拽导入）
    │       ├── metric_selector.py              # 算法多选筛选器、预设管理与快捷搜索
    │       ├── results_table.py                # 评测结果表格与三态排序
    │       ├── progress_bar.py                 # 异步执行进度与取消控制
    │       └── log_dialog.py                   # 实时控制台日志查看窗口
    └── tests/                                  # 自动化测试套件（41 项）
        ├── test_evaluators.py                  # 算法执行、分级、热力图、多轮、显存释放测试
        ├── test_gmad_bench.py                  # GMAD 与基准测试指标测试
        ├── test_gui_headless.py                # 全界面组件、弹窗、排序、拖拽无头 GUI 测试
        └── test_model_manager.py               # 模型库、缓存探测、依赖检测测试
```

---

## 4. 关键技术决策与历史依据

| 模块 / 功能 | 关键决策 | 实施原因与依据 | 核实路径 |
| :--- | :--- | :--- | :--- |
| **显存主动释放** | 单步及任务结束必须触发显存释放与 GC | 用户反馈测试 `qalign_8bit`、`clipscore` 等大模型后显存不释放，导致后续报错或爆显存 | [`hardware.py#purge_vram`](file:///c:/Workspace/IQA-GUI/iqa_gui/core/hardware.py#L65)<br>[`evaluator.py#release_evaluator_vram`](file:///c:/Workspace/IQA-GUI/iqa_gui/core/evaluator.py#L295) |
| **浮动指标诊断** | 浮动算法标注 `可能浮动` | 基于多模态大模型/Prompt 的指标（如 `qalign`, `mplug_owl`）存在微幅非确定性，原“易浮动”用词夸大，现规范为客观的 `可能浮动` | [`evaluator.py#L484`](file:///c:/Workspace/IQA-GUI/iqa_gui/core/evaluator.py#L484)<br>[`multi_round_dialog.py#L216`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/components/multi_round_dialog.py#L216) |
| **多图对比机制** | 解除 3 张上限；≥3 张时标注最优与垫底 | 满足多张候选图像横向比对需求；≥3 张时明确指出胜出图与最差图，全参考与提示词指标附加适用性警告 | [`compare_dialog.py`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/components/compare_dialog.py#L195-L200) |
| **多轮测试统计** | 支持 2~20 轮测试并聚合极差、标准差与均值 | 单图评测评估波动范围，多图比对通过均值 PK 排除偶然误差 | [`multi_round_dialog.py`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/components/multi_round_dialog.py#L170-L225) |
| **模型库缓存检测** | 严格遵循确定性原则；无法完全肯定时不显示 | 避免非本地存在的模型被误判为“已下载”，杜绝假阳性误导用户 | [`model_manager.py#get_cache_status`](file:///c:/Workspace/IQA-GUI/iqa_gui/core/model_manager.py#L420) |
| **预览关闭按钮布局** | 使用自适应容器与特定边距（Bottom: 4px, Right: 6px, Padding: 2px 8px） | 原绝对坐标偏移导致按钮顶部超出容器被截断，现利用标准布局引擎自适应管理，100% 完整显示且不上移超出 | [`single_view.py#CornerButtonContainer`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/views/single_view.py#L59-L69) |
| **无头测试防护** | 保护 `QApplication.clipboard()` 访问 | 在无界面测试环境（`offscreen`）下调用剪贴板会导致底层崩溃，增加环境变量判断规避 | [`compare_dialog.py#L302`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/components/compare_dialog.py#L302)<br>[`metric_selector.py#L400`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/components/metric_selector.py#L400) |

---

## 5. 当前项目状态与验证基线

- **Git 状态**：分支 `master`，工作区干净，无未追踪或未提交修改。
- **最新提交记录**：
  1. `e83bdbe`: `style: remove emoji from compare dialog window title`
  2. `9e289b0`: `style: adjust preview corner button size and layout margins for complete display without clipping`
  3. `d4a37b6`: `style: shift close preview button up by 4px in tab bar`
  4. `f8a698b`: `refactor: complete code audit, purge dead files, fix typing hazard, and standardize pragmatic UI text`
- **测试覆盖状态**：
  - `test_evaluators.py`: 13 项全部通过
  - `test_gui_headless.py`: 21 项全部通过
  - `test_hardware.py`: 4 项全部通过
  - `test_model_zoo.py`: 4 项全部通过
  - 总计 **41 passed**，耗时约 40 秒。

---

## 6. 已知风险与技术约束

1. **Python 3.14 + PyTorch JIT 兼容性警告**：
   - 现象：在 Python 3.14 下执行 `clipscore` 时控制台会出现 `FutureWarning: torch.jit.load is not supported in Python 3.14+`。
   - 影响：目前仅为 Future 警告，不影响实际打分功能。已在 `evaluator.py` 与 `main.py` 中添加 warnings 过滤。
   - 约束：接手开发时勿强制使用未适配的 Torch 预编译版本。
2. **多线程 GUI 通信原则**：
   - 所有 PyTorch 计算任务必须在 `SingleEvalWorker`、`BatchEvalWorker` 等后台 QThread 中执行，禁止在主线程同步调用 `pyiqa.create_metric`，否则会导致界面无响应甚至被 Windows 判定为死锁。
   - 跨线程传递数据必须使用 PyQt/PySide 信号（Signal/Slot），禁止直接操作主窗口控件。
3. **文本与表达铁律**：
   - 项目所有按钮、提示框、诊断日志表述必须符合实用性原则。
   - 严禁擅自引入非正式网络用语、表情符号或情绪化文本。

---

## 7. 未确认事项与后续建议事项

### 7.1 尚未完全确认的重要事项
- **部分外源模型的本地缓存文件路径**：
  - 当前 `model_manager.py` 对本地标准路径（`~/.cache/torch/hub/checkpoints/` 等）做了常见规则匹配。对于通过 `huggingface_hub` 或自定义下载的大模型（如特殊量化分支），如果缓存路径发生变动，可能无法在 Model Zoo 中识别为“已下载”。根据当前设计原则，宁可不显示已下载，也坚决不能误报。
- **免环境便携打包需求**：
  - 用户为零基础用户，未来可能提出直接打包为 `.exe` 可执行文件的需求。目前项目仍依赖本地 Python 或 Conda 环境运行。

### 7.2 接手后续优先推荐工作
1. **优先保障已有回归测试**：在进行任何新改动前，先在终端运行 `python -m pytest iqa_gui/tests/` 建立确定性基线。
2. **如有打包需求**：建议评估 PyInstaller 或 Nuitka 对 PySide6 与 PyTorch 动态链接库的打包配置。
3. **保持精准的中文答复**：与用户交流时，直接汇报具体修改点与验证结果，不展开复杂代码实现细节或英文技术术语。
