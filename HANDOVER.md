# IQA-GUI 项目接手移交文档

**文档版本**：v1.3  
**生成时间**：2026-10-19  
**代码基线 Git 提交**：HEAD（分支 `master`，41/41 测试通过）  
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

### 2.1 依赖环境与双 Python 配置（已实机验证）
- **操作系统**：Windows 11 (64-bit)
- **硬件显卡**：NVIDIA GeForce RTX 4060 Laptop GPU（8GB 显存，驱动版本 617.14）
- **双 Python 环境区分**：
  - **环境一：系统默认 Python 3.14.3（CPU 模式）**
    - 路径：`C:\Program Files\Python314\python.exe`
    - 调用命令：`python`
    - 核心库版本：`torch 2.14.1+cpu`（无 CUDA 加速）、`PySide6 6.11.2`、`pyiqa 0.1.16`
    - 测试状态：41 项单元测试全绿通过。适合轻量功能测试与通用界面逻辑验证。
  - **环境二：GPU 加速 Python 3.12.10（CUDA 显卡加速模式，日常推荐）**
    - 路径：`C:\Users\loven\AppData\Local\Programs\Python\Python312\python.exe`
    - 调用命令：`py -3.12` 或直接使用绝对路径
    - 核心库版本：`torch 2.14.1+cu126`（完整 CUDA 12.6 支持）、`PySide6 6.12.0`、`pyiqa 0.1.16`
    - 测试状态：41 项单元测试全绿通过。支持调用 RTX 4060 硬件算力与显存，适合大模型评测与生产使用。
- **关键防误导警示（Windows `py` 启动器陷阱）**：
  > [!IMPORTANT]
  > 用户电脑上注册有自由线程解释器 `C:\Program Files\Python314\python3.14t.exe`。若在终端**直接输入不带参数的 `py` 命令**，系统默认会分流至该自由线程环境，而该环境下**未安装任何项目依赖库**（会立即报错 `ModuleNotFoundError: No module named 'torch'`）。
  > **因此严禁使用裸 `py` 命令！** 必须严格遵循：
  > - 调用 3.14 CPU 环境：使用 `python`
  > - 调用 3.12 GPU 环境：使用 `py -3.12`

### 2.2 启动与验证脚本
- **常规启动**：[`run_gui.bat`](file:///c:/Workspace/IQA-GUI/run_gui.bat)（默认使用 `python` 以 Python 3.14 CPU 模式启动，若失败则自动回退至 `py -3.12`）
- **GPU 加速启动（推荐日常使用）**：[`run_gui_gpu.bat`](file:///c:/Workspace/IQA-GUI/run_gui_gpu.bat)（显式调用 Python 3.12 GPU 加速环境，启用 RTX 4060 硬件加速，失败时通过 `py -3.12` 重试）
- **独立脚本入口**：[`run_gui.py`](file:///c:/Workspace/IQA-GUI/run_gui.py)（CPU 模式：`python run_gui.py`；GPU 模式：`py -3.12 run_gui.py`）
- **双环境自动化测试命令**：
  - Python 3.14 (CPU 环境测试)：
    ```bash
    python -m pytest iqa_gui/tests/ -v
    ```
  - Python 3.12 (GPU 环境测试)：
    ```bash
    py -3.12 -m pytest iqa_gui/tests/ -v
    ```
  *(注：测试套件内部已自动配置 `QT_QPA_PLATFORM=offscreen`，在无头或后台执行时无需手动声明环境变量，两套环境下全部 41 项测试均 100% 稳定通过)*

---

## 3. 代码架构与模块职责

```text
c:\Workspace\IQA-GUI/
├── run_gui.py / run_gui.bat / run_gui_gpu.bat   # 启动入口与批处理
├── IQA-PyTorch/                                # 本地克隆的上游算法库（已加入 .gitignore，提供测试样本图片与参考实现）
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
| **显存主动释放** | 单步及任务结束必须触发显存释放与 GC | 用户反馈测试 `qalign_8bit`、`clipscore` 等大模型后显存不释放，导致后续报错或爆显存 | [`main_window.py#_on_purge_vram_clicked`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/main_window.py#L283)<br>[`evaluator.py#release_vram`](file:///c:/Workspace/IQA-GUI/iqa_gui/core/evaluator.py#L338) |
| **浮动指标诊断** | 浮动算法标注 `可能浮动` | 基于多模态大模型/Prompt 的指标（如 `qalign`, `mplug_owl`）存在微幅非确定性，原“易浮动”用词夸大，现规范为客观的 `可能浮动` | [`evaluator.py#L232`](file:///c:/Workspace/IQA-GUI/iqa_gui/core/evaluator.py#L232)<br>[`multi_round_dialog.py#L216`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/components/multi_round_dialog.py#L216) |
| **多图对比机制** | 解除 3 张上限；≥3 张时标注最优与垫底 | 满足多张候选图像横向比对需求；≥3 张时明确指出胜出图与最差图，全参考与提示词指标附加适用性警告 | [`compare_dialog.py`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/components/compare_dialog.py#L195-L200) |
| **多轮测试统计** | 支持 2~20 轮测试并聚合极差、标准差与均值 | 单图评测评估波动范围，多图比对通过均值 PK 排除偶然误差 | [`multi_round_dialog.py`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/components/multi_round_dialog.py#L170-L225) |
| **模型库缓存检测** | 严格遵循确定性原则；无法完全肯定时不显示 | 避免非本地存在的模型被误判为“已下载”，杜绝假阳性误导用户 | [`model_manager.py#is_cached`](file:///c:/Workspace/IQA-GUI/iqa_gui/core/model_manager.py#L589-L627) |
| **预览关闭按钮布局与顶部边距** | 自适应容器规范（Right: 16px，Bottom: 3px，高度 22px，统一 `TabCornerButton` 样式）及外层顶部边距设为 0px | 确保按钮与下方展示卡片彻底独立拉开 5px 间隙（满足 $\ge 4$px），在 TabBar 内绝对视觉垂直居中且四角圆角边框无裁切；同时将单图视图外层顶部边距由 6px 压缩为 0px（`setContentsMargins(6, 0, 6, 6)`），消除与主导航栏之间的无效空隙，使选项卡与按钮整体向上紧凑贴合 | [`single_view.py#CornerButtonContainer`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/views/single_view.py#L59-L69)<br>[`single_view.py#_init_ui`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/views/single_view.py#L91-L97)<br>[`theme.py#TabCornerButton`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/theme.py#L186) |
| **无头测试防护** | 保护 `QApplication.clipboard()` 访问 | 在无界面测试环境（`offscreen`）下调用剪贴板会导致底层崩溃，增加环境变量判断规避 | [`compare_dialog.py#L302`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/components/compare_dialog.py#L302)<br>[`metric_selector.py#L302-L306`](file:///c:/Workspace/IQA-GUI/iqa_gui/ui/components/metric_selector.py#L302-L306) |

---

## 5. 当前项目状态与验证基线

- **Git 状态**：分支 `master`，工作区干净（已完成代码与移交文档的同步提交）。
- **最新提交记录**：
  1. `HEAD (当前提交)`: `style: compress single view top margin to 0px for compact layout`
  2. `323a688`: `style: optimize preview close button vertical centering, gap to display, and tab bar height`
  2. `99ca378`: `docs: add handover documentation (HANDOVER.md)`
  3. `e83bdbe`: `style: remove emoji from compare dialog window title`
  4. `9e289b0`: `style: adjust preview corner button size and layout margins for complete display without clipping`
  5. `d4a37b6`: `style: shift close preview button up by 4px in tab bar`
  6. `f8a698b`: `refactor: complete code audit, purge dead files, fix typing hazard, and standardize pragmatic UI text`
- **测试覆盖状态（双环境双重实测验证）**：
  - **Python 3.14 (CPU 环境)**：`python -m pytest iqa_gui/tests/ -v` -> **41 passed**（耗时约 17.8 秒）
  - **Python 3.12 (GPU 环境)**：`py -3.12 -m pytest iqa_gui/tests/ -v` -> **41 passed**（耗时约 17.3 秒）
  - 各套件覆盖明细：
    - `test_evaluators.py`: 13 项全部通过（核心算法、分级、热力图、多轮、显存释放）
    - `test_gmad_bench.py`: 5 项全部通过（GMAD 对抗与基准相关性度量）
    - `test_gui_headless.py`: 16 项全部通过（包含关闭/展开预览按钮属性与状态切换测试）
    - `test_model_manager.py`: 7 项全部通过（配置读写、模型目录、缓存探测、依赖检测）

---

## 6. 已知风险与技术约束

1. **Python 3.14 + PyTorch JIT 兼容性与运行环境推荐**：
   - 现象：在 Python 3.14 下执行涉及 PyTorch JIT 序列化的指标（如 `clipscore`）时，控制台会出现 `FutureWarning: torch.jit.load is not supported in Python 3.14+`。
   - 影响与处理：目前仅为 Future 弃用警告，不影响实际打分功能。已在 `evaluator.py` 与 `main.py` 中添加全局 warnings 过滤。
   - 运行建议：日常性能评估、批量评测及大参数量模型建议优先使用 Python 3.12 GPU 环境（通过 [`run_gui_gpu.bat`](file:///c:/Workspace/IQA-GUI/run_gui_gpu.bat) 启动）。既可获得 RTX 4060 硬件显存与 CUDA 并行加速，也能规避 Python 3.14 实验期针对 PyTorch JIT 的兼容性噪声。
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
  - 用户为零基础用户，未来可能提出直接打包为 `.exe` 可执行文件的需求。目前项目依赖本地已安装的原生 Python 环境（Python 3.12 或 Python 3.14）运行（系统未安装 Conda）。

### 7.2 接手后续优先推荐工作
1. **优先保障已有回归测试**：在进行任何新改动前，先在终端运行 `python -m pytest iqa_gui/tests/`（或 `py -3.12 -m pytest iqa_gui/tests/`）建立确定性基线。
2. **如有打包需求**：建议评估 PyInstaller 或 Nuitka 对 PySide6 与 PyTorch 动态链接库的打包配置。
3. **保持精准的中文答复**：与用户交流时，直接汇报具体修改点与验证结果，不展开复杂代码实现细节或英文技术术语。
