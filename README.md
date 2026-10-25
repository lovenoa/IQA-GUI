# IQA-GUI 图像质量评估工作台

基于 PySide6 和 pyiqa 的 Windows 桌面工具，支持单图、多图与多轮比较、文件夹逐张评测、gMAD 差异对检索、CSV 标注数据集基准及模型管理。

## 启动

已有依赖时，双击 run_gui.bat。启动器依次选择本项目 .venv、当前用户的 Python312、PATH 中的 python；不依赖 py 启动器注册，也不写死用户名。run_gui_gpu.bat 使用同一选择过程，先显示真实 CUDA 状态。程序默认自动选择可用 GPU，CUDA 不可用时使用 CPU；已有 preferred_device 配置仍会被尊重。

命令行也可运行：

```powershell
python -m iqa_gui.main
```

本轮验证使用 Python 3.12.10；不把历史文档中的 Python314/双环境记录作为当前安装事实。

## 新环境准备

1. 安装 Python 3.12，在项目目录创建环境：python -m venv .venv。
2. 用 .venv\Scripts\python.exe 作为后续命令的解释器，先选择适合设备的 torch/torchvision 构建。CPU/CUDA 安装命令以 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 为准；不要直接把另一台电脑的 CUDA 构建照搬到新设备。
3. 在新建环境中安装：.venv\Scripts\python.exe -m pip install -r requirements-dev.txt。只使用程序可换为 requirements.txt。
4. 检查：.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"。

requirements 固定本机验证过的直接依赖，未锁定所有传递依赖。Torch 二进制按设备单独选择。本轮没有重建全新环境或执行安装；不同平台/后续版本仍需独立验证。

现有环境通过界面安装缺失扩展时，会固定所有已安装分发版本，遇到冲突停止；这也可能意味着某些新扩展暂时无法装入当前环境。clip 导入名会正确映射为 openai-clip 分发。安装后需重启并重新检查 CUDA；安装正在写入时关闭窗口会等待，不强制终止 pip。此机制不修复已经损坏的环境，也不验证第三方安装包的全部兼容性。

## 使用边界

- 全参考 FR 指标需要对应原图。基准页可填写目标图列、MOS/DMOS 分数列、参考图列及独立参考目录。CSV 第一行必须是表头，列名要与输入框完全一致。
- MOS 通常越高越好，DMOS 通常越低越好，应按所用数据集说明选择。相关系数保留真实负相关；RMSE 是同一批样本内非负斜率线性校准后的误差，不能称作独立测试集的泛化精度。
- CLIPScore 只在有文字描述的单图/多图页使用，显示图文相似度，不参与 S/A/B/C 画质分级。其他评级属于经验提示，不合成为“整体合格”。
- FID、SFID、Inception Score 等需要整组图片的指标，不能在当前逐张流程中计算，选择器已过滤；算法管理页仍可查看其资料。没有伪装成已实现的图片集工作台。
- 主窗口同一时间运行一个任务，完成清理后才能开始下一项。可请求停止；正在进行的一次模型推理可能需要等待，关窗会自动等待线程安全结束。
- 镜像适配覆盖 pyiqa 直接访问 Hugging Face 的权重 URL。已导入的 Hugging Face 客户端需重启；PyTorch、GitHub 等下载源不受此开关控制。
- 缓存状态只表示已知资源文件存在，不保证文件内容完整；完整依赖无法确认时显示未知，不冒充已下载。

## 测试

```powershell
python -m pytest iqa_gui/tests/ -q
python -m pytest iqa_gui/tests/ --run-integration -q
```

默认回归使用生成的临时图片、临时配置和日志，不需要上游克隆中的示例图，也不要求个人模型缓存。已缓存模型的集成测试默认跳过；第二条命令会启用它们，无所需缓存时明确跳过，并禁止测试主动下载权重。GPU 测试需要实际可用 CUDA；没有 CUDA 时跳过。

可从 pytest 输出区分 passed / skipped；不能用离线替身测试宣称全部模型可运行。独立上游 IQA-PyTorch 克隆仅作参考，不是运行时自动导入目标，也没有被本次修复修改。

## 仓库导航

| 位置 | 内容 |
| --- | --- |
| iqa_gui/ | 程序代码及 tests/ 回归测试 |
| docs/ | [工程维护说明](docs/HANDOVER.md)、[UI 设计目标](docs/UI专业风格设计目标.txt) |
| run_gui.bat / run_gui_gpu.bat / run_gui.py | 启动入口 |
| requirements.txt / requirements-dev.txt | 运行及开发依赖 |
| IQA-PyTorch/ | 本地独立上游参考仓库，未纳入本项目 Git |

UI 设计目标属于设计参考，不等于已实现功能清单。运行日志、Python 缓存和本地虚拟环境由 .gitignore 排除。

## 最近修复（2026-10-25）

- 修复排序造成的结果错行、复选框重复切换和缺失数据的比较统计。
- 修复设置覆盖及保存失败；统一显存开关、后台取消和关窗收尾。
- 校正 CLIPScore 解释、基准评分方向及误差口径，补齐 CSV 参考图映射。
- 完善缓存识别、镜像范围、受约束依赖安装与隔离测试。

验证范围与维护合同见[工程维护说明](docs/HANDOVER.md)。一次性审计报告、实施计划及验证压缩包已移至仓库外归档。
