"""Model metadata, categorization, and cache inspection for pyiqa models."""

from __future__ import annotations

import os
import sys
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import torch.hub

try:
    import pyiqa
    from pyiqa.models.inference_model import DEFAULT_CONFIGS
except ImportError:
    pyiqa = None
    DEFAULT_CONFIGS = {}

# Categories for friendly grouping
CATEGORY_FR_TRADITIONAL = "FR_TRADITIONAL"      # 全参考-传统统计学算法 (无需下载模型权重，速度极快)
CATEGORY_FR_PERCEPTUAL = "FR_PERCEPTUAL"        # 全参考-深度感知模型 (基于神经网络特征，符合人眼感知)
CATEGORY_NR_TRADITIONAL = "NR_TRADITIONAL"      # 无参考-自然场景统计 (无需原图，通过图像统计特性评分)
CATEGORY_NR_DEEP = "NR_DEEP"                    # 无参考-深度学习模型 (经海量人眼评分训练，预测主观打分)
CATEGORY_AESTHETIC = "AESTHETIC"                # 美学质量评估 (构图、色彩、艺术美感打分)
CATEGORY_MULTIMODAL = "MULTIMODAL_VLM"          # 多模态/大模型 (CLIP、大视觉语言模型及图文匹配)
CATEGORY_DISTRIBUTION = "DISTRIBUTION"          # 数据集分布指标 (FID、Inception Score生成质量评估)

CATEGORY_NAMES = {
    CATEGORY_FR_TRADITIONAL: "全参考 - 传统基础 (PSNR/SSIM等，极速免权重)",
    CATEGORY_FR_PERCEPTUAL: "全参考 - 深度感知 (LPIPS/DISTS等，贴合人眼)",
    CATEGORY_NR_TRADITIONAL: "无参考 - 统计分析 (BRISQUE/NIQE等，无需原图)",
    CATEGORY_NR_DEEP: "无参考 - 深度预测 (MUSIQ/MANIQA等，主观打分)",
    CATEGORY_AESTHETIC: "美学评分 - 构图色彩 (NIMA/LAION等，艺术美感)",
    CATEGORY_MULTIMODAL: "大模型/多模态 (CLIPScore/Q-Align等，图文理解)",
    CATEGORY_DISTRIBUTION: "生成分布 - 集合评估 (FID/SFID等，AI生成评估)",
}


@dataclass
class DependencyStatus:
    """Status of system library dependencies for a metric."""
    is_ready: bool
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    safe_install_command: str = ""
    warning_message: str = ""


@dataclass
class MetricInfo:
    """Detailed metadata for a single IQA metric."""
    name: str                           # pyiqa internal name
    display_name: str                   # Friendly display name with Chinese
    mode: str                           # 'FR' (Full Reference) or 'NR' (No Reference)
    category: str                       # One of CATEGORY_*
    lower_better: bool                  # True if lower score means better quality
    score_range: str                    # e.g., "[0, 1]", "[0, 100]", "[0, +∞)"
    description_zh: str                 # Detailed plain Chinese description
    requires_weights: bool = False      # Whether it requires pre-trained weights
    default_weight_file: Optional[str] = None  # Expected weight file in cache
    paper_reference: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    required_packages: List[str] = field(default_factory=list)
    optional_packages: List[str] = field(default_factory=list)
    is_discovered: bool = False

    @property
    def direction_str(self) -> str:
        return "越低越好 (Lower is better)" if self.lower_better else "越高越好 (Higher is better)"


# Curated rich metadata dictionary for all primary IQA-PyTorch algorithms
_METRIC_METADATA: Dict[str, Dict] = {
    # 1. 全参考传统基础
    "psnr": {
        "display_name": "PSNR (峰值信噪比)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": False,
        "score_range": "[0, ~50+ dB]",
        "description_zh": "最经典、计算最快的画质客观指标。衡量图像受损后的像素均方误差，数值越高表示失真越小，但不能完全反映人眼主管感受。",
        "requires_weights": False,
        "tags": ["经典", "极速", "无权重", "推荐"],
    },
    "psnry": {
        "display_name": "PSNR-Y (亮度和灰度峰值信噪比)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": False,
        "score_range": "[0, ~50+ dB]",
        "description_zh": "仅在 Y (亮度) 通道上计算的 PSNR，常用于视频压缩和超分辨率重构算法评估。",
        "requires_weights": False,
        "tags": ["超分辨率", "无权重"],
    },
    "ssim": {
        "display_name": "SSIM (结构相似性-灰度)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "模拟人眼对结构、亮度和对比度的感知。越接近 1 说明两张图片结构越一致，广泛用于图像修复和压缩。",
        "requires_weights": False,
        "tags": ["经典", "极速", "无权重", "推荐"],
    },
    "ssimc": {
        "display_name": "SSIM-C (彩色结构相似性)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "SSIM 的彩色三通道版本，相比单灰度通道能更好地检测色彩失真与偏差。",
        "requires_weights": False,
        "tags": ["彩色", "极速", "无权重"],
    },
    "ms_ssim": {
        "display_name": "MS-SSIM (多尺度结构相似性)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "在不同分辨率尺度下综合计算 SSIM，对于不同人眼观看距离或缩放尺度的失真鲁棒性更好。",
        "requires_weights": False,
        "tags": ["经典", "无权重"],
    },
    "cw_ssim": {
        "display_name": "CW-SSIM (复小波结构相似性)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "基于复小波变换，对轻微几何几何形变、平移或旋转不敏感，适合评估发生轻微对齐位移的图像。",
        "requires_weights": False,
        "tags": ["小波", "平移鲁棒"],
    },
    "fsim": {
        "display_name": "FSIM (特征相似度)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "利用相位一致性和梯度幅度提取图像主要特征，比普通 SSIM 更贴合人眼低层视觉特性。",
        "requires_weights": False,
        "tags": ["高相关性", "无权重"],
    },
    "vsi": {
        "display_name": "VSI (视觉显著性引导相似度)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "引入人眼视觉注视点（显著性）加权，突出人眼最容易注意到的画面主体区域失真。",
        "requires_weights": False,
        "tags": ["视觉显著性", "无权重"],
    },
    "gmsd": {
        "display_name": "GMSD (梯度幅度相似度偏差)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": True,
        "score_range": "[0, 1]",
        "description_zh": "利用高斯梯度算子计算边缘失真偏差，越低越好，运算高效且与主观评分高度相关。",
        "requires_weights": False,
        "tags": ["梯度", "越低越好"],
    },
    "vif": {
        "display_name": "VIF (视觉信息保真度)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": False,
        "score_range": "[0, 1+]",
        "description_zh": "从信息论与人眼自然场景统计角度量化失真，值越大表明保真度越高。",
        "requires_weights": False,
        "tags": ["信息论", "无权重"],
    },
    "mad": {
        "display_name": "MAD (最大视差差异度)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": True,
        "score_range": "[0, 100+]",
        "description_zh": "模拟人眼早期视觉与高级视觉机制的多阶段评估算法，越低越好。",
        "requires_weights": False,
        "tags": ["越低越好"],
    },
    "nlpd": {
        "display_name": "NLPD (归一化拉普拉斯金字塔距离)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": True,
        "score_range": "[0, 1]",
        "description_zh": "基于拉普拉斯金字塔模拟人眼局部对比度掩蔽效应，越低越好。",
        "requires_weights": False,
        "tags": ["越低越好"],
    },
    "dmm": {
        "display_name": "DMM (去偏感知映射全参考)",
        "category": CATEGORY_FR_TRADITIONAL,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "2025年最新 TMM 顶刊成果，采用 SVD 去偏映射消除感知评估偏差。",
        "requires_weights": False,
        "tags": ["最新2025", "无权重"],
    },

    # 2. 全参考深度感知
    "lpips": {
        "display_name": "LPIPS (学得的感知图像块相似度 - AlexNet)",
        "category": CATEGORY_FR_PERCEPTUAL,
        "lower_better": True,
        "score_range": "[0, 1]",
        "description_zh": "生成对抗网络 (GAN) 与图像重建评估的绝对黄金标准！提取深度网络特征对比，数值越小代表越接近原图视觉质感。",
        "requires_weights": True,
        "default_weight_file": "v0.1_alex.pth",
        "tags": ["工业标准", "GAN必用", "越低越好", "推荐"],
    },
    "lpips-vgg": {
        "display_name": "LPIPS-VGG (VGG骨干网感知相似度)",
        "category": CATEGORY_FR_PERCEPTUAL,
        "lower_better": True,
        "score_range": "[0, 1]",
        "description_zh": "使用深层 VGG16 提取特征的 LPIPS，特征更细腻，但计算速度略低于 AlexNet 版本。",
        "requires_weights": True,
        "default_weight_file": "v0.1_vgg.pth",
        "tags": ["深层特征", "越低越好"],
    },
    "dists": {
        "display_name": "DISTS (深度图像结构与纹理相似度)",
        "category": CATEGORY_FR_PERCEPTUAL,
        "lower_better": True,
        "score_range": "[0, 1]",
        "description_zh": "专门解耦图像结构与纹理。即使存在微小纹理变化或轻微几何位移，也能客观评判画质，避免过度惩罚无害纹理变化。",
        "requires_weights": True,
        "default_weight_file": "dists_weights.pth",
        "tags": ["纹理容忍", "越低越好", "高推荐"],
    },
    "pieapp": {
        "display_name": "PieAPP (感知误差概率预测模型)",
        "category": CATEGORY_FR_PERCEPTUAL,
        "lower_better": True,
        "score_range": "[0, ~5]",
        "description_zh": "直接预测人眼判断图像A比图像B质量更优的概率，越低越好。",
        "requires_weights": True,
        "tags": ["人眼概率", "越低越好"],
    },
    "topiq_fr": {
        "display_name": "TOPIQ-FR (拓扑与目标感知全参考)",
        "category": CATEGORY_FR_PERCEPTUAL,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "基于 Transformer 语义特征与低级纹理自适应融合的全参考先进指标，越高越好。",
        "requires_weights": True,
        "tags": ["Transformer", "先进指标"],
    },
    "ahiq": {
        "display_name": "AHIQ (注意力机制自适应层次 IQA)",
        "category": CATEGORY_FR_PERCEPTUAL,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "结合局部自注意力与全局特征交叉比对的先进深度全参考评估算法。",
        "requires_weights": True,
        "tags": ["注意力", "高精度"],
    },

    # 3. 无参考传统统计学
    "brisque": {
        "display_name": "BRISQUE (盲/无参考空间域图像质量评估)",
        "category": CATEGORY_NR_TRADITIONAL,
        "lower_better": True,
        "score_range": "[0, 100]",
        "description_zh": "无需原图！利用自然场景统计规律与支持向量机(SVM)，检测模糊、噪点等失真。越低代表画质越自然高保真。",
        "requires_weights": True,
        "default_weight_file": "brisque_svm_weights.pth",
        "tags": ["无原图", "经典快速", "越低越好", "推荐"],
    },
    "niqe": {
        "display_name": "NIQE (自然图像质量评估器 - 完全无监督)",
        "category": CATEGORY_NR_TRADITIONAL,
        "lower_better": True,
        "score_range": "[0, ~20+]",
        "description_zh": "无需任何人工主观打分标签训练！仅基于自然优质图像统计分布建模，偏离越小（分值越低）画质越优质自然。",
        "requires_weights": False,
        "tags": ["无监督", "无需人工标签", "越低越好", "推荐"],
    },
    "piqe": {
        "display_name": "PIQE (基于块感知的失真评估器)",
        "category": CATEGORY_NR_TRADITIONAL,
        "lower_better": True,
        "score_range": "[0, 100]",
        "description_zh": "无需训练，按图像局部图像块统计空间离散度与噪声，分值越低代表画质越优越清晰。",
        "requires_weights": False,
        "tags": ["零训练", "极速", "越低越好"],
    },
    "ilniqe": {
        "display_name": "IL-NIQE (集成综合自然图像质量评估器)",
        "category": CATEGORY_NR_TRADITIONAL,
        "lower_better": True,
        "score_range": "[0, ~50+]",
        "description_zh": "在色彩空间融合亮度和纹理梯度的增强版 NIQE，对色偏、曝光失真评估更加敏感准确。",
        "requires_weights": False,
        "tags": ["色彩敏感", "越低越好"],
    },

    # 4. 无参考深度学习
    "musiq": {
        "display_name": "MUSIQ (多尺度图像质量 Transformer)",
        "category": CATEGORY_NR_DEEP,
        "lower_better": False,
        "score_range": "[0, 100]",
        "description_zh": "Google 顶会成果，原生支持不同宽高比与任意高分辨率输入！多尺度切片提取特征，无参考综合得分权威准确。",
        "requires_weights": True,
        "tags": ["Google出品", "全分辨率", "主观吻合度极高", "推荐"],
    },
    "maniqa": {
        "display_name": "MANIQA (多维注意力机制无参考 IQA)",
        "category": CATEGORY_NR_DEEP,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "CVPR NTIRE 图像质量评估夺冠算法！结合 ViT 与通道注意力机制，能精细感知复杂场景细节受损情况。",
        "requires_weights": True,
        "tags": ["比赛冠军", "高精度", "推荐"],
    },
    "topiq_nr": {
        "display_name": "TOPIQ-NR (拓扑图引导无参考质量评分)",
        "category": CATEGORY_NR_DEEP,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "顶刊成果，在移动端与网络拍摄图集上主观相关性（SRCC/PLCC）名列前茅。",
        "requires_weights": True,
        "tags": ["先进顶刊", "通用性强"],
    },
    "cnniqa": {
        "display_name": "CNNIQA (卷积神经网络盲质量评估)",
        "category": CATEGORY_NR_DEEP,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "深度学习在无参考画质领域的开山之作，轻量级浅层 CNN 快速打分。",
        "requires_weights": True,
        "tags": ["轻量级"],
    },
    "dbcnn": {
        "display_name": "DBCNN (双通道双线性卷积神经网络)",
        "category": CATEGORY_NR_DEEP,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "一路提取传统合成失真特征，另一路提取真实自然退化特征，擅长处理混合复杂退化图像。",
        "requires_weights": True,
        "tags": ["混合失真"],
    },
    "hyperiqa": {
        "display_name": "HyperIQA (超网络盲图像质量评估)",
        "category": CATEGORY_NR_DEEP,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "通过超网络自适应理解图像内容，根据画面不同场景动态调整打分权重。",
        "requires_weights": True,
        "tags": ["自适应内容"],
    },

    # 5. 美学质量评估
    "nima": {
        "display_name": "NIMA (神经图像美学评分 - AVA 数据集)",
        "category": CATEGORY_AESTHETIC,
        "lower_better": False,
        "score_range": "[1, 10]",
        "description_zh": "Google 著名美学算法！不只关注清晰度，更关注构图、光影、色彩美感，预测专业摄影师大众审美分布分值。",
        "requires_weights": True,
        "tags": ["美学构图", "摄影打分", "推荐"],
    },
    "laion_aes": {
        "display_name": "LAION-Aesthetics (LAION 视觉美学打分器)",
        "category": CATEGORY_AESTHETIC,
        "lower_better": False,
        "score_range": "[1, 10]",
        "description_zh": "Stable Diffusion 数据集筛选所用美学打分器！基于 CLIP 特征预测图像视觉吸引力。",
        "requires_weights": True,
        "tags": ["AIGC美学", "SD筛选", "推荐"],
    },
    "topiq_iaa": {
        "display_name": "TOPIQ-IAA (拓扑美学质量评估模型)",
        "category": CATEGORY_AESTHETIC,
        "lower_better": False,
        "score_range": "[0, 10]",
        "description_zh": "融合全局语义与美感布局的最新深度美学评分模型。",
        "requires_weights": True,
        "tags": ["美学前沿"],
    },

    # 6. 多模态/视觉语言大模型
    "clipscore": {
        "display_name": "CLIPScore (图文匹配度/语义对齐评分)",
        "category": CATEGORY_MULTIMODAL,
        "lower_better": False,
        "score_range": "[0, ~100]",
        "description_zh": "评估画面内容与指定文字描述（提示词 Prompt）的语义贴合程度，AI生图（文生图）必备指标！",
        "requires_weights": True,
        "tags": ["文生图", "图文对齐", "Prompt打分", "推荐"],
    },
    "clipiqa": {
        "display_name": "CLIPIQA (基于 CLIP 对比先验的质量打分)",
        "category": CATEGORY_MULTIMODAL,
        "lower_better": False,
        "score_range": "[0, 1]",
        "description_zh": "利用大模型图文对齐先验，对比“好照片”与“坏照片”语义文本概率计算画质得分，泛化性能极其优越。",
        "requires_weights": True,
        "tags": ["多模态", "Zero-Shot泛化", "推荐"],
    },
    "qalign": {
        "display_name": "Q-Align (大视觉语言模型画质与美学打分)",
        "category": CATEGORY_MULTIMODAL,
        "lower_better": False,
        "score_range": "[1, 5]",
        "description_zh": "基于多模态大语言模型，直接理解图像语义、光照与缺陷，打分具备认知常识级理解能力。",
        "requires_weights": True,
        "tags": ["大语言模型", "顶刊成果"],
    },

    # 7. 数据集分布/生成模型指标
    "fid": {
        "display_name": "FID (Fréchet Inception 距离)",
        "category": CATEGORY_DISTRIBUTION,
        "lower_better": True,
        "score_range": "[0, +∞)",
        "description_zh": "生成对抗网络与扩散模型生成质量核心指标！通过两组图像集合的高维 Inception 特征分布间距评分，越低代表生成越逼真。",
        "requires_weights": True,
        "tags": ["图像集分布", "GAN/扩散模型", "越低越好", "推荐"],
    },
    "sfid": {
        "display_name": "SFID (空间特征 Fréchet 距离)",
        "category": CATEGORY_DISTRIBUTION,
        "lower_better": True,
        "score_range": "[0, +∞)",
        "description_zh": "在 FID 基础上引入空间结构感知，更加准确衡量生成图像的空间结构真实度。",
        "requires_weights": True,
        "tags": ["空间结构", "越低越好"],
    },
    "inception_score": {
        "display_name": "Inception Score (IS 生成多样性与清晰度评分)",
        "category": CATEGORY_DISTRIBUTION,
        "lower_better": False,
        "score_range": "[1, +∞)",
        "description_zh": "衡量生成图像集合的类别可识别度与类别多样性，数值越高表示生成质量与多样性越好。",
        "requires_weights": True,
        "tags": ["多样性", "生成评测"],
    },
}


class ModelManager:
    """Singleton manager for model catalog, metadata, categorization, dependencies, and cache."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or (Path.home() / ".cache" / "torch" / "hub" / "pyiqa"))
        self._all_models: Dict[str, MetricInfo] = {}
        self._pkg_cache: Dict[str, bool] = {}
        self._initialize_catalog()

    def _initialize_catalog(self) -> None:
        """Scan pyiqa available models and combine with metadata."""
        pyiqa_models = pyiqa.list_models() if pyiqa else []

        for name in pyiqa_models:
            cfg = DEFAULT_CONFIGS.get(name, {})
            mode = cfg.get("metric_mode", "FR")
            lower_better = cfg.get("lower_better", False)
            score_range = cfg.get("score_range", "~0, ~1")

            meta = _METRIC_METADATA.get(name.lower(), {})
            is_discovered = name.lower() not in _METRIC_METADATA
            category = meta.get("category")
            if not category:
                # Automatic fallback categorization for discovered models
                if "fid" in name or "inception" in name:
                    category = CATEGORY_DISTRIBUTION
                elif "clip" in name or "qalign" in name or "qrealign" in name:
                    category = CATEGORY_MULTIMODAL
                elif "nima" in name or "aes" in name or "iaa" in name:
                    category = CATEGORY_AESTHETIC
                elif mode == "FR":
                    if name in ["psnr", "psnry", "ssim", "ssimc", "ms_ssim", "cw_ssim", "fsim", "vsi", "gmsd", "vif", "mad", "nlpd", "dmm"]:
                        category = CATEGORY_FR_TRADITIONAL
                    else:
                        category = CATEGORY_FR_PERCEPTUAL
                else:
                    if name in ["brisque", "brisque_matlab", "niqe", "niqe_matlab", "piqe", "ilniqe", "nrqm", "pi", "entropy"]:
                        category = CATEGORY_NR_TRADITIONAL
                    else:
                        category = CATEGORY_NR_DEEP

            display_name = meta.get("display_name", f"{name.upper()} ({mode})")
            description_zh = meta.get("description_zh", f"pyiqa 官方模型 {name}，评估模式：{mode}。")
            requires_weights = meta.get("requires_weights", category not in [CATEGORY_FR_TRADITIONAL, CATEGORY_NR_TRADITIONAL])
            tags = meta.get("tags", [mode, "pyiqa"] + (["新探测"] if is_discovered else []))

            # Infer required & optional packages
            req_pkgs = meta.get("required_packages")
            if req_pkgs is None:
                if "clip" in name:
                    req_pkgs = ["clip", "timm"]
                elif "qalign" in name or "qrealign" in name or "compare" in name:
                    req_pkgs = ["transformers", "datasets"]
                elif any(k in name for k in ["dists", "ahiq", "iqt"]):
                    req_pkgs = ["einops"]
                elif any(k in name for k in ["musiq", "maniqa", "topiq", "tres", "hyper"]):
                    req_pkgs = ["timm"]
                elif category == CATEGORY_FR_TRADITIONAL:
                    req_pkgs = []
                elif category in [CATEGORY_NR_TRADITIONAL, CATEGORY_DISTRIBUTION]:
                    req_pkgs = ["scipy"]
                else:
                    req_pkgs = ["timm"] if requires_weights else []

            opt_pkgs = meta.get("optional_packages")
            if opt_pkgs is None:
                if "qalign" in name or "qrealign" in name:
                    opt_pkgs = ["causal_conv1d", "fla"]
                else:
                    opt_pkgs = []

            info = MetricInfo(
                name=name,
                display_name=display_name,
                mode=mode,
                category=category,
                lower_better=lower_better,
                score_range=score_range,
                description_zh=description_zh,
                requires_weights=requires_weights,
                default_weight_file=meta.get("default_weight_file"),
                tags=tags,
                required_packages=req_pkgs,
                optional_packages=opt_pkgs,
                is_discovered=is_discovered,
            )
            self._all_models[name] = info

    def list_all(self) -> List[MetricInfo]:
        """Return all available models."""
        return list(self._all_models.values())

    def get_model(self, name: str) -> Optional[MetricInfo]:
        """Lookup a model by name."""
        return self._all_models.get(name.lower())

    def filter(
        self,
        mode: Optional[str] = None,
        category: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[MetricInfo]:
        """Filter models by mode, category, and keyword search."""
        res = list(self._all_models.values())
        if mode:
            res = [m for m in res if m.mode.upper() == mode.upper()]
        if category:
            res = [m for m in res if m.category == category]
        if search_query:
            q = search_query.strip().lower()
            res = [
                m for m in res
                if q in m.name.lower() or q in m.display_name.lower() or q in m.description_zh.lower() or any(q in t.lower() for t in m.tags)
            ]
        return res

    def get_categories(self) -> Dict[str, str]:
        """Return category IDs and descriptive labels."""
        return CATEGORY_NAMES.copy()

    def get_all_cache_roots(self) -> List[Path]:
        """Return all valid cache directories (Torch Hub, PyIQA, HuggingFace Hub, and CLIP)."""
        home = Path.home()
        candidates = [
            self.cache_dir,
            home / ".cache" / "torch" / "hub" / "checkpoints",
            home / ".cache" / "torch" / "hub" / "clip",
            home / ".cache" / "torch" / "hub",
            home / ".cache" / "clip",
            home / ".cache" / "huggingface" / "hub",
        ]
        if "HF_HOME" in os.environ:
            candidates.append(Path(os.environ["HF_HOME"]) / "hub")
        if "TORCH_HOME" in os.environ:
            candidates.append(Path(os.environ["TORCH_HOME"]) / "hub")
        return [c for c in candidates if c.exists()]

    def get_cached_weights(self) -> List[str]:
        """List all downloaded weight files present across all cache directories."""
        weights = set()
        valid_exts = {".pth", ".pt", ".bin", ".safetensors", ".mat"}
        for root in self.get_all_cache_roots():
            try:
                for f in root.glob("**/*"):
                    if f.is_file() and f.suffix.lower() in valid_exts and not f.name.endswith(".partial"):
                        weights.add(f.name)
            except Exception:
                pass
        return sorted(list(weights))

    def is_cached(self, model_name: str) -> bool:
        """Check whether the model's weight is present in the cache."""
        import urllib.parse

        info = self.get_model(model_name)
        if not info or not info.requires_weights:
            return True  # Doesn't require weights or unknown

        m_lower = model_name.lower().strip()

        # Known HuggingFace / repository pattern mapping
        hf_map = {
            "qalign": ["one-align", "q-future"],
            "qalign_8bit": ["one-align", "q-future"],
            "qalign_4bit": ["one-align", "q-future"],
            "qrealign": ["one-align", "qrealign"],
            "qrealign-mini": ["one-align", "qrealign"],
            "qrealign-lite": ["one-align", "qrealign"],
            "qrealign-pro": ["one-align", "qrealign"],
            "mplug_owl": ["mplug-owl"],
            "clipscore": ["clip-vit-large-patch14", "vit-l-14"],
            "clipiqa": ["vit-b-32", "clip"],
            "clipiqa+": ["vit-b-32", "clip-iqa%2b", "clipiqa%2b", "rn50", "vitl14"],
            "laion_aes": ["sac%2blogos", "sac+logos", "vit-l-14", "vit-l/14"],
            "musiq": ["musiq"],
            "musiq-spaq": ["musiq_spaq", "musiq"],
            "musiq-ava": ["musiq_ava", "musiq"],
            "maniqa": ["maniqa"],
            "brisque": ["brisque_svm", "brisque_matlab"],
            "fid": ["pt_inception"],
            "fid_dinov2": ["pt_inception", "dinov2"],
            "sfid": ["pt_inception"],
            "inception_score": ["pt_inception"],
        }

        patterns = hf_map.get(m_lower, [])
        if not patterns:
            base = m_lower.split("_")[0].split("-")[0]
            if base in hf_map:
                patterns = hf_map[base]

        # 1. Check HuggingFace hub snapshots
        for root in self.get_all_cache_roots():
            if "huggingface" in str(root).lower() and patterns:
                try:
                    for item in root.iterdir():
                        if item.is_dir() and any(p in item.name.lower() for p in patterns):
                            snapshots = item / "snapshots"
                            if snapshots.exists() and any(snapshots.iterdir()):
                                return True
                except Exception:
                    pass

        # 2. Check explicit default weight file across cache roots
        if info.default_weight_file:
            exp_file = info.default_weight_file.lower()
            for root in self.get_all_cache_roots():
                target = root / info.default_weight_file
                if target.exists() and not target.name.endswith(".partial"):
                    return True
                # Match unquoted name (e.g. sac+logos vs sac%2Blogos)
                try:
                    for f in root.iterdir():
                        if f.is_file() and not f.name.endswith(".partial"):
                            unq = urllib.parse.unquote(f.name).lower()
                            if unq == exp_file or f.name.lower() == exp_file:
                                return True
                except Exception:
                    pass

        # 3. Search across all cache roots
        valid_exts = {".pth", ".pt", ".bin", ".safetensors", ".mat"}
        for root in self.get_all_cache_roots():
            try:
                for f in root.glob("**/*"):
                    if f.is_file() and f.suffix.lower() in valid_exts and not f.name.endswith(".partial"):
                        fname = f.name.lower()
                        unquoted = urllib.parse.unquote(fname).lower()
                        if m_lower in fname or m_lower in unquoted:
                            return True
                        if patterns and any(p in fname or p in unquoted for p in patterns):
                            return True
                        base = m_lower.split("_")[0].split("-")[0]
                        if len(base) >= 4 and (base in fname or base in unquoted):
                            return True
            except Exception:
                pass

        return False

    def check_package_installed(self, pkg_name: str) -> bool:
        """Fast, read-only inspection whether a package is importable without loading whole module."""
        if pkg_name in self._pkg_cache:
            return self._pkg_cache[pkg_name]

        try:
            spec = importlib.util.find_spec(pkg_name)
            installed = spec is not None
        except Exception:
            installed = False

        self._pkg_cache[pkg_name] = installed
        return installed

    def check_dependencies(self, model_name: str) -> DependencyStatus:
        """Inspect dependency health status for a specific model."""
        info = self.get_model(model_name)
        if not info:
            return DependencyStatus(is_ready=True)

        missing_req = [p for p in info.required_packages if not self.check_package_installed(p)]
        missing_opt = [p for p in info.optional_packages if not self.check_package_installed(p)]

        is_ready = len(missing_req) == 0
        safe_cmd = ""
        warning = ""

        if not is_ready:
            pkgs_str = " ".join(missing_req)
            safe_cmd = f'"{sys.executable}" -m pip install {pkgs_str} --no-warn-script-location'
            warning = f"算法 [{info.display_name}] 缺少必要依赖包: {', '.join(missing_req)}"
        elif missing_opt:
            warning = f"算法 [{info.display_name}] 可选硬件加速组件未安装: {', '.join(missing_opt)} (不影响基础运行)"

        return DependencyStatus(
            is_ready=is_ready,
            missing_required=missing_req,
            missing_optional=missing_opt,
            safe_install_command=safe_cmd,
            warning_message=warning,
        )

    def clear_dependency_cache(self) -> None:
        """Clear memory cache after new packages have been installed."""
        self._pkg_cache.clear()

