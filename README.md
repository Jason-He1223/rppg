# rPPG Talking-Face Deepfake Detection

本项目使用 **rPPG 生理信号特征**和**嘴部视觉特征**进行 Talking-Face Deepfake Detection，支持 Lip-only 和 rPPG + Lip Fusion 两种模型。

## 文件说明

**src/**
- `extract_waveforms.py`：RhythmFormer 加载及视频预处理。
- `extract_multiwindow_all.py`：提取多窗口 RhythmFormer latent 和 rPPG waveform，支持 H.264 压缩增强。
- `extract_lip_features.py`：使用 MediaPipe 定位嘴部，通过预训练 ResNet18 提取视觉特征。
- `train_rhythmformer_lip_fusion.py`：模型定义、训练及视频级评估。

**manifests/**
- `dataset_split.csv`：训练、验证、测试集划分。
- `multiwindow_manifest.csv`：视频窗口及 rPPG 特征路径。
- `lip_manifest.csv`：嘴部特征路径。

## 模型

- **Lip-only**：ResNet18 特征 → Temporal Transformer → Attention Pooling → 分类。
- **Fusion**：结合 RhythmFormer latent、rPPG waveform 和嘴部特征进行分类。

## 实验设置

- 数据集：Celeb-DF++ Talking Face
- 窗口长度：160 frames
- 窗口步长：80 frames
- 评估指标：Video-level ROC-AUC

## 注意事项

运行前需要安装相关 Python 依赖，并准备预训练 RhythmFormer 权重。代码中的数据路径需要根据运行环境修改。CSV 文件仅包含数据索引，不包含原始视频或提取后的特征。
