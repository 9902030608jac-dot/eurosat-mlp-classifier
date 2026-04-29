# EuroSAT 三层 MLP 地表覆盖分类

本项目从零实现了一个三层多层感知机（MLP）分类器，用于 EuroSAT RGB 遥感图像的地表覆盖分类。代码只使用 NumPy 进行矩阵运算，没有使用 PyTorch、TensorFlow、JAX 等带自动微分能力的深度学习框架。

## 项目结构

- `autograd.py`：自定义 `Tensor` 类、计算图构建与反向传播。
- `dataloader.py`：EuroSAT 数据读取、图像缩放、数据划分、归一化和翻转增强。
- `model.py`：`Linear` 层和可配置的三层 `MLP` 模型。
- `optimizer.py`：融合交叉熵损失、SGD、Momentum、Weight Decay 和学习率衰减。
- `train.py`：训练循环、验证集评估和最优模型保存。
- `evaluate.py`：测试准确率、混淆矩阵、各类别准确率和错例筛选。
- `hyperparam_search.py`：网格搜索和随机搜索。
- `visualize.py`：训练曲线、第一层权重可视化和错例可视化。
- `main.py`：命令行入口。
- `report.tex`：实验报告 LaTeX 源文件。

## 环境依赖

建议使用 Python 3.8 或以上版本。

```bash
pip install -r requirements.txt
```

依赖包：

- NumPy
- Pillow
- Matplotlib

## 数据集准备

请将 EuroSAT RGB 数据集解压为 `EuroSAT_RGB` 文件夹，并保证内部包含 10 个类别子文件夹：

```text
EuroSAT_RGB/
  AnnualCrop/
  Forest/
  HerbaceousVegetation/
  Highway/
  Industrial/
  Pasture/
  PermanentCrop/
  Residential/
  River/
  SeaLake/
```

推荐目录结构：

```text
hw1/
  EuroSAT_RGB/
  main.py
  ...
```

如果数据集不在项目目录下，可以在运行时通过 `--data-dir` 指定路径。

## 训练最终模型

报告中使用的最终配置如下：

```bash
python3 main.py train \
  --data-dir /path/to/EuroSAT_RGB \
  --epochs 100 \
  --lr 0.003 \
  --hidden-dim1 1024 \
  --hidden-dim2 256 \
  --activation relu \
  --weight-decay 0.001 \
  --momentum 0.9 \
  --batch-size 128 \
  --lr-decay cosine \
  --img-size 32 \
  --augment \
  --save-dir checkpoints/final
```

训练完成后会保存：

- `best_model.npz`：验证集准确率最高的模型权重。
- `norm_stats.npz`：训练集归一化统计量，测试时需要加载。
- `training_curves.png`：训练集/验证集 Loss 曲线和 Accuracy 曲线。
- `first_layer_weights.png`：第一层隐藏层权重可视化。
- `misclassified.png`：测试集错例可视化。

## 测试模型

```bash
python3 main.py test \
  --data-dir /path/to/EuroSAT_RGB \
  --model-path checkpoints/final/best_model.npz
```

测试脚本会输出：

- 测试集 Accuracy。
- 10 个类别的 Confusion Matrix。
- 各类别 Per-class Accuracy。
- 若干错分样本，并生成错例图像。

## 超参数查找

本项目实现了网格搜索和随机搜索，满足作业中“利用网格搜索或随机搜索调节学习率、隐藏层大小、正则化强度等超参数”的要求。

网格搜索命令：

```bash
python3 main.py search \
  --data-dir /path/to/EuroSAT_RGB \
  --search-type grid \
  --epochs 30 \
  --save-dir checkpoints/hparam_search
```

随机搜索命令：

```bash
python3 main.py search \
  --data-dir /path/to/EuroSAT_RGB \
  --search-type random \
  --n-trials 10 \
  --epochs 30 \
  --save-dir checkpoints/random_search
```

实际调参流程分为两步：

1. 先用网格搜索比较学习率、隐藏层大小、激活函数和权重衰减，确定 ReLU、小学习率和较大隐藏层更适合本任务。
2. 再基于网格搜索结果进行定向优化，包括修复 L2 正则重复施加、加入数据增强、使用融合交叉熵损失、更大模型，以及调大学习率和权重衰减并配合 cosine decay。

主要调参记录如下：

| 版本 | 主要配置/改动 | 验证准确率 | 测试准确率 |
| --- | --- | ---: | ---: |
| 初始基线 | 512-64, lr=0.001, step decay, 无增强，存在 L2 重复施加问题 | 69.04% | 68.20% |
| v1 | 修复 L2，加入融合交叉熵，使用水平/垂直翻转增强，cosine decay | 73.48% | 72.00% |
| v2 | 扩大模型到 1024-256，wd=0.0005 | 73.60% | 73.09% |
| 最终配置 | 1024-256, lr=0.003, wd=0.001, cosine decay, 数据增强 | 见实验报告 | 见实验报告 |

最终报告以 `report.tex` 中的实验结果、训练曲线、混淆矩阵和可视化图片为准。

## 提交说明

最终提交时需要确认：

- GitHub 仓库是 Public。
- `README.md` 中包含环境依赖和运行方式。
- 训练好的模型权重上传到 Google Drive 或其他可下载位置。
- `report.tex` / PDF 报告中替换真实的 GitHub 链接和模型权重下载链接。
- PDF 报告包含训练曲线、验证准确率曲线、第一层权重可视化、错例分析、测试准确率和混淆矩阵。
