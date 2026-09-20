# 2026-09-20 方法重审来源登记

查阅日期均为 **2026-09-20**。网页/作者代码为当日读取的在线版本，未固定作者仓库
commit，不宣称复现或完成代码审计。实施时还需固定版本、核对许可证和依赖。
本轮未新增下载文件；没有替换官方 PDF，`references/SHA256SUMS` 对应资产不变。
下表接续 [S01–S14](research-sources.md)，结论见[方法重审](../docs/research/method-review-20260920.md)。

| ID | 一手来源与读取范围 | 证据边界 / 对本项目的意义 |
|---|---|---|
| S15 | **TURN: Fine-tuning Pre-trained Models for Robustness under Noisy Labels**, IJCAI 2024。[论文](https://www.ijcai.org/proceedings/2024/0403.pdf)，§4 算法 1、§5.1，印刷页 3645–3647 | GCE 冻结特征头、逐类损失 GMM、均衡筛选与 FFT；每类抽样数取所有类别选中数的最小值。实验含多种预训练模型；不声称 CLIP 那一行已确认为 B/32。未核到可用作者代码。 |
| S16 | **FINE Samples for Learning with Noisy Labels**, NeurIPS 2021。[论文](https://papers.nips.cc/paper/2021/file/ca91c5464e73d3066825362c3093a45f-Paper.pdf)，§3 / 算法 1 / alignment clusterability 定义 | 用类内 Gram 矩阵主特征向量、投影平方与 GMM；依赖表示结构，不要求文本类别名。理论条件不能当作 AIC 已满足。 |
| S17 | **Fine-Grained Classification with Noisy Labels (SNSCL)**, CVPR 2023。[正文](https://arxiv.org/html/2303.02404v1)，§4–5；[作者代码](https://github.com/1998v7/SNSCL)及 [Vanilla_w_SNSCL.py](https://raw.githubusercontent.com/1998v7/SNSCL/main/Vanilla_w_SNSCL.py) | 检查了可靠性修正、队列、随机特征、实验设置；入口硬编码 ResNet18，100 轮设置，测试用 encoder_q + 分类头，训练有动量分支。代码累计 best test；迁移后只能用允许的 dev 选择。许可证尚未完成审查。 |
| S18 | **Vision-Language Models are Strong Noisy Label Detectors (DeFT)**, NeurIPS 2024。[正文](https://arxiv.org/html/2409.19696v1)，§3–5 / 附录；[作者仓库](https://github.com/HotanLee/DeFT)、[FFT 配置](https://raw.githubusercontent.com/HotanLee/DeFT/main/config/FFT/cifar100.yaml)、[phase2 入口](https://raw.githubusercontent.com/HotanLee/DeFT/main/main_phase2.py) | 双文本提示筛选后视觉微调；配置 ViT-B/16。入口还有不同预训练骨干分支、每轮 test 并报告 best；只能移植符合规则的机制。仓库标示 MIT，复用具体文件仍需核对。 |
| S19 | **Mitigating Endogenous Confirmation Bias in Noisy Label Learning for Vision-Language Models (DKAF)**, AAAI 2026。[官方页面](https://ojs.aaai.org/index.php/AAAI/article/view/39641)、[正文](https://ojs.aaai.org/index.php/AAAI/article/download/39641/43602)，方法与实验设置；[作者仓库](https://github.com/iLearn-Lab/AAAI26-DKAF)、[CUB 配置](https://raw.githubusercontent.com/iLearn-Lab/AAAI26-DKAF/main/config/FFT/cub-200-2011.yaml) | 页面发表于 2026-03-14；ViT-B/16 与语义类名，跨模态筛选/修正/适配。Clothing1M/WebVision 因类别外噪声不执行同样的修正阶段；报告最佳 test，不能原样搬选择协议。未完成代码/许可证审计。 |
| S20 | **CLIPCleaner**, ACM MM 2024。[正文](https://arxiv.org/html/2408.10012v1)，§3.3–3.4 与附录 G.2 | MixFix 渐进扩大监督；文本零样本与冻结视觉分类头替代途径。CLIP 选择器为 B/32，并不表示下游分类器都是 B/32；下游有 ResNet/VGG/Inception 系列、150/300 轮。未核验作者代码。 |
| S21 | **TrustVLM**, 2025。[正文](https://arxiv.org/html/2505.23745v2)，错误检测、结果表及生成图像扩展；[作者仓库](https://github.com/EPFL-IMOS/TrustVLM) README | 主要指标包括 AURC/AUROC/FPR95，不能转述为分类 Top-1 增益。不能笼统说全部方法使用生成图像；不采用其 SD3 外部图像扩展。 |
| S22 | **CANNE: CLIP-Based ANNE Selection for Noisy-Label Learning**, Entropy 2026。[主站](https://www.mdpi.com/1099-4300/28/8/849)摘要及主站检索片段，页面日期 2026-07-30 | 直接打开遇到 429，未完整审阅方法/实验；仅登记离线 CLIP 种子 + 在线 ANNE 思路，不作为已核验实施依据。 |
| S23 | **Robust-CLIP**, IJCNN 2025。[IEEE](https://ieeexplore.ieee.org/document/11227884/)，DOI 10.1109/IJCNN64981.2025.11227884，摘要 | 仅用于发现提示筛选、Semi-LoRA 与半监督路线；未取得完整方法/代码，不与本仓库名称混同。 |
| S24 | **ACD-U: Asymmetric co-teaching with machine unlearning for robust learning with noisy labels**, 2026。[作者预印本](https://arxiv.org/abs/2603.07166)，摘要 | CLIP + CNN 非对称训练，原方案不符合本仓库唯一骨干边界；未作为实施候选。 |

对既有 S05 **TrustCLIP (ACM MM 2025)** 再次访问
[ACM DOI 页](https://dl.acm.org/doi/10.1145/3746027.3755415)仍返回 403。
只保留已有一手目录核实的身份，不推断未读公式；搜索中的同名隐私学习项目不替代该论文。

## 搜索与排除口径

检索组合包括 CLIP/noisy labels/fine-tuning、fine-grained noisy classification、
pretrained visual feature filtering、2025/2026 新工作，以及论文名 + 作者代码。
仅用论文、会议/期刊站点和作者仓库支持技术判断；聚合站线索不充当结果证据。
这是一轮有针对性的研究，不声称穷尽所有论文或找到排行榜选手方案。

官方赛题和新通知以[官方来源复核](official/20260920-source-review.md)为准。
本轮也访问官网赛道索引链接的本赛题排行榜，未取得动态榜单内容；58/80/90 的信息
来源为用户反馈。未登录平台、读取测试图片或联系其他选手。
