# Research Source Register

2026-09-20 补充：新增 S15–S24 及作者代码核对见
[本轮方法重审来源登记](research-sources-20260920.md)。下文保留 9 月 15 日阅读深度。

Reviewed **2026-09-15**. `method sections` means relevant equations and setup
were inspected; it does not mean reproduction. `abstract` supports only the
high-level contribution. Local availability alone does not establish full review.

| ID | Paper / primary source | Evidence inspected | Local artifact |
| --- | --- | --- | --- |
| S01 | Radford et al., **Learning Transferable Visual Models From Natural Language Supervision**, ICML 2021. [arXiv](https://arxiv.org/abs/2103.00020) | Primary abstract; local PDF identity checked | [CLIP](papers/clip-2021.pdf) |
| S02 | Zhou et al., **Learning to Prompt for Vision-Language Models**, IJCV 2022. [arXiv](https://arxiv.org/abs/2109.01134) | Primary abstract; CoOp formulation also checked in S04 §3 | [CoOp](papers/coop-2022.pdf) |
| S03 | Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**, ICLR 2022. [arXiv](https://arxiv.org/abs/2106.09685) | Primary abstract and adaptation mechanism | [LoRA](papers/lora-2022.pdf) |
| S04 | Guo and Gu, **JoAPR: Cleaning the Lens of Prompt Learning for Vision-Language Models**, CVPR 2024. [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_JoAPR_Cleaning_the_Lens_of_Prompt_Learning_for_Vision-Language_Models_CVPR_2024_paper.html) | Local PDF §§3–5.4, printed pp. 28697–28700 | [JoAPR](papers/joapr-2024.pdf) |
| S05 | Zhang et al., **TrustCLIP: Learning from Noisy Labels via Semantic Label Verification and Trust-aligned Gradient Projection**, ACM MM 2025. [DOI](https://doi.org/10.1145/3746027.3755415); [conference list](https://acmmm2025.org/accepted-regular-papers/); [author page](https://guoyanming.github.io/) | Title/authors/venue from primary listings; DOI fetch failed; full method unavailable | Link only |
| S06 | Zhang and Sabuncu, **Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels**, NeurIPS 2018. [arXiv](https://arxiv.org/abs/1805.07836); [full text](https://arxiv.org/html/1805.07836v4) | Abstract and loss formulation | Link only |
| S07 | Wang et al., **Symmetric Cross Entropy for Robust Learning With Noisy Labels**, ICCV 2019. [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.html) | Primary abstract and paper introduction | Link only |
| S08 | Liu et al., **Early-Learning Regularization Prevents Memorization of Noisy Labels**, NeurIPS 2020. [full text](https://arxiv.org/html/2007.00151v2) | §§4.2–4.3; ELR versus ELR+ | Link only |
| S09 | Li et al., **DivideMix: Learning with Noisy Labels as Semi-supervised Learning**, ICLR 2020. [arXiv](https://arxiv.org/abs/2002.07394) | Primary abstract: mixture partition, two-network training, co-refinement | Link only |
| S10 | Ren et al., **Balanced Meta-Softmax for Long-Tailed Visual Recognition**, NeurIPS 2020. [full text](https://arxiv.org/html/2007.10740v3) | Balanced Softmax formulation and sampler interaction; loss versus full BALMS | Link only |
| S11 | Menon et al., **Long-tail learning via logit adjustment**, ICLR 2021. [full text](https://arxiv.org/html/2007.07314v2) | §§4–5; post-hoc and training-time signs | Link only |
| S12 | Gao et al., **CLIP-Adapter: Better Vision-Language Models with Feature Adapters**. [arXiv preprint, 2021](https://arxiv.org/abs/2110.04544) | Primary abstract: bottleneck and residual feature adaptation | Link only |
| S13 | Pan et al., **NLPrompt: Noise-Label Prompt Learning for Vision-Language Models**, CVPR 2025. [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Pan_NLPrompt_Noise-Label_Prompt_Learning_for_Vision-Language_Models_CVPR_2025_paper.html); [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Pan_NLPrompt_Noise-Label_Prompt_Learning_for_Vision-Language_Models_CVPR_2025_paper.pdf) | Local PDF §§4–6.3, pp. 19966–19968: MAE, OT marginals, loss routing, setup | [NLPrompt](papers/nlprompt-2025.pdf), downloaded 2026-09-15 |
| S14 | Kumar et al., **Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution**, ICLR 2022. [arXiv](https://arxiv.org/abs/2202.10054) | Primary abstract and LP-FT rationale | Link only |

## Reading and implementation limits

- S04 uses RN50 and a largely 16-shot setup in its main experiments. S13 uses
  RN50 by default, with ViT-B/16 variants. AIC requires ViT-B/32 and much larger
  noisy datasets. A local port is an adaptation, not an exact reproduction.
- S05 remains a follow-up item. A failed DOI fetch does not prove that no lawful
  public manuscript exists. Do not infer its equations from its title.
- Author repositories [JoAPR](https://github.com/yunncheng/JoAPR) and
  [NLPrompt](https://github.com/qunovo/NLPrompt) are implementation leads.
  Code behavior and licenses require inspection before reuse.
- The [synthesis](../docs/research/literature.md) summarizes primary evidence.
  New mechanisms in [candidates](../docs/research/candidates.md) are proposals.
