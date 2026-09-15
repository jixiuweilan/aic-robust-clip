# Reference Archive

Retrieved on **2026-09-15**. Files are kept with descriptive ASCII names so
scripts and remote runners can handle them consistently. Verify bytes with
[`SHA256SUMS`](SHA256SUMS).

## Organizer documents

| Local file | Source | Purpose |
| --- | --- | --- |
| [`official/task-rules.pdf`](official/task-rules.pdf) | [Official task PDF](https://www.aicomp.cn/wp-content/uploads/2026/04/%E9%9D%A2%E5%90%91%E5%99%AA%E5%A3%B0%E6%A0%87%E7%AD%BE%E6%95%B0%E6%8D%AE%E7%9A%84%E7%BB%86%E7%B2%92%E5%BA%A6%E5%9B%BE%E5%83%8F%E8%AF%86%E5%88%AB%E9%B2%81%E6%A3%92%E5%BE%AE%E8%B0%83-4.pdf) | Task definition, constraints, metrics, and submission format |
| [`official/challenge-track-process.pdf`](official/challenge-track-process.pdf) | [Official track process PDF](https://www.aicomp.cn/wp-content/uploads/2026/04/%E9%99%84%E4%BB%B6%EF%BC%9A%E7%AC%AC%E5%85%AB%E5%B1%8AAIC%E7%AE%97%E6%B3%95%E5%A4%A7%E8%B5%9B%E7%AE%97%E6%B3%95%E6%8C%91%E6%88%98%E8%B5%9B%E9%81%93%E6%AF%94%E8%B5%9B%E6%B5%81%E7%A8%8B-2.pdf) | Challenge-track stage flow |
| [`official/challenge-track-notice.pdf`](official/challenge-track-notice.pdf) | [Official track notice PDF](https://www.aicomp.cn/wp-content/uploads/2026/04/%E5%85%B3%E4%BA%8E%E4%B8%BE%E5%8A%9E%E7%AC%AC%E5%85%AB%E5%B1%8A%E5%85%A8%E7%90%83%E6%A0%A1%E5%9B%AD%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E7%B2%BE%E8%8B%B1%E5%A4%A7%E8%B5%9B%E7%AE%97%E6%B3%95%E6%8C%91%E6%88%98%E8%B5%9B%E9%81%93%E7%AB%9E%E8%B5%9B%E7%9A%84%E9%80%9A%E7%9F%A5-3.pdf) | Eligibility, schedule, team rules, awards, and contacts |

## Papers named by the organizer

| Local file | Primary source | Why it matters |
| --- | --- | --- |
| [`papers/clip-2021.pdf`](papers/clip-2021.pdf) | [Radford et al., ICML 2021](https://arxiv.org/abs/2103.00020) | Required CLIP foundation model |
| [`papers/coop-2022.pdf`](papers/coop-2022.pdf) | [Zhou et al., IJCV 2022](https://arxiv.org/abs/2109.01134) | Learnable prompt context for CLIP |
| [`papers/lora-2022.pdf`](papers/lora-2022.pdf) | [Hu et al., ICLR 2022](https://arxiv.org/abs/2106.09685) | Low-rank parameter-efficient adaptation |
| [`papers/joapr-2024.pdf`](papers/joapr-2024.pdf) | [Guo and Gu, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_JoAPR_Cleaning_the_Lens_of_Prompt_Learning_for_Vision-Language_Models_CVPR_2024_paper.html) | Prompt learning under noisy labels |

## Additional research material

- [NLPrompt, CVPR 2025](papers/nlprompt-2025.pdf), downloaded 2026-09-15 from
  the [official CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Pan_NLPrompt_Noise-Label_Prompt_Learning_for_Vision-Language_Models_CVPR_2025_paper.pdf).
- [Research source register](research-sources.md): 14 references covering
  adaptation, noisy-label learning, long-tail correction, and feature
  preservation. It records primary URLs and the evidence inspected.
- [Literature synthesis and experimental design](../docs/research-notes.md).

The organizer also names **TrustCLIP: Learning from Noisy Labels via Semantic
Label Verification and Trust-aligned Gradient Projection** (ACM MM 2025,
[DOI 10.1145/3746027.3755415](https://doi.org/10.1145/3746027.3755415)). The
full method was not obtained in the 2026-09-15 review. This repository records
the canonical link and primary publication listings; public-manuscript
availability remains unresolved.

## Implementation sources

- [OpenAI CLIP repository](https://github.com/openai/CLIP)
- [OpenAI CLIP ViT-B/32 model card on Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)
- [Official JoAPR implementation](https://github.com/yunncheng/JoAPR)

These links are reference material, not vendored dependencies. Review each
upstream license before copying code. The archived PDFs retain their original
copyright and license terms; do not redistribute them beyond the permissions
granted by their publishers and the competition organizer.
