# Competition Brief

This document is a working summary of the organizer's materials for the 2026
AIC task **Robust Fine-Grained Image Recognition Fine-Tuning with Noisy
Labels**. The submission rules and public notices were rechecked on 2026-09-20;
see the [source review](../references/official/20260920-source-review.md) for access limits.
The organizer's latest notice always takes precedence over this summary.

## Task

Design a robust fine-tuning method for fine-grained classification of natural
plants and animals. Training labels contain noise; the later online stages also
contain long-tailed class distributions. The organizer evaluates predictions
on balanced, manually verified test sets.

The required backbone is OpenAI CLIP ViT-B/32. Suggested directions include
prompt tuning, adapters, LoRA, robust losses, sample filtering, pseudo-label
refinement, representation constraints, and mitigation of catastrophic
forgetting.

## Stage data

| Stage | Classes | Training images | Test images | Noted characteristics |
| --- | ---: | ---: | ---: | --- |
| Preliminary | 500 | 103,218 | 24,967 | Noisy training labels |
| Second round | 750 | 148,695 | 37,444 | Noisy labels and long tail |
| Semifinal | 500 | 90,197 | 24,912 | Noisy labels and long tail |

All stages are single-label classification. Training images are organized by
class directory. Test labels are hidden. The organizer notes that some image
files may appear truncated to ordinary viewers while remaining readable by
Pillow.

## Hard constraints

1. The backbone must be CLIP ViT-B/32; no other or larger vision foundation
   model may replace it.
2. Only the current stage's official dataset may be used. Earlier-stage data
   cannot be carried into later stages.
3. The test set is prediction-only and cannot participate in training by any
   supervised, self-supervised, or unsupervised method.
4. No external public/private dataset or manually added labels may be used.
5. Pretrained weights are limited to OpenAI's public CLIP ViT-B/32 weights,
   obtained through OpenAI CLIP or the corresponding Hugging Face package.
6. Commercial closed-model APIs and online foundation-model inference cannot
   replace the core recognition flow.
7. The final result must come from one model or one inference flow. Ensembles,
   model fusion, and voting are prohibited.
8. The complete training path, including noise screening, must be reproducible
   from submitted code. Manual cleaning cannot be a required preprocessing
   step.
9. Competition data is limited to this competition and must not be leaked or
   reused elsewhere.

## Evaluation

The online metric is Top-1 accuracy:

```text
accuracy = correct predictions / total test images
```

The preliminary score is used for familiarization and advancement but is not
part of the final online aggregate. The organizer states that the second-round
score contributes 40% and the semifinal score contributes 60%. A submission
below a published baseline may be treated as invalid.

The final combines objective evaluation with an offline defense. The defense
focuses on innovation, technical completeness, reproducibility, and
presentation; later final-round instructions are authoritative.

## Prediction submission

Create a headerless CSV in this form:

```csv
xxxxxxxxxxxx.jpg,0001
xxxxxxxxxxxy.jpg,0123
xxxxxxxxxxxz.jpg,0456
```

Requirements:

- exactly two fields per row: exact test filename and class ID;
- preserve filename case and extension;
- class ID is exactly four decimal digits, left-padded with zeroes;
- name the file `pred_results.csv`;
- place that file in a ZIP archive for upload.

2026-09-20 复核：以上两列、四位类别编号、文件名和 ZIP 要求与本地赛题 PDF
第 5–6 页第十三节及现行赛题页一致。已取得条款未规定 ZIP 外部名称或 CSV 行序；
不要把示例排版中的空格解释为强制字符。无表头、无额外空白、根目录只放一个 CSV
是本项目采用的保守输出约定。复赛作品材料的压缩包命名不能擅自套用到初赛预测包。

`aic-validate-submission` 可以检查打包前的裸 CSV，这不表示官网接受裸 CSV 上传。
只有提供 `--expected-files` 才核验覆盖、提供 `--class-map` 才核验类别成员；应将
expected-files 与实际同赛段测试包独立核对。语法通过不代表上传成功或已产生分数，
校验器不检查账户、入口、时间窗、额度或后台评测状态。本次占位审查及用户确认的
延迟出分见[审查记录](research/placeholder-review-20260920.md)。

The same prediction format applies to the first three online stages. Later
stages also require reproducible code, run instructions, the complete
train/validation/inference path, environment details, and a runnable model or
container. The semifinal additionally requires a technical PDF. Final-round
deliverables are subject to a later notice.

## Track schedule and participation

原 4 月赛道通知给出的计划如下；其中“9 月中旬前开始复赛”等初步安排不能代替
下述 9 月新通知与平台的实际日程：

- registration opened on 2026-04-28; each task's registration deadline in the
  registration system is authoritative;
- the second round was planned to start by mid-September 2026;
- the semifinal was planned to start by 2026-10-10;
- the national final was planned for mid-to-late November 2026, with its date
  and location to be announced separately.

2026-09-20 查阅[9 月 14 日复赛通知（〔2026〕38号）](https://www.aicomp.cn/notice/notice-1/5278.html)：

- 报名截止 9 月 20 日 18:00；初赛晋级要求在 9 月 20 日 20:00 前提交初赛结果并在
  排行榜上显示成绩。以上为中国赛事通知中的当地时间（北京时间），不要混淆两个截止点。
- 复赛每日可提交两次，取复赛最高成绩；通知明确说明算分可能延迟。
  本次查阅内容未明确初赛每日额度，不套用旧年份的“不限次数/每小时刷新”说法。
- 赛题 4 的复赛基准线为 40；这不是初赛格式判定或空白分数的解释。
- 官网[排行榜入口](https://reg.aicomp.cn/special/phb/list)与登录后的“我的参赛信息—详情”
  需要用户核对赛题、赛段及个人提交状态。具体复赛起止时间见该通知附件 1，作品要求
  见附件 2；本次浏览工具未成功读取这两个附件，不在这里猜测细节。
- 公开赛题页仍列初赛 24,967 张测试图片；它未公开完整文件名单或 SHA-256。
  因此只能确认数量口径未变，不能仅凭公开网页保证平台当前数据包与本地逐文件一致。

Teams may contain one to three students from the same institution, with up to
two advisors. The track notice lists a fee of CNY 500 per team. Confirm all
deadlines, eligibility, fees, and deliverables in the live registration system
before acting on them.

## Contacts

- Task QQ group: `1090224462`
- Task email: `zerens@njust.edu.cn`
- Challenge-track QQ group: `981069628`
- Organizer email: `office@aicomp.cn`

## Official sources

- [Task page](https://www.aicomp.cn/tracks/tracks-1/3714.html)
- [Challenge-track notice](https://www.aicomp.cn/notice/notice-1/3629.html)
- [Second-round notice, 2026-09-14](https://www.aicomp.cn/notice/notice-1/5278.html)
- [Competition home and registration entry](https://www.aicomp.cn/)
- Archived copies and hashes: [`../references/README.md`](../references/README.md)
