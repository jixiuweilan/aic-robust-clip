# 不改写身份的 ZIP 路径迁移与进程缓存

2026-09-19 更新。运行时仅影响映射后的图片读取路径，`resolve_archive()` 调用接口不变。manifest、split、缓存、HEAD3、检查点持久化格式均不变；审计仍使用显式传入的路径。本机验证只使用合成档案，不是服务器 CUDA 或模型效果证据。

## 支持范围与映射

支持 Linux／WSL2 的本地 Linux 文件系统；WSL2 资产放 Linux 卷，不用 Windows 盘或远端文件系统。`AIC_ARCHIVE_LOCATIONS` 指向下列 JSON，路径和摘要必须替换为当前赛段已核验的真实值：

```json
{
  "schema_version": 1,
  "reason": "迁移到独立 4060 的 Linux 卷，保留冻结身份",
  "archives": {
    "/冻结manifest中的/train.zip": {
      "path": "/本机实际位置/train.zip",
      "sha256": "替换为冻结archive_identity中的64位小写SHA256"
    }
  }
}
```

每个实际任务进程都要设置环境变量，不能只在另一个校验进程中设置。映射未开启或原路径未列入时保留原有行为。启用的映射文件缺失、格式错误、重复键、摘要与审计身份不一致都拒绝读取；不会回退旧路径。映射内容每次计算摘要并重复读取比对，即使全部 stat 字段不变也不能复用旧解析结果。

## 新缓存的行为

- 先用系统 `inotify` 建立监控，再对目标 ZIP 完整计算 SHA-256；无需新增依赖。哈希期间写入或属性变化即拒绝该次读取。
- 缓存键包含 PID、目标路径、预期摘要，同时核对 stat 身份和待处理事件。每个 worker 第一次完整哈希；正常只读期间不逐样本重算整包哈希。
- 写入／属性变化使缓存失效，并同步关闭当前进程读取器持有的旧 ZIP 句柄，再强校验内容。即使等长写入并恢复时间戳、甚至模拟 stat 完全不变，也由事件触发重新校验。
- 删除、替换、移动、监控丢失、队列溢出、监控读取错误均拒绝继续；监控不可用明确报错，不降级到时间戳判断。失败现场保留，由负责人调查，不能自动重试掩盖问题。
- spawn worker 使用独立缓存；fork／PID 切换清除继承状态并关闭本进程继承的描述符，不消费父进程队列。退出使用进程 finalizer／atexit 释放监控，内核也会在进程退出时释放 fd；已打开的 ZIP 句柄同步失效。
- 每个新进程都要完整校验；单独运行一个 SHA 校验进程不能预热后续训练进程。首次整包扫描可能较慢，应记录为冷启动。

资产和映射在训练期间仍必须不可变。这不是文件锁，也不保证防御任意并发篡改；事件处理与读取之间仍有竞态边界。inotify 不覆盖远端文件系统发起的修改，也不报告所有 mmap 写入。依据：[Linux inotify(7) 手册](https://man7.org/linux/man-pages/man7/inotify.7.html)，查阅日期 2026-09-19。不要把此修复描述成通用安全防护机制。

## 验收与使用

确定性回归覆盖 stat 完全不变的等长改写、时间戳恢复、校验期间写入、属性事件、替换／删除、监控不可用／丢失／溢出、跨读取器旧句柄失效、真实 fork/spawn 隔离与正常读取哈希次数；不依赖 sleep 或重跑。原来的冻结身份兼容和 worker 生命周期测试继续保留。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -p test_relocation.py -v
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -v
.conda/aic-robust-clip/bin/python -m compileall -q src tests
.conda/aic-robust-clip/bin/python -m pip check
git diff --check
```

全部用例必须实际执行、零失败零跳过。测试数量以本次交付回执为准，不沿用历史 60/61/90/91 的数量。当前两台 4060 的部署与训练边界见[任务说明](4060-next-round.md)；B04 原失败必须在对应组员机器上复验。旧结果、失败报告、资产不得覆盖。正在运行的 T4 不部署本次修改。
