# 偏好对齐

从 DPO 和 rollout logprob 入手，理解 chosen/rejected 偏好优化的核心张量。

| 模块 | 学习重点 |
| --- | --- |
| [DPO 序列级损失：让 chosen 相对 rejected 更可能](./dpo_sequence_loss/dpo_sequence_loss.md) | 拆解 MiniMind `train_dpo.py` 中 reference/policy logprob margin 的计算。 |
| [逐 token logprob：rollout 后如何评估生成概率](./per_token_logps/per_token_logps.md) | 复刻 `rollout_engine.compute_per_token_logps`，理解 RL/GRPO/PPO 中生成 token 的概率记录。 |
