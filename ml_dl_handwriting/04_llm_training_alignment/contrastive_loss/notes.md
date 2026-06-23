# Contrastive Loss Notes

## 易错点
- 正负样本维度对不齐。
- 忘记 temperature。
- 没归一化导致点积受范数影响。

## 面试追问
- InfoNCE 和交叉熵的关系？
- in-batch negatives 是什么？
