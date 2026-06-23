# Dropout Notes

## 易错点
- 推理时还随机丢弃。
- 训练时没有除以 keep_prob。
- p 和 keep_prob 混淆。

## 面试追问
- Dropout 为什么能缓解过拟合？
- BN 和 Dropout 同时使用有什么注意点？
