# DBSCAN Core Notes

## 易错点
- `min_samples` 是否包含自身要说明，本模块包含自身。
- 噪声点后续可能变成边界点。
- eps 对尺度敏感。

## 面试追问
- DBSCAN 和 KMeans 的区别？
- DBSCAN 如何处理不同密度簇？
