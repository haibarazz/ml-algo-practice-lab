# 维护手册

本项目采用“源码目录为主、`docs/` 为网站镜像”的发布方式。

## 内容源

- `ml_dl_handwriting/`: 机器学习与深度学习手撕模块。
- `projects/minimind/`: MiniMind 项目拆解模块。
- `docs/`: VitePress 网站入口、配置和生成后的 Markdown 镜像。

除 `docs/index.md`、`docs/guide.md`、`docs/contributing.md`、`docs/maintenance.md` 这类站点壳页面外，模块页面应优先修改源码目录，再运行同步脚本生成。

## 同步规则

`scripts/sync_docs.py` 会：

- 清理并重建 `docs/ml_dl_handwriting/`
- 清理并重建 `docs/projects/minimind/`
- 跳过 `projects/minimind/source/`
- 为存在同名 Notebook 的模块页追加 Colab / ModelScope 入口
- 把 `## 参考答案与解析` 自动包进 VitePress 折叠块

## 发布规则

GitHub Pages workflow 位于 `.github/workflows/deploy.yml`。推送到 `main` 后，GitHub Actions 会安装依赖、同步文档、构建 VitePress，并部署到 Pages。

首次启用仓库 Pages 时，需要在 GitHub 仓库页面设置一次发布源：

```text
Settings -> Pages -> Build and deployment -> Source -> GitHub Actions
```

如果 Actions 在 `actions/configure-pages` 步骤报错：

```text
HttpError: Not Found
Get Pages site failed. Please verify that the repository has Pages enabled and configured to build using GitHub Actions
```

通常不是 VitePress 构建失败，而是仓库还没有启用 Pages，或 Pages Source 还没有设为 `GitHub Actions`。完成上面的设置后，重新运行 workflow 即可。

项目站点默认 base 为：

```text
/ml-algo-practice-lab/
```

如果以后改仓库名，需要同步修改 `docs/.vitepress/config.mts` 中的默认 `base`。

## 评论区

站点预留了页面评论区能力。启用前需要先确认仓库 Discussions 的分类和公开互动方式，避免评论入口过早暴露在未准备好的页面上。
