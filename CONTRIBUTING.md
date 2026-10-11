# 开发约定

1. 接受仓库邀请，clone 后执行 `uv sync --locked`。
2. 开始任务前：`git switch main`，`git pull --ff-only`。
3. 创建短期分支，例如 `git switch -c feat/product-retrieval`。
4. 保持模块 service 与 LangChain tool 包装分离；输入输出使用共享 schema。
5. 为本模块提供固定输入输出样例和有意义的测试。不要在导入模块时下载模型或调用外部 API。
6. 提交前运行 `uv run ruff check .`、`uv run ruff format --check .`、`uv run pytest`。
7. 推送分支并创建 PR，写清行为变化、测试结果、数据依赖和接口影响。
8. 一位同伴审阅后 squash merge；公共 schema 变更需受影响模块成员参与。

main 保持可运行。默认 CI 不需要密钥、GPU 或真实外部 API。离线训练独立运行，记录代码 commit、数据划分、配置、种子和模型版本。

分支保护能否强制执行取决于 GitHub 仓库套餐。即便未启用保护，也按 PR 约定协作。
