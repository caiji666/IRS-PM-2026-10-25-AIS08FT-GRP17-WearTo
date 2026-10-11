# Travel Style Agent · Group 17

共同开发的 LangChain Agent 项目。当前是可安装、可测试的协作骨架，尚未实现真实天气、KG、社交检索、电商检索、FashionCLIP、MLP 或图片生成。

## 快速开始

安装 Git 和 uv（https://docs.astral.sh/uv/getting-started/installation/），然后：

```bash
git clone https://github.com/caiji666/travel-style-agent.git
cd travel-style-agent
uv sync --locked
uv run python -m travel_agent.demo
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Python 统一为 3.11。提交 `pyproject.toml` 和 `uv.lock`；新增依赖使用 `uv add`，在同一 PR 提交二者。不要手动编辑锁文件。

`demo` 是无密钥的确定性 fixture 演示，不包含自主 Agent 决策，也不能作为模型效果证明。算法工具目前明确返回 `not_implemented`，用于冻结接口。

## 既定分工

- Student 1：`understanding/` — 用户输入、结构化需求、天气与 KG 推理。
- Student 2：`inspiration/` — 情境向量、社交/Web 参考检索与排序。
- Student 3：`products/` — 商品检索、向量与 Top-K 排序。
- Student 4：`outfits/`、`training/` — Polyvore、离线训练、组合与评分。
- Student 5：`recommendation/` — 硬约束、最终排序、用户修正与预览。

公共文件 `schemas.py`、`agent.py`、依赖与 CI 需要受影响成员共同审阅。主 Agent 集成、界面、部署及全系统评价的负责人待团队认领。

## LangChain 接入方式

模块内部业务函数写在 `service.py`，对外 LangChain 工具写在 `tools.py`；主 Agent 只在 `agent.py` 集中创建和注册。无需每个人创建独立 Agent。

`build_agent(model)` 接受配置好的聊天模型实例或 LangChain 模型标识。真实模型需另装其 provider 包并配置凭据。示例（使用 OpenAI 集成时）：

```bash
uv sync --locked --extra openai
cp .env.example .env
# 在本地 .env 填写 AGENT_MODEL（provider:model 格式）及所需密钥
```

```python
import os
from dotenv import load_dotenv
from travel_agent.agent import build_agent

load_dotenv()
agent = build_agent(os.environ["AGENT_MODEL"])
result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我准备旅行穿搭"}]},
    config={"recursion_limit": 12},
)
```

以上为接线方式；工具未完成前不能产生真实推荐。当前没有持久化会话；后续统一加入 checkpointer 和 session/thread ID。预算与最终有效性必须由确定性代码校验，prompt 不能替代约束检查。

## 团队协作

阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [接口约定](docs/contracts.md)。

main → 功能分支 → PR → CI → 同伴审阅 → squash merge。不得将完整数据集、用户上传图像、密钥或模型权重直接加入 Git；保存获取说明、版本与校验和即可。
