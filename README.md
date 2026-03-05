# context-management-skill

> AI Agent Context 管理完整指南：防止模型失忆与 API Token 超限报错。

---

## 背景

多轮对话 Agent 有两类 Context 问题，性质不同，解法也不同：

| 问题 | 表现 | 性质 |
|------|------|------|
| API 报错 | `context_length_exceeded` 直接崩 | 硬约束，必须裁剪 |
| AI 失忆 | 早期关键信息被稀释，模型"忘记"重要上下文 | 软约束，需压缩 + 检索 |

本 Skill 提供五层完整解决方案，每层均附可直接运行的 Python 代码。

---

## 五层解决方案

| 方案 | 解决问题 | 实现难度 | 优先级 |
|------|---------|---------|--------|
| 分层裁剪 + 工具按需传入 | API 报错 / Token 浪费 | 低 | ⭐ 必须做 |
| 渐进式滚动摘要压缩 | AI 失忆 / Context 膨胀 | 中 | ⭐ 必须做 |
| 任务级 Context 隔离 | 根本性架构解法 | 高 | 强烈推荐 |
| 长期记忆 RAG 化 | 历史记忆无限膨胀 | 中 | 推荐 |
| Prompt Caching | 重复 Token 计费 | 低 | 降成本用 |

---

## 快速上手

### 安装依赖

```bash
pip install tiktoken          # Token 计数（可选，无则退回字符估算）
pip install anthropic         # 使用 Claude Haiku 做摘要压缩
# 或
pip install openai            # 使用 GPT-4.1-nano 做摘要压缩
```

### 方案一：分层裁剪 + 工具按需传入

```python
from scripts.context_budget import ContextBudget, trim_messages, select_tools_for_query

budget = ContextBudget(model="claude-3-7")   # 安全阈值 80K Token

# 每次调用 LLM 前检查
if budget.is_over_budget(messages):
    messages = trim_messages(messages, budget, short_term_keep=5, verbose=True)

# 工具按需传入，不全塞
relevant_tools = select_tools_for_query(
    all_tools=ALL_TOOLS,
    query=current_query,
    top_k=5,
    always_include=["file_read"]   # 基础工具白名单
)
```

裁剪顺序：P5（历史RAG）→ P4（工具Schema）→ P3（早期历史截断）→ P2（极端情况）→ **P1 永不裁剪**

### 方案二：渐进式滚动摘要压缩（核心方案）

```python
from scripts.rolling_summary import ContextMemory

memory = ContextMemory(
    short_term_window=5,           # 保留最近 5 轮完整对话
    compress_provider="anthropic", # Haiku 压缩，约 $0.001/次
)

# 每轮对话结束后调用（自动触发渐进式压缩）
memory.add_turn(user_message, assistant_response)

# 构建传给 LLM 的 messages
messages = memory.build_context(system_prompt=SYSTEM_PROMPT, current_query=query)

# 查看状态
print(memory.stats())
# → {"short_term_turns": 5, "summary_count": 3, "total_tokens": 540}

# 持久化 / 跨进程恢复
saved = memory.to_json()
memory = ContextMemory.from_json(saved, short_term_window=5)
```

**关键：渐进式，每轮触发，不要等攒够再批量压缩。**

实测 30 轮对话效果：

| 指标 | 数值 |
|------|------|
| 原始全量保留 | 1,392 Token |
| 滚动压缩后 | 540 Token |
| 压缩率 | **61.2%** |
| 第 30 轮 Token 数 | ≈ 540（趋于稳定，不再线性增长） |

### 方案三：任务级 Context 隔离

```python
class MemoryManager:
    def build_context(self, subtask: str, task_type: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            *self._get_relevant_turns(task_type, n=3),     # 只取相关轮次
            *self._get_relevant_summaries(task_type),       # 只取相关摘要
            {"role": "user", "content": subtask},
        ]
```

即使会话进行 100 轮，每次 LLM 调用的实际 Token 数几乎不变（通常 < 4K）。

### 方案四：长期记忆 RAG 化

```python
# 任务完成后写入向量库
store.add(text=key_conclusions, metadata={"task_type": task_type})

# 下次调用前检索，固定约 300-500 Token，与历史长度无关
relevant = store.search(current_query, top_k=3, min_similarity=0.7)
```

### 方案五：Prompt Caching

```python
# Anthropic — 在固定 system 块标记 cache_control
{"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}}

# OpenAI GPT-4.1 系列 — 无需配置，固定前缀自动缓存，读取 50% 折扣
```

---

## 文件结构

```
context-management-skill/
├── SKILL.md                         # LobsterAI Skill 入口（供 Agent 读取）
├── scripts/
│   ├── rolling_summary.py           # ContextMemory 类：渐进式滚动压缩 + 持久化
│   └── context_budget.py            # ContextBudget 类：Token 计数 + 分层裁剪 + 工具选取
└── references/
    ├── context-tiers.md             # 五级优先级规则 + 任务级隔离实现细节
    └── compression-patterns.md      # 压缩 Prompt 模板 + RAG 策略 + Caching 配置
```

---

## 设计原则

1. **工具列表永远按需传入**，不全塞 — 解决 60% 的 Context 问题
2. **摘要压缩用渐进式滚动**，不用批量压缩 — 避免信息断层
3. **每次 LLM 调用前检查 Context Budget** — 超出前主动裁剪，不等报错
4. **长期记忆走 RAG** — 不走全量注入
5. **System Prompt 开启 Prompt Caching** — 固定前缀只付一次

---

## 支持模型

`context_budget.py` 内置以下模型的安全 Token 阈值（Context Window × 85%）：

**Anthropic Claude 4 系列（2025-2026）**

| 模型 ID | Context Window | 安全阈值 |
|---------|---------------|---------|
| claude-opus-4-5, claude-opus-4 | 200K | 170,000 |
| claude-sonnet-4-5, claude-sonnet-4 | 200K | 170,000 |
| claude-haiku-4-5, claude-haiku-4 | 200K | 170,000 |
| claude-3-7-sonnet, claude-3-5-sonnet | 200K | 80,000（保守值） |
| claude-3-haiku | 200K | 15,000（低成本模型保守值） |

**OpenAI 系列（2025-2026）**

| 模型 ID | Context Window | 安全阈值 |
|---------|---------------|---------|
| gpt-4.1, gpt-4.1-mini, gpt-4.1-nano | **1M** | 850,000 |
| gpt-5, gpt-5-mini | 400K | 340,000 |
| o3, o4-mini, o3-mini | 200K | 170,000 |
| gpt-4o | 128K | 100,000 |
| gpt-4o-mini | 128K | 50,000 |

**Google Gemini 系列（2025-2026）**

| 模型 ID | Context Window | 安全阈值 |
|---------|---------------|---------|
| gemini-3-pro | **10M** | 1,000,000（取保守上限） |
| gemini-2.5-pro, gemini-2.5-flash | **1M** | 850,000 |
| gemini-2.0-flash | **1M** | 850,000 |
| gemini-1.5-pro, gemini-1.5-flash | **1M** | 850,000 |
| gemini-2.0-flash-lite | 128K | 100,000 |

**开源模型**

| 模型 ID | Context Window | 安全阈值 |
|---------|---------------|---------|
| llama-4-scout | **10M** | 1,000,000（取保守上限） |
| llama-4-maverick | **1M** | 850,000 |
| llama-3.1（8B/70B/405B） | 128K | 100,000 |
| deepseek-v3, deepseek-r1 | 128K | 100,000 |

> **注意：** 安全阈值 = Context Window × 85%，为 API 请求和模型输出各留余量。对于超大 context（1M+）模型，实际有效性能通常在 60–70% 处开始下降，建议在架构设计上仍采用分层裁剪 + RAG，而非依赖单次塞满。


---

## License

MIT
