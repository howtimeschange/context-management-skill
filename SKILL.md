---
name: context-management
description: >
  AI Agent Context 管理完整指南：防止 API Token 超限报错与模型失忆。
  五层解决方案：分层裁剪、渐进式滚动摘要压缩、任务级 Context 隔离、长期记忆 RAG 化、Prompt Caching。
  使用场景：设计多轮对话系统、Agent 记忆架构、解决 context_length_exceeded 报错、
  优化 Token 成本、实现长期记忆管理、构建 Memory Manager 组件。
  包含可直接运行的 Python 脚本和详细参考文档。
---

# Context 管理：防止失忆与 API 报错

## 问题诊断

两类问题，解法不同：

| 问题 | 表现 | 性质 |
|------|------|------|
| API 报错 | `context_length_exceeded` 直接报错 | 硬约束，必须裁剪 |
| AI 失忆 | 早期关键信息被稀释，模型"忘记"重要上下文 | 软约束，需压缩+检索 |

## 五层解决方案快速选型

| 方案 | 解决问题 | 实现难度 | 优先级 |
|------|---------|---------|--------|
| 分层裁剪 + 工具按需传入 | API 报错 / Token 浪费 | 低 | ⭐ 必须做 |
| 渐进式滚动摘要压缩 | AI 失忆 / Context 膨胀 | 中 | ⭐ 必须做 |
| 任务级 Context 隔离 | 根本性解法 | 高 | 强烈推荐 |
| 长期记忆 RAG 化 | 历史记忆无限膨胀 | 中 | 推荐 |
| Prompt Caching | 重复 Token 计费 | 低 | 降成本用 |

---

## 方案一：分层裁剪 + 工具按需传入

### 使用 context_budget.py

```python
from scripts.context_budget import ContextBudget, trim_messages, select_tools_for_query

# 1. 检查是否超出 Budget
budget = ContextBudget(model="claude-3-7")  # 自动取安全阈值 80K
if budget.is_over_budget(messages):
    messages = trim_messages(messages, budget, short_term_keep=5, verbose=True)

# 2. 工具按需传入（替代全量塞入）
relevant_tools = select_tools_for_query(
    all_tools=ALL_TOOLS,
    query=current_query,
    top_k=5,
    always_include=["file_read"]  # 基础工具白名单
)
# 只传 relevant_tools，而非 ALL_TOOLS
```

裁剪优先级：P5（历史RAG）→ P4（工具Schema）→ P3（早期历史压缩）→ P2（极端情况）→ P1永不裁剪

详细优先级规则见 `references/context-tiers.md`

---

## 方案二：渐进式滚动摘要压缩（核心）

### 使用 rolling_summary.py

```python
from scripts.rolling_summary import ContextMemory

# 初始化（每个会话一个实例）
memory = ContextMemory(
    short_term_window=5,        # 保留最近 5 轮完整对话
    compress_provider="anthropic",  # 用 Haiku 压缩，成本约 $0.001/次
    token_budget=8000,
)

# 每轮对话结束后调用（自动触发渐进式压缩）
memory.add_turn(user_message, assistant_response)

# 构建传给 LLM 的 messages
messages = memory.build_context(
    system_prompt=SYSTEM_PROMPT,
    current_query=new_user_query
)

# 查看 Token 使用情况
print(memory.stats())
# → {"short_term_turns": 5, "summary_count": 3, "total_tokens": 1200}

# 持久化 / 恢复会话状态
saved = memory.to_json()
memory = ContextMemory.from_json(saved, short_term_window=5)
```

**关键原则：渐进式（每轮触发），不要批量压缩。**
压缩效果：2000 Token/轮 → 200 Token，约 10x 压缩比，成本 $0.001/次。

更多压缩策略和 Prompt 模板见 `references/compression-patterns.md`

---

## 方案三：任务级 Context 隔离

每个子任务只取它需要的 Context，而非整个会话历史：

```python
class MemoryManager:
    def build_context(self, subtask: str, task_type: str) -> list[dict]:
        """只为当前子任务构建精准 Context"""
        return [
            # P1: 系统提示
            {"role": "system", "content": self.system_prompt},
            # P2: 与当前任务类型相关的最近 N 轮
            *self._get_relevant_turns(task_type, n=3),
            # P3: 压缩后的摘要（只取相关 task_type）
            *self._get_relevant_summaries(task_type),
            # 当前问题
            {"role": "user", "content": subtask},
        ]
```

即使会话进行 100 轮，每次 LLM 调用的实际 Token 数几乎不变（通常 < 4K）。

---

## 方案四：长期记忆 RAG 化

```python
# ✅ 正确：RAG 检索（Context 固定约 500 Token）
from your_vector_db import VectorStore

store = VectorStore()

# 任务完成后写入
store.add(text=key_conclusions, metadata={"task_type": task_type, "ts": time.time()})

# 下次调用前检索
relevant = store.search(current_query, top_k=3, min_similarity=0.7)
context_injection = "\n".join(r.text for r in relevant)   # ≈ 300-500 Token
```

长期记忆占用与历史长度无关，始终固定。
写入策略和去重方案见 `references/compression-patterns.md` → "RAG 长期记忆模式"

---

## 方案五：Prompt Caching

**Anthropic：** 在固定 system content 块添加 `"cache_control": {"type": "ephemeral"}`
**OpenAI：** GPT-4.1 系列无需额外配置，固定前缀自动缓存，读取 50% 折扣

关键：动态内容（当前时间、用户信息）放 user message，不要插入 system。

配置代码见 `references/compression-patterns.md` → "Prompt Caching 配置"

---

## 设计原则（必须遵守）

1. **工具列表永远按需传入**，不全塞 — 这一条解决 60% 的 Context 问题
2. **摘要压缩用渐进式滚动**，不用批量压缩 — 避免信息断层
3. **每次 LLM 调用前检查 Context Budget**，超出前主动裁剪，不等报错
4. **长期记忆走 RAG**，不走全量注入
5. **System Prompt 开启 Prompt Caching**，固定前缀只付一次

---

## 可用脚本

- `scripts/rolling_summary.py` — ContextMemory 类，渐进式滚动摘要，含持久化
- `scripts/context_budget.py` — ContextBudget 类，Token 计数、分层裁剪、工具按需选取

## 参考文档

- `references/context-tiers.md` — 五级优先级详细规则、任务级隔离实现
- `references/compression-patterns.md` — 压缩 Prompt 模板、RAG 策略、Prompt Caching 配置
