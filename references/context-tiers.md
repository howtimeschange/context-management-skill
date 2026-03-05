# Context 分层裁剪参考

## 五级优先级体系

| 优先级 | 内容 | 策略 | Token 估算 |
|--------|------|------|------------|
| P1 永不丢 | System Prompt + 当前任务指令 + 工作记忆(L6) | 始终保留 | ~2-4K |
| P2 尽量保 | 最近 3-5 轮对话（L4 短期记忆） | 超限时才裁 | ~2-5K |
| P3 可压缩 | 更早的历史对话 | 压成摘要后保留 | ~500T（摘要后） |
| P4 按需注入 | 工具列表 Schema | 向量检索 Top-3~5 | ~500-1K |
| P5 按需注入 | 长期记忆 / 知识库 | RAG 检索 Top-3 | ~300-500T |

## 工具列表的 Token 代价

100 个工具 Schema ≈ 5,000–10,000 Token（最大的 Context 杀手）

**按需传入策略：**
1. 将所有工具的 name + description 向量化，存入本地向量库
2. 每次 LLM 调用前，用当前 query 做相似度检索
3. 只将 Top-K（3-5 个）工具 Schema 注入 Context
4. 对于必须始终可用的基础工具（如 file_read），单独维护白名单

## 裁剪触发条件

```python
CONTEXT_BUDGET = {
    "gpt-4.1":       100_000,  # 安全阈值，留 buffer
    "claude-3-7":     80_000,
    "claude-haiku":   15_000,
    "gpt-4.1-mini":   40_000,
}

def should_trim(messages, model):
    current_tokens = count_tokens(messages)
    budget = CONTEXT_BUDGET.get(model, 50_000)
    return current_tokens > budget * 0.85   # 85% 时触发裁剪
```

## 裁剪执行顺序

1. 先丢 P5（历史 RAG 注入片段，下次重新检索）
2. 再丢 P4（工具列表，下次重新检索）
3. 再压缩 P3（调用 rolling_summary 脚本）
4. 最后只有在极端情况下才截断 P2 最老的 1-2 轮

**P1 内容永远不裁剪，即使超出 Budget 也不能删。**

## 任务级 Context 隔离（根本解法）

```
全量会话历史（可能数万 Token）
        │
        │  Memory.build_context(current_subtask)
        │  精准过滤：只取与当前子任务相关的层
        ▼
当前子任务 Context（精准裁剪后，< 4K Token）
        │
        ▼ 才传给 LLM
```

即使会话进行 100 轮，每次 LLM 调用的实际 Token 数几乎不增加。

### 实现要点
- 每个子任务启动时声明 `task_type`（如 `code_review`、`data_analysis`）
- Memory Manager 按 `task_type` 过滤相关记忆层
- 工作记忆（L6）只保存当前子任务的中间状态，完成后写入长期记忆
