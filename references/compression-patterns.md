# 滚动摘要压缩模式参考

## 核心原则：渐进式 vs 批量

| 模式 | 时机 | 问题 |
|------|------|------|
| ❌ 批量压缩 | 攒够 N 轮再压 | 信息断层，压缩时可能丢失近期重要上下文 |
| ✅ 渐进式滚动 | 每轮结束后压最老一轮 | 摘要始终是最新状态，无断层 |

## 渐进式滚动压缩伪代码

```python
# 短期记忆窗口大小
SHORT_TERM_WINDOW = 5  # 保留最近 5 轮完整对话

# 每轮对话结束后调用
def rolling_compress(short_term: list, summary_memory: list, cheap_model):
    if len(short_term) > SHORT_TERM_WINDOW:
        oldest = short_term.pop(0)           # 取出最老一轮
        summary = cheap_model.call(
            f"将以下对话压缩成 2-3 句关键结论，保留事实、决策、数字：\n{oldest}"
        )                                     # 约 200 Token，原文可能 2000 Token
        summary_memory.append(summary)        # 压缩率约 10x
    return short_term, summary_memory
```

## 压缩质量 Prompt 模板

```
你是精确的信息压缩器。将以下对话轮次压缩为 2-3 句话的关键结论。

规则：
- 保留：具体数字、关键决策、用户明确偏好、错误信息
- 丢弃：闲聊、重复、过渡性语言
- 格式：每条结论独立一行，以"·"开头

对话内容：
{oldest_turn}

关键结论（2-3句）：
```

## 成本估算

| 操作 | Token 消耗 | 成本（Haiku/GPT-4.1-nano） |
|------|-----------|--------------------------|
| 压缩单轮对话（约 2K Token → 200 Token） | input 2K + output 200 | ~$0.001 |
| 100 轮对话全部压缩 | 100 × $0.001 | ~$0.10 |
| 节省的 Token 价值 | 减少 100K Token 主模型调用 | ~$1-10（取决于模型） |

**结论：用 1% 的成本换 90% 的 Context 节省，ROI 极高。**

## RAG 长期记忆模式

```python
# ❌ 错误：全量注入（Context 随时间线性增长）
context = system_prompt + ALL_historical_summaries + current_query

# ✅ 正确：RAG 检索注入（Context 占用固定）
relevant = vector_store.search(current_query, top_k=3)
context = system_prompt + relevant_memories + current_query
# 长期记忆始终只占约 300-500 Token，与历史长度无关
```

### 向量库写入时机
- 每个子任务完成后，将关键结论写入向量库
- 写入前先去重（余弦相似度 > 0.95 视为重复）
- 元数据字段：`task_type`、`timestamp`、`importance`（high/medium/low）

### 检索策略
- 用当前 query 做相似度检索，Top-3 最相关片段
- 可叠加时间衰减权重：score = similarity * (0.9 ^ days_ago)
- 若 Top-1 相似度 < 0.7，说明长期记忆中无相关内容，跳过注入

## Prompt Caching 配置

### Anthropic (Claude)
```python
# system prompt 超过 1024 Token 自动触发缓存
# cache_control 标记固定前缀
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": long_system_content,
                "cache_control": {"type": "ephemeral"}  # 5分钟TTL
            },
            {"type": "text", "text": user_query}
        ]
    }
]
```

### OpenAI (GPT-4.1 系列)
```python
# 超过 1024 Token 的前缀自动缓存，读取 50% 折扣
# 无需额外配置，确保 system prompt 内容固定即可
# 频繁变化的内容放到 user message，不放 system
```

### 命中率优化技巧
- System Prompt 内容永远放在 messages 的最前面
- 动态内容（日期、用户信息）放到 user message 里，不要插入 system
- 同一 system prompt 在多个请求间保持字节级一致
