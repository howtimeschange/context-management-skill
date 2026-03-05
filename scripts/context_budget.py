"""
context_budget.py — Context Budget 检查与分层裁剪工具

用法：
    from context_budget import ContextBudget, trim_messages

    budget = ContextBudget(model="claude-3-7")
    messages = [...]  # 原始 messages 列表
    trimmed = trim_messages(messages, budget)

依赖：
    pip install tiktoken  # 可选，无则退回字符估算
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


# ──────────────────────────────────────────
# Model Context Budgets（安全阈值，留 15% buffer）
# ──────────────────────────────────────────

MODEL_BUDGETS: dict[str, int] = {
    # Anthropic
    "claude-3-7":           80_000,
    "claude-3-5-sonnet":    80_000,
    "claude-haiku":         15_000,
    "claude-haiku-4-5":     15_000,
    # OpenAI
    "gpt-4.1":             100_000,
    "gpt-4.1-mini":         40_000,
    "gpt-4.1-nano":         15_000,
    "gpt-4o":               80_000,
    "gpt-4o-mini":          20_000,
    # Google
    "gemini-2.5-pro":      900_000,
    "gemini-2.5-flash":    900_000,
    # 默认
    "default":              50_000,
}

# 触发裁剪的阈值（占 budget 的比例）
TRIM_THRESHOLD = 0.85


# ──────────────────────────────────────────
# Token 计数
# ──────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """估算文本 token 数"""
    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 3)


def count_messages_tokens(messages: list[dict], model: str = "gpt-4") -> int:
    """统计 messages 列表总 token 数"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Anthropic 多模态格式
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += count_tokens(block["text"], model)
        elif isinstance(content, str):
            total += count_tokens(content, model)
        total += 4  # message overhead
    return total


# ──────────────────────────────────────────
# ContextBudget 配置
# ──────────────────────────────────────────

@dataclass
class ContextBudget:
    model: str = "default"
    max_tokens: int = 0          # 0 = 从 MODEL_BUDGETS 自动取
    trim_threshold: float = TRIM_THRESHOLD

    def __post_init__(self):
        if self.max_tokens == 0:
            self.max_tokens = MODEL_BUDGETS.get(self.model, MODEL_BUDGETS["default"])

    @property
    def trim_at(self) -> int:
        return int(self.max_tokens * self.trim_threshold)

    def is_over_budget(self, messages: list[dict]) -> bool:
        return count_messages_tokens(messages, self.model) > self.trim_at

    def usage_report(self, messages: list[dict]) -> dict:
        used = count_messages_tokens(messages, self.model)
        return {
            "model": self.model,
            "used_tokens": used,
            "budget": self.max_tokens,
            "trim_at": self.trim_at,
            "usage_pct": round(used / self.max_tokens * 100, 1),
            "over_budget": used > self.trim_at,
        }


# ──────────────────────────────────────────
# 分层裁剪逻辑
# ──────────────────────────────────────────

def classify_message(msg: dict) -> Literal["p1", "p2", "p3", "p4", "p5"]:
    """
    将单条 message 分类到优先级层。
    调用方需要在 message 中附加 _priority 字段，或通过内容模式判断。

    优先级规则：
      p1 = system role，或含 _priority="p1" 标记
      p2 = 最近 3-5 轮 user/assistant（由 trim_messages 按索引判断）
      p3 = 较早的 user/assistant 历史
      p4 = 含工具定义（tools injection）的消息
      p5 = 含 [历史摘要记忆] 或 _priority="p5" 标记的消息
    """
    priority = msg.get("_priority")
    if priority:
        return priority

    role = msg.get("role", "")
    content = msg.get("content", "") if isinstance(msg.get("content"), str) else ""

    if role == "system":
        return "p1"
    if "[TOOL_SCHEMA]" in content or "tool_schema" in msg:
        return "p4"
    if "[历史摘要记忆]" in content or "[SUMMARY_MEMORY]" in content:
        return "p5"
    return "p2"   # 默认短期记忆，由 trim_messages 降级到 p3


def trim_messages(
    messages: list[dict],
    budget: ContextBudget,
    short_term_keep: int = 5,
    verbose: bool = False,
) -> list[dict]:
    """
    按五级优先级裁剪 messages，直到 token 数低于 budget.trim_at。

    裁剪顺序（从低到高）：
    1. 丢弃 P5（RAG 注入的历史记忆片段）
    2. 丢弃 P4（工具 Schema 注入）
    3. 压缩 P3（早期历史，截断内容）
    4. 截断 P2 最老的轮（极端情况）
    P1 永不裁剪。
    """
    if not budget.is_over_budget(messages):
        return messages

    # 分离 system（P1）和对话消息
    system_msgs = [m for m in messages if m.get("role") == "system"]
    dialog_msgs = [m for m in messages if m.get("role") != "system"]

    def current_tokens():
        return count_messages_tokens(system_msgs + dialog_msgs, budget.model)

    def log(msg):
        if verbose:
            print(f"[context_budget] {msg}")

    # 标记短期记忆（最近 short_term_keep 对 user/assistant）
    user_asst = [(i, m) for i, m in enumerate(dialog_msgs)
                 if m.get("role") in ("user", "assistant")]
    recent_indices = set(i for i, _ in user_asst[-short_term_keep * 2:])

    # Step 1: 丢弃 P5（含 [历史摘要记忆] 的消息）
    before = current_tokens()
    dialog_msgs = [m for m in dialog_msgs if "[历史摘要记忆]" not in str(m.get("content", ""))]
    if verbose and current_tokens() < before:
        log(f"P5 裁剪：{before}T → {current_tokens()}T")
    if not budget.is_over_budget(system_msgs + dialog_msgs):
        return system_msgs + dialog_msgs

    # Step 2: 丢弃 P4（工具 Schema 注入消息）
    before = current_tokens()
    dialog_msgs = [m for m in dialog_msgs
                   if not ("[TOOL_SCHEMA]" in str(m.get("content", "")) or m.get("_priority") == "p4")]
    if verbose and current_tokens() < before:
        log(f"P4 裁剪：{before}T → {current_tokens()}T")
    if not budget.is_over_budget(system_msgs + dialog_msgs):
        return system_msgs + dialog_msgs

    # Step 3: 压缩 P3（早期历史对话，截断到 200 字符摘要）
    new_dialog = []
    for i, m in enumerate(dialog_msgs):
        if i in recent_indices or m.get("role") not in ("user", "assistant"):
            new_dialog.append(m)
        else:
            # 截断早期历史内容
            content = str(m.get("content", ""))
            if len(content) > 300:
                short_content = content[:200] + "…[已截断]"
                new_dialog.append({**m, "content": short_content})
            else:
                new_dialog.append(m)
    dialog_msgs = new_dialog
    log(f"P3 压缩后：{current_tokens()}T")
    if not budget.is_over_budget(system_msgs + dialog_msgs):
        return system_msgs + dialog_msgs

    # Step 4: 极端情况——移除 P2 最老的 1-2 对
    user_asst_pairs = []
    i = 0
    while i < len(dialog_msgs) - 1:
        if (dialog_msgs[i].get("role") == "user" and
                dialog_msgs[i+1].get("role") == "assistant"):
            user_asst_pairs.append((i, i+1))
            i += 2
        else:
            i += 1

    removed = 0
    for u_idx, a_idx in user_asst_pairs[:2]:   # 最多移除最老 2 对
        if budget.is_over_budget(system_msgs + dialog_msgs):
            # 重新计算 idx（因为列表在变化）
            dialog_msgs = [m for j, m in enumerate(dialog_msgs)
                           if j not in (u_idx - removed, a_idx - removed)]
            removed += 2
            log(f"P2 强制裁剪 1 对：{current_tokens()}T")

    return system_msgs + dialog_msgs


# ──────────────────────────────────────────
# 工具按需传入（向量化检索替代全量塞入）
# ──────────────────────────────────────────

def select_tools_for_query(
    all_tools: list[dict],
    query: str,
    top_k: int = 5,
    always_include: list[str] | None = None,
) -> list[dict]:
    """
    从工具列表中按需选取与 query 最相关的 Top-K 工具。

    简单版：基于关键词匹配（无需向量库）
    生产版：替换为 sentence-transformers 或 OpenAI embeddings 做余弦检索

    Args:
        all_tools: 完整工具列表，每个工具格式：{"name": ..., "description": ...}
        query: 当前用户 query
        top_k: 返回最多 top_k 个工具
        always_include: 永远包含的基础工具名列表（白名单）
    """
    always_include = always_include or []

    # 强制包含白名单工具
    must_have = [t for t in all_tools if t.get("name") in always_include]
    candidates = [t for t in all_tools if t.get("name") not in always_include]

    # 简单关键词评分
    query_words = set(query.lower().split())

    def score(tool: dict) -> float:
        text = f"{tool.get('name', '')} {tool.get('description', '')}".lower()
        return sum(1 for w in query_words if w in text)

    ranked = sorted(candidates, key=score, reverse=True)
    selected = must_have + ranked[:max(0, top_k - len(must_have))]

    return selected[:top_k]


# ──────────────────────────────────────────
# CLI 测试入口
# ──────────────────────────────────────────

if __name__ == "__main__":
    budget = ContextBudget(model="claude-haiku", max_tokens=2000)

    # 构造超出 budget 的 messages
    messages = [
        {"role": "system", "content": "你是一个助手" * 10},
        {"role": "user", "content": "[历史摘要记忆]\n· 用户做数据分析\n· 文件在 /data/"},
        {"role": "assistant", "content": "好的，我来帮你分析" * 50},
        {"role": "user", "content": "第二轮问题" * 50},
        {"role": "assistant", "content": "第二轮回答" * 50},
        {"role": "user", "content": "第三轮问题" * 50},
        {"role": "assistant", "content": "第三轮回答" * 50},
        {"role": "user", "content": "当前问题"},
    ]

    report = budget.usage_report(messages)
    print(f"裁剪前：{report['used_tokens']}T ({report['usage_pct']}%), budget={report['budget']}T")

    trimmed = trim_messages(messages, budget, verbose=True)
    report2 = budget.usage_report(trimmed)
    print(f"裁剪后：{report2['used_tokens']}T ({report2['usage_pct']}%)")
    print(f"消息数：{len(messages)} → {len(trimmed)}")

    # 工具选取演示
    tools = [
        {"name": "file_read", "description": "读取文件内容"},
        {"name": "web_search", "description": "搜索互联网"},
        {"name": "code_exec", "description": "执行 Python 代码"},
        {"name": "send_email", "description": "发送电子邮件"},
        {"name": "db_query", "description": "查询数据库"},
        {"name": "chart_gen", "description": "生成图表和可视化"},
    ]
    selected = select_tools_for_query(tools, "帮我分析数据并生成图表", top_k=3,
                                       always_include=["file_read"])
    print(f"\n工具按需选取（query='帮我分析数据并生成图表'）：")
    for t in selected:
        print(f"  · {t['name']}: {t['description']}")
