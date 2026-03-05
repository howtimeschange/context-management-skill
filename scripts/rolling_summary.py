"""
rolling_summary.py — 渐进式滚动摘要压缩核心实现

用法：
    from rolling_summary import ContextMemory
    memory = ContextMemory(short_term_window=5)
    memory.add_turn(user_msg, assistant_msg)
    context = memory.build_context()

依赖：
    pip install anthropic openai tiktoken
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Literal

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


# ──────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────

@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_text(self) -> str:
        return f"[{self.role.upper()}]\n{self.content}"


@dataclass
class Summary:
    text: str
    covers_turns: int   # 此摘要覆盖了多少轮对话
    created_at: float = field(default_factory=time.time)


# ──────────────────────────────────────────
# Token 计数
# ──────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """估算文本 token 数，无 tiktoken 时退回字符估算"""
    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            pass
    # 粗略估算：中文约 1.5 char/token，英文约 4 char/token
    return max(1, len(text) // 3)


# ──────────────────────────────────────────
# 压缩 Prompt
# ──────────────────────────────────────────

COMPRESS_PROMPT = """\
你是精确的信息压缩器。将以下对话轮次压缩为 2-3 句话的关键结论。

规则：
- 保留：具体数字、关键决策、用户明确偏好、错误信息、文件路径
- 丢弃：闲聊、重复内容、过渡性语言
- 格式：每条结论独立一行，以"·"开头

对话内容：
{turn_text}

关键结论（2-3句）："""


def compress_turn_anthropic(turn_text: str) -> str:
    """使用 Anthropic Claude Haiku 压缩单轮对话"""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": COMPRESS_PROMPT.format(turn_text=turn_text)}],
    )
    return response.content[0].text.strip()


def compress_turn_openai(turn_text: str) -> str:
    """使用 OpenAI GPT-4.1-nano 压缩单轮对话"""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        max_tokens=300,
        messages=[{"role": "user", "content": COMPRESS_PROMPT.format(turn_text=turn_text)}],
    )
    return response.choices[0].message.content.strip()


def compress_turn(turn_text: str, provider: Literal["anthropic", "openai"] = "anthropic") -> str:
    """统一压缩入口，失败时降级为截断"""
    try:
        if provider == "anthropic":
            return compress_turn_anthropic(turn_text)
        return compress_turn_openai(turn_text)
    except Exception as e:
        # 降级：直接截断前 200 字符 + 标注
        print(f"[rolling_summary] 压缩失败，降级截断: {e}")
        return f"·（摘要失败，截断）{turn_text[:200]}..."


# ──────────────────────────────────────────
# 核心：ContextMemory
# ──────────────────────────────────────────

class ContextMemory:
    """
    六层记忆中 L3（摘要记忆）+ L4（短期记忆）的实现。

    用法：
        memory = ContextMemory(short_term_window=5)
        memory.add_turn("用户说了什么", "AI 回复了什么")
        messages = memory.build_context()   # 传给 LLM 的 messages 列表
    """

    def __init__(
        self,
        short_term_window: int = 5,
        compress_provider: Literal["anthropic", "openai"] = "anthropic",
        token_budget: int = 8000,
    ):
        self.short_term_window = short_term_window
        self.compress_provider = compress_provider
        self.token_budget = token_budget

        self._short_term: list[tuple[Turn, Turn]] = []   # [(user_turn, assistant_turn), ...]
        self._summaries: list[Summary] = []

    # ── 写入 ──────────────────────────────

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        """每轮对话结束后调用，自动触发渐进式压缩"""
        user_turn = Turn(role="user", content=user_content)
        asst_turn = Turn(role="assistant", content=assistant_content)
        self._short_term.append((user_turn, asst_turn))

        # 渐进式滚动：超出窗口时压缩最老一轮
        if len(self._short_term) > self.short_term_window:
            self._compress_oldest()

    def _compress_oldest(self) -> None:
        """将最老一轮压缩成摘要"""
        oldest_user, oldest_asst = self._short_term.pop(0)
        combined = f"{oldest_user.to_text()}\n\n{oldest_asst.to_text()}"
        compressed = compress_turn(combined, self.compress_provider)
        self._summaries.append(Summary(text=compressed, covers_turns=1))
        print(f"[rolling_summary] 压缩 1 轮 → {count_tokens(combined)}T → {count_tokens(compressed)}T")

    # ── 读取 / 构建 Context ────────────────

    def build_context(
        self,
        system_prompt: str = "",
        current_query: str = "",
    ) -> list[dict]:
        """
        构建传给 LLM 的 messages 列表。

        返回格式：
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "[摘要记忆]\n...\n\n[当前问题]\n..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
        """
        messages: list[dict] = []

        # P1: System Prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # P3: 摘要记忆（拼成一段注入）
        if self._summaries:
            summary_block = "\n".join(f"· {s.text}" if not s.text.startswith("·") else s.text
                                       for s in self._summaries)
            summary_prefix = f"[历史摘要记忆]\n{summary_block}\n\n"
        else:
            summary_prefix = ""

        # P2: 短期记忆（最近 N 轮完整对话）
        for i, (u, a) in enumerate(self._short_term):
            if i == 0 and summary_prefix:
                # 将摘要拼在第一条 user 消息前
                messages.append({"role": "user", "content": summary_prefix + u.content})
            else:
                messages.append({"role": "user", "content": u.content})
            messages.append({"role": "assistant", "content": a.content})

        # 当前 query（如果有）
        if current_query:
            if not self._short_term and summary_prefix:
                messages.append({"role": "user", "content": summary_prefix + current_query})
            else:
                messages.append({"role": "user", "content": current_query})

        return messages

    # ── 状态 ──────────────────────────────

    def stats(self) -> dict:
        short_tokens = sum(
            count_tokens(u.content) + count_tokens(a.content)
            for u, a in self._short_term
        )
        summary_tokens = sum(count_tokens(s.text) for s in self._summaries)
        return {
            "short_term_turns": len(self._short_term),
            "summary_count": len(self._summaries),
            "short_term_tokens": short_tokens,
            "summary_tokens": summary_tokens,
            "total_tokens": short_tokens + summary_tokens,
        }

    def to_json(self) -> str:
        """持久化状态"""
        return json.dumps({
            "short_term": [
                {"user": u.content, "assistant": a.content, "ts": u.timestamp}
                for u, a in self._short_term
            ],
            "summaries": [
                {"text": s.text, "covers_turns": s.covers_turns, "created_at": s.created_at}
                for s in self._summaries
            ],
        }, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: str, **kwargs) -> "ContextMemory":
        """从持久化状态恢复"""
        obj = json.loads(data)
        memory = cls(**kwargs)
        for item in obj.get("short_term", []):
            u = Turn(role="user", content=item["user"], timestamp=item["ts"])
            a = Turn(role="assistant", content=item["assistant"], timestamp=item["ts"])
            memory._short_term.append((u, a))
        for item in obj.get("summaries", []):
            memory._summaries.append(
                Summary(text=item["text"], covers_turns=item["covers_turns"],
                        created_at=item["created_at"])
            )
        return memory


# ──────────────────────────────────────────
# CLI 测试入口
# ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== ContextMemory 演示 ===\n")
    mem = ContextMemory(short_term_window=3, compress_provider="anthropic")

    # 模拟 6 轮对话（会触发 3 次压缩）
    turns = [
        ("帮我分析用户留存数据", "好的，我需要留存数据的文件路径或数据库连接"),
        ("文件在 /data/retention_2024.csv", "已读取，共 12万行，7日留存率 42%"),
        ("对比上个月的数据", "上月7日留存率 38%，环比提升 4 个百分点"),
        ("找出留存率最低的用户群体", "18-24岁新用户留存最低，仅 28%"),
        ("分析原因", "主要原因：首次使用无引导（占比 61%）、加载慢（占比 23%）"),
        ("给出改进建议", "建议优化新手引导流程，目标提升至 35%"),
    ]

    for i, (user, asst) in enumerate(turns):
        mem.add_turn(user, asst)
        stats = mem.stats()
        print(f"轮 {i+1}: 短期={stats['short_term_turns']}轮, "
              f"摘要={stats['summary_count']}条, "
              f"总Token≈{stats['total_tokens']}")

    print("\n--- 最终 Context（build_context 输出）---")
    messages = mem.build_context(system_prompt="你是数据分析助手", current_query="生成分析报告")
    for msg in messages:
        role = msg["role"].upper()
        content_preview = msg["content"][:100].replace("\n", " ")
        print(f"[{role}] {content_preview}...")

    print(f"\n--- 持久化状态 ---")
    print(mem.to_json()[:500] + "...")
