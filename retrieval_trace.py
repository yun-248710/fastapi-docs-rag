# ============================================================================
# retrieval_trace.py —— 极简检索 tracing（数据层 + 展示层）
#
# 【为什么不上 Phoenix / Langfuse】
#   我们的检索不是标准 LangChain retriever，是自定义的
#   「向量阈值过滤 + BM25 补召 + 联动闸门」三段决策。
#   Phoenix 的 LangChain instrumentation 只能自动看到 LLM 调用，
#   **看不到这些检索决策** —— 要看就必须手动埋 span。
#   既然要手写，就先不引入框架：把决策路径存成结构化 dict，
#   需要 UI 时把这套 dict 转成 OTel span attributes 导出即可（本机已装 opentelemetry-sdk），
#   检索逻辑一行都不用改。
#
# 【用法】
#   python retrieval_trace.py "FastAPI 怎么配置跨域 CORS？"      # 单条，打印决策路径
#   python retrieval_trace.py                                    # 跑几条示例
#
#   代码里：
#     from retrieval_trace import trace_query, render, save
#     t = trace_query(ta, question)
#     print(render(t)); save(t)
# ============================================================================

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TRACE_FILE = ROOT / "traces.jsonl"


def load_ta():
    """按路径加载被测模块（文件名带连字符，不能直接 import）"""
    spec = importlib.util.spec_from_file_location("ta", ROOT / "14-ai_agent_assistant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace_query(ta, question: str) -> dict:
    """跑一次检索，把决策路径整条记录下来"""
    t: dict = {}
    ta.final_context_docs(question, trace=t)
    return t


def save(trace: dict, path: Path = TRACE_FILE) -> Path:
    """追加到 JSONL —— 一行一条，可以事后回看"当时为什么这么判" """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    return path


def shorter(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _width(s: str) -> int:
    """显示宽度：中日韩字符占 2 列，其余占 1 列。
    不这么做的话中文列用 Python 的 str.ljust 会对不齐（它按字符数算，不按显示宽度）。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)


def pad(s: str, n: int) -> str:
    return s + " " * max(0, n - _width(s))


def render(t: dict) -> str:
    """把决策路径渲染成可以直接读、也可以直接拿去面试讲的样子"""
    out = []
    line = "─" * 66
    thr = t.get("threshold")
    cands = t.get("vector_candidates", [])
    kept = [c for c in cands if c.get("score", 0) >= thr]
    dropped = [c for c in cands if c.get("score", 0) < thr]

    out.append("═" * 66)
    out.append(f"查询：{t.get('question')}")
    out.append(f"阈值={thr}   向量召回 k={t.get('vector_k')}   "
               f"过阈值 {len(kept)} 条 / 被拦 {len(dropped)} 条")
    out.append(line)

    out.append("① 向量路候选（按分数降序，★=过阈值）")
    for i, c in enumerate(cands, 1):
        mark = "★" if c.get("score", 0) >= thr else " "
        score = c.get("score")
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "  -   "
        out.append(f"   {mark} [{i}] {pad(score_s, 8)}{pad(shorter(c.get('src',''), 32), 34)}"
                   f"{shorter(c.get('preview',''), 40)}")

    out.append(line)
    out.append(f"② 联动闸门：{t.get('gate')}")

    bm = t.get("bm25_added", [])
    out.append(f"③ BM25 补召：{len(bm)} 条")
    for c in bm:
        score = c.get("score")
        score_s = f"BM25={score:.2f}" if isinstance(score, (int, float)) else "BM25=  -  "
        out.append(f"      {pad(score_s, 12)}{pad(shorter(c.get('src',''), 32), 34)}"
                   f"{shorter(c.get('preview',''), 40)}")

    out.append(line)
    fin = t.get("final", [])
    out.append(f"④ ★ 最终进上下文（编号 = 回答里的 [n]），共 {len(fin)} 块")
    if not fin:
        out.append("      (空 —— 判定为库外问题，不允许 BM25 补召)")
    for i, c in enumerate(fin, 1):
        tag = f"[{c.get('via','?')}]"
        out.append(f"      {pad(f'[{i}] {tag}', 12)}{pad(shorter(c.get('src',''), 32), 34)}"
                   f"{shorter(c.get('preview',''), 40)}")
    out.append("═" * 66)
    return "\n".join(out)


if __name__ == "__main__":
    ta = load_ta()
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    questions = args or [
        "FastAPI 怎么配置跨域 CORS？",      # 正常命中
        "FastAPI 怎么处理表单数据？",        # 检索失败（正确文件没进窗口）
        "今天天气怎么样？",                  # 库外 → 应触发闸门
    ]
    for q in questions:
        t = trace_query(ta, q)
        print(render(t))
        save(t)
        print()
    print(f"已追加到 {TRACE_FILE.name}（{len(questions)} 条）")
