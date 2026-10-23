# ============================================================================
# eval_fastapi.py —— 跑分脚本（给出 L1/L2 两个总数）
#
# 与 test_retrieval.py 的分工：
#   test_retrieval.py ：断言式回归（CI 用），只告诉你"过/不过"
#   eval_fastapi.py   ：跑分统计，告诉你"现在几比几"、失败清单是什么
# 两者共用 eval_cases.py（题）和 eval_judge.py（判据），不会出现口径漂移。
#
# 用法：python eval_fastapi.py
# ============================================================================

import importlib.util
from pathlib import Path

# 文件名带连字符不能直接 import，用 importlib 加载
spec = importlib.util.spec_from_file_location(
    "ta", Path(__file__).parent / "14-ai_agent_assistant.py"
)
ta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ta)

# 评测集定义与判据都抽出去了 —— 跑分脚本和 pytest 套件共用同一份
from eval_cases import CASES, TOTAL, case_category, CATEGORY_LABELS
from eval_judge import (
    judge_kb, judge_oob, rel_path, src_match, JUDGE_SYSTEM, JUDGE_HUMAN,
)


def l1_retrieval(kind, q, l1_kws, expected):
    """第一级：只测检索层，不调模型。

    返回 (是否通过, 召回条数, 失败原因, 来源是否命中, 上下文)。
    来源是否命中要单独返回：L1 是「来源 ∧ 探针」的复合判据，
    只看 L1 会把"检索修好了"这件事藏起来。
    """
    docs = ta.final_context_docs(q)
    if kind == "oob":
        ok, why = judge_oob(docs)
        return ok, len(docs), why, ok, docs
    ok, why = judge_kb(docs, l1_kws, expected)
    srcs = [rel_path(d.metadata.get("source", "")) for d in docs]
    return ok, len(docs), why, src_match(srcs, expected), docs


# ---- L2b：语义裁判（只在关键词判负时调用，每次一条模型调用）----
_judge_chain = None


def judge_answer(question: str, answer: str, docs) -> bool:
    """关键词判负后复核：答案是否真的答对了。

    ⚠️ 裁判失败（限流/超时）一律判负 —— 宁可少给分，也不要"因为裁判挂了所以算过"。
    """
    global _judge_chain
    if _judge_chain is None:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", JUDGE_SYSTEM),
            ("human", JUDGE_HUMAN),
        ])
        _judge_chain = prompt | ta.llm.bind(temperature=0) | StrOutputParser()

    context = "\n\n".join(d.page_content for d in docs)[:4000] or "（无召回内容）"
    try:
        out = _judge_chain.invoke(
            {"question": question, "context": context, "answer": answer}
        ).strip().upper()
        return out.startswith("PASS")
    except Exception:
        return False


def l2_eval(q, l2_kws, docs):
    """第二级：测最终回答。

    返回 (关键词是否命中, 最终是否通过)。
    关键词做快速过滤：命中即通过（免费、确定性）；没命中才调裁判复核，
    因为关键词只能测"有没有写出某个符号"，测不了"解释对不对"。
    """
    ans = ta.ask(q)
    kw_ok = any(k.lower() in ans.lower() for k in l2_kws)
    if kw_ok:
        return True, True
    return False, judge_answer(q, ans, docs)


if __name__ == "__main__":
    print("=" * 70)
    print(f"FastAPI 语料 RAG 评测（{TOTAL} 题｜L1 含来源归因｜L1/L2 词表分离）")
    print(f"当前生成模型：{ta.MODEL_ID}")
    print("=" * 70)

    l1_pass = l2_pass = l2_kw_pass = 0
    judge_calls = judge_flips = 0
    l2_errors = []              # 单题异常（429 限流 / 超时等），不能让整批崩掉
    l1_fail_src_only = 0
    l2_kw_all = 0
    fails = []
    # 分层统计：只有一个总数会掩盖"哪一层不行"
    by_cat = {}

    for case in CASES:
        kind, q, l1_kws, l2_kws, expected = case
        l1, n_docs, why, src_ok, docs = l1_retrieval(kind, q, l1_kws, expected)
        cat = case_category(case)
        row = by_cat.setdefault(cat, {"n": 0, "src": 0, "l1": 0})
        row["n"] += 1
        row["src"] += bool(src_ok)
        row["l1"] += bool(l1)

        # ⚠️ 单题独立 try：踩过的坑 —— 一次 429 把整批题打断，前面全白跑。
        try:
            kw_ok, l2 = l2_eval(q, l2_kws, docs)
            err = ""
        except Exception as e:
            kw_ok, l2 = False, False
            err = f"{type(e).__name__}: {str(e)[:70]}"
            l2_errors.append((q, err))

        l2_kw_pass += kw_ok
        if not kw_ok and err == "":
            judge_calls += 1
            judge_flips += bool(l2)
        l1_pass += l1
        l2_pass += l2
        if not l1 and why == "来源✗":
            l1_fail_src_only += 1
        if not l1 and why == "关键词✗":
            l2_kw_all += 1

        mark = "ERR" if err else ("OK " if (l1 and l2) else ("WARN" if l2 else "FAIL"))
        detail = f"（{why}）" if why else ""
        l2_note = "（裁判改判）" if (not kw_ok and l2) else ""
        extra = f"（召回 {n_docs} 块）" if kind == "oob" else ""
        print(f"[{mark}] L1:{'PASS' if l1 else 'FAIL'}{detail}  "
              f"L2:{'PASS' if l2 else 'FAIL'}{l2_note}  [{cat}] {q}{extra}")
        if not l1:
            fails.append((q, why, n_docs, cat))

    print("-" * 70)
    print(f"检索命中  {l1_pass}/{TOTAL}")
    if l2_errors and len(l2_errors) == TOTAL:
        # 全部异常时绝不能打印 0 分 —— 那会被误读成"模型答得烂"
        print(f"回答通过  —/{TOTAL}（全部调用异常，无有效结果）")
    else:
        valid = TOTAL - len(l2_errors)
        print(f"回答通过（关键词） {l2_kw_pass}/{valid}   ← 只认'答案里写没写那个符号'")
        print(f"回答通过（+裁判）   {l2_pass}/{valid}   ← 裁判复核 {judge_calls} 条，改判通过 {judge_flips} 条")
    if l2_errors:
        print(f"⚠️ 调用异常  {len(l2_errors)}/{TOTAL} 题（结果不完整，先修调用再谈分数）")
    print(f"其中「来源对、关键词没覆盖」：{l2_kw_all} 条")
    print("-" * 70)

    # 分层报告：一个总数会掩盖"哪一层不行"
    print("分层结果（L1 = 来源 ∧ 探针；来源列更接近检索真实水位）")
    print(f"  {'层':<10}{'题数':>5}{'来源命中':>10}{'L1':>10}")
    order = ["tutorial", "advanced", "reference", "how-to", "deployment", "async", "python-types", "oob"]
    for cat in order:
        if cat not in by_cat:
            continue
        r = by_cat[cat]
        label = CATEGORY_LABELS.get(cat, cat)
        print(f"  {label:<10}{r['n']:>5}{r['src']:>7}/{r['n']:<3}{r['l1']:>6}/{r['n']}")
    print("-" * 70)

    if l2_errors:
        print("调用异常清单（前 3 条）：")
        for q, e in l2_errors[:3]:
            print(f"  · {q}")
            print(f"    {e}")
        print("提示：若报 429 / SetLimitExceeded，去火山方舟控制台「开通管理」")
        print("      调整或关闭该模型的「安全体验模式」。各模型额度是分开计量的。")
        print("-" * 70)

    if fails:
        print("L1 失败清单：")
        for q, why, n, cat in fails:
            print(f"  · [{cat}/{why}] 召回{n}块  {q}")
    print("=" * 70)
    print("判读：L1 FAIL + L2 PASS = 模型在兜底（检索层有问题）")
    print("      L1 PASS + L2 FAIL = 资料给了没用上（生成层有问题）")
    print("      L1 来源✗        = 检索到了字面，但捞错了文件（数据/排序问题）")
