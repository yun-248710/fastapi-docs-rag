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
from eval_cases import CASES, TOTAL
from eval_judge import judge_kb, judge_oob


def l1_retrieval(kind, q, l1_kws, expected):
    """第一级：只测检索层，不调模型。返回 (是否通过, 召回条数, 失败原因)"""
    docs = ta.final_context_docs(q)
    ok, why = judge_oob(docs) if kind == "oob" else judge_kb(docs, l1_kws, expected)
    return ok, len(docs), why


def l2_answer_ok(q, l2_kws) -> bool:
    """第二级：测最终回答。期望词【任一】命中即通过（答案可能与语料不同语言）"""
    ans = ta.ask(q).lower()
    return any(k.lower() in ans for k in l2_kws)


if __name__ == "__main__":
    print("=" * 70)
    print(f"FastAPI 语料 RAG 评测（{TOTAL} 题｜L1 含来源归因｜L1/L2 词表分离）")
    print(f"当前生成模型：{ta.MODEL_ID}")
    print("=" * 70)

    l1_pass = l2_pass = 0
    l2_errors = []              # 单题异常（429 限流 / 超时等），不能让整批崩掉
    l1_fail_src_only = 0
    l2_kw_all = 0
    fails = []

    for kind, q, l1_kws, l2_kws, expected in CASES:
        l1, n_docs, why = l1_retrieval(kind, q, l1_kws, expected)
        # ⚠️ 单题独立 try：踩过的坑 —— 一次 429 把整批 29 题打断，前面全白跑。
        try:
            l2 = l2_answer_ok(q, l2_kws)
            err = ""
        except Exception as e:
            l2, err = False, f"{type(e).__name__}: {str(e)[:70]}"
            l2_errors.append((q, err))
        l1_pass += l1
        l2_pass += l2
        if not l1 and why == "来源✗":
            l1_fail_src_only += 1
        if not l1 and why == "关键词✗":
            l2_kw_all += 1

        mark = "ERR" if err else ("OK " if (l1 and l2) else ("WARN" if l2 else "FAIL"))
        detail = f"（{why}）" if why else ""
        extra = f"（召回 {n_docs} 块）" if kind == "oob" else ""
        print(f"[{mark}] L1:{'PASS' if l1 else 'FAIL'}{detail}  "
              f"L2:{'PASS' if l2 else 'FAIL'}  [{kind}] {q}{extra}")
        if not l1:
            fails.append((q, why, n_docs))

    print("-" * 70)
    print(f"检索命中  {l1_pass}/{TOTAL}")
    if l2_errors and len(l2_errors) == TOTAL:
        # 全部异常时绝不能打印 0/29 —— 那会被误读成"模型答得烂"
        print(f"回答通过  —/{TOTAL}（全部调用异常，无有效结果）")
    elif l2_errors:
        print(f"回答通过  {l2_pass}/{TOTAL - len(l2_errors)}"
              f"（有效题数；另有 {len(l2_errors)} 题调用异常，未计入）")
    else:
        print(f"回答通过  {l2_pass}/{TOTAL}")
    if l2_errors:
        print(f"⚠️ 调用异常  {len(l2_errors)}/{TOTAL} 题（结果不完整，先修调用再谈分数）")
    print(f"其中「来源对、关键词没覆盖」：{l2_kw_all} 条")
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
        for q, why, n in fails:
            print(f"  · [{why}] 召回{n}块  {q}")
    print("=" * 70)
    print("判读：L1 FAIL + L2 PASS = 模型在兜底（检索层有问题）")
    print("      L1 PASS + L2 FAIL = 资料给了没用上（生成层有问题）")
    print("      L1 来源✗        = 检索到了字面，但捞错了文件（数据/排序问题）")
