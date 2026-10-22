# ============================================================================
# audit_probes.py —— 评测探针体检器
#
# 为什么需要它：
#   L1 判据是「期望关键词出现在召回里」。但如果探针词本身选得不好，
#   失败的原因就可能是【尺子歪了】而不是【系统坏了】，两者会混在一起分不清。
#   实测踩到过：探针 "BaseModel" 根本不在期望页 body-nested-models.md 里，
#   那条题永远不可能 PASS，却一直被记成"检索失败"。
#
# 探针的四个判据：
#   ① 该词在期望页里出现          否则是死题（永远不可能 PASS）
#   ② 该词出现在该页的【多个块】里   ⚠️ 关键：召回的单位是「块」不是「页」，
#                                   词在页里出现 2 次但挤在同一个块里，照样会漏。
#                                   必须用与主脚本【完全相同】的切分参数来数。
#   ③ 不能太泛（全库/本页比值）      "Path" 全库 840 次，它在召回里出现不能证明捞对了页
#   ④ 尽量不用纯代码符号            文档用 {* ../../docs_src/xxx.py *} 引代码，正文里符号稀疏
#
# 用法：
#   python audit_probes.py            # 只报有问题的题
#   python audit_probes.py --all      # 全量报告
#   python audit_probes.py -c "怎么给 Pydantic 模型的字段加约束？"   # 单题细看
# ============================================================================

import collections
import importlib.util
import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- 载入评测集（复用 eval_cases.py 的唯一定义，避免两处维护）----
# 体检器只需要「题面」，不需要加载被测模块（那会连带加载向量库和 BM25，白等 10 秒）。
# 从 eval_cases 直接取，启动从 ~10 秒降到 <1 秒。
from eval_cases import CASES

ROOT = Path(__file__).parent / "docs_fastapi"

# ⚠️ 必须与 14-ai_agent_assistant.py 里的切分参数保持一致，
#    否则体检出来的"块数"跟真实召回对不上。
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " "],
)

# 英文停用词（探针不能是这些）
STOP = set("""the and for with you your that this are can not use using from will
have has been when what which how all any its into out our their they them
more most some such only also than then there these those very just about
above after again against between both but did does doing down during each
few further had here him his itself like made make many might must now off
once other over own same should still take than that them then through too
under until up upon were what when where which while who whom why would""".split())

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def load_corpus():
    pages, counter = {}, collections.Counter()
    for p in sorted(ROOT.rglob("*.md")):
        rel = p.relative_to(ROOT).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace")
        pages[rel] = text
        counter.update(TOKEN_RE.findall(text))
    return pages, counter


def verdict(chunks_hit: int, n_chunks: int, in_corpus: int, in_page: int):
    """返回 (等级, 说明)。等级：OK / DEAD / FRAGILE / TOO_GENERIC"""
    if chunks_hit == 0:
        return "DEAD", "❌ 死题：探针在该页的任何块里都不存在"
    if chunks_hit == 1 and n_chunks >= 3:
        return "FRAGILE", f"⚠️ 脆：只落在 1/{n_chunks} 块里，容易漏"
    ratio = in_corpus / max(in_page, 1)
    if ratio > 25:
        return "TOO_GENERIC", f"⚠️ 太泛：全库是本页的 {ratio:.0f} 倍，命中不代表捞对页"
    return "OK", f"✅ 良好（落在 {chunks_hit}/{n_chunks} 块）"


def candidates(splits, corpus_counter):
    """挑候选探针：落在多个块里、且不太泛"""
    within = collections.Counter()
    for c in splits:
        within.update(set(TOKEN_RE.findall(c)))          # 按块去重计
    rows = []
    for w, hit_chunks in within.items():
        if hit_chunks < 2 or w.lower() in STOP or len(w) < 4:
            continue
        n_page = len(re.findall(re.escape(w), "\n".join(splits)))
        n_corpus = corpus_counter[w]
        rows.append((w, hit_chunks, n_page, n_corpus, n_corpus / max(n_page, 1)))
    # 先看覆盖的块数（越稳），再看泛化比值（越小越专）
    rows.sort(key=lambda r: (-r[1], r[4]))
    return rows[:8]


def audit_one(q, kws, exp, pages, corpus_counter, verbose=False):
    key = exp + ".md"
    page_text = pages.get(key, "")
    if not page_text:
        print(f"  ❌ 期望页不存在：{key}")
        return "DEAD"

    # 用与主脚本一致的参数切块，数「该页一共几块 / 探针落在几块里」
    splits = SPLITTER.split_text(page_text)
    n_chunks = len(splits)

    worst, lines = "OK", []
    order = ["OK", "TOO_GENERIC", "FRAGILE", "DEAD"]
    for kw in kws:
        hit = sum(1 for c in splits if kw in c)
        n_page = len(re.findall(re.escape(kw), page_text))
        n_corpus = len(re.findall(re.escape(kw), "\n".join(pages.values())))
        lvl, msg = verdict(hit, n_chunks, n_corpus, n_page)
        lines.append(f"    探针 {kw!r:14} 落在 {hit}/{n_chunks} 块  "
                     f"(本页 {n_page} 次 / 全库 {n_corpus} 次)   {msg}")
        if order.index(lvl) > order.index(worst):
            worst = lvl

    if verbose or worst != "OK":
        print("-" * 78)
        print(f"  [{worst}] {q}")
        print(f"    期望页：{key}（{len(page_text)} 字符 → {n_chunks} 块）")
        for ln in lines:
            print(ln)
        print("    ── 候选探针（落在多个块里、且不太泛，按稳定性排序）──")
        for w, hit, np_, nc, ratio in candidates(splits, corpus_counter):
            print(f"       {w:<26} 落在 {hit}/{n_chunks} 块   本页 {np_:>3} 次 / 全库 {nc:>4} 次   比值 {ratio:.1f}")
    return worst


if __name__ == "__main__":
    pages, corpus_counter = load_corpus()
    print("=" * 78)
    print(f"评测探针体检　语料 {len(pages)} 篇")
    print("=" * 78)

    show_all = "--all" in sys.argv
    only = None
    if "-c" in sys.argv:
        only = sys.argv[sys.argv.index("-c") + 1]

    stats = collections.Counter()
    for kind, q, l1_kws, l2_kws, exp in CASES:
        if kind != "kb":
            continue
        if only and only not in q:
            continue
        stats[audit_one(q, l1_kws, exp, pages, corpus_counter, verbose=show_all)] += 1

    print("=" * 78)
    print("汇总：" + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print("DEAD/FRAGILE/TOO_COMMON 的题，先修探针再谈系统优化。")
    print("=" * 78)
