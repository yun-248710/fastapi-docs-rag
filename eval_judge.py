# ============================================================================
# eval_judge.py —— 评测判据的【唯一】实现
#
# 为什么单独抽出来：
#   eval_fastapi.py（跑分）和 test_retrieval.py（pytest 回归）必须用【同一套判据】。
#   两处各写一份 → 测试过了但跑分不过（或反过来），这种不一致最难查。
#
# 判据定义：
#   kb  ：L1期望词命中【且】来源落在期望文件里
#   oob ：召回为空（召回非空 = 噪声泄漏）
# ============================================================================

DOCS_MARKER = "docs_fastapi/"


def norm_path(path: str) -> str:
    """统一路径分隔符，便于跨平台做子串匹配"""
    return str(path).replace("\\", "/")


def rel_path(path: str) -> str:
    """把绝对路径收敛成 docs_fastapi 下的相对路径（保留扩展名）"""
    s = norm_path(path)
    i = s.find(DOCS_MARKER)
    return s[i + len(DOCS_MARKER):] if i >= 0 else s


def src_match(rel_srcs, expected) -> bool:
    """来源匹配必须卡【路径边界】，不能用朴素子串。

    ⚠️ 踩过的坑：expected="tutorial/query-params" 会匹配到
    "tutorial/query-params-str-validations.md"（前缀撞车），把捞错文件判成捞对。
    "tutorial/body" 同理撞 body-fields / body-nested-models / body-updates。

    expected 支持 str 或 tuple/list（多个可接受来源）——一个问题常有多个正确页，
    比如"APIRouter 有哪些参数"既能答 reference/apirouter，也能答
    tutorial/big-applications。只认一个会把答对判成答错。
    """
    if isinstance(expected, str):
        exp_list = (expected,)
    elif expected is None:
        exp_list = ()
    else:
        exp_list = tuple(expected)

    for r in rel_srcs:
        r_noext = r[:-3] if r.endswith(".md") else r
        for expected_one in exp_list:
            exp = expected_one.rstrip("/")
            if r_noext == exp or r_noext.startswith(exp + "/"):
                return True
    return False


def judge_kb(docs, l1_kws, expected):
    """库内题判定。返回 (是否通过, 失败原因)

    失败原因分三种，便于定位：
      "召回为空"     —— 一条都没过阈值
      "关键词✗"      —— 来源对了，但召回的片段不含任何探针词（可能抽错块）
      "来源✗"        —— 期望词出现了，但正确文件没进上下文（捞错文件）
      "关键词✗来源✗" —— 两者都挂

    ⚠️ 探针用 any() 而不是 all()：探针是"这页特有的证据"，命中任意一个
    就说明捞到的是这页的内容。旧版要求全部命中，实测把两条正确召回判成失败
    —— 正确页进了窗口，但承载某个词的块落在另一段里。
    """
    if not docs:
        return False, "召回为空"
    text = "\n".join(d.page_content for d in docs)
    srcs = [rel_path(d.metadata.get("source", "")) for d in docs]

    kw_ok = any(k in text for k in (l1_kws or []))
    src_ok = src_match(srcs, expected) if expected else True

    if kw_ok and src_ok:
        return True, ""
    if not kw_ok and not src_ok:
        return False, "关键词✗来源✗"
    if not kw_ok:
        return False, "关键词✗"
    return False, "来源✗"


def judge_oob(docs):
    """库外题判定：召回为空 = 正确"""
    n = len(docs)
    return (n == 0), ("" if n == 0 else f"泄漏{n}块")


def format_docs(docs, limit=8) -> str:
    """把召回结果格式化成可直接读的决策路径 —— 用于测试失败信息，省去再跑一次诊断"""
    if not docs:
        return "      (上下文为空)"
    lines = []
    for i, d in enumerate(docs[:limit], 1):
        src = rel_path(d.metadata.get("source", "?"))
        preview = d.page_content[:50].replace("\n", " ")
        lines.append(f"      [{i}] {src} | {preview}…")
    if len(docs) > limit:
        lines.append(f"      …… 另有 {len(docs) - limit} 块")
    return "\n".join(lines)
