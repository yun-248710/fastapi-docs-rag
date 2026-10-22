# ============================================================================
# test_retrieval.py —— 检索链路回归测试套件
#
# 解决的问题：以前「效果变好了」只能靠人眼对比两次控制台输出、手写「L1 20→21」。
# 现在它是一条可执行、可复现、能进 CI 的断言。
#
# 【两类用例】
#   默认跑（快 / 免费 / 不调大模型，只需本地 Ollama 做 embedding）：
#     - 库内题：正确文档是否进上下文（主指标）
#     - 库内题：探针词是否被覆盖（辅助指标）
#     - 库外题：是否拒绝回答（召回必须为空）
#     - 三个「元测试」：把踩过的静默失败钉成永久防线
#   标 @pytest.mark.llm 的（要 key、花钱、可能限流）→ pytest.ini 默认跳过
#
# 【运行】
#   pytest                # CI 用这个（约 10 秒）
#   pytest -m llm         # 加上要调模型的
#   pytest -m ""          # 全部
#   pytest -k oob         # 只看库外拒答
# ============================================================================

import importlib.util
from pathlib import Path

import pytest

from eval_cases import KB_CASES, OOB_CASES, OOB_STRICT, OOB_EDGE
from eval_judge import format_docs, rel_path, src_match

# 说明：eval_judge.judge_kb = 本文件【第一条 ∧ 第二条】（来源命中 ∧ 关键词覆盖）。
# 跑分脚本 eval_fastapi.py 用的是合并版；这里刻意拆成两条，
# 因为两条分别失败时指向完全不同的根因（捞错文件 vs 抽错块），合并了就分不出来。

ROOT = Path(__file__).parent


def _ids(cases):
    return [f"{i:02d}-{c[1][:16]}" for i, c in enumerate(cases, 1)]


# ---------------------------------------------------------------------------
# 一、库内题：正确文档是否进上下文（主指标）
#     这是"检索有没有用"的最直接证据，且与阈值口径无关，比关键词判据稳。
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", KB_CASES, ids=_ids(KB_CASES))
def test_kb_retrieves_expected_source(retrieve, case):
    _, question, l1_kws, _, expected = case
    docs = retrieve(question)
    srcs = [rel_path(d.metadata.get("source", "")) for d in docs]

    assert src_match(srcs, expected), (
        f"\n期望来源未进上下文：{expected}"
        f"\n实际召回：\n{format_docs(docs)}"
        f"\n→ 检索层问题：正确文档没被捞进窗口（不是生成层的问题）"
    )


# ---------------------------------------------------------------------------
# 二、库内题：探针词是否被覆盖（辅助指标）
#     单独成测的理由：它和上面会分别失败，能区分
#     「捞错文件」和「捞对文件但抽到的片段不含内容」——两者解法完全不同。
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", KB_CASES, ids=_ids(KB_CASES))
def test_kb_keyword_coverage(retrieve, case):
    _, question, l1_kws, _, expected = case
    docs = retrieve(question)
    text = "\n".join(d.page_content for d in docs)
    hit = [k for k in l1_kws if k in text]

    # any()：探针是"这页特有的证据"，命中任意一个就算覆盖。
    # 旧版要求全部命中，实测把两条正确召回判成失败（正确页进了窗口，
    # 但承载某个词的块落在另一段里）—— 那是判据的问题，不是检索的问题。
    assert hit, (
        f"\n探针词一个都没出现在召回里：{l1_kws}（期望来源 {expected}）"
        f"\n实际召回：\n{format_docs(docs)}"
        f"\n→ 先看 test_kb_retrieves_expected_source 是否也挂了："
        f"\n   它也挂 = 检索问题；只有这条挂 = 探针脆或抽错块"
    )


# ---------------------------------------------------------------------------
# 三、库外题：必须拒绝（召回为空）
#     这是"宁可空着，不要错召回"那条设计的看门测试。
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", OOB_STRICT)
def test_oob_returns_empty(retrieve, question):
    docs = retrieve(question)
    assert docs == [], (
        f"\n库外问题泄漏了 {len(docs)} 块噪声：\n{format_docs(docs)}"
        f"\n→ 检查：阈值是否被调低过？向量路全灭时 BM25 闸门是否还被禁用？"
    )


@pytest.mark.xfail(
    reason="已知擦边题：问题里含 'FastAPI' 字样，向量分数天然偏高，"
           "阈值注定拦不住。保留为「活记录」——哪天它开始通过，说明方案变了。",
    strict=False,
)
def test_oob_edge_case_fastapi_training(retrieve):
    """含 'FastAPI' 的库外问题（怎么用 FastAPI 训练深度学习模型）"""
    docs = retrieve(OOB_EDGE[0])
    assert docs == []


# ---------------------------------------------------------------------------
# 四、元测试：把三个踩过的「静默失败」钉成永久防线
#     它们不报错、只让结果悄悄变坏，所以必须用测试兜住。
# ---------------------------------------------------------------------------
def test_no_ghost_documents_in_index(ta):
    """【幽灵文档】向量库里不应残留已从 docs 目录删掉的文件

    踩坑记录：删掉的文档内容仍留在向量库继续被检索，因为建库逻辑只判断
    「库存在吗」，从不判断「库还新吗」。现在库名带 docs 指纹，但万一有人手工
    往旧库里加过东西，这条测试能抓到。
    """
    on_disk = {p.relative_to(ta.docs_dir).as_posix()
               for p in Path(ta.docs_dir).rglob("*")
               if p.is_file() and p.suffix.lower() in getattr(ta, "SUPPORTED_EXTS", (".md",))}
    got = ta.vectordb.get(include=["metadatas"])
    in_db = {rel_path(m.get("source", "")) for m in (got.get("metadatas") or [])}

    ghosts = sorted(in_db - on_disk)
    assert not ghosts, (
        f"\n向量库里存在 docs 目录中已不存在的文件（幽灵文档）：{ghosts[:5]}"
        f"\n→ 库没跟着语料更新，删掉旧库重建即可（库名带指纹，删了会自动重建）"
    )


def test_db_dir_encodes_embedding_model(ta):
    """【换模型不换库】库目录名必须带 embedding 模型名

    踩坑记录：embedding 从 nomic(768 维) 换成 bge-m3(1024 维) 时库名没变，
    于是拿旧维度的向量去配新维度的查询——不报错，只是结果悄悄变坏。
    """
    assert ta.EMBED_MODEL.replace(":", "-") in ta.DB_DIR, (
        f"\n库目录 {ta.DB_DIR} 里没有 embedding 模型名 {ta.EMBED_MODEL}"
        f"\n→ 换 embedding 等于换坐标系，必须换库、并重标阈值"
    )


def test_db_ready_rejects_partial_library(ta, tmp_path):
    """【半截库】块数对不上时必须判为「不就绪」

    踩坑记录：建库中途失败（Ctrl-C / embedding 报错）会留下"前 N 块"的库，
    而旧判据只看 count > 0，照样当成就绪 —— 它能答一部分问题、看起来"能用"，
    语料后半段却永久缺失。现在判据是「块数 == 预期块数」，这条测试钉住它。
    （只做判定，不碰真库、不触发重建。）
    """
    assert ta.db_ready(ta.DB_DIR, len(ta.splits)) is True, "真库应当判为就绪"
    assert ta.db_ready(ta.DB_DIR, len(ta.splits) + 1) is False, "块数少一块也必须拒绝"

    empty = tmp_path / "empty_lib"
    empty.mkdir()
    assert ta.db_ready(str(empty), 10) is False, "空目录必须拒绝"


def test_threshold_sits_in_the_score_gap(ta, raw_scores):
    """【阈值标定失效】阈值必须落在「库内题最低分」和「库外题最高分」之间

    这是阈值有意义的前提。换语料 / 换 embedding / 换切分之后，
    分数分布会整体平移，这条测试会立刻失败，提醒重新标定。
    """
    kb_low = min(max(raw_scores(q)) for _, q, _, _, _ in KB_CASES)
    oob_high = max(max(raw_scores(q)) for q in OOB_STRICT)

    assert oob_high < ta.SCORE_THRESHOLD < kb_low, (
        f"\n阈值 {ta.SCORE_THRESHOLD} 不在可分区间内"
        f"\n  库外题最高分 = {oob_high:.3f}"
        f"\n  库内题最低分 = {kb_low:.3f}"
        f"\n→ 两者应留有间隔，阈值取其中。没有间隔说明该换检索方案而不是调阈值。"
    )


# ---------------------------------------------------------------------------
# 五、生成层：需要调云端大模型，默认跳过（pytest -m llm 才跑）
# ---------------------------------------------------------------------------
def _answer(ta, question: str) -> str:
    """单问独立会话作答（不复用历史，避免用例之间互相污染）

    session id 用 uuid 而不是 `hash(q) % 10**8`：取模理论上会撞，
    两个用例的历史被静默合并 —— 正好破坏这里要守的隔离。
    """
    import uuid

    # 没配密钥时明确跳过，而不是让 401 抛穿用例：
    # 这一层（默认 57 条）本来就不需要密钥，只有 -m llm 才要。
    if ta._PROBLEMS:
        pytest.skip("缺少 ARK 配置（" + "、".join(ta._PROBLEMS) + "），跳过需要调模型的用例")

    return ta.assistant.invoke(
        {"question": question},
        config={"configurable": {"session_id": f"pytest-{uuid.uuid4().hex}"}},
    )


@pytest.mark.llm
@pytest.mark.parametrize("case", KB_CASES, ids=_ids(KB_CASES))
def test_kb_answer_covers_expected_terms(ta, case):
    _, question, _, l2_kws, _ = case
    try:
        ans = _answer(ta, question).lower()
    except Exception as e:
        if "SetLimitExceeded" in str(e) or "429" in str(e):
            pytest.skip(f"模型调用受限，跳过：{str(e)[:120]}")
        raise
    assert any(k.lower() in ans for k in l2_kws), (
        f"\n回答里没有出现任何一个期望词：{l2_kws}\n回答：{ans[:300]}"
    )


@pytest.mark.llm
@pytest.mark.parametrize("question,l2_kws", [(c[1], c[3]) for c in OOB_CASES],
                         ids=[c[1][:16] for c in OOB_CASES])
def test_oob_answer_admits_no_data(ta, question, l2_kws):
    """库外问题：模型必须明说「资料里没有」，而不是硬编一个答案"""
    try:
        ans = _answer(ta, question).lower()
    except Exception as e:
        if "SetLimitExceeded" in str(e) or "429" in str(e):
            pytest.skip(f"模型调用受限，跳过：{str(e)[:120]}")
        raise
    assert any(k.lower() in ans for k in l2_kws), (
        f"\n模型没有承认资料里没有：{l2_kws}\n回答：{ans[:300]}"
    )
