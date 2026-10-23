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

from eval_cases import (
    KB_CASES, OOB_CASES, OOB_STRICT, OOB_EDGE,
    PARAPHRASE_GROUPS, NOISE_VARIANTS, PARAPHRASE_MIN_RATE, KNOWN_NOISE_FAILURES,
)
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
@pytest.mark.needs_index
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


@pytest.mark.needs_index
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


def test_l2_terms_do_not_leak_from_question():
    """【判据不能抄题】L2 期望词不得出现在题面里

    2026-10-23 发现的问题：45 道库内题里，**42 道（93%）的 L2 期望词就写在问题里**
    ——「怎么给查询参数加长度和正则校验？」的期望词是 `长度/正则/校验`。
    于是 L2 退化成「会不会复述问题」，50/50 这个数测不出任何东西。

    现在 L2 全部换成「答案里才会出现」的 API 符号（`item_id` / `response_model` /
    `exclude_unset`…），这条测试防止有人再把题面词写回去。
    这是一条**结构性防线**：只要它绿着，L2 就不可能靠复述通过。
    """
    leaks = []
    for _, q, _, l2, _ in KB_CASES:
        hit = [k for k in l2 if k.lower() in q.lower()]
        if hit:
            leaks.append((q, hit))

    assert not leaks, (
        "\n以下题目的 L2 期望词出现在题面里（判据泄题，指标会退化成复述检测）：\n"
        + "\n".join(f"  · {q}  →  {h}" for q, h in leaks)
        + "\n→ 换成「答案里才会出现」的词（API 符号 / 参数名 / 结论），别再抄题面"
    )
    # 顺带禁止空判据：空列表会让 any() 永远为 False，题目变成永远 FAIL
    empty = [q for _, q, _, l2, _ in KB_CASES if not l2]
    assert not empty, f"这些题没有 L2 期望词：{empty}"


def test_regression_suite_forces_translation_off(ta):
    """【回归套件必须保持离线免费】conftest 必须把查询翻译压成关闭

    应用默认是**开**的（每问多一次模型调用，换 L1 45→47）。但测试不能有这个成本：
    110 条用例会从「本地 embedding、零成本、无 key 可跑」变成「要联网、要花钱、会限流」。
    所以 conftest.py 在加载模块前注入 QUERY_TRANSLATION=0，这条测试钉住它。

    ⚠️ 这里失败说明 conftest 的注入被绕过了（比如有人先 import 了模块）——
    先查 import 顺序，不要改断言。
    """
    assert ta.QUERY_TRANSLATION is False, (
        "回归套件里 QUERY_TRANSLATION 被打开了 —— 会开始调模型、联网、花钱。"
        "看 conftest.py 顶部那行 os.environ 是否还在、是否在 import 之前。"
    )
    q = "FastAPI 怎么定义路径参数？"
    assert ta._translate_for_bm25(q) == q, "翻译关闭时应当原样返回"


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


def test_import_does_not_build_index():
    """【import 副作用】导入模块不应读语料 / 开向量库 / 建 BM25

    踩坑记录：这些事原来写在模块级，于是 `import` 一次要 5~10 秒 ——
    跑分脚本、trace 渲染、IDE 补全、pytest 收集阶段全都要先付这笔钱，
    而它们大多数根本不检索。

    ⚠️ 必须用子进程测：本会话里 ta 早被别的用例触发过，
       进程内的 _INDEX["ready"] 一定是 True，测不出 import 时刻的状态。
    """
    import json
    import subprocess
    import sys

    ta_file = ROOT / "14-ai_agent_assistant.py"
    code = (
        "import importlib.util, json, time\n"
        f"t = time.time()\n"
        f"spec = importlib.util.spec_from_file_location('ta', r'{ta_file}')\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "print(json.dumps({'ready': m._INDEX['ready'], 'sec': round(time.time() - t, 2)}))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=180,
                       env={**__import__("os").environ, "QUERY_TRANSLATION": "0"})
    assert r.returncode == 0, f"子进程 import 失败：\n{r.stderr[-800:]}"

    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["ready"] is False, (
        "\nimport 阶段就把索引建好了 —— 惰性化失效了。"
        "\n→ 检查是否有模块级代码又调用了 _ensure_index() / build_splits() / Chroma(...)"
    )

    # 光看 ready 还不够：如果有人把 build_splits() 留在模块级、只是不打开库，
    # ready 仍是 False，但语料已经被读完了。用副作用痕迹兜住。
    noise = r.stdout + r.stderr
    for marker in ("已加载持久化知识库", "正在构建知识库向量库", "已给", "Building prefix dict"):
        assert marker not in noise, (
            f"\nimport 阶段出现了「{marker}」—— 说明读语料 / 建 BM25 仍在模块级执行。"
            f"\n完整输出：\n{noise[-800:]}"
        )

    # 耗时只做宽松护栏（第三方包本身的导入就要几秒，不归我们管），
    # 目的是拦住"把建库写回模块级"这种回退（那会让 import 变成 20 秒以上）。
    assert info["sec"] < 15, f"\nimport 耗时 {info['sec']}s，模块级仍有重活"


def test_check_runtime_reports_missing_docs(ta, monkeypatch, tmp_path):
    """【前置检查】语料目录不存在时，必须给出能照做的提示，而不是抛原始异常

    对外部方案的修正：原写法用 SystemExit，在 pytest 里会中断整个 session；
    这里 check_runtime 只**返回**问题清单，由调用方决定退出还是跳过。
    """
    monkeypatch.setattr(ta, "docs_dir", str(tmp_path / "no_such_dir"))
    problems = ta.check_runtime()
    assert any("找不到语料目录" in p for p in problems), (
        f"\n语料缺失没有被报出来：{problems}"
    )
    # 空目录也要报：目录在、但一个可索引文件都没有，会建出空库
    (tmp_path / "empty_docs").mkdir()
    monkeypatch.setattr(ta, "docs_dir", str(tmp_path / "empty_docs"))
    problems = ta.check_runtime()
    assert any("没有可索引的文件" in p for p in problems), (
        f"\n空语料目录没有被报出来：{problems}"
    )


def test_readme_claim_about_comment_ratio_holds(ta):
    """【文档与代码一致】README 说「注释+文档字符串约 45%」，代码必须真的如此

    起因：外部审查三次指出 README 里的代码行数过期（860 → 1080 → 1113）。
    根因是这类数字**改代码就过期**，所以 README 已改成模糊表述；
    但既然还留了一个比例数字，就得有人守着 —— 这条就是那道防线。

    如果将来有人重构时把注释删掉一半，这条会失败，
    提醒同步 README 的说法（而不是让 README 悄悄变成假话）。
    """
    src = Path(ta.__file__).read_text(encoding="utf-8").splitlines()
    total = len(src)
    comment = sum(1 for l in src if l.strip().startswith("#"))

    in_doc = False
    doc = 0
    for l in src:
        s = l.strip()
        if s.startswith(('"""', 'r"""')):
            in_doc = not in_doc
            continue
        if in_doc and s and not s.startswith("#"):
            doc += 1

    ratio = (comment + doc) / total
    assert 0.35 <= ratio <= 0.55, (
        f"\n注释与文档字符串占比 {ratio:.0%}，已不在 README 承诺的 35%~55% 区间"
        f"\n  # 注释 {comment} 行 / docstring {doc} 行 / 总行 {total}"
        f"\n→ 同步修改 README 项目结构里那一句，别让文档变成假话"
    )


# ---------------------------------------------------------------------------
# 四·五、行为评测：一致性与鲁棒性
#   前面的用例测的是"这句话能不能查到"，测不到：
#     ① 同一问题换个说法，结果会不会漂（一致性）
#     ② 用户打错字、中英混输、乱加标点还稳不稳（鲁棒性）
#   这两类才是真实用户在做的。阈值按 2026-10-22 的实测基线留了一档余量：
#   一致性 11/14 = 79%，鲁棒性 4/5 与基线一致。
# ---------------------------------------------------------------------------
def _src_hit(retrieve, question: str, expected) -> bool:
    docs = retrieve(question)
    return src_match([rel_path(d.metadata.get("source", "")) for d in docs], expected)


@pytest.mark.parametrize("group", PARAPHRASE_GROUPS, ids=[g[0] for g in PARAPHRASE_GROUPS])
def test_paraphrase_consistency(retrieve, group):
    """同一问题的多种问法，命中率不应显著下降

    只断命中率不断"每一句都必须中"：换问法天然有波动，要求 100% 会把
    正常抖动变成红叉。阈值 60% 是防回归用的，不是及格线。
    """
    name, expected, questions = group
    hits = [(q, _src_hit(retrieve, q, expected)) for q in questions]
    rate = sum(1 for _, ok in hits if ok) / len(hits)

    misses = [q for q, ok in hits if not ok]
    assert rate >= PARAPHRASE_MIN_RATE, (
        f"\n【{name}】换问法命中率 {rate:.0%} < {PARAPHRASE_MIN_RATE:.0%}"
        f"\n没命中的问法：{misses}"
        f"\n→ 说明检索对措辞敏感：向量没抓住语义，或 BM25 被某些词带偏"
    )


def _noise_params():
    params = []
    for item in NOISE_VARIANTS:
        name, expected, base, noisy, kind = item
        marks = []
        if noisy in KNOWN_NOISE_FAILURES:
            marks.append(pytest.mark.xfail(
                reason="已知：单字错别字（数→树）会打断语义检索，留作活记录",
                strict=False,
            ))
        params.append(pytest.param(item, marks=marks, id=f"{name}-{kind}"))
    return params


@pytest.mark.parametrize("item", _noise_params())
def test_robustness_no_worse_than_base(retrieve, item):
    """加了噪声的问法，结果不能比基线问法差

    基线本来就查不到的问题会 skip：那种情况测的是已知的检索缺口，
    不是噪声的影响，混在一起会把两种原因算到一笔账上。
    """
    name, expected, base, noisy, kind = item
    if not _src_hit(retrieve, base, expected):
        pytest.skip(f"【{name}】基线问法本身未命中，无法把差异归因于噪声")

    docs = retrieve(noisy)
    assert src_match([rel_path(d.metadata.get("source", "")) for d in docs], expected), (
        f"\n【{name}】噪声（{kind}）让检索变差了"
        f"\n基线：{base}  ✓"
        f"\n噪声：{noisy}  ✗"
        f"\n实际召回：\n{format_docs(docs)}"
        f"\n→ 用户输入不会总是干净的，这类退化要靠查询纠错或更强的语义模型解决"
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
