# ============================================================================
# conftest.py —— pytest 共享装置
#
# 核心目的：把「加载被测模块」和「检索结果」都做成会话级缓存。
#
# 关于加载成本（2026-10-23 改）：
#   被测模块现在是**惰性索引** —— import 只加载代码（<1 秒），
#   读语料 / 开向量库 / jieba 建 BM25 都推迟到第一次真正检索。
#   所以下面的 session 级 fixture 主要是为了缓存「检索结果」，不再是为了 import。
#   没有 Ollama 的机器上，检索类用例会整场 skip 并说明原因（见文件末尾的 hook）。
# ============================================================================

import importlib.util
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

# ⚠️ 回归套件必须保持「离线、免费、无 key」：强制关掉查询翻译。
#    应用默认是开的（每问多一次模型调用换 +2 道题），但测试不能有这个成本，
#    否则 110 条用例会开始联网、花钱、被限流。
#    必须在加载被测模块之前设置 —— 这个开关是模块级读取的。
os.environ["QUERY_TRANSLATION"] = "0"

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TA_FILE = ROOT / "14-ai_agent_assistant.py"


def _ollama_alive(timeout: float = 2.0) -> bool:
    """探测 Ollama 端口。

    ⚠️ 为什么在 pytest 层探测，而不是让被测模块抛异常：
    embedding 全靠本地 Ollama，没起来时每条检索用例都会以 ConnectionError 变红 ——
    那是**环境没准备好**，不是代码有问题。让人看到 111 条红叉，比看到
    "全部跳过：Ollama 未启动" 要糟得多：后者能立刻指出下一步做什么。
    """
    url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    u = urlparse(url)
    try:
        with socket.create_connection((u.hostname or "localhost", u.port or 11434), timeout):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    """给需要索引的用例打 marker；没有 Ollama 时只跳过这一类。

    ⚠️ 为什么不整场 skip（上一版做法）：那样"CI 绿"是假的 ——
       连「L2 判据不泄题」「库名带模型名」这类纯元测试都不会跑。
       现在按依赖分级：
         · 用到 retrieve / raw_scores 的 → 自动标 needs_index（必然要 embedding）
         · 手工标了的（ghost / db_ready）→ 直接用 ta.vectordb / ta.splits，同样要
         · 其余（判据、命名、import 惰性、check_runtime 错误分支）→ 零依赖，永远能跑
    """
    for item in items:
        if {"retrieve", "raw_scores"} & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.needs_index)

    if _ollama_alive():
        return

    reason = "需要本地 Ollama（embedding）；先跑 ollama serve && ollama pull bge-m3"
    for item in items:
        if "needs_index" in item.keywords:
            item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def ta():
    """整个测试会话只加载一次被测模块"""
    spec = importlib.util.spec_from_file_location("ta", TA_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def retrieve(ta):
    """按问题缓存 final_context_docs —— 同一个问题只检索一次，多个用例复用"""
    cache: dict = {}

    def _retrieve(question: str):
        if question not in cache:
            cache[question] = ta.final_context_docs(question)
        return cache[question]

    return _retrieve


@pytest.fixture(scope="session")
def raw_scores(ta):
    """按问题缓存「原始向量分数」—— 用于阈值的可分性断言"""
    cache: dict = {}

    def _scores(question: str):
        if question not in cache:
            cache[question] = [s for _, s in ta.search(question)]
        return cache[question]

    return _scores
