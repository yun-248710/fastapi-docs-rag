# ============================================================================
# conftest.py —— pytest 共享装置
#
# 核心目的：把「加载被测模块」和「检索结果」都做成会话级缓存。
#
# 为什么必须缓存：
#   导入 14-ai_agent_assistant.py 会触发：读 .env → 加载向量库 → 遍历 3929 个块
#   建 BM25 索引（jieba 分词），本机约 5~10 秒。若每个用例都加载一次，
#   55 个用例就是几分钟。session 级 fixture 只加载一次。
#   检索结果同理：一个用例跑过的问题，后面的用例直接复用，不再调 embedding。
# ============================================================================

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TA_FILE = ROOT / "14-ai_agent_assistant.py"


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
