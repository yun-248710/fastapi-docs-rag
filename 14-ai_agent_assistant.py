# ============================================================================
# 14 学习助手智能体 · API 版（RAG + 多轮记忆 + 引用溯源）
#
# 与 12 的差别：生成模型从「本地 Ollama」换成「火山方舟」（OpenAI 兼容协议），
#               并在检索侧加了相关性闸门 + 混合检索，修掉答非所问 / 复读问题。
#
# 组合了前面所有 demo 的能力：
#   02 模型 API（ChatOpenAI，走 OpenAI 兼容协议）
#   09 多轮对话记忆（RunnableWithMessageHistory）
#   10 向量数据库（Chroma + OllamaEmbeddings，embedding 仍在本机）
#   11 本地知识库 RAG（检索增强生成）
#
# 运行前确认：
#   1. Ollama 已启动（embedding 还用它，别关）
#   2. 已拉取向量模型：ollama pull bge-m3      ← 多语言，中文问英文语料必须用它
#   3. 知识库文档放在 ./docs_fastapi 下（md/docx/pdf/txt 均可）
#   4. .env 里配好三样：ARK_API_KEY / ARK_BASE_URL / ARK_MODEL
#      ⚠️ 模型 ID 只在 .env 里配一处，代码不设默认值（见下面第 0 节的注释）
#
# 用法：
#   python 14-ai_agent_assistant.py           # 交互模式
#   python 14-ai_agent_assistant.py --test    # 单问测试模式
# ============================================================================

import hashlib   # 用于计算文档哈希
import json
import os
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from langchain_core.documents import Document
from dotenv import load_dotenv

# 显式指定「和本脚本同目录」的 .env，不受运行时工作目录影响
load_dotenv(Path(__file__).parent / ".env")

from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- 0. API 配置（全部从 .env 读，代码里不出现明文密钥） ----------------
API_KEY = os.environ.get("ARK_API_KEY")

# ⚠️ 所有配置项刻意【不设默认值】—— 避免「双重配置源」。
# 踩过的坑：这里曾写成 os.environ.get("ARK_MODEL", "某个模型ID")，而 .env 里也有一行
# ARK_MODEL。load_dotenv() 会把它读进 os.environ，于是【默认值成了死代码】：
# 改了这行不生效、也不报错，只是静默用 .env 的值。
# 规则：同一个配置只留一处。缺失要炸，比"悄悄用了另一个值"便宜得多。
# ⚠️ 但**炸的时机**很关键 —— 见 _require_config 的说明：不在 import，而在真要调模型时。
#
# 📌 BASE_URL 原来没有守这条规则（`os.environ.get("ARK_BASE_URL", "https://ark.cn-...")`）：
#    .env 里变量名打错（比如写成 ARK_BASE_URI）就会静默回落到北京区默认值，
#    报出来的是鉴权/模型错误，而不是"你的配置没生效"。2026-09-23 审查后改成必填。
#
# 占位符判断串必须和 .env.example 里的模板**保持一致**。
# ⚠️ 踩过的坑：这里曾只判 `startswith("ark-在这里")`，而 .env.example 里写的是
# `ark-your-key-here` —— 于是【照抄模板的用户能通过这道检查】，一路跑到第一次提问
# 才拿到 401。检查没拦住它要拦的那件事。现在两边对齐。
_KEY_PLACEHOLDERS = ("在这里", "your-key", "placeholder")


def _config_problems() -> list:
    """返回缺失/无效的配置项（空列表 = 配置齐全）。"""
    problems = []
    if not os.environ.get("ARK_MODEL"):
        problems.append("ARK_MODEL")
    if not os.environ.get("ARK_BASE_URL"):
        problems.append("ARK_BASE_URL")
    key = os.environ.get("ARK_API_KEY") or ""
    if not key or any(p in key for p in _KEY_PLACEHOLDERS):
        problems.append("ARK_API_KEY（缺失或仍是占位符）")
    return problems


_PROBLEMS = _config_problems()


def _require_config() -> None:
    """在真正要调生成模型前调用（ask / 交互入口），一次列全缺什么。

    ⚠️ 为什么不在这里的模块级直接 raise（2026-10-22 改）：
    这段是模块级代码，一 raise，**没有密钥的人连 pytest 都跑不了** —— 57 条检索测试
    会全部在 import 阶段挂掉，可它们根本不调大模型、也根本不需要密钥。
    而 README 写着"默认不调大模型、可直接进 CI"，等于给访客挖了个坑：
    clone 下来第一条命令就报错，看起来像项目跑不起来。

    → 所以：import 只读取配置，缺了就用占位值把客户端构造出来；
      真正调模型的地方再当场炸，且一次列全所有缺失项。
    """
    if _PROBLEMS:
        raise SystemExit(
            "缺少配置：" + "、".join(_PROBLEMS) + "\n"
            "请在 ic_demo/.env 里补齐（模板见 .env.example）：\n"
            "  ARK_API_KEY=ark-...\n"
            "  ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3\n"
            "  ARK_MODEL=<控制台「开通管理」里的 Model ID>\n"
            "（只跑检索回归不需要密钥：pytest 默认不调生成模型）"
        )


MODEL_ID = os.environ.get("ARK_MODEL") or "missing-model"
BASE_URL = os.environ.get("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
API_KEY = os.environ.get("ARK_API_KEY")
if not API_KEY or any(p in API_KEY for p in _KEY_PLACEHOLDERS):
    API_KEY = "missing-ark-key"      # 占位：让客户端能构造，调用前必被 _require_config 拦下


# ---------------- 1. 知识库构建（文档加载 -> 分割 -> 向量化入库） ----------------
ROOT = Path(__file__).resolve().parent    # 脚本所在目录：所有路径都以它为基准

# ⚠️ 路径全部基于 ROOT，不用 "./xxx"。
# .env 早就用 Path(__file__).parent 免疫了工作目录，但库目录和语料目录还是相对的 ——
# 换个目录起脚本就会"找不到语料"，或者更糟：在别处悄悄另建一个库。
# 2026-09-23 审查补上。
#
# 库目录名带上 docs 的内容指纹：文档增删改 → 指纹变 → 自动建新库，
# 不会读到过期数据（幽灵文档那个坑）。⚠️ 但指纹本身有两个已知边界，见 docs_fingerprint。
DB_DIR_BASE = str(ROOT / "chroma_db_ta")
docs_dir = str(ROOT / "docs_fastapi")       # ← 换语料就改这一行
EMBED_MODEL = "bge-m3"                      # ← 换 embedding 模型就改这一行（多语言）

# 支持的后缀：建库、体检、测试都读这一处。
# （原来 loader_map 藏在建库的 if 块里，测试只好自己写 rglob("*.md")——
#   语料一换成 PDF，那条"幽灵文档"元测试就会把所有 PDF 误判成幽灵。）
SUPPORTED_EXTS = (".pdf", ".docx", ".txt", ".md")

# 建库时排除的文件（只匹配文件名，不看目录）。
#
# ⚠️ release-notes.md 必须排除：它是 FastAPI 的更新日志，几万字、按 PR 逐条罗列，
# **几乎每个主题都被提一句，但没有一个主题说全**。后果是双杀：
#   ① 块数上统治整个库 —— 实测 3929 块里它占 2010 块（51.2%）；
#   ② 分数上统治检索 —— 每个主题都沾边，BM25 上轻松拿 13~20 分
#      （实测 13.41 / 14.14 / 16.19 / 18.99 / 20.50），而真正的教程页向量分只有 0.5x。
# 于是正常的教程页被它挤出窗口：`怎么读取请求头参数` 那题前 5 条全是它。
#
# 这不是"检索算法不好"，是**语料被一个文件统治了** —— 数据层的问题，调阈值/k 都没用。
EXCLUDE_DOCS = {"release-notes.md"}

# 块内容格式版本：**只要改了「每块里放什么」就必须 +1**，并让它进指纹。
# 原因：指纹只看文件名 + mtime，不看内容，所以改了切块/拼装方式而不改版本号的话，
# 旧库会被当成新库直接复用，改动静默失效 —— 又一个静默失败。
#   v1 = 纯正文
#   v2 = 块首补页面一级标题（contextual header）
CHUNK_FORMAT_VERSION = 2



def _index_files(directory: str):
    """列出**真正会进库**的文件（相对路径，统一 / 分隔）。

    指纹和建库共用这一份口径：只算 SUPPORTED_EXTS、且不在 EXCLUDE_DOCS 里的文件。
    原来指纹用 os.walk 遍历全部文件（403 个里 248 张图），动一张 png 就触发全库重建。
    """
    out = []
    for root, _, files in os.walk(directory):
        for fn in sorted(files):
            if fn in EXCLUDE_DOCS:          # 不进库的文件不参与指纹
                continue
            if os.path.splitext(fn)[1].lower() not in SUPPORTED_EXTS:
                continue
            out.append(os.path.relpath(os.path.join(root, fn), directory).replace("\\", "/"))
    return sorted(out)


def docs_fingerprint(directory: str) -> str:
    """用「相对路径 + 文件内容哈希 + 排除清单」算语料指纹。

    📌 只算**真正进库**的文件，并把 EXCLUDE_DOCS 本身也算进去：
      · 不进库的文件（changelog、png）改一下不该触发重建；
      · 但**改了排除清单 = 换了一套索引内容**，必须换库。

    ⚠️ 为什么不再用 mtime（2026-10-23 改，原方案见 git 历史）：
      ① **换机器必失效**：解包/拷贝会改 mtime，指纹一变库名就对不上，
         随包带的向量库永远用不上 —— 每次解压都要白等 1~2 分钟重建。
      ② **保留 mtime 时反而危险**：`cp -p` / `rsync -t` / 备份还原会保留 mtime，
         内容改了指纹却不变 → 旧库继续用，幽灵文档复活，新内容永远进不了库。
      ③ 用**相对路径**而不是 basename：语料里有 14 组同名文件（index.md 就有 14 个），
         跨目录互换内容时 basename 看不出来。

    代价：要读一遍全部语料（154 篇 md，实测 29 ms；mtime 版 12 ms），
    相对建库的 1.8 分钟可忽略。
    """
    h = hashlib.md5()
    for rel in _index_files(directory):
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")                       # 分隔符，防止 ab+c 与 a+bc 撞
        with open(os.path.join(directory, rel), "rb") as fh:
            h.update(hashlib.md5(fh.read()).digest())
    h.update(("|".join(sorted(EXCLUDE_DOCS))).encode("utf-8"))
    h.update(f"chunk-v{CHUNK_FORMAT_VERSION}".encode("utf-8"))
    return h.hexdigest()[:8]


# 库名 = 基础名 + embedding 模型 + docs 指纹。
# ⚠️ 必须带上 embedding 模型名：不同的 embedding 是**不同的坐标系**（nomic 768 维、
# bge-m3 1024 维），换模型后若不换库，db_ready() 会看到旧库「有向量」而跳过重建，
# 于是拿旧模型的向量去配新模型的 query —— 又是一种不报错的静默失败。
#
# 📌 切分参数（chunk_size / overlap / separators）**故意不进库名**：
#    它们由「预期块数」兜住了 —— 改了它们，build_splits() 的块数就变，
#    db_ready() 立刻判定库不完整并重建。少一个要手工同步的命名维度。
DB_DIR = f"{DB_DIR_BASE}_{EMBED_MODEL.replace(':', '-')}_{docs_fingerprint(docs_dir)}"


def db_dir_for_chroma(path: str = DB_DIR) -> str:
    """把库目录转成「相对当前工作目录」的形式再交给 chromadb。

    ⚠️ 踩过的坑（2026-09-23 本机实测，chromadb 1.5.9）：它的 Rust 层读不了
    **含非 ASCII 字符的绝对路径**。同一个库（本机用户名是中文，路径必然带中文），
    三种传法结果完全不同：

        "C:\\Users\\黄计云\\Desktop\\ic_demo\\chroma_db_x"   → InternalError:
                                                              Error loading hnsw index
        "./chroma_db_x"（cwd 就在中文目录下）                 → OK
        "ic_demo/chroma_db_x"（跨一层目录的相对路径）          → OK

    所以问题出在"绝对路径"这个形式上，不在字符本身 —— 相对路径（即使解析后含中文）
    一切正常。这里统一转成相对形式；只有库和 cwd 跨盘符时 relpath 无解，
    才退回绝对路径并明确警告。
    """
    try:
        return os.path.relpath(path)
    except ValueError:
        print(
            f"⚠️ 库目录与当前工作目录不在同一个盘符，只能把绝对路径交给 chromadb。\n"
            f"   若随后报 'Error loading hnsw index'，请先 cd 到 {ROOT} 再运行本脚本。"
        )
        return path


# Ollama 地址只留一处：check_runtime 探测它、make_embedding 连它。
# 换成远端 Ollama（另一台机器跑 embedding）时改这一个变量即可。
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def make_embedding() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def _count_chunks(path: str) -> int:
    """数库里有多少个块；任何失败都返回 -1（视为不可用）。

    ⚠️ 优先直接只读 chroma.sqlite3，而不是 `Chroma(...)._collection.count()`：
    chromadb 会按路径缓存 client，句柄留在进程里，之后就没法 rename 掉这个目录了
    （Windows 会报"另一个程序正在使用此文件"）。我们紧接着要用 rename 挪走半截库，
    所以这里必须保证连接关干净。schema 真变了再退回官方 API。
    """
    db_file = os.path.join(path, "chroma.sqlite3")
    if os.path.exists(db_file):
        import sqlite3

        try:
            con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            try:
                return int(con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            finally:
                con.close()
        except Exception:
            pass

    try:
        vdb = Chroma(persist_directory=db_dir_for_chroma(path), embedding_function=make_embedding())
        return int(vdb._collection.count())
    except Exception:
        return -1


def db_ready(path: str, expected: int) -> bool:
    """库目录存在、且块数【正好等于预期】，才算就绪。

    ⚠️ 三代判据的演进，前两代都被真实事故打回过：

      第一代「目录非空」
        → 建库中途失败留下一个空壳库，下次启动照样跳过重建，于是永远查不到东西。

      第二代「collection.count() > 0」
        → 空壳库挡住了，但**半截库**（0 < count < 全量）照样放行：它能答一部分问题，
          看起来"能用"，语料后半段却永久缺失 —— 比空壳库更难发现。

      第三代「count == expected」（现在）
        → expected 由语料 + 切分参数确定性推导（build_splits），于是
          半截库 / 换语料 / 改切分参数，三种情况都在这里被拦下并触发重建。
    """
    if not os.path.isdir(path) or not os.listdir(path):
        return False
    n = _count_chunks(path)
    if n != expected:
        print(f"⚠️ 库内 {n} 块 ≠ 预期 {expected} 块"
              f"（建库中断 / 语料或切分参数变过）→ 判定为不可用，重建")
        return False
    return True


def _retire(path: str, tag: str) -> None:
    """把旧目录**挪到一边**，而不是删掉。

    ⚠️ 刻意不用删除（os.remove / shutil.rmtree）：
    ① 某些沙箱环境会劫持删除调用并直接抛错（safe-delete shim），rename 永远安全；
    ② 保留现场便于事后排查"上一版库到底装了多少块"。
    挪出来的 `xxx__stale_<时间戳>` 目录确认无误后可手动删除。
    """
    if not os.path.isdir(path):
        return
    target = f"{path}__{tag}_{int(time.time())}"
    try:
        os.rename(path, target)
        print(f"  ⚠️ 旧库已挪到 {os.path.basename(target)}（确认无误后可手动删除）")
    except OSError as e:
        raise SystemExit(
            f"无法挪走旧库 {path}：{e}\n"
            f"通常是另一个进程还开着它（比如另一个终端正在跑本脚本）。\n"
            f"关掉后重试，或手动删除该目录再运行。"
        )


def _loader_map():
    """后缀 → loader 映射。只在真要建库时才 import langchain_community。

    ⚠️ DirectoryLoader 不支持按扩展名映射 loader（默认走 unstructured），
    这里手动按后缀选择，稳定且不依赖 unstructured。
    """
    from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader, TextLoader

    return {
        ".pdf": PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": lambda p: TextLoader(p, encoding="utf-8"),
        ".md": lambda p: TextLoader(p, encoding="utf-8"),
    }


def _title_of(text: str) -> str:
    """取文档的一级标题：`# Path Parameters { #path-parameters }` → `Path Parameters`。

    ⚠️ 为什么要它：切分后每块只剩正文片段，**页面标题丢了**。
    于是问「怎么定义路径参数」时，`path-params.md` 的正文块和别的正文块在向量空间里
    几乎没区别 —— 实测只能排到第 11~21 名，k=4 根本够不着。
    把标题贴回块首，相当于给每块补一句"我在讲什么"（contextual header）。
    """
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("# "):
            return s[2:].split("{")[0].strip()      # 去掉 { #anchor }
        if not s.startswith(("<!--", "---", "```")):
            break                                    # 第一个非空非标题行就停
    return ""


def build_splits() -> list:
    """加载 docs_dir 下所有受支持文档并切分。**纯文本处理，不调 embedding**（秒级）。

    这个函数是「预期块数」的唯一来源，也是建库的唯一入口：
      · db_ready() 用 len(它) 当尺子，判断库完不完整（防半截库）
      · 建库分支复用它做向量化，不重复加载
    因为它同时受【语料】和【切分参数】影响，所以改 chunk_size / overlap / separators
    也会让预期块数变化 → 自动触发重建。这就是切分参数不必进库名的原因。
    """
    if not os.path.isdir(docs_dir):
        raise SystemExit(f"找不到资料目录：{docs_dir}，请先建好并放入文档")

    loader_map = _loader_map()
    docs = []
    skipped = []
    # ⚠️ 两处 sorted 不是洁癖：不排序时 os.walk 的顺序随文件系统而变，入库顺序跟着变，
    # BM25 平局又靠下标打破 —— 两台机器跑同一份语料能差出 1 分，正好让人怀疑自己的改动。
    for root, _, files in sorted(os.walk(docs_dir)):
        for fn in sorted(files):
            if fn in EXCLUDE_DOCS:          # 变更日志这一类：见 EXCLUDE_DOCS 的注释
                skipped.append(fn)
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in loader_map:
                loaded = loader_map[ext](os.path.join(root, fn)).load()
                docs.extend(loaded)

    if skipped:
        print(f"已排除 {len(skipped)} 个文件（EXCLUDE_DOCS）：{', '.join(sorted(set(skipped)))}")

    # 静默失败防护：目录存在、但一个文件都没认出来（比如 loader_map 漏了后缀），
    # 会建出一个空库，之后所有问题都答「资料里没有」。当场炸掉比事后排查便宜。
    if not docs:
        raise SystemExit(
            f"目录 {docs_dir} 下没加载到任何文档，检查 _loader_map() 是否覆盖了这些后缀"
            f"（当前支持：{SUPPORTED_EXTS}）"
        )

    # 分段：按自然段落切，重叠 50 字符防止知识点被切断
    # 📌 已知待办：separators 里的中文标点在英文语料里永不命中（实际只靠 \n\n / \n / 空格），
    #    且 chunk_size=500 会把密集的代码块从中间切断。改它会让块数变化 → 全库重建 +
    #    历史评测分数作废，所以归到"调优"那一轮，不在本次修复里动。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " "],
    )

    if CHUNK_FORMAT_VERSION < 2:
        return splitter.split_documents(docs)

    # v2：逐篇切、给每块块首补页面标题。
    # 逐篇切而不是一次性切全部，是因为要保留"块属于哪篇文档"的对应关系
    # （一次性切会丢掉这层对应；逐篇切的结果与一次性切一致）。
    splits = []
    titled = 0
    for d in docs:
        title = _title_of(d.page_content)
        for ch in splitter.split_documents([d]):
            if title:
                ch.page_content = f"# {title}\n\n{ch.page_content}"
                titled += 1
            splits.append(ch)
    print(f"已给 {titled}/{len(splits)} 块补上页面标题")
    return splits


# ---- 索引不在 import 时构建 ----
# 这里原本是模块级代码：读 154 篇语料 → 切分 → 打开/构建向量库 → jieba 分词建 BM25，
# import 一次 5~10 秒。于是任何"只想拿一个常量"的场景（跑分脚本、trace 渲染、
# IDE 补全、pytest 收集阶段）都要先付这笔钱。
#
# 真正需要索引的入口只有三个：search() / bm25_search() / ask()。
# 现在改成惰性构建，实现见下面 BM25 段的 _ensure_index()。
#
# 对外接口保持不变：`ta.vectordb` / `ta.splits` 仍然可用，
# 由模块级 __getattr__（PEP 562）转发到 _ensure_index() —— 测试一行都不用改。

# ---- BM25 关键词索引（与向量库同源），治跨语言 / 专有名词 miss ----
import re
import jieba
from rank_bm25 import BM25Okapi

# 分词必须过滤标点/停用词：实测一个全角问号「？」的 IDF 高达 4.89，
# 会让完全不相关的块拿到 4.2 分凭空混进上下文（第 3 轮踩的坑）。
#
# 📌 英文虚词已处理（2026-10-22）：翻译出来的查询会带 the/of/how 这类词，
#    现在和 .lower() 一起归一到同一个词表。改分词**不需要重建向量库**
#    —— BM25 索引是每次启动在内存里重建的（见下面的 `_bm25 = ...`）。
_STOPWORDS = {
    "的", "了", "是", "有", "在", "和", "就", "都", "也", "吗", "呢", "啊",
    "什么", "怎么", "怎么样", "为什么", "如何", "哪些", "这个", "那个",
    "我", "你", "他", "她", "它", "这", "那", "一个", "可以", "需要",
    # 英文虚词：语料与查询都过这里，不滤掉会稀释真实关键词的权重
    "the", "and", "for", "with", "you", "your", "that", "this", "are", "can",
    "how", "what", "which", "when", "where", "why", "use", "using", "from",
    "will", "have", "has", "been", "not", "but", "all", "any", "its", "into",
    "our", "their", "they", "them", "there", "these", "those", "about", "is",
    "of", "to", "in", "on", "or", "it", "be", "do", "does", "as", "at", "by",
}
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)

# BM25 补召取几条。**三处共用这一个常量**（原来 3 写死在三个地方：查分处漏改
# 只会让它打印 "BM25 0.00" 错标，不会报错）。
BM25_K = 3

# 查询翻译开关（应用默认开，测试强制关）。
#   开 = 中文问题先翻成英文关键词，**只喂给 BM25**，向量路仍用原句。
#   为什么只给 BM25：向量路（bge-m3）跨语言本来就强，换英文 query 反而会改变它的
#   排序；而 BM25 是纯字面匹配，中文 token 打英文语料命中率接近 0
#   （实测净贡献只有 1 题）。所以缺的那条腿是 BM25，要补就补它。
#   代价：每问多一次模型调用（约 +0.5~1.5s），实测 L1 45/50 → 47/50。
#
#   ⚠️ 默认值的分叉（2026-10-22）：一个环境变量服务两个互相冲突的消费者 ——
#      · 交互应用：慢一点换准确率，值 → 默认开
#      · 回归测试：必须离线免费、一次调用都不能有 → conftest.py 强制注入 0
#   所以默认值在这边设 1，测试侧在 conftest 里覆盖。改默认值前先看那条测试。
QUERY_TRANSLATION = os.environ.get("QUERY_TRANSLATION", "1") == "1"

_TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You translate Chinese technical questions into English search keywords for the "
     "FastAPI documentation. Output ONLY English keywords or a short English phrase, "
     "max 15 words. No explanation, no Chinese, no quotes."),
    ("human", "{question}"),
])
_translator = None
_trans_cache: dict = {}


def _translate_for_bm25(question: str) -> str:
    """中文问题 → 英文关键词（只给 BM25 用；失败就退回原句）

    失败必须退回而不是抛：翻译是"加分项"，不该让整次检索挂掉。
    """
    if not QUERY_TRANSLATION or _PROBLEMS:
        return question
    if question in _trans_cache:
        return _trans_cache[question]

    global _translator
    try:
        if _translator is None:
            # ⚠️ 翻译必须用 temperature=0：复用过对话模型（0.3）时，同一句问法两次
            # 翻译结果不同（"query parameters" vs "query params declaration"），
            # 检索结果跟着漂 —— 那就不再是"可复现的检索"，A/B 也没法做。
            _translator = _TRANSLATE_PROMPT | llm.bind(temperature=0) | StrOutputParser()
        raw = _translator.invoke({"question": question})
        # 防御：模型偶尔夹带中文解释。只取英文词，避免把解释也拿去打分
        words = re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,}", raw)[:15]
        result = " ".join(words) or question
    except Exception:
        result = question
    _trans_cache[question] = result
    return result


def _tokenize(text: str):
    """分词并过滤：丢掉标点、停用词、单字（它们只会制造噪声命中）

    ⚠️ 统一转小写（2026-10-22）：语料里是 `Path` / `Header` / `Cookie`，
    而翻译出来的查询往往是 `header` / `cookie` —— 不归一化等于两边在不同字典里查。
    BM25 索引每次启动在内存里重建，所以改分词**不需要重建向量库**，两边同时生效。
    """
    return [
        w.lower() for w in jieba.cut(text)
        if len(w.strip()) >= 2 and w.strip().lower() not in _STOPWORDS and not _PUNCT_ONLY.match(w)
    ]


_INDEX: dict = {"ready": False, "vectordb": None, "splits": None,
                "bm25": None, "bm25_docs": None}
_INDEX_LOCK = threading.Lock()


def check_runtime(require_llm: bool = False) -> list:
    """体检：返回问题清单（空 = 就绪）。**只报告，不抛异常** —— 谁来处理由调用方决定。

    ⚠️ 为什么不用 SystemExit（外部方案的写法）：
      在 pytest 里它是【中断整个 session】的信号，不是"这条用例失败"。
      库代码不替调用方做决定：脚本入口该退出就自己 sys.exit，测试该 skip 就自己 skip。

    检查顺序按"用户第一眼能看到什么错"排：语料缺失 → Ollama 没起/模型没 pull
    → ARK 配置不全（仅 require_llm）。这三类原始报错分别是 FileNotFoundError、
    连接被拒、401 —— 全都看不出真正原因，所以值得提前探一遍。
    """
    problems = []
    if not os.path.isdir(docs_dir):
        problems.append(f"找不到语料目录：{docs_dir}")
    elif not _index_files(docs_dir):
        problems.append(f"{docs_dir} 下没有可索引的文件（支持：{', '.join(SUPPORTED_EXTS)}）")

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as r:
            names = {m.get("name", "") for m in json.loads(r.read().decode()).get("models", [])}
        # Ollama 返回的 name 带 tag（bge-m3:latest），而配置里通常不写 tag
        if not any(n == EMBED_MODEL or n.split(":")[0] == EMBED_MODEL for n in names):
            problems.append(f"Ollama 里没有 {EMBED_MODEL} —— 先跑：ollama pull {EMBED_MODEL}")
    except Exception as e:
        problems.append(
            f"连不上 Ollama（{OLLAMA_BASE_URL}，{type(e).__name__}）—— 先启动 ollama serve"
        )

    if require_llm:
        problems.extend(_PROBLEMS)
    return problems


def _ensure_index():
    """索引的唯一入口：语料 → 切分 → 向量库 → BM25。幂等，只有第一次真正干活。

    📌 为什么连 build_splits() 也要懒加载，而不是只懒加载向量库：
      它是"预期块数"的唯一来源，而预期块数要读完全部语料才算得出来 —— 读语料
      本身（154 篇 ≈ 1.5 秒）就是建库成本的大头，留在模块级等于白改。

    ⚠️ 已知边界：这是**进程内单例**。多 worker 部署（uvicorn --workers 4）时每个
       进程各建一份，内存 ×N。当前规模（1919 块）可接受；真要上多 worker，
       应该把索引挪到独立服务，或换成支持共享映射的向量库。
    """
    if _INDEX["ready"]:
        return _INDEX["vectordb"], _INDEX["splits"]

    with _INDEX_LOCK:
        if _INDEX["ready"]:                     # 双检：等锁期间别的线程可能已经建好
            return _INDEX["vectordb"], _INDEX["splits"]

        problems = check_runtime()
        if problems:
            raise RuntimeError(
                "知识库无法就绪：\n  · " + "\n  · ".join(problems) +
                "\n（只跑检索回归需要 语料 + Ollama；调生成模型还需要 .env 里的 ARK_* 配置）"
            )

        splits = build_splits()
        if not db_ready(DB_DIR, len(splits)):
            print(f"【首次运行 / 库不完整】正在构建知识库向量库……（{len(splits)} 块）")
            # 必须在打开库句柄**之前**挪走旧库（半截库、或切分参数变过留下的旧库）
            _retire(DB_DIR, "stale")
            vdb = Chroma(persist_directory=db_dir_for_chroma(), embedding_function=make_embedding())

            # 分批入库：一次性 embed 1000+ 个块会把 Ollama 压垮（实测报 tokenize 连接错误），
            # 而且中途失败看不到进度。分批既稳，也能看到进展。
            # 中断也不会污染正式库：进程退出后，下次启动 db_ready 会发现块数对不上 → 重建。
            BATCH = 64
            for i in range(0, len(splits), BATCH):
                vdb.add_documents(splits[i:i + BATCH])
                print(f"  已入库 {min(i + BATCH, len(splits))}/{len(splits)} 块")
            print("【知识库构建完成】")
        else:
            vdb = Chroma(persist_directory=db_dir_for_chroma(), embedding_function=make_embedding())
            print(f"【已加载持久化知识库】{len(splits)} 块")

        # BM25 与向量库同源：都取库里的块，保证两条检索路看到的是同一批数据
        kb = vdb.get()
        bm25_docs = [Document(page_content=t, metadata=m)
                     for t, m in zip(kb["documents"], kb["metadatas"])]
        _INDEX.update(
            ready=True, vectordb=vdb, splits=splits, bm25_docs=bm25_docs,
            bm25=BM25Okapi([_tokenize(d.page_content) for d in bm25_docs]),
        )
        return vdb, splits


def __getattr__(name: str):
    """PEP 562：让 `ta.vectordb` / `ta.splits` 惰性可用，兼容原来的模块级访问。

    只在常规属性查找失败时才被调用，所以对已存在的名字零影响。
    刻意只放行这两个名字：其他拼错的属性照旧抛 AttributeError，
    不会把"写错属性名"变成"悄悄触发建库"。
    """
    if name in ("vectordb", "splits"):
        vdb, sp = _ensure_index()
        return vdb if name == "vectordb" else sp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def bm25_search(question: str, k: int = BM25_K):
    """关键词检索：query 过滤分词后算 BM25，只留有命中的

    query 侧可选加一条英文译文（QUERY_TRANSLATION=1）。注意是**并集不是替换**：
    两条 query 各打一次分，同一块取较高分再排序。
    ⚠️ 为什么不能直接用译文替换原句：实测 "FastAPI 怎么读取 Cookie？" 原句能中
    `cookie-params`（8.30），翻成 "FastAPI read cookies" 后 BM25 反而把
    `advanced/response-cookies` 顶上来，正确页被挤出唯一位子。替换 = 按下葫芦浮起瓢；
    并集 = 两条腿各投一票。
    """
    _ensure_index()                      # 惰性：第一次调用才建 BM25 索引
    bm25, bm25_docs = _INDEX["bm25"], _INDEX["bm25_docs"]

    queries = [question]
    translated = _translate_for_bm25(question)
    if translated != question:
        queries.append(translated)

    best: dict = {}
    for q in queries:
        tokens = _tokenize(q)
        if not tokens:                  # 全是标点/停用词 → 这条 query 放弃补召
            continue
        for i, s in enumerate(bm25.get_scores(tokens)):
            if s > best.get(i, 0.0):
                best[i] = float(s)

    if not best:
        return []
    order = sorted(best, key=lambda i: -best[i])[:k]
    return [(bm25_docs[i], best[i]) for i in order if best[i] > 0]

# ---------------- 2. API 大模型 ----------------
# 注：V4.1 Flash 是深度思考模型，默认会把大量 token 花在内部推理上。
#     教学助手要的是「快 + 短」，这里关掉思考模式（实测 token 消耗降到约 1/3）。
#
# ⚠️ 下面这 4 个参数是**和 MODEL_ID 绑在一起**调好的，但 MODEL_ID 可以在 .env 里换：
#    · extra_body.thinking —— 写死 "disabled"。换成不支持该字段的模型可能被 400 拒绝，
#      也可能被静默忽略，于是注释里"token 降到 1/3"不再成立，而没有任何地方会发现。
#    · max_tokens=800     —— 是按「关了思考」标定的。一旦思考被打开，推理会把 800
#      吃光，回答表现为空/截断，而代码**从不检查 finish_reason**，看起来像"模型变笨"。
#    · temperature=0.3    —— 也影响 L2 的可复现性。
#    → 换模型时请连这几项一起复验；评估生成层质量要重跑 pytest -m llm。
llm = ChatOpenAI(
    model=MODEL_ID,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.3,
    max_tokens=800,
    extra_body={"thinking": {"type": "disabled"}},
)

# ---------------- 3. 助教提示词 ----------------
# 规则要「少而硬」。12 版写了 6 条，本地小模型执行不过来，
# 反而把「资料里没有提到」「补充说明」当字面量单独吐出来（提示词词泄漏）。
#
# ⚠️ 角色描述必须跟着语料走：这里曾写死「《深入理解 AI Agent》这本书的学习助手」，
# 而语料早就换成了 FastAPI 官方文档 —— 模型被要求"按书回答"，却拿到官方文档的片段。
# 换语料时记得同步这一句，否则又是一个不报错的错配。
system_prompt = (
    "你是 FastAPI 官方文档的学习助手，名叫小助。\n"
    "规则：\n"
    "1. 优先依据「知识库参考内容」回答，引用时用 [1][2] 标注编号；\n"
    "2. 若参考内容不足以回答，先直说「资料里没有提到」，再讲你的通用知识，不要假装资料里有；\n"
    "3. 语言简洁友好，面向大学生，回答控制在 300 字以内；\n"
    "4. 不要反问学生。"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("human", "知识库参考内容：\n{context}\n\n学生的问题：{question}"),
])


# ---------------- 4. 检索与格式化 ----------------
# 相关性闸门：低于这个分数的片段直接丢掉，不进上下文。
#
# ⚠️ 阈值不是常数，是**标定出来的**，而且对两样东西敏感：
#    ① 语料        —— 话题越集中，分数整体越高（鸿蒙 PDF 0.5 → AI Agent 书 0.58）
#    ② embedding   —— 不同模型是不同坐标系，分数尺度完全不同（见下表）
#    换任何一个都必须重跑标定，否则就是拿旧尺子量新东西。
#
# 实测标定记录（同一份 FastAPI 语料、同一套 24 道库内题 + 5 道库外题）：
#   nomic-embed-text (768维)：库内 top1 0.47~0.76 ｜ 真库外 0.29~0.39   ← 两带**重叠**，切不干净
#   bge-m3           (1024维)：库内 top1 0.356~0.642 ｜ 真库外 0.161~0.280 ← 中间有干净的缝
# 所以 bge-m3 的阈值取缝的中点 0.32。
#
# ⚠️ 阈值和「标定它的那个模型」是绑定的，不能只留一个裸数字：0.32 只在 bge-m3 的坐标系里
#    是"缝的中点"，换个模型分数尺度整体平移，同一个数就只是"一个数"。
#    所以这里做成表 —— 换 EMBED_MODEL 必须同时给出该模型的标定值，缺了当场炸掉，
#    比"静默拿旧尺子量新东西"便宜。
#    语料换了同理要重标；那一条由 test_retrieval.py::test_threshold_sits_in_the_score_gap
#    这条元测试兜住（阈值必须落在库外最高分与库内最低分之间）。
_THRESHOLD_BY_MODEL = {
    "bge-m3": 0.32,
    # "nomic-embed-text": ???  ← 旧模型：两带重叠（库内 0.47~0.76 ｜ 库外 0.29~0.39），
    #                            换回去请重新标定后再填，别直接抄一个数。
}
SCORE_THRESHOLD = _THRESHOLD_BY_MODEL.get(EMBED_MODEL.split(":")[0])
if SCORE_THRESHOLD is None:
    raise SystemExit(
        f"没有为 embedding 模型 {EMBED_MODEL} 标定过阈值，拒绝用一个没依据的数继续跑。\n"
        f"标定方法（不花 API 额度，见 audit_probes.py 的用法）：\n"
        f"  ① 跑库内题，取「每道题最高分」的最小值  kb_low\n"
        f"  ② 跑库外题，取「每道题最高分」的最大值  oob_high\n"
        f"  ③ 两者之间取缝的中点，填进 _THRESHOLD_BY_MODEL\n"
        f"已有标定记录：{list(_THRESHOLD_BY_MODEL)}"
    )

# ---- 召回窗口（两个必须联动，只改一个会出问题）----
# VECTOR_K          ：向量路取多少个候选。注意这是**片段数，不是文件数**——
#                     长页面会被切成多个块，只取 4 块时经常拿不到含关键词的那一块。
# MAX_CONTEXT_CHUNKS：最终拼进上下文的块数上限。
#   ⚠️ 列表顺序是「向量在前、BM25 在后」，最后整体截断。所以 **k 一变大，
#   BM25 补召就会被挤出名额**（实测 Cookie 那题正是靠 BM25 捞回来的）。
#   → 两个参数必须同步放大，否则增大 k 是净负收益。
#
# 📌 实测记录（2026-09-21，bge-m3 + 29 题）：k=4→10、上限 6→12
#    结果 L1 20→21、L2 26→27，**只涨 1 分，上下文却翻倍** → 已回滚。
#    原因：正确文档的真实排名分布在 2~27 名，不是"多给几个窗口就能捞到"的问题。
#    诊断方法（不花 API 额度）：用 k=60 检索，打印期望文件的真实 index。
#    → 结论：k 不是这套系统的瓶颈。别再在这个旋钮上花时间。
# 📌 实测（2026-10-21｜bge-m3 + 块首补标题 + 排除 changelog｜24 道库内题）：
#      k=4 → 来源 20/24、L1 21/29
#      k=5 → 来源 21/24、L1 23/29   ← 取这个
#      k=6 → 来源 20/24（BM25 一个名额都不剩，反而掉）
#    → 不是「越大越好」，是「给 BM25 留 1 个名额」最好用。
VECTOR_K = 5
MAX_CONTEXT_CHUNKS = 6


# 语料目录名从配置派生，不写死 "docs_fastapi/"：
# 换语料目录后旧写法会静默返回整条绝对路径，把本机路径落进 traces.jsonl。
_DOCS_MARKER = Path(docs_dir).name + "/"


def source_label(d, idx: int | None = None) -> str:
    """统一的来源标签：`文件名 第N页` / `文件名#块号 首行标题`。

    ⚠️ `metadata["page"]` 只有 PDF（PyMuPDFLoader）会给。语料是 md/txt/docx 时它不存在，
    旧写法会静默退化成纯 basename —— 同一文件的多个块标出来一模一样（FastAPI 文档里
    index.md 还有 6 个同名文件），学生看不出引用到底出自哪一段。
    所以没有页码时补一个块标识：`#块号 + 块首标题`（markdown 语料的首行基本就是标题）。
    """
    src = d.metadata.get("source", "未知来源")
    name = os.path.basename(src)
    page = d.metadata.get("page")
    if page is not None:
        return f"{name} 第{page + 1}页"
    head = next((ln.strip("# ").strip() for ln in d.page_content.splitlines() if ln.strip()), "")
    tag = f"#{idx} " if idx is not None else ""
    return f"{name}{tag}{head[:28]}".strip()


def format_docs(docs):
    """把检索到的 document 拼成带编号的纯文本；来源尽量带"是哪一段"的定位信息"""
    if not docs:
        return "（知识库中未检索到相关内容）"
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] 来源：{source_label(d, i)}\n{d.page_content}")
    return "\n\n".join(parts)   # ← 12 版这里漏了 return，导致 context 变成 None


def search(question: str):
    """带分数的检索，返回 (doc, score) 列表；全文唯一的向量检索入口"""
    vdb, _ = _ensure_index()             # 惰性：第一次调用才读语料 / 开库
    return vdb.similarity_search_with_relevance_scores(question, k=VECTOR_K)


def _describe(d, score=None) -> dict:
    """把一块文档收敛成便于记录/展示的短结构"""
    src = str(d.metadata.get("source", "?")).replace("\\", "/").split(_DOCS_MARKER)[-1]
    item = {"src": src, "preview": d.page_content[:80].replace("\n", " ")}
    page = d.metadata.get("page")
    if page is not None:
        item["page"] = page + 1
    if score is not None:
        item["score"] = round(float(score), 4)
    return item


def final_context_docs(question: str, trace: dict | None = None):
    """最终进上下文的块 = 向量阈值过滤 + BM25 关键词补召（去重）

    【可观测性】可选参数 trace：传入一个 dict，函数会把「检索决策路径」写进去，
    字段固定为：question / threshold / vector_k / vector_candidates / vector_kept /
    gate / bm25_added / final（final 每项带 via 标明来自哪条路、score 为该路分数）。

    这是 tracing 的**数据层**，刻意不依赖任何框架：
    需要 UI 时，把这份 dict 转成 OTel span attributes 挂到 Phoenix / Langfuse 上即可，
    不用重写检索逻辑。（Phoenix 的 LangChain instrumentation 只能自动看到 LLM 调用，
    看不到我们自定义的向量+BM25 融合决策，所以这一层必须自己记录。）
    """
    pairs = search(question)
    kept = [(d, s) for d, s in pairs if s >= SCORE_THRESHOLD]        # 主路：语义

    if trace is not None:
        trace["question"] = question
        trace["threshold"] = SCORE_THRESHOLD
        trace["vector_k"] = VECTOR_K
        trace["vector_candidates"] = [_describe(d, s) for d, s in pairs]
        # 兑现上面承诺的「哪几条过了阈值」：原来数据层没这个字段，
        # 只能由渲染层拿 score >= threshold 自己重算。
        trace["vector_kept"] = [_describe(d, s) for d, s in kept]

    if not kept:
        # 向量路全灭 = 系统唯一能表达「这个问题与知识库无关」的信号。
        # BM25 只认「字面出现过没有」，语料里哪怕恰好有同样的句子（实测书里
        # 真有「今天天气怎么样」，作为工具调用示例，BM25 给 6.99 分）也会命中，
        # 用字面巧合去覆盖语义判断，只会把噪声放进上下文。
        # 少召回无害（模型会说「资料里没有提到」），错召回有幻觉风险 → 宁可空着。
        #
        # 📌 已知边界：闸门是「全灭才关」的开关，不是置信度判断。库里只要有一条
        #    擦边过线（比如问题里含 "FastAPI"，向量分天然偏高，见 test_retrieval.py
        #    里那条 xfail），BM25 就会照常开。要更严得改成按 margin 判断，属调优范围。
        if trace is not None:
            trace["gate"] = "触发 → 向量路 0 条过阈值，判定为库外问题，禁用 BM25 补召"
            trace["bm25_added"] = []
            trace["final"] = []
        return []

    vec_keys = {(d.metadata.get("source"), hash(d.page_content)) for d, _ in kept}
    seen = set(vec_keys)
    bm25_added = []
    bm25_q = _translate_for_bm25(question)      # 开了翻译就有值，否则等于原句
    for d, s in bm25_search(question, k=BM25_K):                      # 补路：关键词
        key = (d.metadata.get("source"), hash(d.page_content))
        if key not in seen:
            kept.append((d, s))
            seen.add(key)
            bm25_added.append(_describe(d, s))

    if trace is not None:
        trace["gate"] = "未触发 → 向量路有命中，允许 BM25 补召"
        if bm25_q != question:
            trace["bm25_query"] = bm25_q        # 只有真翻译过才记，避免噪声字段
        trace["bm25_added"] = bm25_added
        trace["final"] = [
            {**_describe(d, s),
             "via": "向量" if (d.metadata.get("source"), hash(d.page_content)) in vec_keys else "BM25"}
            for d, s in kept[:MAX_CONTEXT_CHUNKS]
        ]
    return [d for d, _ in kept[:MAX_CONTEXT_CHUNKS]]


def retrieve(question: str) -> str:
    """给模型看的上下文：向量阈值过滤 + BM25 补召"""
    return format_docs(final_context_docs(question))


def _pick_context(x: dict) -> str:
    """选上下文：调用方已经检索好的就用它，没有才现算。

    ⚠️ 为什么要多这一层：show_sources() 为了打印面板会检索一次，如果这里不认，
    rag_chain 会**再检索一次** —— 一题两次向量检索（本地 embedding 白跑一遍），
    而且**面板打印的名单和真正喂给模型的那批块不是同一次调用的产物**：
    [n] 只是"通常"对得上，不是结构上保证一致。
    现在面板把结果放进输入里传进来，两条路共用同一批块。
    """
    ctx = x.get("context")
    if ctx:
        return ctx
    return format_docs(final_context_docs(x["question"]))


rag_chain = (
    RunnablePassthrough.assign(context=RunnableLambda(_pick_context))
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------- 5. 多轮对话记忆（带裁剪） ----------------
MAX_TURNS = 8   # 只保留最近 8 轮，防止历史无限增长稀释注意力


class TrimmingHistory(InMemoryChatMessageHistory):
    """超出上限时从头砍，保证 prompt 长度可控"""

    def add_messages(self, messages):
        super().add_messages(messages)
        limit = MAX_TURNS * 2          # 一轮 = human + ai 两条
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]


store: dict[str, TrimmingHistory] = {}


def get_session_history(session_id: str) -> TrimmingHistory:
    if session_id not in store:
        store[session_id] = TrimmingHistory()
    return store[session_id]


assistant = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# ---------------- 6. 交互循环 ----------------
SESSION_ID = "student-001"


def show_sources(question: str) -> str:
    """检索**一次**，打印「最终进上下文」的完整名单，并返回给模型用的 context。

    ⚠️ 旧版这里有三次检索：本函数 search 一遍 → final_context_docs 再算一遍 →
    rag_chain 里的 retrieve 又算第三遍（外加 3 次 BM25）。除了本地 embedding 白跑两遍，
    更要紧的是**打印的名单与喂给模型的那批块不同源** —— 出现过「回答引用 [5][6]，
    列表里却只有 [1]~[4]」的对不上。仪表盘必须显示模型真正看到的东西，
    所以现在只算一次：本次返回的 context 由调用方原样传进 chain（见 _pick_context）。
    """
    trace: dict = {}
    docs = final_context_docs(question, trace=trace)

    print("\n--- 进上下文的片段（编号 = 回答里的 [n]）---")
    if not docs:
        print(f"(向量路 0 条过阈值 {SCORE_THRESHOLD}，BM25 补召已禁用 → 上下文为空)")
    for i, item in enumerate(trace["final"], 1):
        loc = item["src"]
        if item.get("page"):
            loc += f" 第{item['page']}页"
        score = item.get("score")
        path = item["via"] if score is None else f"{item['via']} {score:.2f}"
        print(f"[{i}] ({loc} · {path}) {item['preview'][:60]}…")
    dropped = [c["score"] for c in trace["vector_candidates"] if c["score"] < SCORE_THRESHOLD]
    if dropped:
        print(f"(另有 {len(dropped)} 个向量片段未过阈值，相关度 {dropped})")
    print("---")
    return format_docs(docs)


def is_valid_question(q: str) -> bool:
    """拦掉乱码 / 纯数字 / 纯符号——这类输入模型只会硬编一个答案。

    ⚠️ 原版只认中文字符，纯英文提问（"What is ..."）会被误杀。
    现在规则：含中文，**或**含一个 ≥3 字母的英文词，都算有效问题。
    "rh"（2 字母）、"11"、纯标点 → 仍然被拦住。
    """
    if len(q) < 2:
        return False
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in q)
    has_latin_word = re.search(r"[A-Za-z]{3,}", q) is not None
    return has_cjk or has_latin_word


def ask(question: str) -> str:
    """单问测试用：先展示召回来源，再取一次回答。

    ⚠️ 每问用**独立 session**。踩过的坑：早期这里复用全局 SESSION_ID，
    导致评测时 29 道题共用一条对话历史 —— 上一题的回答会进入下一题的上下文，
    用例之间相互污染（实测同配置两次跑 L2 差 2 分，一部分就来自这里）。
    评测要的是「单题独立作答能力」，不是「连续对话能力」。

    ⚠️ session id 改用 uuid：原来是 f"ask-{abs(hash(q)) % 10**8}"，hash 取模理论上会撞，
    两个问题的历史会被静默合并 —— 正好破坏上面要守的隔离。用完即 pop，
    免得 store 无限增长（跑 29 题的批，原来会攒 29 条永不释放）。

    ⚠️ 输入校验下沉到这里：原来只有交互循环调 is_valid_question()，
    --test 路径和测试套件全绕过去了 —— docstring 承诺的拦截只兑现了一半。
    """
    _require_config()      # 真正要调模型了，这里才校验配置（见 _require_config 的说明）
    if not is_valid_question(question):
        return "（问题无效：请用完整的中英文描述；纯数字、纯符号、单个字母不处理）"

    ctx = show_sources(question)
    sid = f"ask-{uuid.uuid4().hex}"
    try:
        return assistant.invoke(
            {"question": question, "context": ctx},
            config={"configurable": {"session_id": sid}},
        )
    finally:
        store.pop(sid, None)


if __name__ == "__main__":
    # 前置体检：把三类最常见的失败（语料目录找不到 / Ollama 没起或模型没 pull /
    # .env 没填）变成一句能照做的提示，而不是几十行 traceback 让人去猜。
    # 放在这里而不是模块级：import 这个模块（评测脚本、trace 工具）不该被运行环境卡住。
    _problems = check_runtime(require_llm=True)
    if _problems:
        print("无法启动，先解决下面几项：", file=sys.stderr)
        for _p in _problems:
            print("  ·", _p, file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("FastAPI 文档助手「小助」已上线（输入 退出 结束对话）")
    print("=" * 60)

    if "--test" in sys.argv:
        # 支持：python 14-ai_agent_assistant.py --test "你的问题"
        argv_q = [a for a in sys.argv[sys.argv.index("--test") + 1:] if not a.startswith("-")]
        # ⚠️ 默认题必须跟着语料走：原来是《AI Agent》书时代的「Agent 的核心公式是什么？」，
        # 语料换成 FastAPI 文档后它必然触发闸门 → 上下文为空，看起来像"系统坏了"。
        q = argv_q[0] if argv_q else "FastAPI 怎么定义路径参数？"
        print(f"\n学生：{q}\n")
        print("小助：", ask(q))
        sys.exit(0)

    while True:
        try:
            q = input("\n学生：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n小助：下次见！")
            break
        if not q:
            continue
        if q in ("退出", "exit", "quit"):
            print("小助：下次见！")
            break
        if not is_valid_question(q):
            print("\n小助：抱歉，我没看懂你的问题，能完整描述一下吗？")
            continue

        ctx = show_sources(q)   # ← 12 版从没被调用过；现在它同时返回给模型用的 context
        print("\n小助：", end="", flush=True)
        # ⚠️ 旧版这里没有 try：401（key 写错）/ 429（限流）/ 网络中断会从 for 里抛穿，
        # 整个程序退出 —— 对话历史全丢，重启还得等 BM25 索引重建。
        # eval_fastapi.py 是按题 try 的，主程序不该比评测脚本更脆。
        try:
            for chunk in assistant.stream(
                {"question": q, "context": ctx},
                config={"configurable": {"session_id": SESSION_ID}},
            ):
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            print("\n（已中断这次回答）", end="")
        except Exception as e:
            print(f"\n[接口出错，本次回答没走完] {type(e).__name__}: {str(e)[:200]}", end="")
        print()
