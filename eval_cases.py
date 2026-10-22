# ============================================================================
# eval_cases.py —— 评测集的【唯一】定义源
#
# 为什么单独抽出来：
#   eval_fastapi.py（跑分脚本）和 test_retrieval.py（pytest 回归套件）都要用它。
#   两处各抄一份 = 迟早对不上（踩过两次：配置两处来源、L1/L2 共用一个词表）。
#
# 字段：(类型, 问题, L1 探针词, L2 期望词, 可接受来源)
#   L1 探针词：英文、取自页面正文、且落在该页多个块里
#              （体检：python audit_probes.py）
#              ⚠️ 命中【任意一个】即算覆盖：探针是"这页特有的证据"，
#                 不是"必须逐字出现的清单"。旧版要求 all()，实测造成两条假阴性
#                 —— 正确页进了窗口，但承载某个词的块没进来，被记成检索失败。
#   L2 期望词：中文或中英并列，任一命中即通过
#   可接受来源：`str` 或 `tuple[str, ...]`。一个问题常有多个正确页
#              （教程页 + reference 页），只认一个会把答对判成答错。
#              匹配用「子路径片段」，因为 index.md 有多个同名文件。
#
# ★ 2026-10-22 重挑：原来 24 道库内题【全部】落在 tutorial/，
#   而 tutorial 只占语料 33%（advanced 34 篇、reference 24 篇、how-to 12 篇、
#   deployment 9 篇、顶层 21 篇，一道题都没有）。
#   所以 "L1 23/29" 只代表教程页水位，不是文档助手的整体水位。
#
#   重挑原则不是"每个文件一道题"（那是覆盖文件，不是覆盖问题），而是
#   **按用户会问什么分层**：基础用法 / 进阶用法 / API 签名 / 迁移与配置 /
#   部署运维 / 概念。语料里的 contributing.md、newsletter.md、management.md
#   这类非技术页不参与评测 —— 用户不会问它们。
# ============================================================================

CASES = [
    # ================= 基础用法（tutorial，25 题）=================
    # 这是产品的核心路径：中文问 → 英文教程页。
    ("kb", "FastAPI 怎么定义路径参数？",            ["item_id", "Path"],  ["路径参数", "item_id"],            "tutorial/path-params"),
    ("kb", "FastAPI 怎么声明查询参数？",            ["skip", "limit"],    ["查询参数", "skip"],               "tutorial/query-params"),
    ("kb", "怎么用 Pydantic 模型接收请求体？",       ["body", "Pydantic"], ["请求体", "Body", "BaseModel"],   "tutorial/body"),
    ("kb", "怎么给 Pydantic 模型的字段加约束？",     ["Field"],            ["字段", "Field"],                  "tutorial/body-fields"),
    ("kb", "怎么定义嵌套的请求体模型？",             ["nested", "submodel"], ["嵌套", "嵌套模型"],            "tutorial/body-nested-models"),
    ("kb", "怎么给查询参数加长度和正则校验？",       ["Query", "min_length"], ["长度", "正则", "校验"],        "tutorial/query-params-str-validations"),
    ("kb", "怎么校验路径参数是数字且大于某个值？",    ["path_params_numeric_validations", "Path"], ["数字", "大于", "Path"],           "tutorial/path-params-numeric-validations"),
    ("kb", "FastAPI 怎么读取请求头参数？",           ["Header"],           ["请求头", "Header"], "tutorial/header-params"),
    ("kb", "FastAPI 怎么读取 Cookie？",             ["Cookie"],           ["Cookie"],                        "tutorial/cookie-params"),
    ("kb", "FastAPI 怎么接收上传的文件？",           ["UploadFile", "File"], ["文件", "UploadFile"],           "tutorial/request-files"),
    ("kb", "FastAPI 怎么处理表单数据？",             ["Form"],             ["表单", "Form"],                   "tutorial/request-forms"),
    ("kb", "怎么限制接口响应返回的字段？",            ["response_model"],   ["response_model", "响应模型"],     "tutorial/response-model"),
    ("kb", "怎么设置接口返回的状态码？",              ["status_code"],      ["状态码", "status_code"],          "tutorial/response-status-code"),
    ("kb", "怎么抛出自定义的 HTTP 错误？",            ["HTTPException", "detail"], ["HTTPException", "异常"],  "tutorial/handling-errors"),
    ("kb", "怎么把一个类当作依赖使用？",              ["CommonQueryParams", "dependencies"], ["依赖", "Depends"],        "tutorial/dependencies/classes-as-dependencies"),
    ("kb", "带 yield 的依赖是怎么工作的？",           ["yield"],            ["yield"],                         "tutorial/dependencies/dependencies-with-yield"),
    ("kb", "怎么让接口在返回后继续执行后台任务？",      ["BackgroundTasks"],  ["后台任务", "BackgroundTasks"],    "tutorial/background-tasks"),
    ("kb", "FastAPI 里怎么做数据库连接？",            ["Session", "SQLModel"], ["数据库", "Session", "SQLModel"], "tutorial/sql-databases"),
    ("kb", "怎么给 FastAPI 接口写测试？",             ["TestClient"],       ["测试", "TestClient"],             "tutorial/testing"),
    ("kb", "怎么实现 OAuth2 密码模式的登录？",        ["OAuth2", "password"], ["OAuth2", "令牌"],              "tutorial/security/simple-oauth2"),
    ("kb", "怎么用 JWT 做令牌认证？",                 ["JWT", "jwt"],        ["JWT", "令牌"],                   "tutorial/security/oauth2-jwt"),
    ("kb", "怎么配置跨域 CORS？",                     ["CORS", "middleware"], ["CORS", "跨域"],                "tutorial/cors"),
    ("kb", "怎么挂载静态文件目录？",                   ["StaticFiles"],      ["静态", "StaticFiles"],            "tutorial/static-files"),
    ("kb", "怎么把请求体里的数据做部分更新？",          ["exclude_unset"],    ["部分更新", "更新", "exclude_unset"], "tutorial/body-updates"),
    ("kb", "怎么把路由拆到多个文件里？",               ["APIRouter", "router"], ["路由", "APIRouter"],         "tutorial/bigger-applications"),

    # ================= 进阶用法（advanced，6 题）=================
    # 这一层是"用过之后才会问"的问题，之前一道题都没覆盖。
    ("kb", "WebSocket 怎么做实时通信？",              ["WebSocket", "websocket"], ["WebSocket", "实时"],       "advanced/websockets"),
    ("kb", "怎么在应用启动和关闭时执行代码？",         ["lifespan", "shutdown"], ["lifespan", "启动", "关闭"], "advanced/events"),
    ("kb", "怎么给每个请求加一层处理（中间件）？",      ["middlewares", "ASGI"], ["中间件", "middleware"],     "advanced/middleware"),
    ("kb", "怎么从环境变量读取应用配置？",             ["settings", "lru_cache"], ["环境变量", "settings", "配置"], ("advanced/settings", "environment-variables")),
    ("kb", "怎么用 Jinja2 返回 HTML 页面？",          ["Jinja2Templates", "templates"], ["模板", "Jinja2", "HTML"], ("advanced/templates", "reference/templating")),
    ("kb", "怎么在测试里替换依赖（比如换成假数据库）？", ["override", "dependency_overrides"], ["override", "替换", "依赖"], "advanced/testing-dependencies"),

    # ================= API 签名（reference，4 题）=================
    # 问"这个类/函数有哪些参数"，和问"怎么做"是两种检索需求。
    ("kb", "APIRouter 有哪些常用参数？",              ["APIRouter"],       ["APIRouter", "prefix", "路由"], ("reference/apirouter", "tutorial/bigger-applications")),
    ("kb", "怎么把模型转成 JSON 兼容的字典？",         ["jsonable_encoder"], ["jsonable_encoder", "编码"],   ("reference/encoders", "tutorial/encoder")),
    ("kb", "UploadFile 有哪些方法和属性？",           ["UploadFile", "filename"], ["UploadFile", "filename", "文件"], ("reference/uploadfile", "tutorial/request-files")),
    ("kb", "HTTP 状态码常量在哪里定义？",             ["status", "HTTP_"], ["status", "状态码"],            ("reference/status", "tutorial/response-status-code")),

    # ================= 迁移与配置（how-to，5 题）=================
    ("kb", "Pydantic v1 升级到 v2 要注意什么？",      ["Pydantic", "v1"],   ["v2", "迁移", "Pydantic"],         "how-to/migrate-from-pydantic-v1-to-pydantic-v2"),
    ("kb", "怎么在生产环境关闭接口文档？",             ["disable", "openapi"], ["关闭", "openapi_url", "文档"], ("how-to/conditional-openapi", "tutorial/metadata")),
    ("kb", "怎么修改 Swagger UI 的默认参数？",        ["swagger_ui_parameters", "Swagger"], ["Swagger", "参数"], ("how-to/configure-swagger-ui", "reference/openapi/docs")),
    ("kb", "FastAPI 里怎么接 GraphQL？",              ["GraphQL", "strawberry"], ["GraphQL", "strawberry"],     "how-to/graphql"),

    # ================= 部署运维（deployment，4 题）=================
    ("kb", "怎么用 Docker 部署 FastAPI？",            ["Docker", "container"], ["Docker", "容器"],           "deployment/docker"),
    ("kb", "生产环境应该开几个 worker 进程？",         ["workers", "uvicorn"], ["worker", "进程", "uvicorn"], "deployment/server-workers"),
    ("kb", "不用容器，怎么手工启动 uvicorn？",         ["uvicorn"],          ["uvicorn", "启动"],                ("deployment/manually", "deployment/concepts")),
    ("kb", "为什么生产环境要用 HTTPS？",               ["HTTPS", "Proxy"],   ["HTTPS", "证书"],                  "deployment/https"),

    # ================= 概念（顶层，2 题）=================
    ("kb", "async/await 和并发是什么关系？",          ["async", "await"],   ["async", "并发", "await"],         "async"),
    ("kb", "Python 的 Optional / Union 类型提示怎么用？", ["type", "hints"],   ["类型", "类型提示", "Union"], "python-types"),

    # ================= 库外（5 题，期望召回为空）=================
    ("oob", "今天天气怎么样？",                       None,              ["资料里没有", "没有提到", "未提到"], None),
    ("oob", "北京有什么好吃的？",                     None,              ["资料里没有", "没有提到", "未提到"], None),
    ("oob", "什么是量子纠缠？",                       None,              ["资料里没有", "没有提到", "未提到"], None),
    ("oob", "Python 的 GIL 是什么？",                None,              ["资料里没有", "没有提到", "未提到"], None),
    ("oob", "怎么用 FastAPI 训练深度学习模型？",       None,              ["资料里没有", "没有提到", "未提到"], None),
]

TOTAL = len(CASES)

# kb / oob 拆分（测试套件和跑分脚本都会用到）
KB_CASES = [c for c in CASES if c[0] == "kb"]
OOB_CASES = [c for c in CASES if c[0] == "oob"]

# ⚠️ 已知的"擦边库外题"：问题里含 "FastAPI" 字样，向量分数天然偏高，
# 注定会泄漏几块噪声。不计入"库外必须为空"的硬性断言，单独作为观察项。
OOB_STRICT = [c[1] for c in OOB_CASES if "FastAPI" not in c[1]]
OOB_EDGE = [c[1] for c in OOB_CASES if "FastAPI" in c[1]]


# ---------------------------------------------------------------------------
# 分层：报告必须按层出数，只有一个总数会掩盖"哪一层不行"
# ---------------------------------------------------------------------------
CATEGORY_LABELS = {
    "tutorial": "基础用法",
    "advanced": "进阶用法",
    "reference": "API 签名",
    "how-to": "迁移与配置",
    "deployment": "部署运维",
    "async": "并发模型",
    "python-types": "类型提示",
    "oob": "库外拒答",
}


def expected_sources(case) -> tuple:
    """把可接受来源统一成 tuple（单字符串也包一层）"""
    exp = case[4]
    if exp is None:
        return ()
    return (exp,) if isinstance(exp, str) else tuple(exp)


def case_category(case) -> str:
    """分层标签：库外题单列，库内题取第一个可接受来源的顶层目录"""
    if case[0] == "oob":
        return "oob"
    srcs = expected_sources(case)
    if not srcs:
        return "unknown"
    return srcs[0].split("/")[0]
