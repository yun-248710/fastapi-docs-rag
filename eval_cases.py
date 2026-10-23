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
    ("kb", "FastAPI 怎么定义路径参数？",            ["item_id", "Path"],  ["item_id", "path_params"],            "tutorial/path-params"),
    ("kb", "FastAPI 怎么声明查询参数？",            ["skip", "limit"],    ["skip", "limit"],               "tutorial/query-params"),
    ("kb", "怎么用 Pydantic 模型接收请求体？",       ["body", "Pydantic"], ["BaseModel", "body"],   "tutorial/body"),
    ("kb", "怎么给 Pydantic 模型的字段加约束？",     ["Field"],            ["Field", "gt"],                  "tutorial/body-fields"),
    ("kb", "怎么定义嵌套的请求体模型？",             ["nested", "submodel"], ["nested", "submodel"],            "tutorial/body-nested-models"),
    ("kb", "怎么给查询参数加长度和正则校验？",       ["Query", "min_length"], ["max_length", "pattern", "Annotated"],        "tutorial/query-params-str-validations"),
    ("kb", "怎么校验路径参数是数字且大于某个值？",    ["path_params_numeric_validations", "Path"], ["Path", "ge", "gt"],           "tutorial/path-params-numeric-validations"),
    ("kb", "FastAPI 怎么读取请求头参数？",           ["Header"],           ["Header", "underscore"], "tutorial/header-params"),
    ("kb", "FastAPI 怎么读取 Cookie？",             ["Cookie"],           ["cookie_params", "Cookie("],                        "tutorial/cookie-params"),
    ("kb", "FastAPI 怎么接收上传的文件？",           ["UploadFile", "File"], ["UploadFile", "File"],           "tutorial/request-files"),
    ("kb", "FastAPI 怎么处理表单数据？",             ["Form"],             ["Form", "multipart"],                   "tutorial/request-forms"),
    ("kb", "怎么限制接口响应返回的字段？",            ["response_model"],   ["response_model", "exclude"],     "tutorial/response-model"),
    ("kb", "怎么设置接口返回的状态码？",              ["status_code"],      ["status_code", "HTTP_201"],          "tutorial/response-status-code"),
    ("kb", "怎么抛出自定义的 HTTP 错误？",            ["HTTPException", "detail"], ["HTTPException", "RequestValidationError"],  "tutorial/handling-errors"),
    ("kb", "怎么把一个类当作依赖使用？",              ["CommonQueryParams", "dependencies"], ["CommonQueryParams", "Depends"],        "tutorial/dependencies/classes-as-dependencies"),
    ("kb", "带 yield 的依赖是怎么工作的？",           ["yield"],            ["depend", "Exception"],                         "tutorial/dependencies/dependencies-with-yield"),
    ("kb", "怎么让接口在返回后继续执行后台任务？",      ["BackgroundTasks"],  ["BackgroundTasks", "BackgroundTask"],    "tutorial/background-tasks"),
    ("kb", "FastAPI 里怎么做数据库连接？",            ["Session", "SQLModel"], ["SQLModel", "Session"], "tutorial/sql-databases"),
    ("kb", "怎么给 FastAPI 接口写测试？",             ["TestClient"],       ["TestClient", "pytest"],             "tutorial/testing"),
    ("kb", "怎么实现 OAuth2 密码模式的登录？",        ["OAuth2", "password"], ["OAuth2PasswordRequestForm", "tokenUrl"],              "tutorial/security/simple-oauth2"),
    ("kb", "怎么用 JWT 做令牌认证？",                 ["JWT", "jwt"],        ["PyJWT", "hash", "token"],                   "tutorial/security/oauth2-jwt"),
    ("kb", "怎么配置跨域 CORS？",                     ["CORS", "middleware"], ["CORSMiddleware", "allow_origins"],                "tutorial/cors"),
    ("kb", "怎么挂载静态文件目录？",                   ["StaticFiles"],      ["StaticFiles", "mount"],            "tutorial/static-files"),
    ("kb", "怎么把请求体里的数据做部分更新？",          ["exclude_unset"],    ["exclude_unset", "model_dump"], "tutorial/body-updates"),
    ("kb", "怎么把路由拆到多个文件里？",               ["APIRouter", "router"], ["APIRouter", "include_router"],         "tutorial/bigger-applications"),

    # ================= 进阶用法（advanced，6 题）=================
    # 这一层是"用过之后才会问"的问题，之前一道题都没覆盖。
    ("kb", "WebSocket 怎么做实时通信？",              ["WebSocket", "websocket"], ["receive_text", "connect"],       "advanced/websockets"),
    ("kb", "怎么在应用启动和关闭时执行代码？",         ["lifespan", "shutdown"], ["lifespan", "shutdown"], "advanced/events"),
    ("kb", "怎么给每个请求加一层处理（中间件）？",      ["middlewares", "ASGI"], ["middleware", "ASGI"],     "advanced/middleware"),
    ("kb", "怎么从环境变量读取应用配置？",             ["settings", "lru_cache"], ["BaseSettings", "lru_cache"], ("advanced/settings", "environment-variables")),
    ("kb", "怎么用 Jinja2 返回 HTML 页面？",          ["Jinja2Templates", "templates"], ["Jinja2Templates", "TemplateResponse"], ("advanced/templates", "reference/templating")),
    ("kb", "怎么在测试里替换依赖（比如换成假数据库）？", ["override", "dependency_overrides"], ["dependency_overrides", "override"], "advanced/testing-dependencies"),

    # ================= API 签名（reference，4 题）=================
    # 问"这个类/函数有哪些参数"，和问"怎么做"是两种检索需求。
    ("kb", "APIRouter 有哪些常用参数？",              ["APIRouter"],       ["prefix", "tags"], ("reference/apirouter", "tutorial/bigger-applications")),
    ("kb", "怎么把模型转成 JSON 兼容的字典？",         ["jsonable_encoder"], ["jsonable_encoder", "encoders"],   ("reference/encoders", "tutorial/encoder")),
    ("kb", "UploadFile 有哪些方法和属性？",           ["UploadFile", "filename"], ["content_type", "filename"], ("reference/uploadfile", "tutorial/request-files")),
    ("kb", "HTTP 状态码常量在哪里定义？",             ["status", "HTTP_"], ["HTTP_200_OK", "status."],            ("reference/status", "tutorial/response-status-code")),

    # ================= 迁移与配置（how-to，5 题）=================
    ("kb", "Pydantic v1 升级到 v2 要注意什么？",      ["Pydantic", "v1"],   ["pydantic.v1", "model_dump"],         "how-to/migrate-from-pydantic-v1-to-pydantic-v2"),
    ("kb", "怎么在生产环境关闭接口文档？",             ["disable", "openapi"], ["openapi_url", "docs_url"], ("how-to/conditional-openapi", "tutorial/metadata")),
    ("kb", "怎么修改 Swagger UI 的默认参数？",        ["swagger_ui_parameters", "Swagger"], ["swagger_ui_parameters", "docExpansion"], ("how-to/configure-swagger-ui", "reference/openapi/docs")),
    ("kb", "FastAPI 里怎么接 GraphQL？",              ["GraphQL", "strawberry"], ["strawberry", "graphene"],     "how-to/graphql"),

    # ================= 部署运维（deployment，4 题）=================
    ("kb", "怎么用 Docker 部署 FastAPI？",            ["Docker", "container"], ["Dockerfile", "container"],           "deployment/docker"),
    ("kb", "生产环境应该开几个 worker 进程？",         ["workers", "uvicorn"], ["gunicorn", "uvicorn"], "deployment/server-workers"),
    ("kb", "不用容器，怎么手工启动 uvicorn？",         ["uvicorn"],          ["serve", "--host"],                ("deployment/manually", "deployment/concepts")),
    ("kb", "为什么生产环境要用 HTTPS？",               ["HTTPS", "Proxy"],   ["证书", "TLS", "certificate"],                  "deployment/https"),

    # ================= 概念（顶层，2 题）=================
    ("kb", "async/await 和并发是什么关系？",          ["async", "await"],   ["coroutine", "parallel"],         "async"),
    ("kb", "Python 的 Optional / Union 类型提示怎么用？", ["type", "hints"],   ["typing", "hints"], "python-types"),

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


# ============================================================================
# 一致性与鲁棒性（2026-10-22 新增）
#
# 为什么单独建两组：前面的 CASES 测的是"这句话能不能查到"，测不到两件事 ——
#   ① 同一问题的不同说法，结果会不会漂（一致性）
#   ② 用户打错字、中英混输、乱加标点，系统还稳不稳（鲁棒性）
# 这两类都是真实用户每天在做的事，而"单句正确率"看不出来。
#
# 参考：ai-engineering-toolkit 的 eval-harness-builder 把行为评测分成
# refusal / consistency / robustness / calibration / adversarial 五类，
# 本项目原先只有 refusal（库外拒答），这里补上中间两类。
# ============================================================================

# (组名, 期望来源, [同一问题的多种问法；第一个是基线问法])
PARAPHRASE_GROUPS = [
    ("路径参数", "tutorial/path-params", [
        "FastAPI 怎么定义路径参数？",
        "路径参数在 FastAPI 里怎么写？",
        "FastAPI 中如何声明路径参数",
        "怎么在路由里拿到 URL 上的 id？",
    ]),
    ("跨域 CORS", "tutorial/cors", [
        "怎么配置跨域 CORS？",
        "跨域怎么设置？",
        "FastAPI 如何允许跨域请求",
        "CORSMiddleware 怎么加",
    ]),
    ("JWT 认证", "tutorial/security/oauth2-jwt", [
        "怎么用 JWT 做令牌认证？",
        "JWT 认证怎么做",
        "FastAPI 怎么生成和校验 token",
    ]),
    ("请求体", "tutorial/body", [
        "怎么用 Pydantic 模型接收请求体？",
        "POST 的数据怎么接？",
        "请求体怎么定义成模型",
    ]),
]

# (组名, 期望来源, 基线问法, 噪声问法, 噪声类型)
# 断言方式：噪声问法不能比基线差。基线本来就查不到的题不参与判定
# （否则测的是已知的检索缺口，不是噪声的影响）。
NOISE_VARIANTS = [
    ("路径参数", "tutorial/path-params",
     "FastAPI 怎么定义路径参数？", "FastAPI 怎么定义路径参树？", "错别字"),
    ("路径参数", "tutorial/path-params",
     "FastAPI 怎么定义路径参数？", "FastAPI怎么定义路径参数", "无空格无问号"),
    ("跨域 CORS", "tutorial/cors",
     "怎么配置跨域 CORS？", "怎么配置跨玉 CORS？", "错别字"),
    ("跨域 CORS", "tutorial/cors",
     "怎么配置跨域 CORS？", "FastAPI 怎么 configure CORS", "中英混输"),
    ("请求头", "tutorial/header-params",
     "FastAPI 怎么读取请求头参数？", "FastAPI 怎么读取请求头参数？？", "重复全角标点"),
    ("请求体", "tutorial/body",
     "怎么用 Pydantic 模型接收请求体？", "怎么用 pydantic 模型接收请求体", "大小写+无问号"),
]


# 已测基线（2026-10-22，bge-m3 + 1919 块 + k=5）：
#   一致性 11/14 命中（79%）——路径参数 4/4、CORS 3/4、JWT 2/3、请求体 2/3
#   鲁棒性 4/5 与基线一致——唯一失败是"路径参树"（错别字打断了语义）
# 阈值按基线留一档余量，用于防回归，不是"及格线"。
PARAPHRASE_MIN_RATE = 0.6

# 已知会失败的噪声问法：不隐藏，标成 xfail 当活记录。
# "路径参树"测出来是唯一一个断掉的 —— 单字替换会把语义整段带偏，
# 说明 bge-m3 对中文错别字不鲁棒。哪天它开始通过，说明补了纠错或换了模型。
KNOWN_NOISE_FAILURES = {"FastAPI 怎么定义路径参树？"}
