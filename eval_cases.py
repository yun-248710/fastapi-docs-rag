# ============================================================================
# eval_cases.py —— 评测集的【唯一】定义源
#
# 为什么单独抽出来：
#   eval_fastapi.py（跑分脚本）和 test_retrieval.py（pytest 回归套件）都要用它。
#   两处各抄一份 = 迟早对不上（今天已经踩过两次：配置两处来源、L1/L2 共用一个词表）。
#
# 字段说明：(类型, 问题, L1期望词, L2期望词, 期望来源文件)
#   L1期望词：英文、取自页面正文、且最好落在该页多个块里
#             （体检工具：python audit_probes.py）
#   L2期望词：中文或中英并列，任一命中即通过
#   期望来源：用「子路径片段」（如 "tutorial/path-params"），因为 index.md 有 6 个同名文件
# ============================================================================

CASES = [
    # ===== kb：中文提问，期望命中英文教程页 =====
    ("kb", "FastAPI 怎么定义路径参数？",            ["item_id"],       ["路径参数", "item_id"],            "tutorial/path-params"),
    ("kb", "FastAPI 怎么声明查询参数？",            ["skip"],          ["查询参数", "skip"],               "tutorial/query-params"),
    ("kb", "怎么用 Pydantic 模型接收请求体？",       ["body"],          ["请求体", "Body", "BaseModel"],     "tutorial/body"),
    ("kb", "怎么给 Pydantic 模型的字段加约束？",     ["Field"],         ["字段", "Field"],                  "tutorial/body-fields"),
    ("kb", "怎么定义嵌套的请求体模型？",             ["nested"],        ["嵌套", "嵌套模型"],                "tutorial/body-nested-models"),
    ("kb", "怎么给查询参数加长度和正则校验？",       ["Query"],         ["长度", "正则", "校验"],            "tutorial/query-params-str-validations"),
    ("kb", "怎么校验路径参数是数字且大于某个值？",    ["Path"],          ["数字", "大于", "Path"],            "tutorial/path-params-numeric-validations"),
    ("kb", "FastAPI 怎么读取请求头参数？",           ["Header"],        ["请求头", "Header"],               "tutorial/header-params"),
    ("kb", "FastAPI 怎么读取 Cookie？",             ["Cookie"],        ["Cookie"],                        "tutorial/cookie-params"),
    ("kb", "FastAPI 怎么接收上传的文件？",           ["UploadFile"],    ["文件", "UploadFile"],             "tutorial/request-files"),
    ("kb", "FastAPI 怎么处理表单数据？",             ["Form"],          ["表单", "Form"],                   "tutorial/request-forms"),
    ("kb", "怎么限制接口响应返回的字段？",            ["response_model"], ["response_model", "响应模型"],     "tutorial/response-model"),
    ("kb", "怎么设置接口返回的状态码？",              ["status_code"],   ["状态码", "status_code"],          "tutorial/response-status-code"),
    ("kb", "怎么抛出自定义的 HTTP 错误？",            ["HTTPException"], ["HTTPException", "异常"],          "tutorial/handling-errors"),
    ("kb", "怎么把一个类当作依赖使用？",              ["Depends"],       ["依赖", "Depends"],                "tutorial/dependencies/classes-as-dependencies"),
    ("kb", "带 yield 的依赖是怎么工作的？",           ["yield"],         ["yield"],                         "tutorial/dependencies/dependencies-with-yield"),
    ("kb", "怎么让接口在返回后继续执行后台任务？",      ["BackgroundTasks"], ["后台任务", "BackgroundTasks"],  "tutorial/background-tasks"),
    ("kb", "FastAPI 里怎么做数据库连接？",            ["Session"],       ["数据库", "Session", "SQLModel"],   "tutorial/sql-databases"),
    ("kb", "怎么给 FastAPI 接口写测试？",             ["TestClient"],    ["测试", "TestClient"],             "tutorial/testing"),
    ("kb", "怎么实现 OAuth2 密码模式的登录？",        ["OAuth2"],        ["OAuth2", "令牌"],                 "tutorial/security/simple-oauth2"),
    ("kb", "怎么用 JWT 做令牌认证？",                 ["JWT"],           ["JWT", "令牌"],                    "tutorial/security/oauth2-jwt"),
    ("kb", "怎么配置跨域 CORS？",                     ["CORS"],          ["CORS", "跨域"],                   "tutorial/cors"),
    ("kb", "怎么挂载静态文件目录？",                   ["StaticFiles"],   ["静态", "StaticFiles"],            "tutorial/static-files"),
    ("kb", "怎么把请求体里的数据做部分更新？",          ["exclude_unset"], ["部分更新", "更新", "exclude_unset"], "tutorial/body-updates"),
    # ===== oob：知识库外，期望召回为空 =====
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
