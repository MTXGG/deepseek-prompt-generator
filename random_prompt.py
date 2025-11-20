import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import json
import requests
import re
import os
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# 0. 配置信息 & API Key
# ==============================================================================
APP_THEME = dbc.themes.PULSE 
PORT_NUMBER = 9989

DEEPSEEK_API_BASE = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL_NAME = "deepseek-chat"
# ⚠️ 请在这里替换为您的 DeepSeek 密钥，或者使用 os.environ.get()
# 在生产环境中，强烈建议使用 os.environ.get('DEEPSEEK_API_KEY', 'YOUR_FALLBACK_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY') 
if not DEEPSEEK_API_KEY:
    # 如果环境变量未设置，则尝试从本地文件或返回错误
    print("警告：DEEPSEEK_API_KEY 环境变量未设置！")

DEEPSEEK_HEADERS = {
    'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
    'Content-Type': 'application/json'
}

# ==============================================================================
# 1. 通用 LLM 调用函数
# ==============================================================================

def llm_api_call(system_prompt: str, user_prompt: str, is_json_output: bool = True) -> Tuple[Optional[Any], Optional[str]]:
    """通用 LLM 调用函数，可用于文本生成或 JSON 格式化"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    payload = {"model": DEEPSEEK_MODEL_NAME, "messages": messages, "stream": False, "temperature": 0.7}
    
    try:
        response = requests.post(DEEPSEEK_API_BASE, headers=DEEPSEEK_HEADERS, json=payload, timeout=90)
        
        if response.status_code != 200:
            return None, f"API Error: Status {response.status_code}, {response.text}"
            
        data = response.json()
        model_reply_str = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        
        if not model_reply_str:
            return None, "API returned no content."

        if not is_json_output:
            # 返回纯文本 (用于创意生成阶段)
            return model_reply_str, None
        
        # 尝试解析 JSON (用于格式化阶段)
        temp_str = model_reply_str
        temp_str = re.sub(r'```json\s*|```', '', temp_str).strip()
        
        # 如果模型返回了纯文本而不是 JSON，或者 JSON 格式不正确，则会引发异常
        structured_data = json.loads(temp_str)
        return structured_data, None
        
    except Exception as e:
        # 在错误信息中显示截断后的原始返回内容，便于调试
        return None, f"API 异常或 JSON 解析错误: {e}. 原始返回: {model_reply_str[:100]}..."

# ==============================================================================
# 2. 第一次调用：AI 创意生成
# ==============================================================================

def get_creative_system_prompt(style: str) -> str:
    """根据风格生成定制化的创意系统提示词，以整合用户输入。"""
    
    base_prompt = (
        "你是一位专业的图像生成提示词灵感大师。你的任务是根据用户提供的**核心主题**和下方**风格要求**，"
        "生成一个**高质量、细节丰富、富有想象力的中文自然语言提示词**。"
        "**硬性要求：** 1. 人物主题必须严格围绕用户输入展开。2. **绝不能包含任何男性元素。** "
        "3. **必须只返回最终的中文提示词，不包含任何解释或额外的文字。**\n\n"
        "**🎯 描述焦点优先级：** 你的描述必须严格遵循以下层次结构："
        "**第一优先级 (主体)：** 极度详细地描述**主要人物、人物间的构图、姿势、动作和服装配饰**，确保主体形象的完整性。"
        # 🎯 关键更新：加强环境描写细节
        "**第二优先级 (环境)：** 在主体描述完成后，请**必须转而对环境/背景、光线/色彩和整体氛围进行同样高水平的细节描写**。环境不应是敷衍的模糊背景，而是**精心构造的、细节丰富的场景**。"
        "**注意：** 尽管背景必须细节丰富，但其功能始终是**烘托主体**，不得在视觉上削弱主要人物。"
    )
    
    # --- 30 种风格的具体要求 (省略大量重复内容，保证逻辑完整性) ---
    if style == "NORMAL":
        instructions = ("**【风格要求：1. 正常 SFW】** 描述必须是**日常、休闲、公共场所**的场景。姿势必须是**简单、静态、非诱惑性**的。服装必须**完全遮盖**。")
    elif style == "ARTISTIC":
        instructions = ("**【风格要求：2. 艺术 SFW】** 描述必须充满**戏剧性、叙事感和情绪深度**。姿势必须是**复杂、动态或具有强烈情感**的。")
    elif style == "GRAND_SFW":
        instructions = ("**【风格要求：3. 宏大 SFW (Grand Scale)】** 描述必须设定在**史诗级、超大规模**的 SFW 场景中。构图必须服从于**场景的震撼力**。")
    elif style == "R16_BREASTS_ONLY":
        instructions = ("**【风格要求：4. R16 擦边 (露胸/禁止露阴)】** 描述必须具有**诱惑性**。**硬性要求：** 必须明确包含**乳房/乳头（Breasts/Nipples）的暴露或特写**。**绝对禁止描述阴部。**")
    elif style == "R16_GENITALS_ONLY":
        instructions = ("**【风格要求：5. R16 擦边 (露阴/禁止露胸/极致反差)】** 上半身必须穿着**完全遮盖**的庄重/日常服装。**硬性要求：** 必须明确包含**阴部/生殖器（Genitals）的暴露或特写**。**绝对禁止描述乳房/乳头。**")
    elif style == "GRAND_NSFW_POSITIVE":
        instructions = ("**【风格要求：6. NSFW 宏大正面 (Majestic Positive Nude)】** 描述必须设定在**宏大、史诗、正面、神圣**的场景中。**硬性要求：** 必须包含**全身裸露（Full Nudity）**，强调**宏大构图和史诗感**。")
    elif style == "GRAND_NSFW_EXPLICIT":
        instructions = ("**【风格要求：7. NSFW 宏大 R-18 (Grand R-18 Explicit)】** 描述必须设定在**宏大、史诗、戏剧化**的场景中。**硬性要求：** 必须包含**乳头**和**阴户/生殖器**的清晰、露骨的描述。")
    elif style == "NSFW_EXPLICIT":
        instructions = ("**【风格要求：8. NSFW R-18 露骨 (Explicit)】** 描述必须是**明确的性主题或露骨的裸露场景**。**硬性要求：** 请在你的描述中**使用 R18 级别的中文关键词**。")
    elif style == "CRIME_CAPTURE":
        instructions = ("**【风格要求：9. R-18 犯罪 (被捕罪徒)】** **主题：** 强调被捕获、被约束。**硬性要求：** 必须包含**乳头和阴户**的明确描写，强调**束缚和无助**。")
    elif style == "CRIME_THIEF_ACTION":
        instructions = ("**【风格要求：10. R-18 犯罪 (夜色盗贼)】** **主题：** 强调在潜入、攀爬中的危险瞬间。**硬性要求：** 必须包含**乳头和阴户**的明确描写，强调**高风险、动态**的姿势。")
    elif style == "CRIME_RITUAL":
        instructions = ("**【风格要求：11. R-18 犯罪 (邪教仪式)】** **主题：** 强调秘密、非法、邪恶的宗教/邪教仪式。**硬性要求：** 必须包含**乳头和阴户**的明确描写，强调**恐怖、神秘、仪式感**。")
    elif style == "CRIME_HUMILIATION":
        instructions = ("**【风格要求：12. R-18 犯罪 (极致羞辱/侵犯类型主题)】** **主题：** 强调**屈服、绝对弱势、公开暴露或被迫顺从**的场景。**硬性要求：** 必须包含**乳头和阴户**的明确描写，强调**约束和绝对的暴露/弱势感**。")
    elif style == "ART_NUDE_NSFW":
        instructions = ("**【风格要求：13. 人体艺术 NSFW (Nude Art)】** 描述必须专注于**人体形态、雕塑感和光影美学**。**硬性要求：** 必须包含**全身裸露**，明确描述**乳头和阴户**，但**绝对排除性行为动作**。")
    elif style == "GRAVURE_R17":
        instructions = ("**【风格要求：14. 写真 R-17 (Suggestive Gravure)】** 描述必须是**高清晰度、商业级**的诱惑写真风格。**硬性要求：** 必须包含强烈暗示，但**绝对禁止描述乳头和阴户**。")
    elif style == "GRAVURE_NSFW":
        instructions = ("**【风格要求：15. 写真 NSFW (Explicit Gravure)】** 描述必须是**露骨、商业级**的成人写真风格。**硬性要求：** 必须包含**乳头和阴户**的明确描写，强调**高清晰度、湿润感和皮肤光泽**。")
    elif style == "COSPLAY_SFW":
        instructions = ("**【风格要求：16. Cosplay SFW】** 描述必须**忠实还原**一个虚构角色的服装、道具和妆容。**服装必须完全遮盖**。")
    elif style == "COSPLAY_R16":
        instructions = ("**【风格要求：17. Cosplay 擦边 (Suggestive Cos)】** 描述必须**忠实还原**虚构角色的服装，但通过**服装的修改、破损或湿透**来增加诱惑力。**硬性要求：** 必须包含强烈擦边暗示，但**绝对禁止描述乳头和阴户**。")
    elif style == "COSPLAY_NSFW":
        instructions = ("**【风格要求：18. Cosplay NSFW (Explicit Cos)】** 描述必须**忠实还原**虚构角色的身份，但在**场景或姿势中展现露骨的 R-18 内容**。**硬性要求：** 必须包含**乳头和阴户**的明确描写。")
    elif style == "UNIFORM_VIOLATION":
        instructions = ("**【风格要求：19. R-18 制服失控 (Uniform Violation)】** **主题：** 强调制服被**撕裂、弄脏、或解开**。**硬性要求：** 必须包含**乳头和阴户**的明确描写。")
    elif style == "WET_OILY_FOCUS":
        instructions = ("**【风格要求：20. R-18 湿身/油光特写 (Wet & Oily Focus)】** **主题：** 纯粹聚焦于**水、油、汗液**在皮肤表面流淌的效果。**硬性要求：** 必须包含**乳头和阴户**的明确描写。")
    elif style == "MYTH_EXPLICIT":
        instructions = ("**【风格要求：21. R-18 神话/古典 (Mythology Explicit)】** **主题：** 设定在**古典、神话**背景下，人物必须是**神祇、圣徒**。**硬性要求：** 必须包含**乳头和阴户**的明确描写。")
    elif style == "VOYEUR_UNAWARE":
        instructions = ("**【风格要求：22. R-18 偷窥视角 (Voyeuristic View)】** **主题：** 强调从**隐蔽、狭窄**的角度捕捉到的**被观察者毫不知情**的私人瞬间。**硬性要求：** 必须包含**乳头和阴户**的明确描写。")
    elif style == "MISTY_WATER_NUDE":
        instructions = ("**【风格要求：23. R-18 雾气弥漫/水景裸体 (Misty/Water Nude)】** **主题：** 专注于**柔和、扩散光和雾气/水汽**对裸体身体的柔化效果。**硬性要求：** 必须包含**乳头和阴户**的明确描写，强调**柔焦和水珠**。")
    elif style == "GOTHIC_ROMANTIC_NUDE":
        instructions = ("**【风格要求：24. R-18 哥特式浪漫裸体 (Gothic Romantic Nude)】** **主题：** 强调**黑暗、忧郁、古典和维多利亚时期**的美学。**硬性要求：** 必须包含**乳头和阴户**的明确描写，聚焦于**深色调和强烈的明暗对比**。")
    elif style == "MINIMALIST_FORM_NUDE":
        instructions = ("**【风格要求：25. R-18 极简主义形态 (Minimalist Form Nude)】** **主题：** 将人体视为**抽象雕塑**，强调**纯粹的线条、几何形状和光影构成**。**硬性要求：** 必须包含**乳头和阴户**的明确描写，强调**锐利的边缘、强烈的明暗对比**。")
    elif style == "NUDE_SOCIETY_NORMAL":
        instructions = ("**【风格要求：26. R-18 裸体社会 (Nude Society Normalcy)】** **主题：** 描绘一个**没有衣物**的社会中的**日常公共场景**。**硬性要求：** 必须包含**乳头和阴户**的明确描写，但强调**写实、日常、社会性**的氛围。")
    elif style == "FASHION_NORMAL":
        instructions = ("**【风格要求：27. 时尚 正常 (Commercial Fashion)】** **主题：** 专注于**高品质的商业/日常服装**展示。**硬性要求：** SFW，服装**完全遮盖**，将焦点置于服装本身。")
    elif style == "FASHION_SFW":
        instructions = ("**【风格要求：28. 时尚 艺术/高定 (Avant-Garde Fashion)】** **主题：** 专注于**前卫、概念性、高定艺术服装**的展示。**硬性要求：** SFW，服装**完全遮盖**，艺术性为核心。")
    elif style == "FASHION_R16":
        instructions = ("**【风格要求：29. 时尚 擦边 (Suggestive Fashion)】** **主题：** 专注于**内衣、泳装或极度透视**的高级时装展示。**硬性要求：** 擦边 R16，**绝对禁止描述乳头和阴户**，但暗示性极强。")
    elif style == "FASHION_NSFW":
        instructions = ("**【风格要求：30. 时尚 NSFW (Explicit Fashion)】** **主题：** 专注于**高度概念性、露骨的时尚大片**。**硬性要求：** 必须包含**乳头和阴户**的明确描写，将**时尚的艺术表现力与 R-18 元素**结合。")
    else:
        instructions = "未知风格，请使用正常 SFW 风格。"
        
    return base_prompt + instructions

def ai_generate_raw_prompt(style: str, user_theme: str) -> Tuple[Optional[str], Optional[str]]:
    """调用 LLM 生成高细节的中文原始提示词"""
    system_prompt = get_creative_system_prompt(style)
    user_prompt = f"用户核心主题：【{user_theme}】。请根据此主题和系统要求，立即开始生成提示词。"
    raw_prompt, error = llm_api_call(system_prompt, user_prompt, is_json_output=False)
    return raw_prompt, error

# ==============================================================================
# 3. 第二次调用：DeepSeek 最终格式化
# ==============================================================================

SYSTEM_PROMPT_FORMATTING = (
    "你是一位专业的 NovelAI/Stable Diffusion 提示词优化大师。你的任务是根据用户提供的**中文原始提示词**，进行**最终格式化**。"
    "核心任务：将提供的中文描述转化为 Danbooru 风格标签串和流畅的英文自然语言描述，并生成一套专业的英文负面提示词。"
    "1. **标签串 (final_tag)：** 必须是 Danbooru 标签，包含质量标签（如 `masterpiece, best quality, ultra detailed`）和背景标签。**必须添加所有细节标签**。"
    "2. **负面提示词 (final_negative)：** 必须是完整、专业的英文负面提示词列表，**必须包含 no males/boys 等排除男性元素的标签**。"
    "你必须以一个**纯 JSON 格式**的字符串作为最终回复，**绝不添加任何额外的文字或解释**。"
    "JSON 结构必须包含以下五个键：'final_tag', 'final_natural', 'final_negative', 'final_chinese_natural', 和 'final_chinese_negative'。"
    "final_chinese_natural 的值必须是你对原始中文提示词的**准确扩写和润色**后的中文版本。"
)

def deepseek_format_prompt(raw_chinese_prompt: str) -> Tuple[Optional[Dict], Optional[str]]:
    """调用 LLM 进行最终的 JSON 格式化"""
    return llm_api_call(SYSTEM_PROMPT_FORMATTING, raw_chinese_prompt, is_json_output=True)


# ==============================================================================
# 4. Dash 应用布局 (保持不变)
# ==============================================================================

app = dash.Dash(__name__, external_stylesheets=[APP_THEME])
server = app.server # 必须保留，供 Gunicorn/Waitress 等 WSGI 服务器调用

def result_card(title, id_name, is_code=False):
    style = {"font-family": "monospace", "white-space": "pre-wrap"} if is_code else {}
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            html.Hr(),
            html.Div(id=id_name, children="点击按钮生成提示词...", style=style)
        ]),
        className="mb-4",
    )

app.layout = dbc.Container([
    html.H1("🌟 AI 提示词多风格生成器 (30 种模式)", className="text-center my-4"),
    html.P("【AI驱动】DeepSeek 模型将处理创意生成和专业格式化两个阶段。", className="text-center mb-4 text-muted"),

    dbc.Row([
        dbc.Col([
            html.Label("输入您的核心主题（例如：手持旗帜的女神，站在战场废墟上）:", className="fw-bold mb-2"),
            dcc.Textarea(
                id='user-theme-input',
                value='一位穿着紧身宇航服的女性，漂浮在太空中', # 默认示例
                placeholder='在此输入您对人物、数量、服装、背景等的核心要求...',
                style={'width': '100%', 'minHeight': 100, 'backgroundColor': '#f8f9fa'},
            ),
        ], md=12, className="mb-4"),
    ]),
    
    # 按钮区域 - 调整为八行布局 (30个按钮)
    html.H5("一、SFW / R16 风格 (日常、艺术、宏大、局部擦边)", className="mt-2 mb-2"),
    dbc.Row([
        dbc.Col(dbc.Button("1. SFW 正常", id="btn-normal", color="primary", className="w-100"), md=3),
        dbc.Col(dbc.Button("2. SFW 艺术", id="btn-art", color="info", className="w-100"), md=3),
        dbc.Col(dbc.Button("3. SFW 宏大场景", id="btn-grand-sfw", color="success", className="w-100"), md=3),
        dbc.Col(dbc.Button("4. R16 擦边 (露胸)", id="btn-r16-breasts", color="warning", className="w-100"), md=3),
    ], className="mb-4"),

    html.H5("二、R16 / R18 宏大/艺术风格", className="mt-2 mb-2"),
    dbc.Row([
        dbc.Col(dbc.Button("5. R16 擦边 (露阴)", id="btn-r16-genitals", color="warning", className="w-100"), md=3),
        dbc.Col(dbc.Button("6. NSFW 宏大正面", id="btn-grand-nsfw-positive", color="dark", className="w-100"), md=3),
        dbc.Col(dbc.Button("7. NSFW 宏大 R-18", id="btn-grand-nsfw-explicit", color="danger", className="w-100"), md=3),
        dbc.Col(dbc.Button("8. R-18 露骨", id="btn-nsfw-explicit", color="danger", className="w-100"), md=3), 
    ], className="mb-4"),

    html.H5("三、写真/ Cosplay 风格", className="mt-2 mb-2"),
    dbc.Row([
        dbc.Col(dbc.Button("9. R17 写真", id="btn-gravure-r17", color="info", className="w-100"), md=3),
        dbc.Col(dbc.Button("10. NSFW 写真", id="btn-gravure-nsfw", color="danger", className="w-100"), md=3),
        dbc.Col(dbc.Button("11. Cosplay SFW", id="btn-cosplay-sfw", color="primary", className="w-100"), md=3),
        dbc.Col(dbc.Button("12. Cosplay 擦边", id="btn-cosplay-r16", color="warning", className="w-100"), md=3),
    ], className="mb-4"),

    html.H5("四、R-18 艺术美学风格 (形态、光影、环境)", className="mt-2 mb-2"),
    dbc.Row([
        dbc.Col(dbc.Button("13. 人体艺术 NSFW", id="btn-art-nude-nsfw", color="danger", className="w-100"), md=3),
        dbc.Col(dbc.Button("14. 雾气弥漫/水景裸体", id="btn-misty-water-nude", color="dark", className="w-100"), md=3), 
        dbc.Col(dbc.Button("15. 哥特式浪漫裸体", id="btn-gothic-romantic-nude", color="dark", className="w-100"), md=3), 
        dbc.Col(dbc.Button("16. 极简主义形态", id="btn-minimalist-form-nude", color="dark", className="w-100"), md=3), 
    ], className="mb-4"),

    html.H5("五、R-18 细分题材 (制服、偷窥、社会)", className="mt-2 mb-2"),
    dbc.Row([
        dbc.Col(dbc.Button("17. 制服失控", id="btn-uniform-violation", color="danger", className="w-100"), md=3),
        dbc.Col(dbc.Button("18. 湿身/油光特写", id="btn-wet-oily-focus", color="dark", className="w-100"), md=3),
        dbc.Col(dbc.Button("19. 神话/古典 R-18", id="btn-myth-explicit", color="dark", className="w-100"), md=3),
        dbc.Col(dbc.Button("20. 偷窥视角", id="btn-voyeur-unaware", color="secondary", className="w-100"), md=3),
    ], className="mb-4"),
    
    html.H5("六、时尚模特/社会题材 (SFW 到 R-18)", className="mt-2 mb-2"),
    dbc.Row([
        dbc.Col(dbc.Button("21. 裸体社会", id="btn-nude-society-normal", color="secondary", className="w-100"), md=3), 
        dbc.Col(dbc.Button("22. 时尚 正常", id="btn-fashion-normal", color="primary", className="w-100"), md=3), 
        dbc.Col(dbc.Button("23. 时尚 艺术/高定", id="btn-fashion-sfw", color="info", className="w-100"), md=3), 
        dbc.Col(dbc.Button("24. 时尚 擦边", id="btn-fashion-r16", color="warning", className="w-100"), md=3), 
    ], className="mb-4"),

    html.H5("七、R-18 Cosplay / 时尚 / 犯罪风格", className="mt-2 mb-2"),
    dbc.Row([
        dbc.Col(dbc.Button("25. Cosplay NSFW", id="btn-cosplay-nsfw", color="danger", className="w-100"), md=3),
        dbc.Col(dbc.Button("26. 时尚 NSFW", id="btn-fashion-nsfw", color="danger", className="w-100"), md=3), 
        dbc.Col(dbc.Button("27. 犯罪 (被捕罪徒)", id="btn-crime-capture", color="secondary", className="w-100"), md=3),
        dbc.Col(dbc.Button("28. 犯罪 (夜色盗贼)", id="btn-crime-thief", color="secondary", className="w-100"), md=3),
    ], className="mb-4"),

    html.H5("八、R-18 犯罪叙事风格 (二)", className="mt-2 mb-2"),
    dbc.Row([
        dbc.Col(dbc.Button("29. 犯罪 (邪教仪式)", id="btn-crime-ritual", color="secondary", className="w-100"), md=3),
        dbc.Col(dbc.Button("30. 犯罪 (极致羞辱)", id="btn-crime-humiliation", color="secondary", className="w-100"), md=3),
    ], className="mb-5"),


    html.Hr(),
    html.H3(id="result-title", children="等待生成...", className="text-center my-4"),

    # 结果展示区域
    dbc.Row([
        dbc.Col(result_card("原始 AI 创意描述 (First Pass)", "output-raw-prompt", is_code=False), md=12),
    ]),
    dbc.Row([
        dbc.Col(result_card("标签串 (Final Danbooru Tags)", "output-tag", is_code=True), md=12),
    ]),
    dbc.Row([
        dbc.Col(result_card("英文自然语言描述 (English Natural Prompt)", "output-natural"), md=6),
        dbc.Col(result_card("中文润色描述 (Chinese Refined Prompt)", "output-chinese-natural"), md=6),
    ]),
    dbc.Row([
        dbc.Col(result_card("负面提示词 (Negative Prompt)", "output-negative", is_code=True), md=12),
    ]),

    dcc.Store(id='style-store', data=None),
    dcc.Loading(id="loading-output", children=html.Div(id="loading-indicator"), type="circle"),
    html.Div(id="dummy-output", style={'display': 'none'})
], fluid=True, className="p-4")

# ==============================================================================
# 5. Dash 回调函数 (保持不变)
# ==============================================================================

# (回调函数逻辑与前文一致，省略以避免重复，但实际部署需包含全部回调函数)
@callback(
    [Output('style-store', 'data'),
     Output('dummy-output', 'children')],
    [Input('btn-normal', 'n_clicks'), Input('btn-art', 'n_clicks'), Input('btn-grand-sfw', 'n_clicks'), Input('btn-r16-breasts', 'n_clicks'),
     Input('btn-r16-genitals', 'n_clicks'), Input('btn-grand-nsfw-positive', 'n_clicks'), Input('btn-grand-nsfw-explicit', 'n_clicks'), Input('btn-nsfw-explicit', 'n_clicks'),
     Input('btn-crime-capture', 'n_clicks'), Input('btn-crime-thief', 'n_clicks'), Input('btn-crime-ritual', 'n_clicks'), Input('btn-crime-humiliation', 'n_clicks'),
     Input('btn-art-nude-nsfw', 'n_clicks'), Input('btn-gravure-r17', 'n_clicks'), Input('btn-gravure-nsfw', 'n_clicks'), Input('btn-cosplay-sfw', 'n_clicks'), 
     Input('btn-cosplay-r16', 'n_clicks'), Input('btn-cosplay-nsfw', 'n_clicks'), Input('btn-uniform-violation', 'n_clicks'), Input('btn-wet-oily-focus', 'n_clicks'),
     Input('btn-myth-explicit', 'n_clicks'), Input('btn-voyeur-unaware', 'n_clicks'),
     Input('btn-misty-water-nude', 'n_clicks'), Input('btn-gothic-romantic-nude', 'n_clicks'), Input('btn-minimalist-form-nude', 'n_clicks'),
     Input('btn-nude-society-normal', 'n_clicks'), Input('btn-fashion-normal', 'n_clicks'), Input('btn-fashion-sfw', 'n_clicks'),
     Input('btn-fashion-r16', 'n_clicks'), Input('btn-fashion-nsfw', 'n_clicks')]
)
def store_style_selection(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    style_map = {
        'btn-normal': "NORMAL", 'btn-art': "ARTISTIC", 'btn-grand-sfw': "GRAND_SFW", 'btn-r16-breasts': "R16_BREASTS_ONLY",
        'btn-r16-genitals': "R16_GENITALS_ONLY", 'btn-grand-nsfw-positive': "GRAND_NSFW_POSITIVE", 'btn-grand-nsfw-explicit': "GRAND_NSFW_EXPLICIT", 'btn-nsfw-explicit': "NSFW_EXPLICIT",
        'btn-crime-capture': "CRIME_CAPTURE", 'btn-crime-thief': "CRIME_THIEF_ACTION", 'btn-crime-ritual': "CRIME_RITUAL", 'btn-crime-humiliation': "CRIME_HUMILIATION",
        'btn-art-nude-nsfw': "ART_NUDE_NSFW", 'btn-gravure-r17': "GRAVURE_R17", 'btn-gravure-nsfw': "GRAVURE_NSFW", 'btn-cosplay-sfw': "COSPLAY_SFW",
        'btn-cosplay-r16': "COSPLAY_R16", 'btn-cosplay-nsfw': "COSPLAY_NSFW",
        'btn-uniform-violation': "UNIFORM_VIOLATION", 'btn-wet-oily-focus': "WET_OILY_FOCUS", 'btn-myth-explicit': "MYTH_EXPLICIT", 'btn-voyeur-unaware': "VOYEUR_UNAWARE",
        'btn-misty-water-nude': "MISTY_WATER_NUDE", 'btn-gothic-romantic-nude': "GOTHIC_ROMANTIC_NUDE", 'btn-minimalist-form-nude': "MINIMALIST_FORM_NUDE",
        'btn-nude-society-normal': "NUDE_SOCIETY_NORMAL", 'btn-fashion-normal': "FASHION_NORMAL", 'btn-fashion-sfw': "FASHION_SFW",
        'btn-fashion-r16': "FASHION_R16", 'btn-fashion-nsfw': "FASHION_NSFW"
    }
    selected_style = style_map.get(button_id)
    
    if selected_style:
        return selected_style, ""
    return dash.no_update, ""


@callback(
    [Output('result-title', 'children'),
     Output('output-raw-prompt', 'children'),
     Output('output-tag', 'children'),
     Output('output-natural', 'children'),
     Output('output-chinese-natural', 'children'),
     Output('output-negative', 'children')],
    [Input('style-store', 'data')],
    [State('user-theme-input', 'value')]
)
def generate_and_display_prompt(selected_style, user_theme):
    
    if not selected_style:
        return "等待生成...", "", "", "", "", ""
    
    if not user_theme or not user_theme.strip():
        error_msg = "❌ 请在上方文本框中输入您的核心主题描述！"
        return error_msg, "N/A", "N/A", "N/A", "N/A", "N/A"
    
    title_text = f"⚙️ 正在执行【DeepSeek 创意生成】... (风格: {selected_style})"
    
    raw_chinese_prompt, gen_error = ai_generate_raw_prompt(selected_style, user_theme)
    
    if gen_error:
        error_msg = f"❌ AI 创意生成失败: {gen_error}"
        return error_msg, "N/A", "N/A", "N/A", "N/A", "N/A"
    
    title_text = f"⚙️ 正在执行【DeepSeek 专业格式化】..."
    
    structured_data, format_error = deepseek_format_prompt(raw_chinese_prompt)
    
    if format_error:
        error_msg = f"❌ DeepSeek 格式化失败: {format_error}"
        return error_msg, raw_chinese_prompt, "N/A", "N/A", "N/A", "N/A"

    final_tag = structured_data.get('final_tag', 'N/A')
    final_natural = structured_data.get('final_natural', 'N/A')
    final_chinese_natural = structured_data.get('final_chinese_natural', 'N/A')
    final_negative = structured_data.get('final_negative', 'N/A')
    
    final_title = f"✅ 最终提示词输出：{selected_style} 风格 (主题已整合)"
    
    return (
        final_title,
        raw_chinese_prompt,
        final_tag,
        final_natural,
        final_chinese_natural,
        final_negative
    )

# ==============================================================================
# 6. 运行应用
# ==============================================================================
if __name__ == '__main__':
    # 在本地运行的入口
    app.run(debug=True, host='0.0.0.0', port=PORT_NUMBER)
