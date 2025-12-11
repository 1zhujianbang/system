import streamlit as st

def load_openai_style():
    """注入模仿 OpenAI Platform 的 CSS 样式"""
    st.markdown("""
        <style>
        /* 全局字体与背景 */
        .stApp {
            font-family: 'Söhne', 'ui-sans-serif', 'system-ui', -apple-system, 'Segoe UI', Roboto, Ubuntu, Cantarell, 'Noto Sans', sans-serif;
            background-color: #ffffff;
            color: #0d0d0d;
        }
        
        /* 侧边栏样式 */
        section[data-testid="stSidebar"] {
            background-color: #f9f9f9;
            border-right: 1px solid #e5e5e5;
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* 隐藏 Streamlit 默认头部装饰 */
        header[data-testid="stHeader"] {
            background-color: transparent;
        }
        
        /* 导航 Radio 按钮改造 */
        .stRadio > label {
            display: none; /* 隐藏标题 */
        }
        
        div[role="radiogroup"] > label {
            background-color: transparent !important;
            border: none;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.2rem;
            border-radius: 6px;
            color: #6e6e80;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        div[role="radiogroup"] > label:hover {
            background-color: #ececf1 !important;
            color: #0d0d0d;
        }
        
        /* 选中状态 */
        div[role="radiogroup"] > label[data-checked="true"] {
            background-color: #ececf1 !important;
            color: #0d0d0d;
            font-weight: 600;
        }

        /* 标题样式 */
        h1, h2, h3 {
            font-family: 'Söhne', sans-serif;
            letter-spacing: -0.01em;
            color: #202123;
        }
        
        /* 按钮样式 - Primary (模仿 OpenAI 黑色/绿色按钮) */
        .stButton > button {
            border-radius: 6px;
            border: 1px solid #e5e5e5;
            background-color: #ffffff;
            color: #0d0d0d;
            font-weight: 500;
            padding: 0.5rem 1rem;
            transition: all 0.1s ease;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        .stButton > button:hover {
            border-color: #d1d1d1;
            background-color: #f7f7f8;
            color: #0d0d0d;
        }
        
        .stButton > button:active {
            background-color: #f0f0f1;
        }

        /* 特定 Primary 按钮覆盖 (如果有特定的 type='primary') */
        .stButton > button[kind="primary"] {
            background-color: #10a37f;
            color: white;
            border: none;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #1a7f64;
        }

        /* 卡片/容器样式 */
        div[data-testid="stMetric"], div[data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #e5e5e5;
            border-radius: 6px;
            padding: 1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        /* 调整 Metric 样式 */
        div[data-testid="stMetricLabel"] {
            font-size: 0.875rem;
            color: #6e6e80;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 600;
            color: #202123;
        }

        /* 输入框样式 */
        .stTextInput > div > div > input, .stTextArea > div > div > textarea {
            border-radius: 6px;
            border-color: #e5e5e5;
            color: #0d0d0d;
        }
        .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
            border-color: #10a37f;
            box-shadow: 0 0 0 1px #10a37f;
        }

        /* 现代化卡片样式 */
        .modern-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #FFFFFF 100%);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            color: white;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .modern-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        }

        .feature-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.75rem 0;
            border: 1px solid #e5e5e5;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: all 0.2s ease;
        }

        .feature-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            border-color: #10a37f;
        }

        /* 渐变按钮样式 */
        .gradient-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        .gradient-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
        }

        /* 响应式网格布局 */
        .responsive-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 1rem 0;
        }

        /* 状态指示器 */
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }

        .status-online {
            background-color: #10a37f;
            box-shadow: 0 0 8px rgba(16, 163, 127, 0.5);
        }

        .status-offline {
            background-color: #ef4444;
            box-shadow: 0 0 8px rgba(239, 68, 68, 0.5);
        }

        .status-warning {
            background-color: #f59e0b;
            box-shadow: 0 0 8px rgba(245, 158, 11, 0.5);
        }

        /* 改进的展开面板 */
        .custom-expander {
            border-radius: 12px;
            border: 1px solid #e5e5e5;
            overflow: hidden;
        }

        .custom-expander > div:first-child {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-bottom: 1px solid #e5e5e5;
            padding: 1rem 1.5rem;
            font-weight: 600;
            color: #334155;
        }

        /* 代码块美化 */
        .stCodeBlock {
            border-radius: 8px;
            border: 1px solid #e5e5e5;
            background-color: #f8fafc;
        }

        /* 表格样式优化 */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        /* 进度条美化 */
        .stProgress > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
        }

        /* 标签页样式 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            border: 1px solid #e5e5e5;
            background-color: #f8fafc;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: white;
            border-bottom-color: white;
        }

        /* 滚动容器优化 */
        .scroll-container {
            max-height: 400px;
            overflow-y: auto;
            border-radius: 8px;
            border: 1px solid #e5e5e5;
            padding: 1rem;
            background-color: #fafbfc;
        }

        /* 工具提示优化 */
        .tooltip {
            position: relative;
            display: inline-block;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        /* 动画效果 */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .fade-in {
            animation: fadeIn 0.3s ease-out;
        }

        /* 移动端适配 */
        @media (max-width: 768px) {
            .responsive-grid {
                grid-template-columns: 1fr;
                gap: 1rem;
            }

            .modern-card, .feature-card {
                padding: 1rem;
                margin: 0.25rem 0;
            }

            .stButton > button {
                width: 100%;
                margin-bottom: 0.5rem;
            }
        }

        </style>
    """, unsafe_allow_html=True)

def render_sidebar_header():
    """渲染侧边栏顶部 Logo 区域"""
    st.sidebar.markdown("""
        <div style="padding-bottom: 1.5rem; padding-left: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);">
                    <span style="color: white; font-weight: bold; font-size: 18px;">📰</span>
                </div>
                <div>
                    <div style="font-weight: 600; font-size: 1rem; color: #202123;">新闻智能体系统</div>
                    <div style="font-size: 0.75rem; color: #6e6e80;">v2.0.0</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_modern_card(title, content, icon="📊", color_class="modern-card"):
    """创建现代化的卡片组件"""
    card_html = f"""
        <div class="{color_class}">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem; margin-right: 0.75rem;">{icon}</span>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 600;">{title}</h3>
            </div>
            <div style="line-height: 1.6;">
                {content}
            </div>
        </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_feature_grid(features):
    """创建功能特性网格"""
    cols = st.columns(min(len(features), 3))
    for i, feature in enumerate(features):
        with cols[i % len(cols)]:
            create_modern_card(
                feature["title"],
                feature["description"],
                feature["icon"],
                "feature-card"
            )

def create_status_indicator(status, label):
    """创建状态指示器"""
    status_class = {
        "online": "status-online",
        "offline": "status-offline",
        "warning": "status-warning"
    }.get(status.lower(), "status-offline")

    indicator_html = f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span class="status-indicator {status_class}"></span>
            <span style="font-weight: 500; color: #374151;">{label}</span>
        </div>
    """
    st.markdown(indicator_html, unsafe_allow_html=True)

def create_responsive_layout(*contents, gap="1rem"):
    """创建响应式布局容器"""
    if len(contents) == 1:
        st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr; gap: {gap}; margin: 1rem 0;">
        """, unsafe_allow_html=True)
        contents[0]()
        st.markdown("</div>", unsafe_allow_html=True)
    elif len(contents) == 2:
        col1, col2 = st.columns(2)
        with col1:
            contents[0]()
        with col2:
            contents[1]()
    elif len(contents) == 3:
        col1, col2, col3 = st.columns(3)
        with col1:
            contents[0]()
        with col2:
            contents[1]()
        with col3:
            contents[2]()
    else:
        # 对于更多内容，使用网格布局
        cols = st.columns(min(len(contents), 4))
        for i, content in enumerate(contents):
            with cols[i % len(cols)]:
                content()

def create_scrollable_container(content_func, height="400px", title=""):
    """创建可滚动容器"""
    if title:
        st.subheader(title)

    scrollable_html = f"""
        <div class="scroll-container" style="max-height: {height};">
    """
    st.markdown(scrollable_html, unsafe_allow_html=True)
    content_func()
    st.markdown("</div>", unsafe_allow_html=True)

def apply_gradient_button():
    """为主要按钮应用渐变样式"""
    st.markdown("""
        <style>
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4) !important;
        }
        </style>
    """, unsafe_allow_html=True)

