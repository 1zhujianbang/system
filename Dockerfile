# 新闻智能体系统Docker镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建数据目录
RUN mkdir -p data/{raw_news,logs,tmp/{raw_news,deduped_news,extracted_events}} \
    && mkdir -p config/agents

# 设置环境变量
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 暴露端口
EXPOSE 8501
EXPOSE 6379

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.core import get_config_manager; get_config_manager()" || exit 1

# 启动命令
CMD ["streamlit", "run", "app.py"]
