#!/bin/bash

# MarketLens 部署脚本

set -e

echo "Starting MarketLens deployment..."

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# 创建必要的目录
echo "Creating directories..."
mkdir -p data/raw_news data/logs data/tmp/raw_news data/tmp/deduped_news data/tmp/extracted_events
mkdir -p config/agents
mkdir -p ssl

# 检查.env文件是否存在
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp config/.env.example .env
    echo "Please update the .env file with your configuration and run this script again."
    exit 1
fi

# 构建Docker镜像
echo "Building Docker images..."
docker-compose build

# 启动服务
echo "Starting services..."
docker-compose up -d

# 等待服务启动
echo "Waiting for services to start..."
sleep 30

# 检查服务状态
echo "Checking service status..."
docker-compose ps

echo "Deployment completed successfully!"
echo "Access the application at http://localhost:8501"