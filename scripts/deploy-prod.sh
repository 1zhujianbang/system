#!/bin/bash

# MarketLens 生产环境部署脚本

set -e

echo "Starting MarketLens production deployment..."

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

# 检查SSL证书是否存在
if [ ! -f ssl/nginx.crt ] || [ ! -f ssl/nginx.key ]; then
    echo "Generating self-signed SSL certificates..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/nginx.key -out ssl/nginx.crt \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
fi

# 检查.env文件是否存在
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp config/.env.example .env
    echo "Please update the .env file with your production configuration and run this script again."
    exit 1
fi

# 构建Docker镜像
echo "Building Docker images..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# 启动生产环境服务
echo "Starting production services..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 等待服务启动
echo "Waiting for services to start..."
sleep 60

# 检查服务状态
echo "Checking service status..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps

# 初始化数据库
echo "Initializing database..."
docker-compose -f docker-compose.yml -f docker-compose.prod.yml exec neo4j neo4j-admin set-initial-password password

echo "Production deployment completed successfully!"
echo "Access the application at https://localhost"