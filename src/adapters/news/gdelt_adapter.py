"""
适配器层 - GDELT 数据源适配器

实现 NewsSource 端口，提供 GDELT 数据抓取功能。
"""
from __future__ import annotations

import asyncio
import csv
import io
import zipfile
from datetime import datetime, timezone
from typing import Optional, List, Dict, AsyncIterator
from urllib.parse import urlparse

import aiohttp

from ...ports.extraction import (
    NewsSource, NewsSourceType, NewsItem, FetchConfig, FetchResult
)
from ...infra import get_logger


class GDELTAdapter(NewsSource):
    """GDELT 数据源适配器"""

    BASE_URL = "http://data.gdeltproject.org/gdeltv2/"
    MASTER_FILE_LIST = "masterfilelist.txt"
    
    # GDELT事件数据文件后缀
    EVENTS_SUFFIX = ".export.CSV.zip"
    
    # GDELT GKG数据文件后缀
    GKG_SUFFIX = ".gkg.csv.zip"

    def __init__(self, name: str = "GDELT", timeout: int = 30):
        self._name = name
        self._timeout = timeout
        self._session = None
        self._logger = get_logger(__name__)

    @property
    def source_type(self) -> NewsSourceType:
        return NewsSourceType.CUSTOM

    @property
    def source_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        # GDELT数据源始终可用
        return True

    async def fetch(self, config: Optional[FetchConfig] = None) -> FetchResult:
        """抓取GDELT数据"""
        config = config or FetchConfig()
        items = []

        try:
            # 使用默认的ThreadedResolver避免Windows上的aiodns兼容性问题
            try:
                from aiohttp.resolver import ThreadedResolver
                resolver = ThreadedResolver()
            except ImportError:
                resolver = None
                
            connector = aiohttp.TCPConnector(resolver=resolver)
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # 获取最新的主文件列表
                master_list_url = f"{self.BASE_URL}{self.MASTER_FILE_LIST}"
                async with session.get(master_list_url) as response:
                    if response.status != 200:
                        text = await response.text()
                        return FetchResult(
                            items=[],
                            total_fetched=0,
                            success=False,
                            error=f"Failed to fetch master file list: {response.status} - {text}",
                            fetch_time=datetime.now(timezone.utc)
                        )

                    # 解析主文件列表内容
                    master_list_content = await response.text()
                    file_urls = self._parse_master_file_list(master_list_content, config)
                    
                    # 下载并解析最新的事件数据文件
                    for file_url in file_urls[:min(config.max_items, 5)]:  # 限制处理文件数量
                        try:
                            news_items = await self._fetch_and_parse_gdelt_file(session, file_url)
                            items.extend(news_items)
                        except Exception as e:
                            self._logger.warning(f"Failed to process GDELT file {file_url}: {e}")
                            continue

            return FetchResult(
                items=items,
                total_fetched=len(items),
                success=True,
                fetch_time=datetime.now(timezone.utc)
            )

        except Exception as e:
            self._logger.error(f"GDELT fetch error: {e}")
            return FetchResult(
                items=[],
                total_fetched=0,
                success=False,
                error=str(e),
                fetch_time=datetime.now(timezone.utc)
            )

    async def fetch_stream(self, config: Optional[FetchConfig] = None) -> AsyncIterator[NewsItem]:
        """流式抓取GDELT数据"""
        result = await self.fetch(config)
        for item in result.items:
            yield item

    def _parse_master_file_list(self, content: str, config: FetchConfig) -> List[str]:
        """解析主文件列表，返回符合条件的文件URL列表"""
        file_urls = []
        
        # 按行分割内容
        lines = content.strip().split('\n')
        
        # 反向遍历以获取最新的文件
        for line in reversed(lines):
            parts = line.strip().split()
            if len(parts) >= 3:
                # 第三个字段是文件URL
                file_url = parts[2]
                
                # 检查是否是我们需要的事件数据文件
                if file_url.endswith(self.EVENTS_SUFFIX):
                    file_urls.append(file_url)
                    
                    # 如果指定了日期范围，检查文件名中的日期
                    if config.from_date or config.to_date:
                        # 从URL中提取日期
                        try:
                            # URL格式类似: http://data.gdeltproject.org/gdeltv2/20230101000000.export.CSV.zip
                            filename = urlparse(file_url).path.split('/')[-1]
                            date_str = filename.split('.')[0]  # 提取日期部分
                            file_date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                            
                            # 检查日期范围
                            if config.from_date and file_date < config.from_date:
                                continue
                            if config.to_date and file_date > config.to_date:
                                continue
                        except Exception:
                            # 如果无法解析日期，跳过日期检查
                            pass
                    
                    # 如果达到最大数量限制，停止
                    if len(file_urls) >= config.max_items:
                        break
        
        return file_urls

    async def _fetch_and_parse_gdelt_file(self, session: aiohttp.ClientSession, file_url: str) -> List[NewsItem]:
        """下载并解析GDELT数据文件"""
        items = []
        
        async with session.get(file_url) as response:
            if response.status != 200:
                self._logger.warning(f"Failed to download GDELT file {file_url}: {response.status}")
                return items

            # 读取ZIP文件内容
            zip_content = await response.read()
            
            # 解压ZIP文件并解析CSV
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                # 获取第一个CSV文件
                csv_filename = zip_file.namelist()[0]
                with zip_file.open(csv_filename) as csv_file:
                    # 使用csv.reader解析
                    decoded_file = io.TextIOWrapper(csv_file, encoding='utf-8')
                    csv_reader = csv.reader(decoded_file, delimiter='\t')
                    
                    # 解析每一行数据
                    for row_num, row in enumerate(csv_reader):
                        try:
                            # GDELT v2事件数据有61列，我们需要提取关键信息
                            # 参考GDELT文档，关键列包括：
                            # 0: GLOBALEVENTID
                            # 1: SQLDATE
                            # 3: Actor1Name
                            # 4: Actor2Name
                            # 5: Actor1CountryCode
                            # 6: Actor2CountryCode
                            # 17: EventCode
                            # 27: GoldsteinScale
                            # 29: NumMentions
                            # 30: NumSources
                            # 31: NumArticles
                            # 34: AvgTone
                            # 39: Actor1Geo_FullName
                            # 40: Actor2Geo_FullName
                            # 53: SOURCEURL
                            
                            if len(row) < 54:
                                continue
                                
                            event_id = row[0]
                            sql_date = row[1]
                            actor1_name = row[3]
                            actor2_name = row[4]
                            event_code = row[17]
                            source_url = row[53]
                            
                            # 构造新闻条目
                            title = f"{actor1_name} -> {event_code} -> {actor2_name}" if actor1_name and actor2_name else f"GDELT Event {event_id}"
                            content = f"Event ID: {event_id}\nDate: {sql_date}\nActors: {actor1_name} -> {actor2_name}\nEvent Code: {event_code}\nSource: {source_url}"
                            
                            # 解析日期
                            published_at = None
                            try:
                                published_at = datetime.strptime(sql_date, "%Y%m%d")
                            except Exception:
                                pass
                            
                            news_item = NewsItem(
                                id=event_id or source_url or f"gdelt-{row_num}",
                                title=title,
                                content=content,
                                source_name=self._name,
                                source_url=source_url,
                                published_at=published_at,
                                author=actor1_name,
                                category=event_code,
                                language="en",  # GDELT数据主要是英文
                                raw_data={
                                    "actor1": actor1_name,
                                    "actor2": actor2_name,
                                    "event_code": event_code,
                                    "date": sql_date,
                                    "url": source_url
                                }
                            )
                            
                            items.append(news_item)
                        except Exception as e:
                            self._logger.warning(f"Failed to parse GDELT row {row_num}: {e}")
                            continue
        
        return items