"""
适配器层 - GDELT API 适配器

实现 NewsSource 端口，提供 GDELT 数据源接入功能。
支持 GDELT 1.0 和 2.0 事件数据、GKG（Global Knowledge Graph）和mentions数据的获取。
"""
from __future__ import annotations

import os
import json
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Any, List, Dict, AsyncIterator
from pathlib import Path

from ...ports.extraction import (
    NewsSource, NewsSourceType, NewsSourcePool,
    NewsItem, FetchConfig, FetchResult
)
from ...infra import get_logger


class GDELTAdapter(NewsSource):
    """GDELT 数据源适配器"""
    
    def __init__(
        self,
        name: str = "GDELT",
        timeout: int = 300,  # GDELT 查询可能需要较长时间
        version: int = 2,  # 默认使用GDELT 2.0
        api_key: Optional[str] = None
    ):
        self._name = name
        self._timeout = timeout
        self._version = version
        self._api_key = api_key or os.getenv("GDELT_API_KEY", "")
        self._logger = get_logger(__name__)
        
    @property
    def source_type(self) -> NewsSourceType:
        return NewsSourceType.GDELT
    
    @property
    def source_name(self) -> str:
        return self._name
    
    def is_available(self) -> bool:
        # GDELT 作为公开数据源，不需要API密钥
        return True
    
    async def fetch(self, config: Optional[FetchConfig] = None) -> FetchResult:
        """获取GDELT数据"""
        config = config or FetchConfig()
        items = []
        
        try:
            # 根据配置决定获取哪种数据
            if config.category and "gkg" in config.category.lower():
                items = await self._fetch_gkg_data(config)
            elif config.category and "mentions" in config.category.lower():
                items = await self._fetch_mentions_data(config)
            else:
                items = await self._fetch_events_data(config)
                
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
        """流式获取GDELT数据"""
        result = await self.fetch(config)
        for item in result.items:
            yield item
    
    async def _fetch_events_data(self, config: FetchConfig) -> List[NewsItem]:
        """获取GDELT事件数据"""
        try:
            # 导入gdelt库
            import gdelt
            
            # 创建GDELT实例
            gd = gdelt.gdelt(version=self._version)
            
            # 解析日期范围
            date_range = self._parse_date_range(config)
            
            # 从extra配置中获取GDELT特定参数
            coverage_param = config.extra.get('coverage', config.max_items == 0)
            translation_param = config.extra.get('translation', None)
            output_format = config.extra.get('output', 'df')
            
            # 执行查询
            df = gd.Search(
                date=date_range,
                table='events',
                coverage=coverage_param,
                translation=translation_param,
                output=output_format
            )
            
            if df is None or df.empty:
                self._logger.info("No GDELT events data found for the given criteria")
                return []
            
            # 根据关键词过滤数据
            if config.keywords:
                df = self._filter_by_keywords(df, config.keywords)
            
            # 限制返回数量
            if config.max_items > 0:
                df = df.head(config.max_items)
            
            # 转换为NewsItem
            items = []
            for _, row in df.iterrows():
                item = self._convert_event_to_newsitem(row)
                if item:
                    items.append(item)
            
            return items
            
        except ImportError:
            self._logger.error("gdelt library not installed. Please install with: pip install gdelt")
            return self._fetch_events_data_fallback(config)  # 使用备用方法
        except Exception as e:
            self._logger.error(f"Error fetching GDELT events data: {e}")
            return []

    async def _fetch_gkg_data(self, config: FetchConfig) -> List[NewsItem]:
        """获取GDELT GKG数据"""
        try:
            # 导入gdelt库
            import gdelt
            
            # 创建GDELT实例
            gd = gdelt.gdelt(version=self._version)
            
            # 解析日期范围
            date_range = self._parse_date_range(config)
            
            # 从extra配置中获取GDELT特定参数
            coverage_param = config.extra.get('coverage', config.max_items == 0)
            translation_param = config.extra.get('translation', None)
            output_format = config.extra.get('output', 'df')
            
            # 执行查询
            df = gd.Search(
                date=date_range,
                table='gkg',
                coverage=coverage_param,
                translation=translation_param,
                output=output_format
            )
            
            if df is None or df.empty:
                self._logger.info("No GDELT GKG data found for the given criteria")
                return []
            
            # 根据关键词过滤数据
            if config.keywords:
                df = self._filter_by_keywords_gkg(df, config.keywords)
            
            # 限制返回数量
            if config.max_items > 0:
                df = df.head(config.max_items)
            
            # 转换为NewsItem
            items = []
            for _, row in df.iterrows():
                item = self._convert_gkg_to_newsitem(row)
                if item:
                    items.append(item)
            
            return items
            
        except ImportError:
            self._logger.error("gdelt library not installed. Please install with: pip install gdelt")
            return self._fetch_gkg_data_fallback(config)  # 使用备用方法
        except Exception as e:
            self._logger.error(f"Error fetching GDELT GKG data: {e}")
            return []

    async def _fetch_mentions_data(self, config: FetchConfig) -> List[NewsItem]:
        """获取GDELT mentions数据"""
        try:
            # 导入gdelt库
            import gdelt
            
            # 创建GDELT实例
            gd = gdelt.gdelt(version=self._version)
            
            # 解析日期范围
            date_range = self._parse_date_range(config)
            
            # 从extra配置中获取GDELT特定参数
            coverage_param = config.extra.get('coverage', config.max_items == 0)
            translation_param = config.extra.get('translation', None)
            output_format = config.extra.get('output', 'df')
            
            # 执行查询
            df = gd.Search(
                date=date_range,
                table='mentions',
                coverage=coverage_param,
                translation=translation_param,
                output=output_format
            )
            
            if df is None or df.empty:
                self._logger.info("No GDELT mentions data found for the given criteria")
                return []
            
            # 根据关键词过滤数据
            if config.keywords:
                df = self._filter_by_keywords_mentions(df, config.keywords)
            
            # 限制返回数量
            if config.max_items > 0:
                df = df.head(config.max_items)
            
            # 转换为NewsItem
            items = []
            for _, row in df.iterrows():
                item = self._convert_mention_to_newsitem(row)
                if item:
                    items.append(item)
            
            return items
            
        except ImportError:
            self._logger.error("gdelt library not installed. Please install with: pip install gdelt")
            return []  # mentions没有备用方法
        except Exception as e:
            self._logger.error(f"Error fetching GDELT mentions data: {e}")
            return []
    
    def _parse_date_range(self, config: FetchConfig) -> str:
        """解析日期范围"""
        if config.from_date and config.to_date:
            # 格式化为GDELT库期望的格式
            start_date = config.from_date.strftime('%Y %m %d')
            end_date = config.to_date.strftime('%Y %m %d')
            return [start_date, end_date]
        elif config.from_date:
            return config.from_date.strftime('%Y %m %d')
        else:
            # 默认返回最近一天
            today = datetime.now()
            return today.strftime('%Y %m %d')
    
    def _filter_by_keywords(self, df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
        """根据关键词过滤事件数据"""
        filtered_df = df.copy()
        
        for keyword in keywords:
            # 检查参与者名称、事件代码、地点等字段
            mask = (
                filtered_df['Actor1Name'].str.contains(keyword, case=False, na=False) |
                filtered_df['Actor2Name'].str.contains(keyword, case=False, na=False) |
                filtered_df['EventCode'].str.contains(keyword, case=False, na=False) |
                filtered_df['ActionGeo_FullName'].str.contains(keyword, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
        
        return filtered_df
    
    def _filter_by_keywords_gkg(self, df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
        """根据关键词过滤GKG数据"""
        filtered_df = df.copy()
        
        for keyword in keywords:
            # 检查主题、人物、组织等字段
            mask = (
                df['V2Themes'].str.contains(keyword, case=False, na=False) |
                df['V2EnhancedThemes'].str.contains(keyword, case=False, na=False) |
                df['V2Persons'].str.contains(keyword, case=False, na=False) |
                df['V2EnhancedPersons'].str.contains(keyword, case=False, na=False) |
                df['V2Organizations'].str.contains(keyword, case=False, na=False) |
                df['V2EnhancedOrganizations'].str.contains(keyword, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
        
        return filtered_df
    
    def _filter_by_keywords_mentions(self, df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
        """根据关键词过滤mentions数据"""
        filtered_df = df.copy()
        
        for keyword in keywords:
            # 检查提及的参与者、事件ID等字段
            mask = (
                df['Actor1Name'].str.contains(keyword, case=False, na=False) |
                df['Actor2Name'].str.contains(keyword, case=False, na=False) |
                df['MentionIdentifier'].str.contains(keyword, case=False, na=False)
            )
            filtered_df = filtered_df[mask]
        
        return filtered_df
    
    def _get_events_column_names(self) -> List[str]:
        """获取GDELT事件数据的列名"""
        return [
            'GLOBALEVENTID', 'SQLDATE', 'MonthYear', 'Year', 'FractionDate',
            'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
            'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
            'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
            'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
            'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
            'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
            'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode',
            'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources',
            'NumArticles', 'AvgTone', 'Actor1Geo_Type', 'Actor1Geo_FullName',
            'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_Lat',
            'Actor1Geo_Long', 'Actor1Geo_FeatureID', 'Actor2Geo_Type',
            'Actor2Geo_FullName', 'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code',
            'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID',
            'ActionGeo_Type', 'ActionGeo_FullName', 'ActionGeo_CountryCode',
            'ActionGeo_ADM1Code', 'ActionGeo_Lat', 'ActionGeo_Long',
            'ActionGeo_FeatureID', 'DATEADDED', 'SOURCEURL'
        ]
    
    def _get_gkg_column_names(self) -> List[str]:
        """获取GDELT GKG数据的列名"""
        return [
            'GKGRECORDID', 'DATE', 'SourceCollectionIdentifier',
            'SourceCommonName', 'DocumentIdentifier', 'V2Counts',
            'V2Themes', 'V2EnhancedThemes', 'V2Locations',
            'V2EnhancedLocations', 'V2Persons', 'V2EnhancedPersons',
            'V2Organizations', 'V2EnhancedOrganizations', 'V2Tone',
            'V2EnhancedDates', 'V2GCAM', 'V2SharingImage',
            'V2RelatedImages', 'V2AllNames', 'V2Amounts', 'V2EnhancedRootClust',
            'V2ParentLocations', 'V2EnhancedParentLocations',
            'V2ArticleSources', 'V2AllOrganizations', 'V2AllLocations'
        ]
    
    def _get_mentions_column_names(self) -> List[str]:
        """获取GDELT mentions数据的列名"""
        return [
            'GLOBALEVENTID', 'EventTimeDate', 'MentionTimeDate', 'MentionType',
            'MentionSourceName', 'MentionIdentifier', 'SentenceID', 'Actor1CharOffset',
            'Actor2CharOffset', 'ActionCharOffset', 'InRawHTML', 'Confidence',
            'MentionDocLen', 'MentionDocTone', 'MentionDocTranslationInfo',
            'Extras', 'SOURCEURL'
        ]
    
    def _convert_event_to_newsitem(self, row: pd.Series) -> Optional[NewsItem]:
        """将GDELT事件数据行转换为NewsItem"""
        try:
            # 提取关键信息
            global_event_id = str(row.get('GLOBALEVENTID', ''))
            sql_date = row.get('SQLDATE', '')
            actor1_name = row.get('Actor1Name', '')
            actor2_name = row.get('Actor2Name', '')
            event_code = row.get('EventCode', '')
            action_geo_name = row.get('ActionGeo_FullName', '')
            source_url = row.get('SOURCEURL', '')
            
            # 构建标题和内容
            title_parts = []
            if actor1_name:
                title_parts.append(actor1_name)
            if event_code:
                title_parts.append(f"事件代码: {event_code}")
            if actor2_name:
                title_parts.append(actor2_name)
            
            title = " ".join(title_parts) or f"GDELT事件 {global_event_id}"
            
            content_parts = [
                f"全球事件ID: {global_event_id}",
                f"日期: {sql_date}",
                f"参与者1: {actor1_name}",
                f"参与者2: {actor2_name}",
                f"事件代码: {event_code}",
                f"地点: {action_geo_name}",
                f"来源URL: {source_url}"
            ]
            
            content = "\n".join(content_parts)
            
            # 转换日期格式
            published_at = None
            if sql_date:
                try:
                    # 尝试不同的日期格式
                    if isinstance(sql_date, str) and len(sql_date) == 8:
                        published_at = datetime.strptime(str(sql_date), '%Y%m%d')
                    elif isinstance(sql_date, (int, float)):
                        date_str = str(int(sql_date))
                        if len(date_str) == 8:
                            published_at = datetime.strptime(date_str, '%Y%m%d')
                except:
                    pass
            
            return NewsItem(
                id=global_event_id,
                title=title,
                content=content,
                source_name=self._name,
                source_url=source_url,
                published_at=published_at,
                author=None,
                category="GDELT_EVENT",
                language="en",  # GDELT数据通常为英文
                raw_data=row.to_dict()
            )
        except Exception as e:
            self._logger.error(f"Error converting GDELT event to NewsItem: {e}")
            return None
    
    def _convert_gkg_to_newsitem(self, row: pd.Series) -> Optional[NewsItem]:
        """将GDELT GKG数据行转换为NewsItem"""
        try:
            gkg_record_id = str(row.get('GKGRECORDID', ''))
            date = row.get('DATE', '')
            source_name = row.get('SourceCommonName', '')
            document_url = row.get('DocumentIdentifier', '')
            themes = row.get('V2EnhancedThemes', '')
            persons = row.get('V2EnhancedPersons', '')
            organizations = row.get('V2EnhancedOrganizations', '')
            
            title = f"GKG记录 {gkg_record_id[:10]}..." if gkg_record_id else "GKG数据记录"
            
            content_parts = [
                f"GKG记录ID: {gkg_record_id}",
                f"日期: {date}",
                f"来源: {source_name}",
                f"主题: {themes}",
                f"人物: {persons}",
                f"组织: {organizations}",
                f"文档URL: {document_url}"
            ]
            
            content = "\n".join(content_parts)
            
            published_at = None
            if date:
                try:
                    published_at = datetime.strptime(str(date), '%Y%m%d')
                except:
                    pass
            
            return NewsItem(
                id=gkg_record_id,
                title=title,
                content=content,
                source_name=self._name,
                source_url=document_url,
                published_at=published_at,
                author=None,
                category="GDELT_GKG",
                language="en",
                raw_data=row.to_dict()
            )
        except Exception as e:
            self._logger.error(f"Error converting GDELT GKG to NewsItem: {e}")
            return None
    
    def _convert_mention_to_newsitem(self, row: pd.Series) -> Optional[NewsItem]:
        """将GDELT mentions数据行转换为NewsItem"""
        try:
            global_event_id = str(row.get('GLOBALEVENTID', ''))
            mention_time = row.get('MentionTimeDate', '')
            mention_source = row.get('MentionSourceName', '')
            mention_url = row.get('MentionIdentifier', '')
            confidence = row.get('Confidence', '')
            
            title = f"Mention - 事件 {global_event_id}" if global_event_id else "Mention记录"
            
            content_parts = [
                f"全球事件ID: {global_event_id}",
                f"提及时间: {mention_time}",
                f"提及来源: {mention_source}",
                f"提及URL: {mention_url}",
                f"置信度: {confidence}"
            ]
            
            content = "\n".join(content_parts)
            
            published_at = None
            if mention_time:
                try:
                    published_at = datetime.strptime(str(mention_time), '%Y%m%d%H%M%S')
                except:
                    try:
                        published_at = datetime.strptime(str(mention_time)[:8], '%Y%m%d')
                    except:
                        pass
            
            return NewsItem(
                id=f"mention_{global_event_id}",
                title=title,
                content=content,
                source_name=self._name,
                source_url=mention_url,
                published_at=published_at,
                author=None,
                category="GDELT_MENTION",
                language="en",
                raw_data=row.to_dict()
            )
        except Exception as e:
            self._logger.error(f"Error converting GDELT mention to NewsItem: {e}")
            return None
    
    async def _fetch_events_data_fallback(self, config: FetchConfig) -> List[NewsItem]:
        """获取GDELT事件数据的备用方法（使用原始URL）"""
        # 由于GDELT事件数据是公开的，我们可以直接从公开URL获取最新的数据
        try:
            import aiohttp
            
            # 获取最新的更新文件
            recent_events_url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(recent_events_url) as response:
                    if response.status != 200:
                        self._logger.error(f"Failed to get GDELT recent updates: {response.status}")
                        return []
                    
                    content = await response.text()
                    # 解析最新更新信息
                    lines = content.strip().split('\n')
                    if lines and len(lines) >= 1:
                        # 获取最新事件文件的URL
                        parts = lines[0].split()
                        if len(parts) >= 2:
                            latest_events_url = parts[1].strip()
                            
                            # 下载最新事件数据
                            async with session.get(latest_events_url) as events_response:
                                if events_response.status != 200:
                                    self._logger.error(f"Failed to get GDELT events: {events_response.status}")
                                    return []
                                
                                # GDELT数据是gzip压缩的tsv格式
                                import gzip
                                from io import StringIO
                                
                                # 读取压缩数据
                                content = await events_response.read()
                                decompressed_content = gzip.decompress(content).decode('utf-8')
                                
                                # 解析TSV数据
                                df = pd.read_csv(
                                    StringIO(decompressed_content),
                                    sep='\t',
                                    header=None,
                                    names=self._get_events_column_names()
                                )
                                
                                # 根据配置过滤数据
                                filtered_df = self._filter_events_data(df, config)
                                
                                # 转换为NewsItem
                                items = []
                                for _, row in filtered_df.iterrows():
                                    item = self._convert_event_to_newsitem(row)
                                    if item:
                                        items.append(item)
                                
                                return items
            
            return []
        except Exception as e:
            self._logger.error(f"Error fetching GDELT events data (fallback): {e}")
            return []

    async def _fetch_gkg_data_fallback(self, config: FetchConfig) -> List[NewsItem]:
        """获取GDELT GKG数据的备用方法（使用原始URL）"""
        try:
            import aiohttp
            
            # 获取GKG数据的URL（这里使用公开的GKG数据）
            gkg_url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(gkg_url) as response:
                    if response.status != 200:
                        self._logger.error(f"Failed to get GDELT GKG updates: {response.status}")
                        return []
                    
                    content = await response.text()
                    lines = content.strip().split('\n')
                    
                    # 寻找GKG相关的更新
                    for line in lines:
                        if 'gkg' in line.lower():
                            parts = line.split()
                            if len(parts) >= 2:
                                gkg_url = parts[1].strip()
                                
                                # 下载GKG数据
                                async with session.get(gkg_url) as gkg_response:
                                    if gkg_response.status != 200:
                                        continue
                                    
                                    import gzip
                                    from io import StringIO
                                    
                                    content = await gkg_response.read()
                                    decompressed_content = gzip.decompress(content).decode('utf-8')
                                    
                                    # 解析GKG TSV数据
                                    df = pd.read_csv(
                                        StringIO(decompressed_content),
                                        sep='\t',
                                        header=None,
                                        names=self._get_gkg_column_names()
                                    )
                                    
                                    # 转换为NewsItem
                                    items = []
                                    for _, row in df.iterrows():
                                        item = self._convert_gkg_to_newsitem(row)
                                        if item:
                                            items.append(item)
                                    
                                    return items
            
            return []
        except Exception as e:
            self._logger.error(f"Error fetching GDELT GKG data (fallback): {e}")
            return []
    
    def _filter_events_data(self, df: pd.DataFrame, config: FetchConfig) -> pd.DataFrame:
        """根据配置过滤事件数据（备用方法使用）"""
        filtered_df = df.copy()
        
        # 根据关键词过滤
        if config.keywords:
            keyword_filter = False
            for keyword in config.keywords:
                # 检查参与者名称中是否包含关键词
                actor1_mask = filtered_df['Actor1Name'].str.contains(keyword, case=False, na=False)
                actor2_mask = filtered_df['Actor2Name'].str.contains(keyword, case=False, na=False)
                if keyword_filter is False:
                    keyword_filter = actor1_mask | actor2_mask
                else:
                    keyword_filter = keyword_filter | (actor1_mask | actor2_mask)
            if keyword_filter is not False:
                filtered_df = filtered_df[keyword_filter]
        
        # 根据日期范围过滤
        if config.from_date:
            filtered_df = filtered_df[pd.to_datetime(filtered_df['SQLDATE'], format='%Y%m%d', errors='coerce') >= config.from_date]
        if config.to_date:
            filtered_df = filtered_df[pd.to_datetime(filtered_df['SQLDATE'], format='%Y%m%d', errors='coerce') <= config.to_date]
        
        # 限制返回数量
        if config.max_items > 0:
            filtered_df = filtered_df.head(config.max_items)
        
        return filtered_df


class GDELTManager(NewsSourcePool):
    """GDELT 数据源管理器"""
    
    _instance: Optional["GDELTManager"] = None
    
    def __init__(self):
        self._sources: Dict[str, NewsSource] = {}
        self._logger = get_logger(__name__)
        
        # 注册默认GDELT源
        self.register(GDELTAdapter(name="GDELT"))
    
    @classmethod
    def get_instance(cls) -> "GDELTManager":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = GDELTManager()
        return cls._instance
    
    def register(self, source: NewsSource) -> None:
        """注册新闻源"""
        self._sources[source.source_name] = source
        self._logger.info(f"Registered GDELT source: {source.source_name}")
    
    def get_source(self, name: str) -> Optional[NewsSource]:
        """获取指定新闻源"""
        return self._sources.get(name)
    
    def list_sources(self) -> List[str]:
        """列出所有新闻源"""
        return list(self._sources.keys())
    
    async def fetch_all(self, config: Optional[FetchConfig] = None) -> Dict[str, FetchResult]:
        """从所有GDELT源获取数据"""
        results = {}
        
        for name, source in self._sources.items():
            if source.is_available():
                try:
                    result = await source.fetch(config)
                    results[name] = result
                except Exception as e:
                    results[name] = FetchResult(
                        items=[],
                        total_fetched=0,
                        success=False,
                        error=str(e)
                    )
        
        return results


def get_gdelt_manager() -> GDELTManager:
    """获取GDELT管理器实例"""
    return GDELTManager.get_instance()