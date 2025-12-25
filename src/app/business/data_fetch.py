from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union

import pandas as pd
try:
    import dask.dataframe as dd
    from dask.distributed import Client, as_completed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

from ...infra.registry import register_tool
from ...infra.paths import tools as Tools
from ...core import ConfigManager, AsyncExecutor, RateLimiter, get_config_manager, get_llm_pool
from ...domain.data_operations import update_entities, update_abstract_map
from ...adapters.news.fetch_utils import fetch_from_multiple_sources, normalize_news_items
from .extraction import llm_extract_events, NewsDeduplicator, persist_expanded_news_to_tmp
from ...infra.file_utils import safe_unlink_multiple, safe_unlink
from ...domain.data_operations import sanitize_datetime_fields, write_jsonl_file
from ...ports.extraction import NewsItem, FetchConfig
from ...adapters.news.gdelt_adapter import GDELTAdapter

# 导入新闻API管理器
from ...core import NewsAPIManager


class GDELTDataProcessor:
    """GDELT数据处理器"""
    
    def __init__(self, use_dask: bool = True):
        self._use_dask = use_dask and DASK_AVAILABLE
        self._gdelt_adapter = GDELTAdapter()
        
    async def process_gdelt_data(
        self, 
        config: FetchConfig,
        normalize_entities: bool = True,
        extract_roles: bool = True,
        format_timestamps: bool = True
    ) -> pd.DataFrame:
        """
        处理GDELT数据的主方法
        
        Args:
            config: 获取配置
            normalize_entities: 是否标准化实体
            extract_roles: 是否提取角色
            format_timestamps: 是否格式化时间戳
            
        Returns:
            处理后的DataFrame
        """
        # 获取原始数据
        raw_data = await self._gdelt_adapter.fetch(config)
        
        if not raw_data.items:
            print("No GDELT data found for the given criteria")
            return pd.DataFrame()
        
        # 将NewsItem转换为DataFrame
        df = self._news_items_to_dataframe(raw_data.items)
        
        # 数据预处理
        df = self._preprocess_data(df)
        
        # 实体标准化
        if normalize_entities:
            df = self._normalize_entities(df)
        
        # 角色提取
        if extract_roles:
            df = self._extract_roles(df)
        
        # 时间戳处理
        if format_timestamps:
            df = self._format_timestamps(df)
        
        return df
    
    def _news_items_to_dataframe(self, news_items: List[NewsItem]) -> pd.DataFrame:
        """将NewsItem列表转换为DataFrame"""
        if not news_items:
            return pd.DataFrame()
        
        # 提取raw_data中的信息
        records = []
        for item in news_items:
            if item.raw_data:
                record = item.raw_data.copy()
                record['id'] = item.id
                record['title'] = item.title
                record['content'] = item.content
                record['source_name'] = item.source_name
                record['source_url'] = item.source_url
                record['published_at'] = item.published_at
                record['author'] = item.author
                record['category'] = item.category
                record['language'] = item.language
                records.append(record)
        
        if records:
            df = pd.DataFrame(records)
            return df
        else:
            return pd.DataFrame()
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        if df.empty:
            return df
        
        # 清理空值
        df = df.fillna('')
        
        # 确保关键列存在
        required_columns = [
            'Actor1Name', 'Actor2Name', 'EventCode', 'SQLDATE',
            'GLOBALEVENTID', 'SOURCEURL', 'ActionGeo_FullName'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # 去除重复行
        df = df.drop_duplicates(subset=['GLOBALEVENTID'], keep='first')
        
        return df
    
    def _normalize_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化实体（去重ActorName）"""
        if df.empty:
            return df
        
        print(f"Normalizing entities in {len(df)} records")
        
        # 对Actor1Name和Actor2Name进行标准化处理
        df['Actor1Name_normalized'] = df['Actor1Name'].apply(self._normalize_actor_name)
        df['Actor2Name_normalized'] = df['Actor2Name'].apply(self._normalize_actor_name)
        
        # 实现去重逻辑 - 合并相同实体的记录
        df = self._deduplicate_entities(df)
        
        return df
    
    def _normalize_actor_name(self, name: str) -> str:
        """标准化参与者名称"""
        if pd.isna(name) or name == '':
            return ''
        
        # 清理名称：去除多余空格，统一大小写等
        normalized = str(name).strip().title()
        
        # 处理常见的名称变体（如缩写、全称等）
        name_mapping = {
            'USA': 'United States',
            'US': 'United States',
            'UK': 'United Kingdom',
            'UAE': 'United Arab Emirates',
            'USSR': 'Soviet Union',
            'CHN': 'China',
            'JPN': 'Japan',
            'GER': 'Germany',
            'FRA': 'France',
            'ITA': 'Italy',
            'CAN': 'Canada',
            'MEX': 'Mexico',
            'BRA': 'Brazil',
            'IND': 'India',
            'AUS': 'Australia',
            'KOR': 'South Korea',
            'PRK': 'North Korea',
            'SA': 'South Africa',
            'NGA': 'Nigeria',
            'EGY': 'Egypt',
            'TUR': 'Turkey',
            'SAU': 'Saudi Arabia',
            # 可以根据需要添加更多映射
        }
        
        # 应用映射
        mapped_name = name_mapping.get(normalized, normalized)
        
        # 移除多余的词缀或描述
        suffixes_to_remove = ['Government', 'Of', 'The', 'Republic', 'State', 'Province', 'City', 'Municipality']
        for suffix in suffixes_to_remove:
            mapped_name = mapped_name.replace(f' {suffix}', '').replace(f'{suffix} ', '')
        
        # 清理多余空格
        mapped_name = ' '.join(mapped_name.split())
        
        return mapped_name
    
    def _deduplicate_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """去重实体，合并相同实体的记录"""
        if df.empty:
            return df
        
        # 创建标准化的Actor组合键，用于识别重复项
        df['actor_pair_normalized'] = df.apply(
            lambda row: tuple(sorted([
                row['Actor1Name_normalized'], 
                row['Actor2Name_normalized']
            ])), axis=1
        )
        
        # 如果需要完全去重，可以基于标准化的实体对进行分组
        # 这里我们保留每个实体对的最新记录
        df_sorted = df.sort_values('SQLDATE', ascending=False)
        df_deduplicated = df_sorted.drop_duplicates(
            subset=['actor_pair_normalized', 'EventCode'], 
            keep='first'
        ).sort_index()  # 恢复原始顺序
        
        print(f"Deduplicated from {len(df)} to {len(df_deduplicated)} records")
        
        return df_deduplicated
    
    def _extract_roles(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取角色信息（Actor1为施事者，Actor2为受事者）"""
        if df.empty:
            return df
        
        print(f"Extracting roles in {len(df)} records")
        
        # 添加角色标签
        df['Actor1_role'] = 'actor'  # 施事者（发起行动的实体）
        df['Actor2_role'] = 'target'  # 受事者（接受行动的实体）
        
        # 根据事件代码确定事件类型和角色关系
        df['event_type'] = df['EventCode'].apply(self._get_event_type)
        df['actor_target_relationship'] = df.apply(self._determine_relationship, axis=1)
        
        # 增强角色信息
        df['actor_description'] = df['Actor1Name_normalized'].apply(self._describe_actor)
        df['target_description'] = df['Actor2Name_normalized'].apply(self._describe_actor)
        
        return df
    
    def _determine_relationship(self, row) -> str:
        """根据事件代码确定施事者和受事者的关系"""
        event_code = str(row['EventCode'])
        if len(event_code) >= 2:
            primary_code = event_code[:2]
        else:
            primary_code = event_code
        
        # 根据CAMEO事件代码定义关系
        relationship_map = {
            '01': 'makes_public_statement_to',      # 发表公开声明
            '02': 'appeals_to',                    # 呼吁
            '03': 'expresses_intent_to_cooperate_with',  # 表达合作意图
            '04': 'consults_with',                  # 咨询
            '05': 'engages_in_diplomatic_cooperation_with',  # 进行外交合作
            '06': 'engages_in_material_cooperation_with',    # 进行物质合作
            '07': 'provides_aid_to',               # 提供援助
            '08': 'yields_to',                     # 屈服于
            '09': 'investigates',                  # 调查
            '10': 'demands_of',                    # 要求
            '11': 'disapproves_of',                # 不赞成
            '12': 'rejects',                       # 拒绝
            '13': 'threatens_with',                # 以...威胁
            '14': 'protests_against',              # 抗议
            '15': 'exhibits_force_posture_towards', # 展示武力姿态
            '16': 'reduces_relations_with',         # 减少与...的关系
            '17': 'coerces',                       # 强制
            '18': 'assaults',                      # 攻击
            '19': 'fights_with',                   # 与...战斗
            '20': 'uses_unconventional_violence_against'  # 对...使用非常规暴力
        }
        
        return relationship_map.get(primary_code, f'interacts_with_code_{primary_code}')
    
    def _describe_actor(self, actor_name: str) -> str:
        """描述参与者类型"""
        if pd.isna(actor_name) or actor_name == '':
            return 'unknown'
        
        # 根据名称模式判断参与者类型
        actor_lower = actor_name.lower()
        
        # 国家类型
        if any(country in actor_lower for country in ['china', 'united states', 'russia', 'france', 'germany', 
                                                     'uk', 'united kingdom', 'japan', 'canada', 'australia']):
            return 'country'
        
        # 组织类型
        if any(org in actor_lower for org in ['party', 'government', 'ministry', 'department', 'agency', 
                                             'committee', 'council', 'organization', 'union']):
            return 'organization'
        
        # 个人类型
        if len(actor_name.split()) <= 3:  # 姓名通常较短
            return 'person'
        
        return 'entity'
    
    def _get_event_type(self, event_code: str) -> str:
        """根据事件代码确定事件类型"""
        if pd.isna(event_code):
            return 'unknown'
        
        # CAMEO事件代码映射
        # 这里只列出一些常见的事件类型，可以根据需要扩展
        event_mapping = {
            '01': 'make_public_statement',
            '02': 'appeal',
            '03': 'express_intent_to_cooperate',
            '04': 'consult',
            '05': 'engage_in_diplomatic_cooperation',
            '06': 'engage_in_material_cooperation',
            '07': 'provide_aid',
            '08': 'yield',
            '09': 'investigate',
            '10': 'demand',
            '11': 'disapprove',
            '12': 'reject',
            '13': 'threaten',
            '14': 'protest',
            '15': 'exhibit_force_posture',
            '16': 'reduce_relations',
            '17': 'coerce',
            '18': 'assault',
            '19': 'fight',
            '20': 'use_unconventional_mass_violence'
        }
        
        code_str = str(event_code)
        # 取前两位作为主要事件类型
        primary_code = code_str[:2] if len(code_str) >= 2 else code_str
        
        return event_mapping.get(primary_code, f'code_{code_str}')
    
    def _format_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """时间戳提取和格式化"""
        if df.empty:
            return df
        
        print(f"Formatting timestamps in {len(df)} records")
        
        # 处理SQLDATE列
        if 'SQLDATE' in df.columns:
            df['timestamp_formatted'] = df['SQLDATE'].apply(self._format_sql_date)
            df['date_parsed'] = df['SQLDATE'].apply(self._parse_sql_date)
            df['year'] = df['date_parsed'].apply(lambda x: x.year if x is not None else None)
            df['month'] = df['date_parsed'].apply(lambda x: x.month if x is not None else None)
            df['day'] = df['date_parsed'].apply(lambda x: x.day if x is not None else None)
            df['week'] = df['date_parsed'].apply(lambda x: x.isocalendar()[1] if x is not None else None)
            df['quarter'] = df['date_parsed'].apply(lambda x: x.quarter if x is not None and hasattr(x, 'quarter') else None)
        
        # 处理其他可能的时间列
        if 'DATEADDED' in df.columns:
            df['dateadded_parsed'] = df['DATEADDED'].apply(self._parse_sql_date)
        
        # 如果有MentionTimeDate（来自mentions表）
        if 'MentionTimeDate' in df.columns:
            df['mention_timestamp'] = df['MentionTimeDate'].apply(self._format_mention_timestamp)
        
        # 创建时间特征
        df = self._create_temporal_features(df)
        
        return df
    
    def _format_sql_date(self, sql_date: Union[str, int, float]) -> Optional[str]:
        """格式化SQLDATE为标准时间戳格式"""
        if pd.isna(sql_date):
            return None
        
        try:
            # SQLDATE格式通常是YYYYMMDD
            date_str = str(int(sql_date)) if isinstance(sql_date, (int, float)) else str(sql_date)
            if len(date_str) == 8:
                dt = datetime.strptime(date_str, '%Y%m%d')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            elif len(date_str) == 14:  # YYYYMMDDHHMMSS 格式
                dt = datetime.strptime(date_str, '%Y%m%d%H%M%S')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                return str(sql_date)
        except Exception:
            return str(sql_date)
    
    def _parse_sql_date(self, sql_date: Union[str, int, float]) -> Optional[datetime]:
        """解析SQLDATE为datetime对象"""
        if pd.isna(sql_date):
            return None
        
        try:
            # SQLDATE格式通常是YYYYMMDD
            date_str = str(int(sql_date)) if isinstance(sql_date, (int, float)) else str(sql_date)
            if len(date_str) == 8:
                return datetime.strptime(date_str, '%Y%m%d')
            elif len(date_str) == 14:  # YYYYMMDDHHMMSS 格式
                return datetime.strptime(date_str, '%Y%m%d%H%M%S')
            else:
                return None
        except Exception:
            return None
    
    def _format_mention_timestamp(self, mention_time: Union[str, int, float]) -> Optional[str]:
        """格式化MentionTimeDate为标准时间戳格式"""
        if pd.isna(mention_time):
            return None
        
        try:
            # MentionTimeDate格式通常是YYYYMMDDHHMMSS
            time_str = str(int(mention_time)) if isinstance(mention_time, (int, float)) else str(mention_time)
            if len(time_str) == 14:
                dt = datetime.strptime(time_str, '%Y%m%d%H%M%S')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            elif len(time_str) == 8:  # 如果只有日期部分
                dt = datetime.strptime(time_str, '%Y%m%d')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                return str(time_str)
        except Exception:
            return str(mention_time)
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间特征"""
        if df.empty:
            return df
        
        # 基于解析后的日期创建更多时间特征
        if 'date_parsed' in df.columns:
            date_col = df['date_parsed']
            
            # 工作日特征
            df['day_of_week'] = date_col.apply(lambda x: x.weekday() if x is not None else None)
            df['is_weekend'] = date_col.apply(lambda x: x.weekday() >= 5 if x is not None else None)
            
            # 季节特征
            df['season'] = date_col.apply(lambda x: self._get_season(x.month) if x is not None else None)
            
            # 是否为月初/月末
            df['is_month_start'] = date_col.apply(lambda x: x.day == 1 if x is not None else None)
            df['is_month_end'] = date_col.apply(lambda x: x.day == x.days_in_month if x is not None and hasattr(x, 'days_in_month') else None)
        
        return df
    
    def _get_season(self, month: int) -> str:
        """根据月份获取季节"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    async def process_large_gdelt_data(
        self, 
        config: FetchConfig,
        chunk_size: int = 10000
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        处理大规模GDELT数据（支持Dask）
        
        Args:
            config: 获取配置
            chunk_size: 分块大小
            
        Returns:
            处理后的DataFrame（pandas或Dask）
        """
        if self._use_dask:
            return await self._process_with_dask(config, chunk_size)
        else:
            return await self.process_gdelt_data(config)
    
    async def _process_with_dask(
        self, 
        config: FetchConfig,
        chunk_size: int = 10000
    ) -> dd.DataFrame:
        """使用Dask处理大规模数据"""
        if not DASK_AVAILABLE:
            print("Dask not available, falling back to pandas")
            return await self.process_gdelt_data(config)
        
        # 由于Dask处理需要特定的数据源，我们模拟分块处理
        # 在实际应用中，这可能需要从文件系统或数据库直接读取
        
        # 先获取所有数据
        raw_data = await self._gdelt_adapter.fetch(config)
        
        if not raw_data.items:
            return dd.from_pandas(pd.DataFrame(), npartitions=1)
        
        # 将数据转换为DataFrame
        df = self._news_items_to_dataframe(raw_data.items)
        
        # 如果数据量大，使用Dask
        if len(df) > chunk_size:
            # 创建Dask DataFrame
            ddf = dd.from_pandas(df, npartitions=max(1, len(df) // chunk_size))
            
            # 应用处理函数
            ddf = ddf.map_partitions(self._process_partition)
            
            return ddf
        else:
            # 数据量小，直接使用pandas处理
            processed_df = self._process_partition(df)
            return dd.from_pandas(processed_df, npartitions=1)
    
    def _process_partition(self, partition_df: pd.DataFrame) -> pd.DataFrame:
        """处理数据分区"""
        # 预处理
        partition_df = self._preprocess_data(partition_df)
        
        # 实体标准化
        partition_df = self._normalize_entities(partition_df)
        
        # 角色提取
        partition_df = self._extract_roles(partition_df)
        
        # 时间戳处理
        partition_df = self._format_timestamps(partition_df)
        
        return partition_df
    
    def integrate_gkg_data(
        self, 
        events_df: pd.DataFrame, 
        gkg_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        集成GKG文件以增强主题/情感分析
        
        Args:
            events_df: 事件数据DataFrame
            gkg_df: GKG数据DataFrame
            
        Returns:
            集成后的DataFrame
        """
        if events_df.empty or gkg_df.empty:
            return events_df
        
        print("Integrating GKG data with events data")
        
        # 这里实现GKG数据与事件数据的集成逻辑
        # 通常通过日期或主题匹配
        # 由于GKG和事件数据的结构不同，我们需要适当的连接逻辑
        
        # 示例：按日期连接（这可能需要根据实际数据结构调整）
        if 'DATE' in gkg_df.columns and 'SQLDATE' in events_df.columns:
            # 将事件数据的SQLDATE转换为GKG格式进行匹配
            events_df['DATE'] = events_df['SQLDATE'].apply(
                lambda x: int(str(x)[:8]) if pd.notna(x) else None
            )
            
            # 合并数据
            merged_df = pd.merge(
                events_df,
                gkg_df,
                on='DATE',
                how='left',
                suffixes=('', '_gkg')
            )
            
            return merged_df
        else:
            # 如果没有匹配的列，直接返回事件数据
            print("Cannot merge events and GKG data: no matching columns found")
            return events_df


def _load_entity_equivs() -> Dict[str, Set[str]]:

@register_tool(
    name="fetch_news_stream",
    description="从所有配置的数据源（当前仅 GNews）获取最新新闻",
    category="Data Fetch"
)
async def fetch_news_stream(
    limit: int = 50,
    sources: Optional[List[str]] = None,
    # GNews 可选参数
    category: Optional[str] = None,
    query: Optional[str] = None,
    from_: Optional[str] = None,
    to: Optional[str] = None,
    nullable: Optional[str] = None,
    truncate: Optional[str] = None,
    sortby: Optional[str] = None,
    in_fields: Optional[str] = None,
    page: Optional[int] = None,
    daily_incremental: bool = False,  # 新增参数：是否按天递增请求
) -> List[Dict[str, Any]]:
    """
    获取全渠道新闻数据。

    Args:
        limit: 每个源获取的最大条数
        sources: 指定源列表 (如 ["GNews-cn"]), 默认为所有可用源
        daily_incremental: 是否按天递增请求数据（从from_开始，每天请求一次，直到to_）

    Returns:
        新闻列表 (List[Dict])
    """
    tools = Tools()
    tools.log(f"[fetch_news_stream] 开始执行，请求源: {sources}, limit: {limit}, daily_incremental: {daily_incremental}")
    
    # 配置驱动的并发上限（使用统一配置管理器）
    config_manager = get_config_manager()
    concurrency = config_manager.get_concurrency_limit("agent1_config")
    
    # 设置日志级别为DEBUG以便获取更多调试信息
    from ...infra.logging import LoggerManager
    LoggerManager.set_level("DEBUG")

    # 初始化 API Pool
    api_pool = NewsAPIManager.get_instance()
    available_sources = api_pool.list_available_sources()
    tools.log(f"[fetch_news_stream] 可用数据源: {available_sources}")
    
    if sources:
        target_sources = [s for s in sources if s in available_sources]
        tools.log(f"[fetch_news_stream] 筛选后目标源: {target_sources}")
    else:
        target_sources = available_sources

    if not target_sources:
        tools.log("Warning: No valid sources to fetch from. API密钥可能未配置。")
        return []

    # 如果启用了按天递增请求，并且提供了from_和to_参数
    if daily_incremental and from_ and to:
        tools.log(f"[fetch_news_stream] 启用按天递增请求: 从 {from_} 到 {to}")
        all_news = []
        
        # 定义所有可用的类别
        categories = ["general", "world", "nation", "business", "technology", "entertainment", "sports", "science", "health"]
        
        # 解析日期
        try:
            from_date = datetime.fromisoformat(from_.replace("Z", "+00:00")).date()
            to_date = datetime.fromisoformat(to.replace("Z", "+00:00")).date()
            
            # 按天递增请求数据
            current_date = from_date
            while current_date <= to_date:
                # 构造当天的日期范围
                day_start = f"{current_date.isoformat()}T00:00:00.000Z"
                day_end = f"{current_date.isoformat()}T23:59:59.999Z"
                
                tools.log(f"[fetch_news_stream] 请求 {current_date.isoformat()} 的数据")
                
                # 为每个类别获取数据
                day_news = []
                for cat in categories:
                    tools.log(f"[fetch_news_stream] 请求 {current_date.isoformat()} 的 {cat} 类别数据")
                    cat_news = await fetch_from_multiple_sources(
                        api_pool=api_pool,
                        source_names=target_sources,
                        concurrency_limit=concurrency,
                        query=query,
                        category=cat,
                        limit=limit,
                        from_=day_start,
                        to=day_end,
                        nullable=nullable,
                        truncate=truncate,
                        sortby=sortby,
                        in_fields=in_fields,
                        page=page,
                    )
                    day_news.extend(cat_news)
                    tools.log(f"[fetch_news_stream] {current_date.isoformat()} 的 {cat} 类别获取到 {len(cat_news)} 条数据")
                
                all_news.extend(day_news)
                tools.log(f"[fetch_news_stream] {current_date.isoformat()} 总共获取到 {len(day_news)} 条数据")
                
                # 移动到下一天
                current_date = current_date + timedelta(days=1)                
        except Exception as e:
            tools.log(f"[fetch_news_stream] 按天递增请求出错: {e}")
            # 如果出错，回退到原来的实现
            return await fetch_from_multiple_sources(
                api_pool=api_pool,
                source_names=target_sources,
                concurrency_limit=concurrency,
                query=query,
                category=category,
                limit=limit,
                from_=from_,
                to=to,
                nullable=nullable,
                truncate=truncate,
                sortby=sortby,
                in_fields=in_fields,
                page=page,
            )
            
        tools.log(f"[fetch_news_stream] 按天递增请求完成，总共获取到 {len(all_news)} 条数据")
        return all_news
    else:
        # 如果指定了类别，则只使用该类别；否则遍历所有类别
        categories = ["general", "world", "nation", "business", "technology", "entertainment", "sports", "science", "health"]
        
        # 为每个类别获取数据并合并
        all_news = []
        for cat in categories:
            cat_news = await fetch_from_multiple_sources(
                api_pool=api_pool,
                source_names=target_sources,
                concurrency_limit=concurrency,
                query=query,
                category=cat,
                limit=limit,
                from_=from_,
                to=to,
                nullable=nullable,
                truncate=truncate,
                sortby=sortby,
                in_fields=in_fields,
                page=page,
            )
            all_news.extend(cat_news)
        
        return all_news

def _load_entity_equivs() -> Dict[str, Set[str]]:
    """
    构建实体及同义词索引：实体库 original_forms + 合并规则的别名。
    返回 dict: 词 -> 同义集合（包含自身）。
    """
    tools = Tools()
    idx: Dict[str, Set[str]] = {}

    def add_forms(key: str, forms: List[str]):
        if not key:
            return
        bucket = idx.setdefault(key, set())
        for f in forms:
            if f:
                bucket.add(f)
        bucket.add(key)

    # 实体库 original_forms
    try:
        if tools.ENTITIES_FILE.exists():
            with open(tools.ENTITIES_FILE, "r", encoding="utf-8") as f:
                ents = json.load(f)
            for name, data in ents.items():
                forms = data.get("original_forms", []) if isinstance(data, dict) else []
                if isinstance(forms, list):
                    flat = []
                    for x in forms:
                        if isinstance(x, str):
                            flat.append(x)
                        elif isinstance(x, list):
                            flat.extend([str(i) for i in x])
                    add_forms(name, flat)
                else:
                    add_forms(name, [])
    except Exception:
        pass

    # 合并规则（alias -> primary），双向加入
    try:
        merge_rules_file = tools.CONFIG_DIR / "entity_merge_rules.json"
        if merge_rules_file.exists():
            with open(merge_rules_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = data.get("merge_rules", {}) if isinstance(data, dict) else {}
            inv: Dict[str, List[str]] = {}
            for alias, primary in rules.items():
                add_forms(alias, [primary])
                inv.setdefault(primary, []).append(alias)
            for primary, aliases in inv.items():
                add_forms(primary, aliases)
    except Exception:
        pass

    # 展开：确保互相包含
    for k, forms in list(idx.items()):
        for f in list(forms):
            if f in idx:
                forms.update(idx[f])
    return idx


def _expand_keywords(keywords: List[str]) -> List[List[str]]:
    """
    将每个关键词扩展为同义集合列表（按输入顺序保留组）。
    """
    idx = _load_entity_equivs()
    groups: List[List[str]] = []
    for kw in keywords:
        kw_norm = (kw or "").strip()
        if not kw_norm:
            continue
        forms = set([kw_norm])
        if kw_norm in idx:
            forms.update(idx[kw_norm])
        groups.append(list(forms))
    return groups


def _build_boolean_query(groups: List[List[str]]) -> str:
    """
    构建gnews.io API查询
    规则：
    1. 每个分组内的词用OR连接
    2. 不同分组之间用AND连接  
    """
    if not groups:
        return ""
    
    query_parts = []
    
    for group in groups:
        if not group:
            continue
            
        # 清理组内的词
        valid_terms = []
        for term in group:
            if isinstance(term, str):
                term = term.strip()
                if term:
                    valid_terms.append(term)
        
        if not valid_terms:
            continue
            
        # 处理单个词
        if len(valid_terms) == 1:
            term = valid_terms[0]
            query_parts.append(f'"{term}"')
        # 处理多个词
        else:
            or_terms = []
            for term in valid_terms:
                or_terms.append(f'"{term}"')
            query_parts.append(f"{' OR '.join(or_terms)}")
    
    if not query_parts:
        return ""
    
    base_query = " AND ".join(query_parts)
    return base_query



@register_tool(
    name="search_news_by_keywords",
    description="按关键词搜索新闻（当前仅 GNews），支持可选时间范围与排序",
    category="Data Fetch"
)
async def search_news_by_keywords(
    keywords: List[str],
    apis: Optional[List[str]] = None,
    limit: int = 10,
    category: Optional[str] = None,
    from_: Optional[str] = None,
    to: Optional[str] = None,
    nullable: Optional[str] = None,
    truncate: Optional[str] = None,
    sortby: Optional[str] = None,
    in_fields: Optional[str] = None,
    page: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    根据关键词列表搜索新闻（GNews），返回合并后的结果。
    """
    tools = Tools()
    if not keywords:
        return []

    # 构造 (A OR A2) AND (B OR B2) 查询串
    groups = _expand_keywords(keywords)
    query_str = _build_boolean_query(groups)
    if not query_str:
        return []

    # 初始化 API Pool
    api_pool = NewsAPIManager.get_instance()

    # 并发上限（使用统一配置管理器）
    config_manager = get_config_manager()
    concurrency = config_manager.get_concurrency_limit("agent2_config")

    available_sources = api_pool.list_available_sources()
    if apis:
        target_sources = [s for s in apis if s in available_sources]
    else:
        target_sources = available_sources

    if not target_sources:
        tools.log("Warning: No valid sources to search from.")
        return []

    # 使用公共函数获取数据
    return await fetch_from_multiple_sources(
        api_pool=api_pool,
        source_names=target_sources,
        concurrency_limit=concurrency,
        query=query_str,
        category=category,
        limit=limit,
        from_=from_,
        to=to,
        nullable=nullable,
        truncate=truncate,
        sortby=sortby,
        in_fields=in_fields,
        page=page,
    )


async def expand_news_by_entities(entities: List[Dict], limit_per_entity: int = 10, time_window_days: int = 30, full_search: bool = False) -> List[Dict]:
    """
    根据实体列表搜索相关新闻，支持使用原始词进行检索

    Args:
        entities: 实体列表，每个实体包含name和original_forms字段
        limit_per_entity: 每个实体搜索的新闻数量限制
        time_window_days: 默认检索时间窗口（天），默认为30天
        full_search: 是否进行全面检索，如果为True则从当前时间向前检索多个30天或更小的天数直到2020年

    Returns:
        搜索到的相关新闻列表
    """
    expanded_news = []
    news_id_set = set()  # 用于去重

    # 获取所有可用的新闻收集器
    news_collectors = []
    api_pool = NewsAPIManager.get_instance()
    available_sources = api_pool.list_available_sources()

    # 与更新后的API池兼容，移除可能的备用逻辑
    for source_name in available_sources:
        try:
            collector = api_pool.get_collector(source_name)
            news_collectors.append(collector)
        except Exception as e:
            Tools().log(f"⚠️ 无法创建新闻收集器 {source_name}: {e}")

    if not news_collectors:
        Tools().log("❌ 未找到可用的新闻收集器")
        return expanded_news

    # 为每个实体搜索相关新闻
    for entity in entities:
        entity_name = entity['name']
        original_forms = entity.get('original_forms', [])

        # 构建使用OR操作符连接的搜索查询：实体名称 + 所有原始词
        all_terms = [entity_name] + original_forms

        # 生成查询批次以避免超过200字符限制
        query_batches = []
        current_batch = []
        current_length = 0

        for term in all_terms:
            quoted_term = f'"{term}"'
            term_length = len(quoted_term)

            # 如果是第一个词，直接添加；否则需要考虑OR操作符的长度
            if current_batch:
                required_length = current_length + 4 + term_length  # 4是" OR "的长度
                if required_length > 200:
                    # 如果添加当前词会超过限制，保存当前批次并开始新批次
                    query_batches.append(" OR ".join(current_batch))
                    current_batch = [quoted_term]
                    current_length = term_length
                else:
                    current_batch.append(quoted_term)
                    current_length = required_length
            else:
                current_batch.append(quoted_term)
                current_length = term_length

        # 添加最后一个批次
        if current_batch:
            query_batches.append(" OR ".join(current_batch))

        Tools().log(f"🔍 为实体 '{entity_name}' 搜索相关新闻，使用OR查询连接 {len(original_forms)} 个原始词...")
        Tools().log(f"   📝 生成了 {len(query_batches)} 个查询批次以避免超过200字符限制")

        # 获取时间范围
        time_ranges = get_time_ranges(time_window_days, full_search)

        for collector in news_collectors:
            try:
                # 对每个查询批次和时间范围进行搜索
                for time_range in time_ranges:
                    start_date = time_range['start']
                    end_date = time_range['end']

                    Tools().log(f"   📅 搜索时间范围: {start_date} 至 {end_date}")

                    for batch_index, batch_query in enumerate(query_batches):
                        Tools().log(f"   📝 执行查询批次 {batch_index + 1}/{len(query_batches)}: '{batch_query}'")

                        try:
                            # 使用搜索功能获取相关新闻，传入时间范围参数
                            search_params = {
                                'keyword' if hasattr(collector, 'search_news_by_keyword') else 'query': batch_query,
                                'limit': limit_per_entity // (len(query_batches) * len(time_ranges)) + 1  # 平均分配限制
                            }

                            # 如果收集器支持时间范围参数，则添加
                            if hasattr(collector, 'search_news_by_keyword'):
                                if 'start_date' in collector.search_news_by_keyword.__code__.co_varnames:
                                    search_params['start_date'] = start_date
                                    search_params['end_date'] = end_date
                                elif 'from_date' in collector.search_news_by_keyword.__code__.co_varnames:
                                    search_params['from_date'] = start_date
                                    search_params['to_date'] = end_date
                            elif hasattr(collector, 'search'):
                                if 'start_date' in collector.search.__code__.co_varnames:
                                    search_params['start_date'] = start_date
                                    search_params['end_date'] = end_date
                                elif 'from_date' in collector.search.__code__.co_varnames:
                                    search_params['from_date'] = start_date
                                    search_params['to_date'] = end_date

                            # 使用更新后的API调用方法
                            # 优先使用search_news_by_keyword方法，确保与更新后的API池兼容
                            if hasattr(collector, 'search_news_by_keyword'):
                                news_list = await collector.search_news_by_keyword(**search_params)
                            elif hasattr(collector, 'search'):
                                news_list = await collector.search(**search_params)
                            else:
                                Tools().log(f"⚠️ 收集器 {collector.__class__.__name__} 没有支持的搜索方法")
                                continue

                            # 为每条新闻添加实体标签并去重
                            for news in news_list:
                                # 生成唯一标识符用于去重
                                news_id = f"{news.get('url', '')}_{news.get('publishedAt', '')}"
                                if news_id not in news_id_set:
                                    news_id_set.add(news_id)
                                    news['expanded_from_entity'] = entity_name
                                    news['search_term'] = batch_query  # 记录使用的搜索词
                                    news['source'] = collector.__class__.__name__.replace('Collector', '').lower()
                                    news['query_batch'] = batch_index + 1  # 记录查询批次
                                    news['search_time_range'] = f"{start_date} to {end_date}"  # 记录搜索时间范围
                                    expanded_news.append(news)
                        except Exception as batch_error:
                            Tools().log(f"⚠️ 查询批次 {batch_index + 1} 执行失败: {batch_error}")
            except Exception as e:
                Tools().log(f"⚠️ 从 {collector.__class__.__name__} 搜索失败: {e}")

    return expanded_news


def get_time_ranges(default_days: int = 30, full_search: bool = False) -> List[Dict]:
    """
    获取搜索的时间范围列表，满足以下要求：
    1. 时间范围只能从2020年至今
    2. 默认检索前30天内的新闻
    3. 全面检索时从当前时间向前检索多个30天或更小的天数直到2020年

    Args:
        default_days: 默认的时间窗口天数，默认为30天
        full_search: 是否进行全面检索

    Returns:
        时间范围列表，每个元素包含start和end日期字符串
    """
    time_ranges = []
    now = datetime.now(timezone.utc)

    # 定义2020年1月1日作为起始时间
    start_date_2020 = datetime(2020, 1, 1, tzinfo=timezone.utc)

    if not full_search:
        # 非全面检索，只返回默认时间窗口
        end_date = now
        start_date = max(start_date_2020, now - timedelta(days=default_days))
        time_ranges.append({
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        })
    else:
        # 全面检索，从当前时间向前检索多个30天或更小的天数直到2020年
        Tools().log("🔄 执行全面检索，从当前时间向前检索多个30天批次直到2020年...")

        end_date = now
        batch_count = 0

        while end_date > start_date_2020:
            start_date = max(start_date_2020, end_date - timedelta(days=default_days))

            # 确保不重复添加相同的时间范围
            if not time_ranges or time_ranges[-1]['start'] != start_date.strftime('%Y-%m-%d'):
                time_ranges.append({
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d')
                })

            batch_count += 1
            end_date = start_date - timedelta(days=1)  # 避免日期重叠

        Tools().log(f"✅ 生成了 {batch_count} 个时间范围批次")

    return time_ranges


def get_recent_entities(time_window_days: int = 30, limit: int = 50) -> List[Dict]:
    """
    获取最近时间窗口内的实体列表，包含原始词信息

    Args:
        time_window_days: 时间窗口（天）
        limit: 返回的实体数量限制

    Returns:
        最近的实体列表，每个实体包含名称和原始词信息
    """
    entities = []
    tools = Tools()

    if not tools.ENTITIES_FILE.exists():
        tools.log("⚠️ 实体库文件不存在")
        return entities

    # 读取实体库
    with open(tools.ENTITIES_FILE, "r", encoding="utf-8") as f:
        entity_data = json.load(f)

    # 根据 first_seen 排序，获取最近的实体
    sorted_entities = sorted(
        entity_data.items(),
        key=lambda x: x[1].get('first_seen', ''),
        reverse=True
    )

    # 过滤时间窗口内的实体
    now = datetime.now(timezone.utc)
    time_window = timedelta(days=time_window_days)

    for entity_name, entity_info in sorted_entities:
        first_seen = entity_info.get('first_seen')
        if first_seen:
            try:
                # 解析时间字符串
                if 'T' in first_seen:
                    # ISO格式时间
                    seen_time = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                else:
                    # 普通格式时间
                    seen_time = datetime.strptime(first_seen, '%Y-%m-%d %H:%M:%S')
                    seen_time = seen_time.replace(tzinfo=timezone.utc)

                # 检查是否在时间窗口内
                if now - seen_time <= time_window:
                    entity_info = {
                        'name': entity_name,
                        'original_forms': entity_data[entity_name].get('original_forms', [])
                    }
                    entities.append(entity_info)
                    if len(entities) >= limit:
                        break
            except Exception as e:
                tools.log(f"⚠️ 解析实体 '{entity_name}' 的时间戳失败: {e}")

    tools.log(f"✅ 获取了 {len(entities)} 个最近实体")
    return entities


async def process_expanded_news(expanded_news: List[Dict], rate_limit: float = 1.0, max_workers: int = 3) -> int:
    """
    处理拓展的新闻，提取实体和事件

    Args:
        expanded_news: 拓展的新闻列表
        rate_limit: 每秒速率限制
        max_workers: 最大并发数

    Returns:
        处理的新闻数量
    """
    processed_count = 0
    tools = Tools()

    # 初始化新闻去重器
    deduplicator = NewsDeduplicator(threshold=tools.get_dedupe_threshold())

    # 创建去重集合（ID去重）
    seen_news = set()

    # 使用统一异步执行器和限速器
    async_executor = AsyncExecutor()
    limiter = RateLimiter(rate_limit) if rate_limit > 0 else None

    async def handle_one(news: Dict) -> int:
        nonlocal processed_count
        try:
            news_id = news.get('id')
            source = news.get('source', 'unknown')
            if news_id:
                news_key = f"{source}:{news_id}"
                if news_key in seen_news:
                    return 0
                seen_news.add(news_key)

            title = news.get('title', '')
            content = news.get('content', '')
            if not title:
                return 0

            news_text = f"{title} {content}".strip()
            if deduplicator.is_duplicate(news_text):
                return 0

            # 应用限速
            if limiter:
                await limiter.acquire_async()

            loop = asyncio.get_running_loop()
            # 这里应该使用LLM API池，不是新闻API池
            from ..core import get_llm_pool
            api_pool = get_llm_pool()
            extracted = await loop.run_in_executor(None, llm_extract_events, title, content, api_pool)

            if extracted:
                all_entities = []
                for ev in extracted:
                    all_entities.extend(ev['entities'])

                if all_entities:
                    published_at = news.get('datetime')
                    if published_at and isinstance(published_at, datetime):
                        published_at = published_at.isoformat()

                    all_entities_original = all_entities
                    update_entities(all_entities, all_entities_original, source, published_at)
                    update_abstract_map(extracted, source, published_at)
                    return 1
        except Exception as e:
            tools.log(f"⚠️ 处理拓展新闻失败: {e}")
        return 0

    tasks = [handle_one(news) for news in expanded_news]
    # 使用AsyncExecutor进行并发执行
    results = await async_executor.run_concurrent_tasks(
        tasks=tasks,
        concurrency=max_workers
    )
    processed_count = sum(results)

    return processed_count


@register_tool(
    name="expand_news_by_recent_entities",
    description="[工作流] 根据最近实体搜索相关新闻并提取事件",
    category="Workflow"
)
async def expand_news_by_recent_entities(
    entity_limit: int = 1,
    time_window_days: int = 30,
    limit_per_entity: int = 120,
    full_search: bool = False,
    rate_limit: float = 1.0,
    max_workers: int = 3
) -> Dict[str, Any]:
    """
    根据最近实体搜索相关新闻的工作流

    Args:
        entity_limit: 实体数量限制
        time_window_days: 时间窗口（天）
        limit_per_entity: 每个实体搜索新闻数量
        full_search: 是否全面检索
        rate_limit: 速率限制
        max_workers: 最大并发数

    Returns:
        处理结果统计
    """
    # 从数据库获取已处理的ID
    from src.adapters.sqlite.store import get_store
    processed_ids = get_store().get_processed_ids()

    # 获取最近实体
    recent_entities = get_recent_entities(time_window_days=time_window_days, limit=entity_limit)
    if not recent_entities:
        tools.log("📭 没有可用的实体进行新闻拓展")
        return {"processed_count": 0, "expanded_news_count": 0}

    # 搜索相关新闻
    tools.log(f"🔍 开始搜索 {len(recent_entities)} 个实体的相关新闻...")
    expanded_news = await expand_news_by_entities(
        recent_entities,
        limit_per_entity=limit_per_entity,
        time_window_days=time_window_days,
        full_search=full_search
    )
    tools.log(f"✅ 共搜索到 {len(expanded_news)} 条相关新闻")

    # 处理搜索到的新闻
    if expanded_news:
        deduped_path = persist_expanded_news_to_tmp(expanded_news, processed_ids)
        processed_count = 0
        if deduped_path and deduped_path.exists():
            tools.log(f"📄 开始处理拓展的新闻 (deduped: {deduped_path.name}) ...")
            news_list = []
            with open(deduped_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        news_list.append(json.loads(line))
                    except Exception as e:
                        tools.log(f"⚠️ 跳过无效行: {e}")
            processed_count = await process_expanded_news(news_list, rate_limit=rate_limit, max_workers=max_workers)
            # 清理 tmp 文件
            try:
                raw_file = tools.RAW_NEWS_TMP_DIR / deduped_path.name.replace("_deduped", "")
                file_paths = [raw_file, deduped_path]
                safe_unlink_multiple(file_paths, "临时")
            except Exception as e:
                tools.log(f"⚠️ 删除临时文件失败: {e}")
        tools.log(f"✅ 成功处理 {processed_count} 条拓展新闻")
        return {"processed_count": processed_count, "expanded_news_count": len(expanded_news)}

    return {"processed_count": 0, "expanded_news_count": 0}

