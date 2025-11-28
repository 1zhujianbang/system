# src/data/data_collector.py
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
from typing import Dict, List, Optional
from ..config.config_manager import UserConfig, DataConfig
from okx.api import API
from okx.app.utils import eprint

class OKXMarketClient:
    """OKX 市场数据客户端"""
    
    def __init__(self, user_config: UserConfig, data_config: DataConfig):
        """
        初始化客户端
        
        Args:
            user_config: 用户配置
            data_config: 数据配置
        """
        self.user_config = user_config
        self.data_config = data_config

        # 初始化 API
        self.api = API(proxy_host=self.data_config.proxy)
        
    def get_trading_pairs(self) -> List[str]:
        """从配置中获取交易对列表"""
        return self.user_config.trading_pairs
    
    def get_timeframe(self) -> str:
        """从配置中获取时间框架，转换为OKX支持的格式"""
        timeframe_map = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M', '3M': '3M'
        }
        config_tf = self.data_config.timeframe
        return timeframe_map.get(config_tf, '1H')  # 默认为1小时
    
    def get_historical_days(self) -> int:
        """从配置中获取历史天数"""
        return self.data_config.historical_days
    
    def _fix_datetime_warning(self):
        """修复datetime警告的替代方法"""
        # 这个警告来自OKX库内部，我们可以在自己的代码中使用正确的方法
        pass
    
    def get_ticker(self, instId: str) -> Optional[Dict]:
        """获取单个交易对行情"""
        try:
            result = self.api.market.get_ticker(instId=instId)
            if result['code'] == '0' and result['data']:
                return result['data'][0]
            else:
                print(f"获取 {instId} 行情失败: {result.get('msg', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"获取 {instId} 行情异常: {str(e)}")
            return None
    
    def get_all_tickers(self) -> Dict[str, Dict]:
        """获取配置中所有交易对的行情"""
        tickers = {}
        symbols = self.get_trading_pairs()
        print(f"正在获取 {len(symbols)} 个交易对的实时行情...")
        
        for instId in symbols:
            ticker_data = self.get_ticker(instId)
            if ticker_data:
                tickers[instId] = ticker_data
                print(f"✅ 成功获取 {instId} 实时数据")
            else:
                print(f"❌ 无法获取 {instId} 实时数据")
            time.sleep(2)  # 限速
        return tickers
    
    def get_kline(self, 
                  instId: str, 
                  bar: str = None, 
                  limit: int = 100,
                  after: str = None) -> Optional[pd.DataFrame]:
        """
        获取K线数据
        
        Args:
            instId: 交易对
            bar: K线周期，如果为None则使用配置中的timeframe
            limit: 数据条数
            after: 在此时间之后的数据
        """
        if bar is None:
            bar = self.get_timeframe()
            
        try:
            # 构建请求参数
            params = {
                'instId': instId,
                'bar': bar,
                'limit': str(limit)
            }
            if after:
                params['after'] = after
                
            result = self.api.market.get_candles(**params)
            if result['code'] == '0':
                return self._parse_candles_data(result['data'])
            else:
                print(f"获取 {instId} K线失败: {result.get('msg', 'Unknown error')}, 参数: bar={bar}")
                return None
        except Exception as e:
            print(f"获取 {instId} K线异常: {str(e)}")
            return None
    
    def _parse_candles_data(self, candles_data: List) -> pd.DataFrame:
        """解析K线数据"""
        if not candles_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(candles_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'
        ])
        
        # 数据类型转换
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def get_historical_klines(self, 
                            instId: str, 
                            bar: str = None,
                            days: int = None) -> pd.DataFrame:
        """
        获取历史K线数据（自动分页）
        
        Args:
            instId: 交易对
            bar: K线周期，如果为None则使用配置中的timeframe
            days: 数据天数，如果为None则使用配置中的historical_days
        """
        if bar is None:
            bar = self.get_timeframe()
        if days is None:
            days = self.get_historical_days()
            
        print(f"获取 {instId} 的 {days} 天数据，时间框架: {bar}")
        
        all_data = pd.DataFrame()
        limit = 240  # 每次最多240条
        
        # 计算需要的总条数
        total_bars = self._calculate_total_bars(bar, days)
        
        if total_bars <= 0:
            print(f"❌ 时间框架 {bar} 和天数 {days} 计算出的条数为0")
            return all_data
        
        print(f"需要获取大约 {total_bars} 条K线数据")
        
        # 分批获取数据
        retrieved_bars = 0
        after = None
        
        while retrieved_bars < total_bars:
            current_limit = min(limit, total_bars - retrieved_bars)
            
            try:
                kline_data = self.get_kline(instId, bar, current_limit, after)
                
                if kline_data is not None and not kline_data.empty:
                    # 如果是第一次获取，直接赋值
                    if all_data.empty:
                        all_data = kline_data
                    else:
                        # 合并数据，确保时间顺序
                        all_data = pd.concat([kline_data, all_data])
                        all_data = all_data[~all_data.index.duplicated(keep='first')]
                        all_data.sort_index(inplace=True)
                    
                    retrieved_bars += len(kline_data)
                    
                    # 设置下一次请求的起始时间
                    if not kline_data.empty:
                        after = kline_data.index[0].strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                    
                    print(f"  ✅ 已获取 {len(kline_data)} 条数据，总计 {retrieved_bars}/{total_bars}")
                else:
                    print(f"  ❌ 获取数据失败，停止请求")
                    break
                
                # 限速，避免请求过快
                time.sleep(12)
                
            except Exception as e:
                print(f"  ❌ 获取数据时发生异常: {str(e)}")
                break
        
        if not all_data.empty:
            print(f"✅ 成功获取 {instId} 的 {len(all_data)} 条历史数据")
        else:
            print(f"❌ 未能获取 {instId} 的历史数据")
            
        return all_data
    
    def _calculate_total_bars(self, bar: str, days: int) -> int:
        """根据时间框架和天数计算需要的K线条数"""
        try:
            if bar.endswith('m'):
                minutes = int(bar[:-1])
                return (days * 24 * 60) // minutes
            elif bar.endswith('H'):
                hours = int(bar[:-1])
                return (days * 24) // hours
            elif bar.endswith('D'):
                return days
            elif bar.endswith('W'):
                return days // 7
            elif bar.endswith('M'):
                return days // 30
            else:
                # 默认按小时计算
                return days * 24
        except:
            # 如果计算失败，返回默认值
            return days * 24
    
    def get_all_historical_klines(self) -> Dict[str, pd.DataFrame]:
        """获取配置中所有交易对的历史K线数据"""
        market_data = {}
        symbols = self.get_trading_pairs()
        print(f"开始获取 {len(symbols)} 个交易对的历史数据...")
        
        success_count = 0
        for instId in symbols:
            print(f"获取 {instId} 的历史数据...")
            kline_data = self.get_historical_klines(instId)
            if kline_data is not None and not kline_data.empty:
                market_data[instId] = kline_data
                success_count += 1
                print(f"✅ 成功获取 {instId} 的历史数据，共 {len(kline_data)} 条")
            else:
                print(f"❌ 获取 {instId} 的历史数据失败")
            time.sleep(2)  # 限速
        
        print(f"历史数据获取完成: 成功 {success_count}/{len(symbols)} 个交易对")
        return market_data
    
    def get_instruments(self, instType: str = "SPOT") -> List[Dict]:
        """获取可交易产品信息"""
        try:
            result = self.api.public.get_instruments(instType=instType)
            if result['code'] == '0':
                instruments = result['data']
                print(f"✅ 获取到 {len(instruments)} 个{instType}产品")
                return instruments
            else:
                print(f"❌ 获取产品信息失败: {result.get('msg', 'Unknown error')}")
                return []
        except Exception as e:
            print(f"❌ 获取产品信息异常: {str(e)}")
            return []
    
    def get_funding_rate(self, instId: str) -> Optional[Dict]:
        """获取资金费率（仅永续合约有效）"""
        try:
            result = self.api.market.get_funding_rate(instId=instId)
            if result['code'] == '0' and result['data']:
                return result['data'][0]
            else:
                print(f"⚠️ 获取 {instId} 资金费率失败: {result.get('msg', 'Not a swap instrument')}")
                return None
        except Exception as e:
            print(f"❌ 获取资金费率异常: {str(e)}")
            return None

    def get_realtime_data(self) -> Dict[str, Dict]:
        """获取所有交易对的实时数据"""
        return self.get_all_tickers()
    
    def validate_instruments(self):
        """验证配置的交易对是否可用"""
        print("验证交易对配置...")
        available_instruments = self.get_instruments("SPOT")
        available_pairs = [inst['instId'] for inst in available_instruments]
        
        configured_pairs = self.get_trading_pairs()
        
        valid_pairs = []
        invalid_pairs = []
        
        for pair in configured_pairs:
            if pair in available_pairs:
                valid_pairs.append(pair)
            else:
                invalid_pairs.append(pair)
        
        print(f"✅ 有效交易对: {valid_pairs}")
        if invalid_pairs:
            print(f"❌ 无效交易对: {invalid_pairs}")
            print(f"💡 建议使用以下格式: BTC-USDT, ETH-USDT, SOL-USDT")
        
        return valid_pairs, invalid_pairs