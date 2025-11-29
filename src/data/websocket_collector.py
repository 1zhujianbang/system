# src/data/websocket_collector.py
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import json
import threading
import websockets
from typing import Dict, List, Optional, Callable, Any
from ..config.config_manager import UserConfig, DataConfig
from okx.websocket.WsPublicAsync import WsPublicAsync
from okx.websocket.WsPublic import WsPublic

class ProxyWebSocketClient:
    """代理WebSocket客户端"""
    
    def __init__(self, proxy_url: str = None):
        """
        初始化代理WebSocket客户端
        
        Args:
            proxy_url: 代理服务器URL，格式: http://username:password@host:port
        """
        self.proxy_url = proxy_url
        self.connector = None
        
    async def create_websocket_connection(self, url: str) -> websockets.WebSocketClientProtocol:
        """创建通过代理的WebSocket连接"""
        if self.proxy_url:
            # 解析代理URL
            proxy_parts = self.proxy_url.replace('http://', '').replace('https://', '').split('@')
            if len(proxy_parts) == 2:
                auth, server = proxy_parts
                username, password = auth.split(':')
                host, port = server.split(':')
            else:
                server = proxy_parts[0]
                username, password = None, None
                host, port = server.split(':')
            
            # 创建代理连接
            proxy_auth = aiohttp.BasicAuth(username, password) if username and password else None
            self.connector = aiohttp.TCPConnector()
            
            # 通过代理连接WebSocket
            return await websockets.connect(
                url,
                proxy=f"http://{host}:{port}",
                proxy_headers=proxy_auth
            )
        else:
            # 直接连接
            return await websockets.connect(url)

class OKXWebSocketCollector:
    """OKX WebSocket数据收集器 - 支持代理转发"""
    
    def __init__(self, user_config: UserConfig, data_config: DataConfig):
        """
        初始化WebSocket收集器
        
        Args:
            user_config: 用户配置
            data_config: 数据配置
        """
        self.user_config = user_config
        self.data_config = data_config
        
        # 代理设置
        self.proxy_client = ProxyWebSocketClient(proxy_url=data_config.proxy)
        
        # WebSocket客户端
        self.ws_public_async = None
        self.ws_public = None
        self.ws_connected = False
        
        # 数据存储
        self.realtime_data = {}
        self.historical_data = {}
        self.instruments_data = {}
        self.funding_rates = {}
        self.mark_prices = {}
        self.open_interest = {}
        self.liquidation_orders = {}
        
        # 回调函数
        self.callbacks = {}
        
        # 线程锁
        self.data_lock = threading.Lock()
        
        # 连接状态监控
        self.connection_stats = {
            'total_messages': 0,
            'last_message_time': None,
            'connection_errors': 0,
            'reconnect_count': 0
        }
        
        # 初始化数据缓冲区
        self._init_data_buffers()
    
    def _init_data_buffers(self):
        """初始化数据缓冲区"""
        symbols = self.get_trading_pairs()
        for symbol in symbols:
            self.realtime_data[symbol] = {}
            self.historical_data[symbol] = pd.DataFrame()
            self.mark_prices[symbol] = {}
            self.funding_rates[symbol] = {}
            self.open_interest[symbol] = {}
    
    def get_trading_pairs(self) -> List[str]:
        """从配置中获取交易对列表"""
        return self.user_config.trading_pairs
    
    def get_timeframe(self) -> str:
        """从配置中获取时间框架"""
        return self.data_config.timeframe
    
    # ==================== 代理连接管理 ====================
    
    async def create_proxied_websocket(self, url: str) -> websockets.WebSocketClientProtocol:
        """创建通过代理的WebSocket连接"""
        return await self.proxy_client.create_websocket_connection(url)
    
    def _setup_proxy_for_okx_library(self):
        """为OKX库设置代理"""
        if self.data_config.proxy:
            import os
            # 设置环境变量
            os.environ['HTTP_PROXY'] = self.data_config.proxy
            os.environ['HTTPS_PROXY'] = self.data_config.proxy
            os.environ['ALL_PROXY'] = self.data_config.proxy
            
            print(f"🔌 已设置代理: {self.data_config.proxy}")
    
    # ==================== WebSocket连接管理 ====================
    
    async def start_async_websocket(self, callbacks: Dict[str, Callable] = None):
        """启动异步WebSocket连接（支持代理）"""
        if callbacks:
            self.callbacks.update(callbacks)
            
        try:
            # 设置代理
            self._setup_proxy_for_okx_library()
            
            # 创建WebSocket客户端
            self.ws_public_async = WsPublicAsync(
                url="wss://ws.okx.com:8443/ws/v5/public",
                proxy_host=self.data_config.proxy  # OKX库支持直接传入代理
            )
            
            await self.ws_public_async.start()
            self.ws_connected = True
            print("✅ 异步WebSocket连接已启动（通过代理）")
            
            # 订阅频道
            await self._subscribe_all_channels_async()
            
        except Exception as e:
            print(f"❌ 启动异步WebSocket失败: {str(e)}")
            self.connection_stats['connection_errors'] += 1
            self.ws_connected = False
    
    def start_sync_websocket(self, callbacks: Dict[str, Callable] = None):
        """启动同步WebSocket连接（支持代理）"""
        if callbacks:
            self.callbacks.update(callbacks)
            
        try:
            # 设置代理
            self._setup_proxy_for_okx_library()
            
            # 创建WebSocket客户端
            self.ws_public = WsPublic(
                url="wss://ws.okx.com:8443/ws/v5/public",
                proxy_host=self.data_config.proxy  # OKX库支持直接传入代理
            )
            
            self.ws_public.start()
            self.ws_connected = True
            print("✅ 同步WebSocket连接已启动（通过代理）")
            
            # 订阅频道
            self._subscribe_all_channels_sync()
            
        except Exception as e:
            print(f"❌ 启动同步WebSocket失败: {str(e)}")
            self.connection_stats['connection_errors'] += 1
            self.ws_connected = False
    
    async def start_custom_websocket(self, callbacks: Dict[str, Callable] = None):
        """启动自定义WebSocket连接（完全控制代理）"""
        if callbacks:
            self.callbacks.update(callbacks)
            
        try:
            # 使用自定义代理连接
            url = "wss://ws.okx.com:8443/ws/v5/public"
            self.custom_ws = await self.create_proxied_websocket(url)
            self.ws_connected = True
            print("✅ 自定义WebSocket连接已启动（通过代理）")
            
            # 启动消息处理循环
            asyncio.create_task(self._custom_message_handler())
            
            # 订阅频道
            await self._subscribe_custom_channels()
            
        except Exception as e:
            print(f"❌ 启动自定义WebSocket失败: {str(e)}")
            self.connection_stats['connection_errors'] += 1
            self.ws_connected = False
    
    async def _custom_message_handler(self):
        """自定义消息处理器"""
        try:
            async for message in self.custom_ws:
                self.connection_stats['total_messages'] += 1
                self.connection_stats['last_message_time'] = datetime.now(timezone.utc)
                
                # 解析消息
                data = json.loads(message)
                
                # 根据频道类型分发处理
                if 'arg' in data and 'channel' in data['arg']:
                    channel = data['arg']['channel']
                    if channel == 'tickers':
                        self._handle_ticker_data(data)
                    elif channel.startswith('candle'):
                        self._handle_candle_data(data)
                    elif channel == 'instruments':
                        self._handle_instruments_data(data)
                    elif channel == 'mark-price':
                        self._handle_mark_price_data(data)
                    elif channel == 'funding-rate':
                        self._handle_funding_rate_data(data)
                    elif channel == 'open-interest':
                        self._handle_open_interest_data(data)
                
        except websockets.exceptions.ConnectionClosed:
            print("❌ WebSocket连接已关闭")
            self.ws_connected = False
        except Exception as e:
            print(f"❌ 消息处理异常: {str(e)}")
    
    async def _subscribe_custom_channels(self):
        """自定义订阅频道"""
        # 订阅产品信息
        await self._send_custom_message({
            "op": "subscribe",
            "args": [{
                "channel": "instruments",
                "instType": "SPOT"
            }]
        })
        
        # 订阅实时行情
        symbols = self.get_trading_pairs()
        ticker_args = [{"channel": "tickers", "instId": symbol} for symbol in symbols]
        await self._send_custom_message({
            "op": "subscribe",
            "args": ticker_args
        })
        
        # 订阅K线数据
        timeframe = self.get_timeframe()
        candle_channel = f"candle{timeframe}"
        candle_args = [{"channel": candle_channel, "instId": symbol} for symbol in symbols]
        await self._send_custom_message({
            "op": "subscribe",
            "args": candle_args
        })
        
        print(f"✅ 自定义订阅完成: {len(symbols)} 个交易对")
    
    async def _send_custom_message(self, message: dict):
        """发送自定义消息"""
        if self.ws_connected and self.custom_ws:
            await self.custom_ws.send(json.dumps(message))
    
    # ==================== 频道订阅方法 ====================
    
    async def _subscribe_all_channels_async(self):
        """异步订阅所有频道"""
        if not self.ws_connected:
            return
            
        # 订阅产品信息
        await self.subscribe_instruments_async()
        
        # 订阅实时行情
        await self.subscribe_tickers_async()
        
        # 订阅K线数据
        await self.subscribe_candles_async()
        
        # 订阅标记价格
        await self.subscribe_mark_price_async()
        
        # 订阅资金费率
        await self.subscribe_funding_rate_async()
        
        # 订阅持仓总量
        await self.subscribe_open_interest_async()
    
    def _subscribe_all_channels_sync(self):
        """同步订阅所有频道"""
        if not self.ws_connected:
            return
            
        # 订阅产品信息
        self.subscribe_instruments_sync()
        
        # 订阅实时行情
        self.subscribe_tickers_sync()
        
        # 订阅K线数据
        self.subscribe_candles_sync()
        
        # 订阅标记价格
        self.subscribe_mark_price_sync()
        
        # 订阅资金费率
        self.subscribe_funding_rate_sync()
        
        # 订阅持仓总量
        self.subscribe_open_interest_sync()
    
    async def subscribe_instruments_async(self):
        """订阅产品信息频道"""
        args = [{
            "channel": "instruments",
            "instType": "SPOT"
        }]
        await self.ws_public_async.subscribe(args, callback=self._handle_instruments_data)
        print("✅ 已订阅产品信息频道")
    
    def subscribe_instruments_sync(self):
        """同步订阅产品信息频道"""
        args = [{
            "channel": "instruments",
            "instType": "SPOT"
        }]
        self.ws_public.subscribe(args, callback=self._handle_instruments_data)
        print("✅ 已订阅产品信息频道")
    
    async def subscribe_tickers_async(self):
        """订阅实时行情频道"""
        symbols = self.get_trading_pairs()
        args = [{"channel": "tickers", "instId": symbol} for symbol in symbols]
        await self.ws_public_async.subscribe(args, callback=self._handle_ticker_data)
        print(f"✅ 已订阅 {len(symbols)} 个交易对的实时行情")
    
    def subscribe_tickers_sync(self):
        """同步订阅实时行情频道"""
        symbols = self.get_trading_pairs()
        args = [{"channel": "tickers", "instId": symbol} for symbol in symbols]
        self.ws_public.subscribe(args, callback=self._handle_ticker_data)
        print(f"✅ 已订阅 {len(symbols)} 个交易对的实时行情")
    
    async def subscribe_candles_async(self):
        """订阅K线数据频道"""
        symbols = self.get_trading_pairs()
        timeframe = self.get_timeframe()
        channel = f"candle{timeframe}"
        args = [{"channel": channel, "instId": symbol} for symbol in symbols]
        await self.ws_public_async.subscribe(args, callback=self._handle_candle_data)
        print(f"✅ 已订阅 {len(symbols)} 个交易对的K线数据 ({timeframe})")
    
    def subscribe_candles_sync(self):
        """同步订阅K线数据频道"""
        symbols = self.get_trading_pairs()
        timeframe = self.get_timeframe()
        channel = f"candle{timeframe}"
        args = [{"channel": channel, "instId": symbol} for symbol in symbols]
        self.ws_public.subscribe(args, callback=self._handle_candle_data)
        print(f"✅ 已订阅 {len(symbols)} 个交易对的K线数据 ({timeframe})")
    
    async def subscribe_mark_price_async(self):
        """订阅标记价格频道"""
        symbols = self.get_trading_pairs()
        args = [{"channel": "mark-price", "instId": symbol} for symbol in symbols]
        await self.ws_public_async.subscribe(args, callback=self._handle_mark_price_data)
        print(f"✅ 已订阅 {len(symbols)} 个交易对的标记价格")
    
    def subscribe_mark_price_sync(self):
        """同步订阅标记价格频道"""
        symbols = self.get_trading_pairs()
        args = [{"channel": "mark-price", "instId": symbol} for symbol in symbols]
        self.ws_public.subscribe(args, callback=self._handle_mark_price_data)
        print(f"✅ 已订阅 {len(symbols)} 个交易对的标记价格")
    
    async def subscribe_funding_rate_async(self):
        """订阅资金费率频道（永续合约）"""
        swap_symbols = [symbol for symbol in self.get_trading_pairs() if "SWAP" in symbol]
        if swap_symbols:
            args = [{"channel": "funding-rate", "instId": symbol} for symbol in swap_symbols]
            await self.ws_public_async.subscribe(args, callback=self._handle_funding_rate_data)
            print(f"✅ 已订阅 {len(swap_symbols)} 个永续合约的资金费率")
    
    def subscribe_funding_rate_sync(self):
        """同步订阅资金费率频道（永续合约）"""
        swap_symbols = [symbol for symbol in self.get_trading_pairs() if "SWAP" in symbol]
        if swap_symbols:
            args = [{"channel": "funding-rate", "instId": symbol} for symbol in swap_symbols]
            self.ws_public.subscribe(args, callback=self._handle_funding_rate_data)
            print(f"✅ 已订阅 {len(swap_symbols)} 个永续合约的资金费率")
    
    async def subscribe_open_interest_async(self):
        """订阅持仓总量频道"""
        symbols = self.get_trading_pairs()
        args = [{"channel": "open-interest", "instId": symbol} for symbol in symbols]
        await self.ws_public_async.subscribe(args, callback=self._handle_open_interest_data)
        print(f"✅ 已订阅 {len(symbols)} 个交易对的持仓总量")
    
    def subscribe_open_interest_sync(self):
        """同步订阅持仓总量频道"""
        symbols = self.get_trading_pairs()
        args = [{"channel": "open-interest", "instId": symbol} for symbol in symbols]
        self.ws_public.subscribe(args, callback=self._handle_open_interest_data)
        print(f"✅ 已订阅 {len(symbols)} 个交易对的持仓总量")
    
    # ==================== 数据处理方法 ====================
    
    def _handle_instruments_data(self, message):
        """处理产品信息数据"""
        try:
            self.connection_stats['total_messages'] += 1
            self.connection_stats['last_message_time'] = datetime.now(timezone.utc)
            
            if 'data' in message and message['data']:
                with self.data_lock:
                    for instrument in message['data']:
                        inst_id = instrument['instId']
                        self.instruments_data[inst_id] = instrument
                
                print(f"📋 更新产品信息: {len(message['data'])} 个产品")
                
                if 'instruments' in self.callbacks:
                    self.callbacks['instruments'](message['data'])
                    
        except Exception as e:
            print(f"❌ 处理产品信息数据异常: {str(e)}")
    
    def _handle_ticker_data(self, message):
        """处理实时行情数据"""
        try:
            self.connection_stats['total_messages'] += 1
            self.connection_stats['last_message_time'] = datetime.now(timezone.utc)
            
            if 'data' in message and message['data']:
                ticker_data = message['data'][0]
                inst_id = ticker_data['instId']
                
                with self.data_lock:
                    self.realtime_data[inst_id] = {
                        'last': float(ticker_data.get('last', 0)),
                        'bid': float(ticker_data.get('bidPx', 0)),
                        'ask': float(ticker_data.get('askPx', 0)),
                        'high_24h': float(ticker_data.get('high24h', 0)),
                        'low_24h': float(ticker_data.get('low24h', 0)),
                        'volume_24h': float(ticker_data.get('vol24h', 0)),
                        'timestamp': datetime.now(timezone.utc)
                    }
                
                if 'ticker' in self.callbacks:
                    self.callbacks['ticker'](inst_id, self.realtime_data[inst_id])
                
                print(f"📊 {inst_id} 实时价格: {self.realtime_data[inst_id]['last']}")
                
        except Exception as e:
            print(f"❌ 处理实时行情数据异常: {str(e)}")
    
    def _handle_candle_data(self, message):
        """处理K线数据"""
        try:
            self.connection_stats['total_messages'] += 1
            self.connection_stats['last_message_time'] = datetime.now(timezone.utc)
            
            if 'data' in message and message['data']:
                candle_data = message['data'][0]
                inst_id = candle_data['instId']
                
                kline = {
                    'timestamp': pd.to_datetime(candle_data[0], unit='ms'),
                    'open': float(candle_data[1]),
                    'high': float(candle_data[2]),
                    'low': float(candle_data[3]),
                    'close': float(candle_data[4]),
                    'volume': float(candle_data[5]),
                    'confirm': candle_data[6] == '1'
                }
                
                with self.data_lock:
                    if inst_id not in self.historical_data:
                        self.historical_data[inst_id] = pd.DataFrame()
                    
                    new_row = pd.DataFrame([kline])
                    new_row.set_index('timestamp', inplace=True)
                    
                    if self.historical_data[inst_id].empty:
                        self.historical_data[inst_id] = new_row
                    else:
                        if kline['timestamp'] not in self.historical_data[inst_id].index:
                            self.historical_data[inst_id] = pd.concat([
                                self.historical_data[inst_id], new_row
                            ])
                            self.historical_data[inst_id] = self.historical_data[inst_id][
                                ~self.historical_data[inst_id].index.duplicated(keep='last')
                            ]
                            self.historical_data[inst_id].sort_index(inplace=True)
                
                if 'candle' in self.callbacks:
                    self.callbacks['candle'](inst_id, kline)
                
                if kline['confirm']:
                    print(f"🕯️  {inst_id} K线确认: {kline['close']} (时间: {kline['timestamp']})")
                
        except Exception as e:
            print(f"❌ 处理K线数据异常: {str(e)}")
    
    def _handle_mark_price_data(self, message):
        """处理标记价格数据"""
        try:
            self.connection_stats['total_messages'] += 1
            self.connection_stats['last_message_time'] = datetime.now(timezone.utc)
            
            if 'data' in message and message['data']:
                mark_data = message['data'][0]
                inst_id = mark_data['instId']
                
                with self.data_lock:
                    self.mark_prices[inst_id] = {
                        'mark_px': float(mark_data.get('markPx', 0)),
                        'timestamp': datetime.now(timezone.utc)
                    }
                
                if 'mark_price' in self.callbacks:
                    self.callbacks['mark_price'](inst_id, self.mark_prices[inst_id])
                
                print(f"🏷️  {inst_id} 标记价格: {self.mark_prices[inst_id]['mark_px']}")
                
        except Exception as e:
            print(f"❌ 处理标记价格数据异常: {str(e)}")
    
    def _handle_funding_rate_data(self, message):
        """处理资金费率数据"""
        try:
            self.connection_stats['total_messages'] += 1
            self.connection_stats['last_message_time'] = datetime.now(timezone.utc)
            
            if 'data' in message and message['data']:
                funding_data = message['data'][0]
                inst_id = funding_data['instId']
                
                with self.data_lock:
                    self.funding_rates[inst_id] = {
                        'funding_rate': float(funding_data.get('fundingRate', 0)),
                        'next_funding_rate': float(funding_data.get('nextFundingRate', 0)),
                        'funding_time': pd.to_datetime(funding_data.get('fundingTime', 0), unit='ms'),
                        'next_funding_time': pd.to_datetime(funding_data.get('nextFundingTime', 0), unit='ms'),
                        'timestamp': datetime.now(timezone.utc)
                    }
                
                if 'funding_rate' in self.callbacks:
                    self.callbacks['funding_rate'](inst_id, self.funding_rates[inst_id])
                
                print(f"💰 {inst_id} 资金费率: {self.funding_rates[inst_id]['funding_rate']:.6f}")
                
        except Exception as e:
            print(f"❌ 处理资金费率数据异常: {str(e)}")
    
    def _handle_open_interest_data(self, message):
        """处理持仓总量数据"""
        try:
            self.connection_stats['total_messages'] += 1
            self.connection_stats['last_message_time'] = datetime.now(timezone.utc)
            
            if 'data' in message and message['data']:
                oi_data = message['data'][0]
                inst_id = oi_data['instId']
                
                with self.data_lock:
                    self.open_interest[inst_id] = {
                        'oi': float(oi_data.get('oi', 0)),
                        'oi_ccy': float(oi_data.get('oiCcy', 0)),
                        'oi_usd': float(oi_data.get('oiUsd', 0)),
                        'timestamp': datetime.now(timezone.utc)
                    }
                
                if 'open_interest' in self.callbacks:
                    self.callbacks['open_interest'](inst_id, self.open_interest[inst_id])
                
                print(f"📈 {inst_id} 持仓总量: {self.open_interest[inst_id]['oi']:.2f}")
                
        except Exception as e:
            print(f"❌ 处理持仓总量数据异常: {str(e)}")
    
    # ==================== 数据获取方法 ====================
    
    def get_realtime_data(self, symbol: str = None) -> Dict:
        """获取实时数据"""
        with self.data_lock:
            if symbol:
                return self.realtime_data.get(symbol, {})
            return self.realtime_data.copy()
    
    def get_historical_data(self, symbol: str = None) -> Dict[str, pd.DataFrame]:
        """获取历史K线数据"""
        with self.data_lock:
            if symbol:
                return {symbol: self.historical_data.get(symbol, pd.DataFrame())}
            return self.historical_data.copy()
    
    def get_mark_prices(self, symbol: str = None) -> Dict:
        """获取标记价格"""
        with self.data_lock:
            if symbol:
                return self.mark_prices.get(symbol, {})
            return self.mark_prices.copy()
    
    def get_funding_rates(self, symbol: str = None) -> Dict:
        """获取资金费率"""
        with self.data_lock:
            if symbol:
                return self.funding_rates.get(symbol, {})
            return self.funding_rates.copy()
    
    def get_open_interest(self, symbol: str = None) -> Dict:
        """获取持仓总量"""
        with self.data_lock:
            if symbol:
                return self.open_interest.get(symbol, {})
            return self.open_interest.copy()
    
    def get_instruments_data(self, symbol: str = None) -> Dict:
        """获取产品信息"""
        with self.data_lock:
            if symbol:
                return self.instruments_data.get(symbol, {})
            return self.instruments_data.copy()
    
    # ==================== 连接管理 ====================
    
    async def stop_async_websocket(self):
        """停止异步WebSocket连接"""
        if self.ws_public_async and self.ws_connected:
            await self.ws_public_async.close()
            self.ws_connected = False
            print("✅ 异步WebSocket连接已停止")
    
    def stop_sync_websocket(self):
        """停止同步WebSocket连接"""
        if self.ws_public and self.ws_connected:
            self.ws_public.stop()
            self.ws_connected = False
            print("✅ 同步WebSocket连接已停止")
    
    async def stop_custom_websocket(self):
        """停止自定义WebSocket连接"""
        if self.custom_ws and self.ws_connected:
            await self.custom_ws.close()
            self.ws_connected = False
            print("✅ 自定义WebSocket连接已停止")
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.ws_connected
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return self.connection_stats.copy()
    
    # ==================== 工具方法 ====================
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        with self.data_lock:
            return {
                'realtime_data_count': len(self.realtime_data),
                'historical_data_count': {symbol: len(df) for symbol, df in self.historical_data.items()},
                'mark_prices_count': len(self.mark_prices),
                'funding_rates_count': len(self.funding_rates),
                'open_interest_count': len(self.open_interest),
                'instruments_count': len(self.instruments_data),
                'websocket_connected': self.ws_connected,
                'connection_stats': self.connection_stats
            }


# 使用示例和测试
async def proxy_websocket_demo():
    """代理WebSocket使用示例"""
    from src.config.config_manager import UserConfig, DataConfig
    
    # 配置 - 设置代理
    user_config = UserConfig(
        trading_pairs=['BTC-USDT', 'ETH-USDT', 'SOL-USDT'],
        initial_capital=10000.0,
        risk_appetite='moderate'
    )
    
    data_config = DataConfig(
        timeframe="1H",
        historical_days=30,
        proxy="http://username:password@proxy-server:8080"  # 替换为实际代理
    )
    
    # 创建WebSocket收集器
    collector = OKXWebSocketCollector(user_config, data_config)
    
    # 定义回调函数
    def on_ticker_update(symbol, data):
        print(f"🚀 {symbol} 价格更新: {data['last']}")
    
    def on_candle_update(symbol, kline):
        if kline['confirm']:
            print(f"📈 {symbol} K线确认: {kline['close']}")
    
    callbacks = {
        'ticker': on_ticker_update,
        'candle': on_candle_update
    }
    
    # 启动WebSocket（选择一种方式）
    print("选择连接方式:")
    print("1. 异步WebSocket (使用OKX库)")
    print("2. 同步WebSocket (使用OKX库)") 
    print("3. 自定义WebSocket (完全控制代理)")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    try:
        if choice == "1":
            await collector.start_async_websocket(callbacks)
        elif choice == "2":
            collector.start_sync_websocket(callbacks)
        elif choice == "3":
            await collector.start_custom_websocket(callbacks)
        else:
            print("使用默认异步方式")
            await collector.start_async_websocket(callbacks)
        
        # 运行一段时间
        print("WebSocket运行中... 按 Ctrl+C 停止")
        await asyncio.sleep(60)
        
    except KeyboardInterrupt:
        print("\n正在停止WebSocket...")
    finally:
        # 停止连接
        if choice == "1" or choice == "3":
            await collector.stop_async_websocket()
        else:
            collector.stop_sync_websocket()
        
        # 打印统计信息
        stats = collector.get_connection_stats()
        print(f"\n连接统计:")
        print(f"总消息数: {stats['total_messages']}")
        print(f"连接错误: {stats['connection_errors']}")
        print(f"最后消息时间: {stats['last_message_time']}")
    
    return collector

if __name__ == "__main__":
    asyncio.run(proxy_websocket_demo())