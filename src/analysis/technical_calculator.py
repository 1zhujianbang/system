# src/analysis/technical_calculator.py
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional

class TechnicalCalculator:
    """技术指标计算器"""
    
    def __init__(self, rsi_period: int = 14, 
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 bb_period: int = 20, bb_std: int = 2,
                 atr_period: int = 14):
        """
        初始化技术指标计算器
        
        Args:
            rsi_period: RSI周期
            macd_fast: MACD快线周期
            macd_slow: MACD慢线周期  
            macd_signal: MACD信号线周期
            bb_period: 布林带周期
            bb_std: 布林带标准差
            atr_period: ATR周期
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含所有技术指标的DataFrame
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # 基础OHLCV数据（确保存在）
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")
        
        # 1. 移动平均线
        df = self._calculate_moving_averages(df)
        
        # 2. RSI
        df = self._calculate_rsi(df)
        
        # 3. MACD
        df = self._calculate_macd(df)
        
        # 4. 布林带
        df = self._calculate_bollinger_bands(df)
        
        # 5. ATR
        df = self._calculate_atr(df)
        
        # 6. 成交量指标
        df = self._calculate_volume_indicators(df)
        
        # 7. 价格变化率
        df = self._calculate_price_changes(df)
        
        # 8. 其他衍生指标
        df = self._calculate_derived_indicators(df)
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线"""
        # 简单移动平均
        df['ma_5'] = talib.SMA(df['close'], timeperiod=5)
        df['ma_10'] = talib.SMA(df['close'], timeperiod=10)
        df['ma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['ma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ma_200'] = talib.SMA(df['close'], timeperiod=200)
        
        # 指数移动平均
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算RSI"""
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # RSI衍生指标
        df['rsi_signal'] = 0
        df.loc[df['rsi'] < 30, 'rsi_signal'] = 1  # 超卖
        df.loc[df['rsi'] > 70, 'rsi_signal'] = -1  # 超买
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD"""
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'], 
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow, 
            signalperiod=self.macd_signal
        )
        
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # MACD信号
        df['macd_cross'] = 0
        df.loc[(macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1)), 'macd_cross'] = 1  # 金叉
        df.loc[(macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1)), 'macd_cross'] = -1  # 死叉
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带"""
        upper, middle, lower = talib.BBANDS(
            df['close'],
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )
        
        df['bollinger_upper'] = upper
        df['bollinger_middle'] = middle
        df['bollinger_lower'] = lower
        
        # 布林带位置和宽度
        df['bollinger_position'] = (df['close'] - lower) / (upper - lower)
        df['bollinger_width'] = (upper - lower) / middle
        
        # 布林带信号
        df['bb_signal'] = 0
        df.loc[df['close'] < lower, 'bb_signal'] = 1  # 下轨下方，可能反弹
        df.loc[df['close'] > upper, 'bb_signal'] = -1  # 上轨上方，可能回调
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算ATR（平均真实波幅）"""
        df['atr'] = talib.ATR(
            df['high'], df['low'], df['close'],
            timeperiod=self.atr_period
        )
        
        # ATR比率（相对于价格）
        df['atr_ratio'] = df['atr'] / df['close']
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标"""
        # 成交量移动平均
        df['volume_ma_5'] = talib.SMA(df['volume'], timeperiod=5)
        df['volume_ma_20'] = talib.SMA(df['volume'], timeperiod=20)
        
        # 成交量比率
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # 量价关系
        df['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
        
        # OBV（能量潮）
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        return df
    
    def _calculate_price_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格变化率"""
        # 收益率
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_20'] = df['close'].pct_change(20)
        
        # 价格波动率
        df['volatility_5'] = df['returns_1'].rolling(5).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        
        # 高低价范围
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['open_close_range'] = (df['close'] - df['open']) / df['open']
        
        return df
    
    def _calculate_derived_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算其他衍生指标"""
        # 价格位置（相对于近期高低点）
        df['price_position_20'] = (df['close'] - df['low'].rolling(20).min()) / \
                                 (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        
        # 动量指标
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # 威廉指标
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CCI（商品通道指数）
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """获取所有特征列名"""
        base_features = ['open', 'high', 'low', 'close', 'volume']
        technical_features = [
            'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_200',
            'ema_12', 'ema_26',
            'rsi', 'rsi_signal',
            'macd', 'macd_signal', 'macd_hist', 'macd_cross',
            'bollinger_upper', 'bollinger_lower', 'bollinger_middle',
            'bollinger_position', 'bollinger_width', 'bb_signal',
            'atr', 'atr_ratio',
            'volume_ma_5', 'volume_ma_20', 'volume_ratio', 
            'volume_price_trend', 'obv',
            'returns_1', 'returns_5', 'returns_20',
            'volatility_5', 'volatility_20',
            'high_low_range', 'open_close_range',
            'price_position_20', 'momentum_5', 'momentum_10',
            'williams_r', 'cci'
        ]
        return base_features + technical_features
    
    def validate_features(self, df: pd.DataFrame, required_features: List[str]) -> List[str]:
        """
        验证数据是否包含所需特征
        
        Args:
            df: 数据DataFrame
            required_features: 需要的特征列表
            
        Returns:
            缺失的特征列表
        """
        missing_features = []
        for feature in required_features:
            if feature not in df.columns:
                missing_features.append(feature)
        return missing_features