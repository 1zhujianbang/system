import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
import gym
from gym import spaces
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TorchTradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, lookback_window=50):
        super(TorchTradingEnvironment, self).__init__()
        
        # 数据预处理 - 填充NaN值
        self.df = df.reset_index(drop=True).fillna(method='bfill').fillna(method='ffill')
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # 动作空间: [-1, 1] 连续动作
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # 状态空间维度
        self.state_dim = self._get_state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def _get_state_dim(self):
        """计算状态维度"""
        base_features = 4  # OHLC
        technical_features = 17
        account_features = 4
        history_features = self.lookback_window * 2
        
        return base_features + technical_features + account_features + history_features
    
    def _get_features(self, step):
        """获取完整状态特征"""
        if step < self.lookback_window:
            step = self.lookback_window
        
        row = self.df.iloc[step]
        
        # 基础价格特征 - 使用对数收益率归一化
        price_features = [
            np.log(row['open'] / row['close']),
            np.log(row['high'] / row['close']), 
            np.log(row['low'] / row['close']),
            0.0  # close相对于自己为0
        ]
        
        # 技术指标特征 - 进行归一化
        technical_features = [
            (row['ma_5'] - row['close']) / row['close'],
            (row['ma_10'] - row['close']) / row['close'],
            (row['ma_20'] - row['close']) / row['close'],
            (row['ma_50'] - row['close']) / row['close'],
            (row['ma_200'] - row['close']) / row['close'],
            (row['ema_12'] - row['close']) / row['close'],
            (row['ema_26'] - row['close']) / row['close'],
            (row['rsi'] - 50) / 50,  # RSI归一化到[-1,1]
            row['macd'] / (abs(row['close']) + 1e-8),
            row['macd_signal'] / (abs(row['close']) + 1e-8),
            (row['bollinger_upper'] - row['close']) / row['close'],
            (row['bollinger_middle'] - row['close']) / row['close'],
            (row['bollinger_lower'] - row['close']) / row['close'],
            row['atr'] / row['close'],
            np.log(row['volume'] + 1),
            np.log(row['volume_ma_5'] + 1),
            row['volume_ratio'] - 1
        ]
        
        # 账户状态特征
        account_features = [
            self.balance / self.initial_balance,
            self.position,  # 直接使用持仓比例，不用价格
            self.total_value / self.initial_balance,
            np.clip(self.returns, -1, 10)  # 限制收益率范围
        ]
        
        # 历史价格序列特征 - 使用对数收益率
        history_features = []
        current_price = row['close']
        for i in range(step - self.lookback_window, step):
            if i >= 0:
                hist_row = self.df.iloc[i]
                price_return = np.log(hist_row['close'] / current_price)
                volume_ratio = np.log(hist_row['volume'] / (row['volume'] + 1e-8) + 1)
                history_features.extend([price_return, volume_ratio])
            else:
                history_features.extend([0, 0])
        
        # 组合所有特征
        features = np.array(price_features + technical_features + account_features + history_features, 
                           dtype=np.float32)
        
        # 处理异常值
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features = np.clip(features, -10, 10)  # 限制特征范围
        
        return features
    
    def reset(self):
        """重置环境"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0  # 持仓数量
        self.position_value = 0.0  # 持仓价值
        self.total_value = self.initial_balance
        self.returns = 0.0
        self.trades = []
        self.done = False
        self.max_drawdown = 0.0
        self.peak_value = self.initial_balance
        
        return self._get_features(self.current_step)
    
    def step(self, action):
        """执行动作"""
        if self.done:
            return self._get_features(self.current_step), 0, True, {}
        
        current_price = self.df.iloc[self.current_step]['close']
        
        # 修复动作维度问题
        if isinstance(action, np.ndarray):
            action = action[0]  # 从数组中提取标量值
        else:
            action = float(action)

        action = np.clip(action, -1, 1)  # 确保动作在有效范围内
        
        # 计算目标持仓价值
        target_value = action * self.total_value
        current_position_value = self.position * current_price
        
        # 执行交易
        trade_value = target_value - current_position_value
        
        # 交易成本和滑点
        transaction_cost = abs(trade_value) * self.transaction_cost
        
        if abs(trade_value) > self.total_value * 0.01:  # 最小交易阈值
            # 更新持仓和余额
            if trade_value > 0:  # 买入
                shares_to_buy = trade_value / current_price
                self.position += shares_to_buy
                self.balance -= trade_value + transaction_cost
            else:  # 卖出
                shares_to_sell = abs(trade_value) / current_price
                self.position = max(0, self.position - shares_to_sell)  # 不能卖空
                self.balance += abs(trade_value) - transaction_cost
            
            self.trades.append({
                'step': self.current_step,
                'action': action,
                'price': current_price,
                'value': trade_value
            })
        
        # 更新总资产和收益
        self.total_value = self.balance + self.position * current_price
        self.returns = (self.total_value - self.initial_balance) / self.initial_balance
        
        # 更新最大回撤
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
        current_drawdown = (self.peak_value - self.total_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # 移动到下一步
        self.current_step += 1
        
        # 检查是否结束
        if self.current_step >= len(self.df) - 1:
            self.done = True
        
        # 计算奖励
        reward = self._calculate_reward(action, current_price)
        
        # 获取新状态
        next_state = self._get_features(self.current_step)
        
        return next_state, reward, self.done, {
            'total_value': self.total_value,
            'returns': self.returns,
            'max_drawdown': self.max_drawdown
        }
    
    def _calculate_reward(self, action, price):
        """平衡多空策略的奖励函数"""
        row = self.df.iloc[self.current_step]
        
        action_scalar = float(action)
        
        # 1. 基础收益奖励
        portfolio_return = (self.total_value - self.initial_balance) / self.initial_balance
        returns_reward = portfolio_return * 0.05
        
        # 2. 方向性奖励
        directional_reward = 0
        
        # 技术指标判断
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        bb_position = row['bollinger_position'] if not pd.isna(row['bollinger_position']) else 0.5
        macd = row['macd'] if not pd.isna(row['macd']) else 0
        macd_signal = row['macd_signal'] if not pd.isna(row['macd_signal']) else 0
        
        # 买入信号奖励
        buy_signals = 0
        if rsi < 35:
            buy_signals += 0.05
        if bb_position < 0.2:
            buy_signals += 0.05
        if macd > macd_signal:
            buy_signals += 0.05
        
        # 卖出信号奖励  
        sell_signals = 0
        if rsi > 65:
            sell_signals += 0.05
        if bb_position > 0.8:
            sell_signals += 0.05
        if macd < macd_signal:
            sell_signals += 0.05
        
        # 方向一致性奖励
        if buy_signals >= 2 and action_scalar > 0.2:
            directional_reward += 0.1
            # print(f"✅ 正确买入! RSI: {rsi:.1f}, 布林带: {bb_position:.2f}, 动作: {action_scalar:.2f}")
        elif sell_signals >= 2 and action_scalar < -0.2:
            directional_reward += 0.1
            # print(f"✅ 正确卖出! RSI: {rsi:.1f}, 布林带: {bb_position:.2f}, 动作: {action_scalar:.2f}")
        elif buy_signals >= 2 and action_scalar < -0.2:
            directional_reward -= 0.09  # 逆势卖出惩罚
            # print(f"❌ 错误卖出! RSI: {rsi:.1f}, 动作: {action_scalar:.2f}")
        elif sell_signals >= 2 and action_scalar > 0.2:
            directional_reward -= 0.09  # 逆势买入惩罚
            # print(f"❌ 错误买入! RSI: {rsi:.1f}, 动作: {action_scalar:.2f}")
        
        # 3. 持仓平衡奖励
        position_value = self.position * price
        position_ratio = position_value / self.total_value
        
        if abs(action_scalar) > 0.5:  # 大幅动作
            if action_scalar > 0.5 and position_ratio < 0.8:  # 大幅买入且未超买
                balance_reward = 0.1
            elif action_scalar < -0.5 and position_ratio > -0.8:  # 大幅卖出且未超卖
                balance_reward = 0.1
            else:
                balance_reward = -0.15  # 过度持仓惩罚
        else:
            balance_reward = 0
        
        # 4. 空头盈利奖励 - 特别奖励卖出盈利
        if action_scalar < -0.3 and len(self.trades) > 0:
            last_trade = self.trades[-1]
            if last_trade['value'] < 0:  # 卖出交易
                # 检查价格是否下跌
                if self.current_step > 0:
                    prev_price = self.df.iloc[self.current_step-1]['close']
                    price_change = (price - prev_price) / prev_price
                    if price_change < -0.01:  # 价格下跌1%
                        short_profit_reward = 0.1
                        print(f"🎯 空头盈利! 价格下跌: {price_change*100:.1f}%")
                    else:
                        short_profit_reward = 0
                else:
                    short_profit_reward = 0
            else:
                short_profit_reward = 0
        else:
            short_profit_reward = 0
        
        # 5. 交易频率惩罚
        if len(self.trades) > 50:  # 过多交易
            frequency_penalty = -0.01 * len(self.trades)
        else:
            frequency_penalty = 0
        
        # 组合奖励
        total_reward = (
            returns_reward * 0.3 +
            directional_reward * 0.4 +
            balance_reward * 0.1 +
            short_profit_reward * 0.3 +
            frequency_penalty * 0.1
        )
        
        return np.clip(total_reward, -10, 10)

    
    def _technical_consistency_reward(self, action):
        """技术指标一致性奖励"""
        row = self.df.iloc[self.current_step]
        reward = 0
        
        # RSI信号
        if row['rsi'] < 30 and action > 0:  # 超卖买入
            reward += 0.1
        elif row['rsi'] > 70 and action < 0:  # 超买卖出
            reward += 0.1
        
        # MACD信号
        if (row['macd'] > row['macd_signal'] and action > 0):
            reward += 0.05
        elif (row['macd'] < row['macd_signal'] and action < 0):
            reward += 0.05
        
        return reward

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):  # 减小网络规模
        super(ActorNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),  # 使用Tanh防止梯度爆炸
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
        )
        
        self.mu = nn.Linear(hidden_dim // 2, 1)
        self.sigma = nn.Linear(hidden_dim // 2, 1)
        
        # 初始化
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0.0)
        nn.init.orthogonal_(self.sigma.weight, gain=0.01)
        nn.init.constant_(self.sigma.bias, -1.0)  # 初始较小的方差
        
    def forward(self, state):
        features = self.network(state)
        mu = torch.tanh(self.mu(features))  # [-1, 1]
        sigma = F.softplus(self.sigma(features)) + 1e-6  # 确保正值
        
        return mu, sigma

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state):
        return self.network(state)

class PPOAgent:
    def __init__(self, state_dim, lr_actor=1e-5, lr_critic=3e-5, gamma=0.99, 
             gae_lambda=0.95, clip_epsilon=0.1, ppo_epochs=4, batch_size=64, entropy_coef=0.02):
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        
        # 网络
        self.actor = ActorNetwork(state_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-6)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-6)
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # 策略参数
        self.confidence_threshold = 0.7  # 信心阈值，用于大胆下单
        self.exploration_decay = 0.995   # 探索衰减


    
    def get_action(self, state, training=True):
        """动作选择"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, sigma = self.actor(state_tensor)
            
            if (torch.isnan(mu).any() or torch.isnan(sigma).any()):
                mu = torch.zeros_like(mu)
                sigma = torch.ones_like(sigma) * 0.5
            
            # 基于市场状态调整探索
            if training:
                # 检查市场是否处于卖出信号区域
                rsi = state[7] if len(state) > 7 else 50
                bb_pos = state[14] if len(state) > 14 else 0.5
                
                if rsi > 60 or bb_pos > 0.7:  # 卖出信号区域
                    # 鼓励卖出探索
                    if np.random.random() < 0.3:  # 30%概率强制卖出探索
                        mu = torch.clamp(mu, max=-0.3)  # 偏向卖出
                        sigma = sigma * 0.8  # 减少探索
                
                sigma = torch.clamp(sigma, min=0.2, max=0.8)
            
            dist = Normal(mu, sigma)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            value = self.critic(state_tensor)
            
        # 确保返回正确的格式
        action_value = float(action.cpu().numpy()[0])
        action_log_prob_value = float(action_log_prob.cpu().numpy()[0])
        value_value = float(value.cpu().numpy()[0])
        
        # 强制多空平衡探索
        if training:
            current_position = state[21] if len(state) > 21 else 0  # 持仓比例
            
            if current_position > 0.5 and np.random.random() < 0.2:
                # 持仓过高时，鼓励卖出
                action_value = min(action_value, -0.3)  # 偏向卖出
            elif current_position < -0.5 and np.random.random() < 0.2:
                # 空头过高时，鼓励买入
                action_value = max(action_value, 0.3)  # 偏向买入
        
        action_value = np.clip(action_value, -1, 1)
        
        # 返回标量动作值
        return action_value, action_log_prob_value, value_value
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(float(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
    
    def compute_advantages_and_returns(self, last_value=0):
        """优势计算"""
        if len(self.rewards) == 0:
            return np.array([]), np.array([])
        
        # 确保所有值都是标量
        rewards_clean = [float(r) for r in self.rewards]
        values_clean = [float(v) for v in self.values]
        dones_clean = [bool(d) for d in self.dones]
        
        # 增强卖出交易的奖励
        enhanced_rewards = []
        for i, (reward, action) in enumerate(zip(rewards_clean, self.actions)):
            action_val = float(action)
            if action_val < -0.1:  # 卖出动作
                if reward > 0:
                    enhanced_reward = reward * 3.0  # 大幅奖励成功卖出
                else:
                    enhanced_reward = reward * 0.5  # 减轻失败卖出的惩罚
            else:
                enhanced_reward = reward
                
            enhanced_rewards.append(enhanced_reward)
        
        # 标准化奖励
        scaled_rewards = np.array(enhanced_rewards, dtype=np.float32)
        if len(scaled_rewards) > 1:
            reward_std = scaled_rewards.std()
            if reward_std > 0:
                scaled_rewards = scaled_rewards / (reward_std + 1e-8)
        
        # 获取最后状态的价值估计
        if len(self.states) > 0:
            states_tensor = torch.FloatTensor(np.array(self.states)).to(device)
            with torch.no_grad():
                last_value_tensor = self.critic(states_tensor[-1:])
                last_value = float(last_value_tensor.cpu().numpy()[0])
        else:
            last_value = 0.0
        
        advantages = []
        returns = []
        gae = 0
        
        # 确保values数组是标量
        values = np.array(values_clean + [last_value], dtype=np.float32)
        rewards = np.array(scaled_rewards, dtype=np.float32)
        dones = np.array(dones_clean, dtype=bool)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = float(last_value)
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = float(values[t + 1])
            
            delta = float(rewards[t]) + self.gamma * next_value * next_non_terminal - float(values[t])
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, float(gae))
            returns.insert(0, float(gae + values[t]))
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        if len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 0:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """针对多空平衡的PPO更新"""
        if len(self.states) < self.batch_size:
            return
        
        # 分析多空行为
        buy_actions = 0
        sell_actions = 0
        profitable_buys = 0
        profitable_sells = 0
        
        for i, (action, reward) in enumerate(zip(self.actions, self.rewards)):
            action = float(action)
            if action > 0.1:
                buy_actions += 1
                if reward > 0.5:
                    profitable_buys += 1
            elif action < -0.1:
                sell_actions += 1
                if reward > 0.5:
                    profitable_sells += 1
        
        total_actions = len(self.actions)
        buy_ratio = buy_actions / total_actions if total_actions > 0 else 0
        sell_ratio = sell_actions / total_actions if total_actions > 0 else 0
        buy_success_rate = profitable_buys / buy_actions if buy_actions > 0 else 0
        sell_success_rate = profitable_sells / sell_actions if sell_actions > 0 else 0
        
        print(f"买入比例: {buy_ratio:.2f}, 卖出比例: {sell_ratio:.2f}")
        print(f"买入成功率: {buy_success_rate:.2f}, 卖出成功率: {sell_success_rate:.2f}")
        
        # 计算优势函数和回报
        advantages, returns = self.compute_advantages_and_returns()
        
        if len(advantages) == 0:
            return
        
        # 对不平衡策略进行调整
        if sell_ratio < 0.1:  # 卖出动作过少
            print("⚠️ 卖出动作不足，加强卖出奖励")
            # 放大卖出动作的奖励
            for i in range(len(self.actions)):
                if self.actions[i] < -0.1:
                    advantages[i] = advantages[i] * 2.0
        
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array([float(a) for a in self.actions])).to(device)
        old_log_probs = torch.FloatTensor(np.array([float(lp) for lp in self.log_probs])).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        # PPO更新
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 演员网络更新
                self.actor_optimizer.zero_grad()
                
                mu, sigma = self.actor(batch_states)
                
                if (torch.isnan(mu).any() or torch.isnan(sigma).any()):
                    continue
                    
                dist = Normal(mu, sigma)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 对卖出动作给予额外关注
                sell_mask = (batch_actions < -0.1).float()
                sell_bonus = sell_mask.mean() * 0.2  # 鼓励卖出
                
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(torch.clamp(log_ratio, -5, 5))
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy + sell_bonus
                
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.3)
                self.actor_optimizer.step()
                
                # 评论家网络更新
                self.critic_optimizer.zero_grad()
                current_values = self.critic(batch_states)
                critic_loss = F.mse_loss(current_values, batch_returns)
                
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.3)
                self.critic_optimizer.step()
        
        self._clear_buffer()
    
    def _clear_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

class PPOTrainer:
    def __init__(self, env, agent, max_episodes=1000, max_steps=500,  # 减少步数
                 update_interval=512, save_interval=50):  # 减少更新间隔
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.update_interval = update_interval
        self.save_interval = save_interval
        
        self.episode_returns = []
        self.episode_lengths = []
        
    def train(self):
        """训练循环"""
        print("开始训练PPO交易智能体...")
        
        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.max_steps):
                # 获取动作
                action, log_prob, value = self.agent.get_action(state)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.store_transition(state, action, log_prob, reward, value, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # 定期更新
                if len(self.agent.states) >= self.update_interval:
                    self.agent.update()
                
                if done:
                    break
            
            # 最终更新
            if len(self.agent.states) > 0:
                self.agent.update()
            
            self.episode_returns.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            if episode % 10 == 0:
                avg_return = np.mean(self.episode_returns[-10:]) if len(self.episode_returns) >= 10 else episode_reward
                print(f"Episode {episode}, Return: {episode_reward:.2f}, "
                    f"Avg Return: {avg_return:.2f}, Length: {episode_length}, Total Value: {info.get('total_value', 0):.2f}")
            
            if episode % self.save_interval == 0 and episode > 0:
                self.agent.save_model(f"models/pth/ppo_trading_agent_{episode}.pth")
                print(f"模型已保存: models/pth/ppo_trading_agent_{episode}.pth")
        
        # 保存最终模型
        self.agent.save_model("models/pth/ppo_trading_agent_final.pth")
        print("最终模型已保存: models/pth/ppo_trading_agent_final.pth")
        
        return self.episode_returns, self.episode_lengths
    
    def plot_training_progress(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_returns)
        plt.title('Episode Returns')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        
        plt.tight_layout()
        plt.show()

def main():
    # 加载数据
    df = pd.read_csv('models/data/1D/BTC_USDT_1D_5years_20251130_193559.csv')
    
    # 创建环境
    env = TorchTradingEnvironment(df, lookback_window=30, initial_balance=10000)
    
    # 创建智能体
    agent = PPOAgent(state_dim=env.state_dim)
    
    # 训练
    trainer = PPOTrainer(env, agent, max_episodes=500, max_steps=200)  # 减少训练规模
    returns, lengths = trainer.train()
    
    # 绘制训练进度
    trainer.plot_training_progress()
    
    return agent, returns

if __name__ == "__main__":
    agent, returns = main()