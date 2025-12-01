import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TorchTradingEnvironment(gym.Env):
    def __init__(self, df, 
                 initial_balance=10000, 
                 transaction_cost=0.001, 
                 lookback_window=50,
                 allow_short=False):  # 新增：是否允许卖空
        super(TorchTradingEnvironment, self).__init__()
        
        # 数据预处理 - 填充NaN值
        self.df = df.reset_index(drop=True).fillna(method='bfill').fillna(method='ffill')
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.lookback_window = int(lookback_window)
        self.allow_short = bool(allow_short)
        
        # 动作空间: [-1, 1] 连续动作
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # 状态维度
        self.state_dim = self._get_state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def _get_state_dim(self):
        base_features = 4      # OHLC log returns
        technical_features = 17
        account_features = 5   # balance, position, total_value, returns, position_ratio
        history_features = self.lookback_window * 2
        
        return base_features + technical_features + account_features + history_features
    
    def _get_features(self, step):
        if step < self.lookback_window:
            step = self.lookback_window
        row = self.df.iloc[step]
        
        # 当前价格
        current_price = row['close']
        
        # 基础价格特征 - 对数收益率相对于收盘价
        price_features = [
            np.log(row['open'] / current_price + 1e-8),
            np.log(row['high'] / current_price + 1e-8),
            np.log(row['low'] / current_price + 1e-8),
            0.0  # close / close = 1 → log(1)=0
        ]
        
        # 技术指标特征 - 归一化
        def safe_div(a, b):
            return a / (b + 1e-8)
        
        technical_features = [
            safe_div(row['ma_5'] - current_price, current_price),
            safe_div(row['ma_10'] - current_price, current_price),
            safe_div(row['ma_20'] - current_price, current_price),
            safe_div(row['ma_50'] - current_price, current_price),
            safe_div(row['ma_200'] - current_price, current_price),
            safe_div(row['ema_12'] - current_price, current_price),
            safe_div(row['ema_26'] - current_price, current_price),
            (row['rsi'] - 50) / 50 if not pd.isna(row['rsi']) else 0.0,
            safe_div(row['macd'], current_price),
            safe_div(row['macd_signal'], current_price),
            safe_div(row['bollinger_upper'] - current_price, current_price),
            safe_div(row['bollinger_middle'] - current_price, current_price),
            safe_div(row['bollinger_lower'] - current_price, current_price),
            safe_div(row['atr'], current_price),
            np.log(row['volume'] + 1),
            np.log(row['volume_ma_5'] + 1),
            (row['volume_ratio'] - 1) if not pd.isna(row['volume_ratio']) else 0.0
        ]
        
        # 账户特征
        position_value = self.position * current_price
        position_ratio = position_value / (self.total_value + 1e-8)
        returns = (self.total_value - self.initial_balance) / (self.initial_balance + 1e-8)
        
        account_features = [
            self.balance / self.initial_balance,
            self.position,  # 绝对股数（可正可负）
            self.total_value / self.initial_balance,
            np.clip(returns, -1, 5),  # 限制收益 [-100%, +500%]
            np.clip(position_ratio, -2, 2)  # 允许2倍杠杆（若做空）
        ]
        
        # 历史特征：对数收益率 + volume ratio
        history_features = []
        for i in range(step - self.lookback_window, step):
            if i >= 0:
                hist_row = self.df.iloc[i]
                hist_price = hist_row['close']
                price_return = np.log(hist_price / (current_price + 1e-8))
                volume_ratio = np.log((hist_row['volume'] + 1) / (row['volume'] + 1))
                history_features.extend([price_return, volume_ratio])
            else:
                history_features.extend([0.0, 0.0])
        
        # 组合 & 清理
        features = np.array(
            price_features + technical_features + account_features + history_features,
            dtype=np.float32
        )
        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        features = np.clip(features, -5.0, 5.0)  # 严格限制范围
        
        return features

    def reset(self):
        self.current_step = int(self.lookback_window)
        self.balance = float(self.initial_balance)
        self.position = 0.0         # 持仓股数（>0多头，<0空头）
        self.total_value = float(self.initial_balance)
        self.prev_total_value = float(self.initial_balance)
        self.returns = 0.0
        self.trades = []
        self.done = False
        self.max_drawdown = 0.0
        self.peak_value = float(self.initial_balance)
        
        return self._get_features(self.current_step)
    
    def _compute_total_value(self, price):
        return self.balance + self.position * price

    def step(self, action):
        if self.done:
            return self._get_features(self.current_step), 0, True, {}
        
        current_price = float(self.df.iloc[self.current_step]['close'])
        
        # 标准化动作
        if hasattr(action, '__len__'):
            action_scalar = float(action[0])
        else:
            action_scalar = float(action)
        action_scalar = np.clip(action_scalar, -1, 1)
        
        # === 🚀 核心修复：稳健仓位计算 ===
        current_position_value = self.position * current_price
        current_total_value = self._compute_total_value(current_price)
        
        # 目标仓位比例：-1=全空仓，0=清仓，+1=满仓（用全部balance买入）
        target_position_ratio = action_scalar
        
        # 计算目标持仓股数（不加杠杆）
        if self.allow_short:
            # 允许卖空：target_position_value 可为负（需保证金，此处简化）
            target_position_value = target_position_ratio * self.balance
        else:
            # 禁止卖空：仅用余额买入，不能为负
            target_position_ratio = np.clip(target_position_ratio, 0, 1)
            target_position_value = target_position_ratio * self.balance
        
        # 计算目标股数（用余额建仓）
        if abs(current_price) < 1e-8:
            target_shares = 0.0
        else:
            target_shares = target_position_value / current_price

        # 动态波动缩放
        lookback = min(20, self.current_step)
        if lookback >= 5:
            recent_prices = self.df.iloc[self.current_step - lookback : self.current_step]['close'].values
            price_std = np.std(recent_prices) if len(recent_prices) > 1 else 1e-8
            volatility = price_std / (current_price + 1e-8)
        else:
            volatility = 0.02
        vol_factor = np.clip(1.0 / (1 + 10 * volatility), 0.2, 1.0)
        target_shares *= vol_factor

        # 需要交易的股数
        trade_shares = target_shares - self.position
        
        # === 执行交易 ===
        trade_value = 0.0
        transaction_cost = 0.0
        
        if abs(trade_shares) > 1e-8:
            trade_value = trade_shares * current_price
            transaction_cost = abs(trade_value) * self.transaction_cost
            
            # 🔒 安全检查：买入不能透支
            if trade_shares > 0:  # 买入
                max_affordable = self.balance * (1 - self.transaction_cost)
                if trade_value + transaction_cost > max_affordable:
                    # 按可用资金缩放
                    scale = max_affordable / (trade_value + transaction_cost + 1e-8)
                    trade_shares *= scale
                    trade_value = trade_shares * current_price
                    transaction_cost = abs(trade_value) * self.transaction_cost
            
            # 扣除成本，更新仓位
            self.balance -= trade_value + transaction_cost
            self.position += trade_shares
            
            # 数值保护
            self.balance = max(self.balance, 1e-8)
            
            # 记录交易
            self.trades.append({
                'step': self.current_step,
                'action': action_scalar,
                'price': current_price,
                'shares': trade_shares,
                'value': trade_value,
                'cost': transaction_cost
            })
        
        # 更新总资产
        self.total_value = self._compute_total_value(current_price)
        self.total_value = max(self.total_value, 1e-8)  # 防止≤0
        
        # 🚨 异常检测：防数值爆炸
        if self.total_value > self.initial_balance * 1000:  # >100万即异常（初始1万）
            print(f"⚠️ 异常高资产: {self.total_value:.2f}，重置为初始值")
            self.total_value = self.initial_balance
            self.balance = self.initial_balance
            self.position = 0.0
        
        # 收益与回撤更新
        self.returns = (self.total_value - self.initial_balance) / (self.initial_balance + 1e-8)
        
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
        drawdown = (self.peak_value - self.total_value) / (self.peak_value + 1e-8)
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # === 奖励计算 ===
        reward = self._calculate_reward(action_scalar, current_price)
        
        # 下一步
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True
        
        # 记录用于下步reward计算
        self.prev_total_value = current_total_value

        # 熔断保护
        if self.total_value < self.initial_balance * 0.7:
            self.done = True
            reward = -5.0
            print(f"💀 熔断: {self.total_value:.1f} < 70% 初始资金")
        
        return (
            self._get_features(self.current_step), 
            reward, 
            self.done, 
            {
                'total_value': self.total_value,
                'returns': self.returns,
                'max_drawdown': self.max_drawdown,
                'balance': self.balance,
                'position': self.position
            }
        )
    
    def _calculate_reward(self, action, price):
        # 当前总资产
        curr_total_value = self.total_value
        prev_total_value = self.prev_total_value
        
        # 单步收益率（核心信号）
        step_return = (curr_total_value - prev_total_value) / (prev_total_value + 1e-8)
        step_return = np.clip(step_return, -0.3, 0.3)  # 防极端值
        
        # === 🛡️ 安全奖励设计（仅用环境状态）===
        reward = 0.0
        
        # 1. 基础收益奖励（缩放）
        reward += step_return * 3.0
        
        # 2. 回撤惩罚（关键！）
        if self.peak_value > 0:
            drawdown = (self.peak_value - curr_total_value) / self.peak_value
            # 轻度回撤(<15%)：不罚；中度(15%~30%)：线性惩罚；重度(>30%)：重罚
            if drawdown > 0.15:
                excess_dd = drawdown - 0.15
                reward -= excess_dd * 20.0  # 每超1%回撤扣0.2分
        
        # 3. 波动调整奖励：用近期价格波动衡量风险
        lookback = min(20, self.current_step)
        if lookback >= 5:
            recent_prices = self.df.iloc[max(0, self.current_step-lookback):self.current_step]['close'].values
            if len(recent_prices) > 1:
                price_std = np.std(recent_prices)
                volatility = price_std / (price + 1e-8)
                # 波动大时，同样收益应奖励更低（风险调整）
                vol_penalty = np.clip(volatility * 5, 0, 1.0)
                reward -= abs(step_return) * vol_penalty * 0.5
        
        # 4. 交易成本与抖动惩罚
        reward -= abs(action) * self.transaction_cost * 5  # 成本惩罚
        
        if hasattr(self, '_last_action'):
            if abs(action) < 0.05 and abs(self._last_action) < 0.05:
                reward -= 0.01  # 避免微动作抖动
        self._last_action = action
        
        # 5. 极端仓位惩罚（防赌徒心理）
        position_value = abs(self.position) * price
        if position_value > self.balance * 1.5:  # >150% 杠杆（即使没做空，余额不足也危险）
            reward -= 0.5
        
        return float(np.clip(reward, -2.0, 2.0))

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        
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
        )
        
        self.mu = nn.Linear(hidden_dim // 2, 1)
        self.sigma = nn.Linear(hidden_dim // 2, 1)
        
        # 稳健初始化
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0.0)
        nn.init.orthogonal_(self.sigma.weight, gain=0.01)
        nn.init.constant_(self.sigma.bias, -1.5)  # 更小初始方差
        
    def forward(self, state):
        features = self.network(state)
        mu = torch.tanh(self.mu(features))  # [-1, 1]
        sigma = F.softplus(self.sigma(features)) + 1e-4  # sigma >= 0.0001
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
    def __init__(self, state_dim, lr_actor=3e-5, lr_critic=1e-4, 
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 ppo_epochs=4, batch_size=64, entropy_coef=0.01):
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        
        self.actor = ActorNetwork(state_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)
        
        self._clear_buffer()
    
    def get_action(self, state, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, sigma = self.actor(state_tensor)
            
            # NaN保护
            if torch.isnan(mu).any() or torch.isnan(sigma).any():
                mu = torch.zeros_like(mu)
                sigma = torch.ones_like(sigma) * 0.1
            
            dist = Normal(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor)
        
        action_val = float(action.cpu().numpy()[0])
        log_prob_val = float(log_prob.cpu().numpy()[0])
        value_val = float(value.cpu().numpy()[0])
        
        action_val = np.clip(action_val, -1, 1)
        
        return action_val, log_prob_val, value_val
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(float(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
    
    def compute_advantages_and_returns(self, last_value=0):
        if len(self.rewards) == 0:
            return np.array([]), np.array([])
        
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        # 标准化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages.astype(np.float32), returns.astype(np.float32)
    
    def update(self):
        if len(self.states) < self.batch_size:
            return
        
        # 获取最后价值估计
        with torch.no_grad():
            last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(device)
            last_value = self.critic(last_state).item()
        
        advantages, returns = self.compute_advantages_and_returns(last_value)
        
        if len(advantages) == 0:
            return
        
        states = torch.FloatTensor(np.stack(self.states)).to(device)
        actions = torch.FloatTensor(self.actions).unsqueeze(1).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).unsqueeze(1).to(device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(device)
        
        for _ in range(self.ppo_epochs):
            idx = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                batch_idx = idx[start:start+self.batch_size]
                if len(batch_idx) < 2:
                    continue
                    
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                
                # Actor loss
                mu, sigma = self.actor(batch_states)
                dist = Normal(mu, sigma)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Critic loss
                values = self.critic(batch_states)
                critic_loss = F.mse_loss(values, batch_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
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
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


class PPOTrainer:
    def __init__(self, env, agent, max_episodes=500, max_steps=500,
                 update_interval=256, save_interval=50):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.update_interval = update_interval
        self.save_interval = save_interval
        
        self.episode_returns = []
        self.episode_lengths = []
    
    def train(self):
        print("🚀 开始安全训练 PPO 交易智能体...")
        
        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            for step in range(self.max_steps):
                action, log_prob, value = self.agent.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.store_transition(state, action, log_prob, reward, value, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
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
                avg_last10 = np.mean(self.episode_returns[-10:]) if len(self.episode_returns) >= 10 else episode_reward
                print(f"Ep {episode:3d} | R: {episode_reward:6.2f} | Avg10: {avg_last10:6.2f} | "
                      f"Len: {episode_length} | Value: {info['total_value']:8.2f} | DD: {info['max_drawdown']:.3f}")
            
            if episode % self.save_interval == 0 and episode > 0:
                path = f"models/pth/ppo_trading_agent_ep{episode:04d}.pth"
                self.agent.save_model(path)
                print(f"💾 模型保存: {path}")
        
        # 保存最终模型
        self.agent.save_model("models/pth/ppo_trading_agent_final.pth")
        print("✅ 训练完成，最终模型已保存")
        return self.episode_returns, self.episode_lengths
    
    def plot_training_progress(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_returns, alpha=0.7)
        plt.title('Episode Returns')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths, alpha=0.7, color='orange')
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    df = pd.read_csv('models/data/1D/BTC_USDT_1D_5years_20251130_193559.csv')
    print("📊 数据加载成功:", df.shape)
    print("📅 时间范围:", df['timestamp'].iloc[0] if 'timestamp' in df else 'N/A', 
            "→", df['timestamp'].iloc[-1] if 'timestamp' in df else 'N/A')
    
    
    # 创建环境
    env = TorchTradingEnvironment(
        df, 
        initial_balance=10000,
        transaction_cost=0.001,
        lookback_window=30,
        allow_short=True
    )
    
    print("🔧 环境创建成功 | 状态维度:", env.state_dim)
    
    agent = PPOAgent(state_dim=env.state_dim)
    
    trainer = PPOTrainer(
        env, 
        agent, 
        max_episodes=300,   # 合理训练轮次
        max_steps=500,      # 每轮最多500步
        update_interval=128 # 更频繁更新
    )
    
    returns, lengths = trainer.train()
    trainer.plot_training_progress()
    
    return agent, returns


if __name__ == "__main__":
    agent, returns = main()