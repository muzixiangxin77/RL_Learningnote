import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Actor-Critic网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        # Actor网络 - 输出动作均值和标准差
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic网络 - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """前向传播"""
        # Actor输出
        actor_features = self.actor(state)
        action_mean = self.actor_mean(actor_features)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)

        # Critic输出
        value = self.critic(state)

        return action_mean, action_std, value

    def get_action(self, state):
        """采样动作"""
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, action_log_prob, value

    def evaluate_actions(self, state, action):
        """评估动作"""
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return action_log_prob, value, entropy


class PPOBuffer:
    """经验回放缓冲区"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def store(self, state, action, reward, value, log_prob, done):
        """存储transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self):
        """获取所有数据并清空缓冲区"""
        data = {
            'states': torch.cat(self.states, dim=0),
            'actions': torch.cat(self.actions, dim=0),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32).unsqueeze(1),
            'values': torch.cat(self.values, dim=0),
            'log_probs': torch.cat(self.log_probs, dim=0),
            'dones': torch.tensor(self.dones, dtype=torch.float32).unsqueeze(1)
        }
        self.clear()
        return data

    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []


class PPOAgent:
    """PPO智能体"""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, epochs=10,
                 batch_size=64, value_coef=0.5, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # 网络和优化器
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

        # 经验缓冲区
        self.buffer = PPOBuffer()

    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(state)

        return action.cpu().numpy()[0], log_prob, value

    def compute_gae(self, rewards, values, dones, next_value):
        """计算广义优势估计(GAE)"""
        advantages = torch.zeros_like(rewards)
        gae = 0

        # 将next_value添加到values末尾
        values = torch.cat([values, next_value], dim=0)

        # 从后向前计算GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, next_state, done):
        """更新策略"""
        # 获取缓冲区数据
        data = self.buffer.get()
        states = data['states'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        rewards = data['rewards'].to(self.device)
        values = data['values'].to(self.device)
        dones = data['dones'].to(self.device)

        # 计算next_value
        with torch.no_grad():
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, _, next_value = self.actor_critic.get_action(next_state)
            if done:
                next_value = torch.zeros_like(next_value)

        # 计算优势和回报
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        for _ in range(self.epochs):
            # 生成随机索引
            indices = np.random.permutation(len(states))

            # 分批更新
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 评估当前策略
                log_probs, state_values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )

                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # 计算裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * batch_advantages

                # 计算损失
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(state_values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                # 更新网络
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

        return loss.item()


def train_ppo():
    """训练PPO智能体"""
    # 创建环境
    env = gym.make('Pendulum-v1')

    # 获取状态和动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 创建智能体
    agent = PPOAgent(state_dim, action_dim)

    # 训练参数
    max_episodes = 500
    max_steps = 200
    update_interval = 2048  # 每收集这么多步更新一次

    # 训练循环
    total_steps = 0
    episode_rewards = []

    print("开始训练PPO智能体在Pendulum-v1环境...")
    print(f"设备: {agent.device}")

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # 选择动作
            action, log_prob, value = agent.select_action(state)

            # 动作裁剪到环境范围
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储transition
            agent.buffer.store(
                torch.FloatTensor(state).unsqueeze(0),
                torch.FloatTensor(action).unsqueeze(0),
                reward,
                value,
                log_prob,
                done
            )

            episode_reward += reward
            total_steps += 1
            state = next_state

            # 更新策略
            if total_steps % update_interval == 0:
                loss = agent.update(next_state, done)
                print(f"步数: {total_steps}, 损失: {loss:.4f}")

            if done:
                break

        episode_rewards.append(episode_reward)

        # 打印训练信息
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{max_episodes}, "
                  f"平均奖励: {avg_reward:.2f}, "
                  f"最近奖励: {episode_reward:.2f}")

    env.close()
    print("训练完成!")

    # 测试智能体
    print("\n测试训练好的智能体...")
    test_env = gym.make('Pendulum-v1', render_mode='human')

    for test_episode in range(20):
        state, _ = test_env.reset()
        test_reward = 0

        for _ in range(max_steps):
            action, _, _ = agent.select_action(state)
            action = np.clip(action, test_env.action_space.low, test_env.action_space.high)
            state, reward, terminated, truncated, _ = test_env.step(action)
            test_reward += reward

            if terminated or truncated:
                break

        print(f"测试Episode {test_episode + 1}, 奖励: {test_reward:.2f}")

    test_env.close()


if __name__ == "__main__":
    train_ppo()