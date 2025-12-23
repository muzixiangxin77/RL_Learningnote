import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt

# 设置随机种子以确保可复现性
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# 超参数配置
class Config:
    ENV_NAME = 'Pendulum-v1'
    GAMMA = 0.99  # 折扣因子
    TAU = 0.005  # 软更新参数
    ACTOR_LR = 3e-4  # Actor学习率
    CRITIC_LR = 3e-4  # Critic学习率
    ALPHA_LR = 3e-4  # 温度参数学习率
    BUFFER_SIZE = 1000000  # 经验回放缓冲区大小
    BATCH_SIZE = 256  # 批次大小
    MAX_EPISODES = 500  # 最大训练轮数
    MAX_STEPS = 200  # 每轮最大步数
    HIDDEN_DIM = 256  # 隐藏层维度
    AUTO_ENTROPY = True  # 是否自动调整熵系数
    INIT_ALPHA = 0.2  # 初始温度参数
    START_TIMESTEPS = 1000  # 开始训练前的随机探索步数


# Actor网络（高斯策略）
class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(GaussianActor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # 重参数化技巧
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        # 计算log_prob，考虑tanh变换
        log_prob = normal.log_prob(x_t)
        # 应用tanh的雅可比行列式修正
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean)

        return action, log_prob, mean

    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            return y_t


# Critic网络（双Q网络）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1网络
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Q2网络
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        # Q1
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        # Q2
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1)
        )

    def size(self):
        return len(self.buffer)


# SAC智能体
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.max_action = max_action
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.auto_entropy = config.AUTO_ENTROPY

        # Actor网络
        self.actor = GaussianActor(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)

        # Critic网络（双Q网络）
        self.critic = Critic(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        # 温度参数（熵系数）
        if self.auto_entropy:
            # 目标熵：-动作维度
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.ALPHA_LR)
        else:
            self.alpha = config.INIT_ALPHA

        # 经验回放
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)

        # 训练步数计数器
        self.total_it = 0

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor.get_action(state, deterministic)
        action = action.cpu().numpy()[0] * self.max_action
        return action

    def train(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return None, None, None, None

        self.total_it += 1

        # 从经验回放中采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device) / self.max_action  # 归一化到[-1, 1]
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ==================== 更新Critic ====================
        with torch.no_grad():
            # 采样下一个动作
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            # 计算目标Q值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # SAC的目标值包含熵项
            if self.auto_entropy:
                target_q = target_q - self.alpha.detach() * next_log_probs
            else:
                target_q = target_q - self.alpha * next_log_probs

            target_q = rewards + (1 - dones) * self.gamma * target_q

        # 获取当前Q估计
        current_q1, current_q2 = self.critic(states, actions)

        # 计算Critic损失（MSE损失）
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ==================== 更新Actor ====================
        # 重新采样动作以计算策略损失
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        # SAC的Actor损失：最大化 Q - α*log_prob
        if self.auto_entropy:
            actor_loss = (self.alpha.detach() * log_probs - q_new).mean()
        else:
            actor_loss = (self.alpha * log_probs - q_new).mean()

        # 优化Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ==================== 更新温度参数 ====================
        alpha_loss = None
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            alpha_loss = alpha_loss.item()

        # ==================== 软更新目标网络 ====================
        self.soft_update(self.critic, self.critic_target)

        return (
            critic_loss.item(),
            actor_loss.item(),
            self.alpha.item() if self.auto_entropy else self.alpha,
            current_q1.mean().item()
        )

    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy else None,
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        if self.auto_entropy and checkpoint['log_alpha'] is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'])
            self.alpha = self.log_alpha.exp()


# 训练函数
def train_sac():
    config = Config()

    # 创建环境
    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"环境: {config.ENV_NAME}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}, 最大动作值: {max_action}")

    # 创建智能体
    agent = SACAgent(state_dim, action_dim, max_action, config)

    # 训练记录
    episode_rewards = []
    critic_losses = []
    actor_losses = []
    alpha_values = []
    q_values = []

    # 总步数计数
    total_steps = 0

    # 训练循环
    for episode in range(config.MAX_EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        episode_critic_loss = []
        episode_actor_loss = []
        episode_alpha = []
        episode_q_value = []

        for step in range(config.MAX_STEPS):
            # 前期随机探索
            if total_steps < config.START_TIMESTEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.replay_buffer.add(state, action, reward, next_state, float(done))

            # 训练（只有在收集足够经验后才开始）
            if total_steps >= config.START_TIMESTEPS:
                critic_loss, actor_loss, alpha, q_value = agent.train(config.BATCH_SIZE)
                if critic_loss is not None:
                    episode_critic_loss.append(critic_loss)
                    episode_actor_loss.append(actor_loss)
                    episode_alpha.append(alpha)
                    episode_q_value.append(q_value)

            episode_reward += reward
            state = next_state
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        if episode_critic_loss:
            critic_losses.append(np.mean(episode_critic_loss))
            actor_losses.append(np.mean(episode_actor_loss))
            alpha_values.append(np.mean(episode_alpha))
            q_values.append(np.mean(episode_q_value))

        # 打印训练信息
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            alpha_str = f"{alpha_values[-1]:.4f}" if alpha_values else "N/A"
            print(f"Episode {episode + 1}/{config.MAX_EPISODES}, "
                  f"总步数: {total_steps}, "
                  f"平均奖励: {avg_reward:.2f}, "
                  f"当前奖励: {episode_reward:.2f}, "
                  f"Alpha: {alpha_str}, "
                  f"缓冲区大小: {agent.replay_buffer.size()}")

    env.close()

    # 保存模型
    agent.save("sac_model.pth")
    print("\n模型已保存到 sac_model.pth")

    # 绘制训练曲线
    plot_results(episode_rewards, critic_losses, actor_losses, alpha_values, q_values)

    return agent, episode_rewards


# 绘制结果
def plot_results(rewards, critic_losses, actor_losses, alpha_values, q_values):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 奖励曲线
    axes[0, 0].plot(rewards, label='Episode Reward', alpha=0.6)
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10) / 10, mode='valid')
        axes[0, 0].plot(range(9, len(rewards)), moving_avg, label='Moving Average (10)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Critic损失
    if critic_losses:
        axes[0, 1].plot(critic_losses)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Critic Loss')
        axes[0, 1].set_title('Critic Loss (Twin Q-Networks)')
        axes[0, 1].grid(True, alpha=0.3)

    # Actor损失
    if actor_losses:
        axes[0, 2].plot(actor_losses)
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Actor Loss')
        axes[0, 2].set_title('Actor Loss (Policy Gradient)')
        axes[0, 2].grid(True, alpha=0.3)

    # Alpha值（温度参数）
    if alpha_values:
        axes[1, 0].plot(alpha_values)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Alpha (Temperature)')
        axes[1, 0].set_title('Entropy Coefficient (Auto-tuned)')
        axes[1, 0].grid(True, alpha=0.3)

    # Q值
    if q_values:
        axes[1, 1].plot(q_values)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Q-Value')
        axes[1, 1].set_title('Q-Value Estimates')
        axes[1, 1].grid(True, alpha=0.3)

    # 最后一个子图：奖励分布
    if len(rewards) > 0:
        axes[1, 2].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(np.mean(rewards), color='r', linestyle='--',
                           label=f'Mean: {np.mean(rewards):.2f}')
        axes[1, 2].set_xlabel('Reward')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Reward Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sac_training_results.png', dpi=150)
    print("训练曲线已保存到 sac_training_results.png")
    plt.show()


# 测试函数
def test_agent(agent, env_name='Pendulum-v1', num_episodes=20):
    env = gym.make(env_name, render_mode='rgb_array')
    test_rewards = []

    print("\n开始测试...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < 200:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            step += 1

        test_rewards.append(episode_reward)
        print(f"测试Episode {episode + 1}: 总奖励 = {episode_reward:.2f}")

    env.close()

    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    print(f"\n测试结果: 平均奖励 = {avg_reward:.2f} ± {std_reward:.2f}")

    return test_rewards


# 主函数
if __name__ == "__main__":
    print("=" * 60)
    print("SAC算法训练开始")
    print("SAC关键特性:")
    print("1. 最大熵强化学习 - 平衡探索与利用")
    print("2. 双Q网络 - 减少过估计偏差")
    print("3. 随机策略 - 高斯策略网络")
    print("4. 自动温度调整 - 自适应熵系数")
    print("=" * 60)

    # 训练
    agent, rewards = train_sac()

    print("\n" + "=" * 60)
    print("训练完成,开始测试")
    print("=" * 60)

    # 测试
    test_rewards = test_agent(agent)

    print("\n训练和测试完成!")
    print(f"最终训练平均奖励 (最后10轮): {np.mean(rewards[-10:]):.2f}")
    print(f"测试平均奖励: {np.mean(test_rewards):.2f}")