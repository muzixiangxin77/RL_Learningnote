import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    BUFFER_SIZE = 1000000  # 经验回放缓冲区大小
    BATCH_SIZE = 256  # 批次大小
    MAX_EPISODES = 500  # 最大训练轮数
    MAX_STEPS = 200  # 每轮最大步数
    EXPL_NOISE = 0.1  # 探索噪声标准差
    POLICY_NOISE = 0.2  # 目标策略平滑噪声
    NOISE_CLIP = 0.5  # 噪声裁剪范围
    POLICY_FREQ = 2  # 策略延迟更新频率
    HIDDEN_DIM = 256  # 隐藏层维度
    START_TIMESTEPS = 10000  # 开始训练前的随机探索步数


# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.max_action


# Critic网络 - TD3使用两个Q网络
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

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


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


# TD3智能体
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.max_action = max_action
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.policy_noise = config.POLICY_NOISE
        self.noise_clip = config.NOISE_CLIP
        self.policy_freq = config.POLICY_FREQ
        self.expl_noise = config.EXPL_NOISE

        # Actor网络
        self.actor = Actor(state_dim, action_dim, max_action, config.HIDDEN_DIM).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, config.HIDDEN_DIM).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)

        # Critic网络（双Q网络）
        self.critic = Critic(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        # 经验回放
        self.replay_buffer = ReplayBuffer(config.BUFFER_SIZE)

        # 训练步数计数器
        self.total_it = 0

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()

        if add_noise:
            noise = np.random.normal(0, self.max_action * self.expl_noise, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def train(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return None, None, None

        self.total_it += 1

        # 从经验回放中采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            # 选择下一个动作并添加裁剪噪声（目标策略平滑）
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            # 计算目标Q值，取两个Q网络的最小值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # 获取当前Q估计
        current_q1, current_q2 = self.critic(states, actions)

        # 计算Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟策略更新
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            # 计算Actor损失
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()

            # 优化Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

            actor_loss = actor_loss.item()

        return critic_loss.item(), actor_loss, current_q1.mean().item()

    def soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


# 训练函数
def train_td3():
    config = Config()

    # 创建环境
    env = gym.make(config.ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"环境: {config.ENV_NAME}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}, 最大动作值: {max_action}")

    # 创建智能体
    agent = TD3Agent(state_dim, action_dim, max_action, config)

    # 训练记录
    episode_rewards = []
    critic_losses = []
    actor_losses = []
    q_values = []

    # 总步数计数
    total_steps = 0

    # 训练循环
    for episode in range(config.MAX_EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        episode_critic_loss = []
        episode_actor_loss = []
        episode_q_value = []

        for step in range(config.MAX_STEPS):
            # 前期随机探索
            if total_steps < config.START_TIMESTEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, add_noise=True)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存储经验
            agent.replay_buffer.add(state, action, reward, next_state, float(done))

            # 训练（只有在收集足够经验后才开始）
            if total_steps >= config.START_TIMESTEPS:
                critic_loss, actor_loss, q_value = agent.train(config.BATCH_SIZE)
                if critic_loss is not None:
                    episode_critic_loss.append(critic_loss)
                    episode_q_value.append(q_value)
                if actor_loss is not None:
                    episode_actor_loss.append(actor_loss)

            episode_reward += reward
            state = next_state
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        if episode_critic_loss:
            critic_losses.append(np.mean(episode_critic_loss))
            q_values.append(np.mean(episode_q_value))
        if episode_actor_loss:
            actor_losses.append(np.mean(episode_actor_loss))

        # 打印训练信息
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{config.MAX_EPISODES}, "
                  f"总步数: {total_steps}, "
                  f"平均奖励: {avg_reward:.2f}, "
                  f"当前奖励: {episode_reward:.2f}, "
                  f"缓冲区大小: {agent.replay_buffer.size()}")

    env.close()

    # 保存模型
    agent.save("td3_model.pth")
    print("\n模型已保存到 td3_model.pth")

    # 绘制训练曲线
    plot_results(episode_rewards, critic_losses, actor_losses, q_values)

    return agent, episode_rewards


# 绘制结果
def plot_results(rewards, critic_losses, actor_losses, q_values):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 奖励曲线
    axes[0, 0].plot(rewards, label='Episode Reward', alpha=0.6)
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
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
        axes[1, 0].plot(actor_losses)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Actor Loss')
        axes[1, 0].set_title('Actor Loss (Delayed Update)')
        axes[1, 0].grid(True, alpha=0.3)

    # Q值
    if q_values:
        axes[1, 1].plot(q_values)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Q-Value')
        axes[1, 1].set_title('Q-Value Estimates')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('td3_training_results.png', dpi=150)
    print("训练曲线已保存到 td3_training_results.png")
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
            action = agent.select_action(state, add_noise=False)
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
    print("TD3算法训练开始")
    print("TD3关键改进:")
    print("1. 双Q网络 - 减少过估计")
    print("2. 延迟策略更新 - 降低方差")
    print("3. 目标策略平滑 - 平滑Q值估计")
    print("=" * 60)

    # 训练
    agent, rewards = train_td3()

    print("\n" + "=" * 60)
    print("训练完成,开始测试")
    print("=" * 60)

    # 测试
    test_rewards = test_agent(agent)

    print("\n训练和测试完成!")
    print(f"最终训练平均奖励 (最后10轮): {np.mean(rewards[-10:]):.2f}")
    print(f"测试平均奖励: {np.mean(test_rewards):.2f}")