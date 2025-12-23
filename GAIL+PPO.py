import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal #Normal 用于动作分布（连续动作空间）。

device=torch.device("cpu")

env_name="Pendulum-v1"
hidden_dim=128
lr_actor=3e-4
lr_critic=1e-3
lr_disc=3e-4
gamma=0.99
lambda_gae=0.95 #用于平衡偏差和方差,在TD(0)和MC之间平衡
clip_eps=0.2
epochs_ppo=10 #每次收集数据后PPO更新多少轮
batch_size=32 #批量采样大小
bc_epochs=50 #BC预训练轮数
GAIL_steps=20000 #GAIL总训练步数
expert_samples=1000 #专家样本数量

#策略网络
class Actor(nn.Module):
    def __init__(self,state_dim,action_dim,max_action):
        super(Actor,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh()
        )
        self.mu_head=nn.Linear(hidden_dim,action_dim)
        self.log_std=nn.Parameter(torch.zeros(action_dim))#可学习的标准差
        self.max_action=max_action

    def forward(self,state):
        x=self.net(state)
        mu=torch.tanh(self.mu_head(x))*self.max_action#限制动作范围
        std=torch.exp(self.log_std)
        return Normal(mu,std)  #输出动作分布（高斯分布Normal(mu, std)）:环境是连续动作空间，策略是随机性的（探索用）

#价值网络
class Critic(nn.Module):
    def __init__(self,state_dim):
        super(Critic,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self,state):
        return self.net(state)

#判别器
class Discriminator(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Discriminator,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(state_dim+action_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )

    def forward(self,state,action):
        cat=torch.cat([state,action],dim=-1)
        return self.net(cat)


# 针对 Pendulum 的规则专家数据
# 实际项目中，这里应该是加载 pickle 文件的代码。
def get_expert_data(env, num_samples):
    states, actions = [], []
    state, _ = env.reset()

    print(f"正在收集 {num_samples} 条专家数据...")
    for _ in range(num_samples):
        # 简单的基于物理规则的PID控制器逻辑 (仅适用于倒立摆)
        # 目标是保持 theta = 0 (直立)
        th, thdot = state[0], state[2]  # 观测值转换
        # 实际上 pendulum env state 是 [cos(theta), sin(theta), theta_dot]
        # 我们用简单的 heuristic 近似
        theta = np.arctan2(state[1], state[0])
        # 专家策略：简单的 PD 控制
        u = -2.0 * theta - 1.0 * thdot
        action = [np.clip(u, -2.0, 2.0)]

        states.append(state)
        actions.append(action)

        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state if not (terminated or truncated) else env.reset()[0]

    return np.array(states), np.array(actions)

#BC,PPO更新
class GAILAgent:
    def __init__(self,state_dim,action_dim,max_action):
        self.actor=Actor(state_dim,action_dim,max_action).to(device)
        self.critic=Critic(state_dim).to(device)
        self.disc=Discriminator(state_dim,action_dim).to(device)

        self.opt_actor=optim.Adam(self.actor.parameters(),lr=lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=lr_disc)

    #阶段1：行为克隆BC预训练:使用专家(s,a)对，通过MSE损失让Actor的mu接近专家动作（忽略std，假设确定性策略）。
    def pretrain_bc(self,expert_states,expert_actions):
        print("Begin BC Pretrain... ")
        dataset=torch.utils.data.TensorDataset(
            torch.FloatTensor(expert_states).to(device),
            torch.FloatTensor(expert_actions).to(device)
        )
        loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

        for epoch in range(bc_epochs):
            total_loss=0
            for s,a_expert in loader:
                dist=self.actor(s)
                loss=nn.MSELoss()(dist.loc,a_expert)

                self.opt_actor.zero_grad()
                loss.backward()
                self.opt_actor.step()
                total_loss+=loss.item()

            if (epoch+1)%10==0:
                print(f"BC Epoch {epoch + 1}/{bc_epochs}, Loss: {total_loss / len(loader):.4f}")
        print("BC pretrain finished!")

    #阶段2：GAIL（PPO+Discriminator）：
    def update(self,buffer,expert_states,expert_actions):
        #准备数据
        states=torch.FloatTensor(np.array(buffer['states'])).to(device)
        actions = torch.FloatTensor(np.array(buffer['actions'])).to(device)
        old_log_probs = torch.FloatTensor(np.array(buffer['log_probs'])).to(device)
        next_states = torch.FloatTensor(np.array(buffer['next_states'])).to(device)
        dones = torch.FloatTensor(np.array(buffer['dones'])).to(device)

        #更新判别器
        #专家数据
        exp_idx=np.random.randint(0,len(expert_states),size=states.size(0))
        exp_s=torch.FloatTensor(expert_states[exp_idx]).to(device)
        exp_a=torch.FloatTensor(expert_actions[exp_idx]).to(device)

        #loss:
        real_prob=self.disc(exp_s,exp_a)
        fake_prob=self.disc(states,actions.detach())

        loss_disc=-torch.mean(torch.log(real_prob+1e-8)+torch.log(1-fake_prob+1e-8))

        self.opt_disc.zero_grad()
        loss_disc.backward()
        self.opt_disc.step()


        #计算GAIL奖励
        with torch.no_grad():
            d_prob=self.disc(states,actions)
            gail_rewards=-torch.log(1-d_prob+1e-8)

        #计算优势函数（GAE）
        with torch.no_grad():
            values=self.critic(states)
            next_values=self.critic(next_states)
            deltas=gail_rewards+gamma*next_values*(1-dones)-values
            advantages=torch.zeros_like(deltas)
            adv=0
            for t in reversed(range(len(deltas))):
                adv=deltas[t]+gamma*lambda_gae*(1-dones[t])*adv
                advantages[t]=adv
            returns=advantages+values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(epochs_ppo):
            dist=self.actor(states)
            cur_log_probs=dist.log_prob(actions).sum(dim=-1)
            cur_values=self.critic(states)

            ratios=torch.exp(cur_log_probs-old_log_probs)

            #PPO loss
            surr1=ratios*advantages
            surr2=torch.clamp(ratios,1-clip_eps,1+clip_eps)*advantages
            actor_loss=-torch.min(surr1,surr2).mean()

            #Critic loss
            critic_loss=nn.MSELoss()(cur_values,returns)#对比自己的评分(cur_values)与实际表现(returns)，调整评分标准

            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()
        return loss_disc.item(),gail_rewards.mean().item()


def main():
    # 环境初始化
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 1. 获取专家数据
    expert_states, expert_actions = get_expert_data(env, expert_samples)

    # 2. 初始化 Agent
    agent = GAILAgent(state_dim, action_dim, max_action)

    # 3. BC 初始化 (预训练)
    agent.pretrain_bc(expert_states, expert_actions)

    # 4. GAIL 训练循环
    state, _ = env.reset()
    buffer = {'states': [], 'actions': [], 'log_probs': [], 'next_states': [], 'dones': []}

    print(">>> Begin GAIL train...")
    episode_rewards = []
    gail_rewards_history = []

    curr_ep_reward = 0
    for step in range(GAIL_steps):
        # 收集数据
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist = agent.actor(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action_np = action.cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        # 存入 Buffer
        buffer['states'].append(state)
        buffer['actions'].append(action_np)
        buffer['log_probs'].append(log_prob.cpu().item())
        buffer['next_states'].append(next_state)
        buffer['dones'].append(float(done))

        state = next_state
        curr_ep_reward += reward  # 注意：这里记录的是环境真实奖励，用于评估效果，不是用来训练的

        if done:
            state, _ = env.reset()
            episode_rewards.append(curr_ep_reward)
            print(f"Step {step}, Ep Reward (True Env): {curr_ep_reward:.2f}")
            curr_ep_reward = 0

        # 只有当 Buffer 满了之后才更新 (模拟 On-Policy)
        if len(buffer['states']) >= 2048:  # PPO batch size
            disc_loss, avg_gail_reward = agent.update(buffer, expert_states, expert_actions)
            gail_rewards_history.append(avg_gail_reward)

            # 清空 Buffer
            buffer = {'states': [], 'actions': [], 'log_probs': [], 'next_states': [], 'dones': []}

            if step % 2048 == 0:
                print(f"  --> Update: Disc Loss: {disc_loss:.4f}, Avg GAIL Reward: {avg_gail_reward:.4f}")

    # 简单的绘图
    plt.plot(episode_rewards)
    plt.title("True Environment Rewards over Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()



if __name__ == "__main__":
    main()













