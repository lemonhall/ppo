import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np
import time

print("正在创建和训练模型...")
# 创建训练环境
train_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=0)
train_env = VecFrameStack(train_env, n_stack=4)

# 创建并训练模型
model = DQN("CnnPolicy", train_env, verbose=1, buffer_size=10000)
print("开始训练模型...")
model.learn(total_timesteps=20000)  # 训练2万步
print("训练完成！")

# 保存模型
model.save("breakout_dqn_model")
print("模型已保存为 breakout_dqn_model.zip")

# 创建可视化环境来观看AI玩游戏
print("\n现在让我们看看AI是怎么玩Breakout的！")
# 创建一个带有图形显示的环境
demo_env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")

# 因为模型期望的是经过预处理的观察（灰度、缩放等），
# 我们需要手动应用一些预处理
from stable_baselines3.common.atari_wrappers import AtariWrapper
demo_env = AtariWrapper(demo_env)

obs, info = demo_env.reset()
total_reward = 0
episode_count = 0

print("开始演示！按 Ctrl+C 停止...")
try:
    while episode_count < 5:  # 玩5个回合
        # 将单个观察堆叠成4帧（模型期望的格式）
        stacked_obs = np.repeat(obs[np.newaxis, :, :, :], 4, axis=0)
        stacked_obs = stacked_obs.transpose(1, 2, 3, 0)  # 调整维度顺序
        stacked_obs = stacked_obs[np.newaxis, :]  # 添加batch维度
        
        # 让模型决定动作
        action, _states = model.predict(stacked_obs, deterministic=True)
        
        # 执行动作
        obs, reward, terminated, truncated, info = demo_env.step(action)
        total_reward += reward
        
        # 添加小延迟让人类能看清
        time.sleep(0.03)
        
        if terminated or truncated:
            episode_count += 1
            print(f"第{episode_count}回合结束，得分: {total_reward}")
            total_reward = 0
            obs, info = demo_env.reset()
            time.sleep(1)  # 回合间停顿

except KeyboardInterrupt:
    print("\n演示被用户停止")

demo_env.close()
print("演示结束！")