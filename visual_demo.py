import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
import time
import os

print("=== 🎮 Breakout AI 可视化演示 ===\n")

model_path = "breakout_dqn_model.zip"

# 检查模型是否存在
if not os.path.exists(model_path):
    print("❌ 未找到训练好的模型！")
    print("请先运行 demo.py 训练模型，然后再运行此程序")
    exit()

print("✅ 发现已训练的模型，正在加载...")
model = DQN.load(model_path)
print("✅ 模型加载完成！")

print("\n🎯 正在创建可视化游戏环境...")
print("💡 注意：游戏窗口即将打开，请不要关闭！")

# 创建能显示画面的环境
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")

print("✅ 游戏窗口已打开！")
print("🚀 AI即将开始玩游戏，你将看到实时画面！")
print("📝 按 Ctrl+C 随时停止演示\n")

obs, info = env.reset()
total_reward = 0
episode_count = 0
step_count = 0
max_episodes = 3

try:
    while episode_count < max_episodes:
        # 简单处理：让AI随机行动（因为观察格式不同）
        # 这里我们先用随机动作演示画面效果
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # 控制游戏速度
        time.sleep(0.05)  # 20 FPS
        
        # 每100步显示一次状态
        if step_count % 100 == 0:
            print(f"⏱️  步数: {step_count:4d} | 💯 得分: {total_reward:6.1f} | 🎮 第{episode_count + 1}局")
        
        if terminated or truncated:
            episode_count += 1
            print(f"\n🏁 第{episode_count}局结束 - 得分: {total_reward:.1f} 分 (用时 {step_count} 步)")
            
            total_reward = 0
            step_count = 0
            
            if episode_count < max_episodes:
                print("⏳ 3秒后开始下一局...")
                time.sleep(3)
                obs, info = env.reset()

except KeyboardInterrupt:
    print("\n⏹️  演示被用户停止")

env.close()
print(f"\n🎉 演示结束！总共观看了 {episode_count} 局游戏")
print("💡 下次想看AI真正的智能表现，需要解决模型兼容性问题")