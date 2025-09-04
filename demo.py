import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
import numpy as np
import time
import os

model_path = "breakout_dqn_model.zip"

# 检查是否已经有训练好的模型
if os.path.exists(model_path):
    print(f"发现已保存的模型：{model_path}")
    choice = input("是否直接加载模型开始演示？(y/n): ").lower()
    if choice == 'y' or choice == 'yes' or choice == '':
        print("正在加载已保存的模型...")
        model = DQN.load(model_path)
        print("模型加载完成！")
    else:
        print("将重新训练模型...")
        # 创建训练环境
        train_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=0)
        train_env = VecFrameStack(train_env, n_stack=4)
        
        # 创建并训练模型
        model = DQN("CnnPolicy", train_env, verbose=1, buffer_size=10000)
        print("开始训练模型...")
        model.learn(total_timesteps=20000)
        print("训练完成！")
        
        # 保存模型
        model.save("breakout_dqn_model")
        print(f"模型已保存为 {model_path}")
else:
    print("未找到已训练的模型，开始训练...")
    # 创建训练环境
    train_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=0)
    train_env = VecFrameStack(train_env, n_stack=4)
    
    # 创建并训练模型
    model = DQN("CnnPolicy", train_env, verbose=1, buffer_size=10000)
    print("开始训练模型...")
    model.learn(total_timesteps=20000)
    print("训练完成！")
    
    # 保存模型
    model.save("breakout_dqn_model")
    print(f"模型已保存为 {model_path}")

# 现在开始可视化演示！
print("\n=== 开始游戏演示 ===\n")
print("注意：由于技术限制，将显示AI游戏进度而不是实时画面")
print("但你可以看到AI的得分和表现统计")

# 创建和训练环境相同格式的演示环境
visual_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=42)
visual_env = VecFrameStack(visual_env, n_stack=4)

print("游戏环境已创建！")
print("接下来你将看到AI玩Breakout的实时统计")
print("按 Ctrl+C 可以随时停止\n")

obs = visual_env.reset()
total_reward = 0
episode_count = 0
max_episodes = 5  # 演示5局
step_count = 0

try:
    while episode_count < max_episodes:
        # 让训练好的AI决定动作
        action, _states = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, rewards, dones, infos = visual_env.step(action)
        total_reward += rewards[0]
        step_count += 1
        
        # 每50步输出一次进度
        if step_count % 50 == 0:
            print(f"步数: {step_count:4d} | 当前得分: {total_reward:6.1f} | 第{episode_count + 1}局进行中...")
        
        if dones[0]:
            episode_count += 1
            print(f"\n🎯 第{episode_count}局游戏结束 - 最终得分: {total_reward:.1f} 分")
            print(f"   本局用了 {step_count} 步")
            
            # 重置计数器
            total_reward = 0
            step_count = 0
            
            if episode_count < max_episodes:
                print(f"\n准备开始第{episode_count + 1}局...")
                time.sleep(1)  # 局间休息
                
except KeyboardInterrupt:
    print("\n演示被用户停止")

visual_env.close()
print(f"\n🎮 游戏演示结束！AI总共玩了{episode_count}局游戏")
print("\n💡 提示：如果想看到真正的游戏画面，需要安装额外的可视化库")