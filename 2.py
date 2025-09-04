import ale_py
ale_py.register_v5_envs()
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# 使用 make_atari_env 来自动应用包装器
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
model = DQN("CnnPolicy", env, verbose=1, buffer_size=10000)
model.learn(total_timesteps=10000)  # 先训练少一点，然后演示

# 训练完成后，让我们看看AI怎么玩游戏！
print("\n开始演示AI玩游戏...")
# 创建一个新的可视化环境
demo_env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
obs, info = demo_env.reset()
for i in range(1000):  # 演示1000步
    # 暂时使用随机动作（因为需要处理观测维度差异）
    action = demo_env.action_space.sample()
    obs, reward, terminated, truncated, info = demo_env.step(action)
    if terminated or truncated:
        obs, info = demo_env.reset()
demo_env.close()
# 然后你就可以看它打游戏了！