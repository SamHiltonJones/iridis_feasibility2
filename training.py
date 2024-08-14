import os
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
import torch as th
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.side_channel import SideChannel, OutgoingMessage
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv  
from typing import Callable
from gym import spaces
import gym
from bayes_opt import BayesianOptimization
import uuid

class TupleToBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TupleToBoxWrapper, self).__init__(env)
        self.observation_space = spaces.Box(
            low=np.concatenate([space.low for space in env.observation_space.spaces]),
            high=np.concatenate([space.high for space in env.observation_space.spaces]),
            dtype=np.float32
        )

    def observation(self, observation):
        return np.concatenate(observation)

def create_env(env_path, worker_id=0, time_scale=5.0, no_graphics=True):
    channel = EngineConfigurationChannel()
    reward_channel = RewardSideChannel()
    unity_env = UnityEnvironment(env_path, side_channels=[channel, reward_channel], worker_id=worker_id, no_graphics=no_graphics, base_port=1)
    channel.set_configuration_parameters(time_scale=time_scale)
    
    gym_env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
    gym_env = TupleToBoxWrapper(gym_env)
    
    return gym_env, reward_channel

class RewardSideChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("12345678-1234-5678-1234-567812345678"))
        self.reward_scale1 = 1.0
        self.reward_scale2 = 1.0
        self.reward_scale3 = 1.0

    def on_message_received(self, msg):
        pass  

    def send_reward_scales(self, scale1, scale2, scale3):
        self.reward_scale1 = scale1
        self.reward_scale2 = scale2
        self.reward_scale3 = scale3
        
        message = f"{scale1},{scale2},{scale3}"
        outgoing_message = OutgoingMessage()
        outgoing_message.write_string(message)  
        self.queue_message_to_send(outgoing_message)


def get_save_path(base_path, model_name, trained_path=False):
    date_str = datetime.now().strftime("%d%m%Y")
    version = 0
    if trained_path:
        save_path = f"{base_path}/{model_name}_{date_str}_v{version}.zip"
        while os.path.exists(save_path):
            version += 1
            save_path = f"{base_path}/{model_name}_{date_str}_v{version}.zip"
    else:
        save_path = f"{base_path}/{model_name}_{date_str}_v{version}"
        while os.path.exists(save_path):
            version += 1
            save_path = f"{base_path}/{model_name}_{date_str}_v{version}"
    return save_path[:-4] if trained_path else save_path

class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_interval, log_file, verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.log_file = log_file
        self.step_count = 0
        self.total_reward = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        self.total_reward += np.sum(self.locals['rewards'])

        if self.step_count % self.log_interval == 0:
            with open(self.log_file, 'a') as f:
                f.write(f'Step: {self.step_count}, Reward: {self.total_reward}\n')
            print(f'Logged reward at step {self.step_count}: {self.total_reward}')
            self.total_reward = 0
        return True

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def evaluate_model(env, model, reward_channel, n_episodes=5):
    """
    Run the environment for a given number of episodes and return the average normalized reward.
    
    :param env: The environment.
    :param model: The trained model.
    :param reward_channel: The reward side channel used to scale rewards.
    :param n_episodes: Number of episodes to evaluate the model.
    :return: The normalized moving average of the rewards.
    """
    all_rewards = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            
            normalized_reward = reward / (reward_channel.reward_scale1 + reward_channel.reward_scale2 + reward_channel.reward_scale3)
            episode_reward += normalized_reward
        
        all_rewards.append(episode_reward)

    moving_average_reward = np.mean(all_rewards)
    return moving_average_reward

def train_and_evaluate(env, model, reward_channel, total_timesteps=500000, n_episodes=50, callbacks=None):
    """
    Train the model and then evaluate it.
    
    :param env: The environment.
    :param model: The model to be trained.
    :param reward_channel: The reward side channel used to scale rewards.
    :param total_timesteps: Total number of timesteps for training.
    :param n_episodes: Number of episodes to evaluate the model.
    :param callbacks: List of callbacks for training.
    :return: The normalized moving average of the rewards after evaluation.
    """
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callbacks)
    
    performance = evaluate_model(env, model, reward_channel, n_episodes=n_episodes)
    
    return performance

def bayesian_optimization(reward_channel, env, model):
    def black_box_function(scale1, scale2, scale3):
        reward_channel.send_reward_scales(scale1, scale2, scale3)
        
        checkpoint_callback = CheckpointCallback(save_freq=250000, save_path=save_path_scene1, name_prefix=model_name)
        reward_logging_callback = RewardLoggingCallback(log_interval=500, log_file=f'reward_log_{model_name}.txt')
        callbacks = [checkpoint_callback, reward_logging_callback]
        
        performance = train_and_evaluate(env, model, reward_channel, total_timesteps=500000, n_episodes=10, callbacks=callbacks)
        return performance

    pbounds = {'scale1': (0.1, 5.0), 'scale2': (0.1, 5.0), 'scale3': (0.1, 5.0)}
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(init_points=2, n_iter=10)
    return optimizer.max

if __name__ == '__main__':
    scene1_env = r"C:\iridis_feasibility\iridis_build\3DPos.exe"
    
    env, reward_channel = create_env(scene1_env, worker_id=0, time_scale=5.0, no_graphics=True)
    
    env = DummyVecEnv([lambda: env])

    base_path = 'logs_models'
    trained_models_path = 'trained_models'
    os.makedirs(trained_models_path, exist_ok=True)

    model_name = 'simple_model_env_v8'
    save_path_scene1 = get_save_path(base_path, model_name)

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[dict(pi=[128, 256, 256, 128], vf=[128, 256, 256, 128])])

    tensorboard_log_path_scene1 = get_save_path("./logs_graphs", model_name)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    initial_learning_rate = 5e-5
    model = PPO("MlpPolicy", env, verbose=2, tensorboard_log=tensorboard_log_path_scene1,
                policy_kwargs=policy_kwargs, learning_rate=initial_learning_rate,
                device=device, ent_coef=0.01)

    best_scales = bayesian_optimization(reward_channel, env, model) 
    print(f"Best reward scales found: {best_scales}")

    final_model_path_scene1 = get_save_path(trained_models_path, model_name, trained_path=True)
    model.save(final_model_path_scene1)
    env.close()
