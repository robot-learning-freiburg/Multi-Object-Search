import os
from typing import Callable
import numpy as np
import igibson

from src.general.SB3.save_model_callback import SaveModel
from torchvision import models
try:
    import gym
    import torch as th
    import torch.nn as nn
    
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


"""
Example training code using stable-baselines3 PPO for PointNav task.
"""

#from VisTranNet import ViT


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            if key in ["proprioception", "task_obs"]:
                #self.proprioceptive_dim = subspace.shape[0]
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU())
                
            elif key == "image":
                n_input_channels = subspace.shape[0]  # channel last
                
                
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                
                test_tensor = th.zeros([n_input_channels, subspace.shape[1], subspace.shape[2]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                    
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key == "image_global":#["image","image_global","rgb","depth"]:
                feature_size = 256
                n_input_channels = subspace.shape[0]  # channel last
                
                
                
                cnn = models.resnet18(pretrained=True)
                #num_ftrs = cnn.fc.in_features
                #cnn.avgpool = nn.AvgPool2d(3,stride=1)
                #cnn.fc = nn.Linear(num_ftrs, 128)
                
                test_tensor = th.zeros([n_input_channels, subspace.shape[1], subspace.shape[2]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                    
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            else:
                raise ValueError("Unknown observation key: %s" % key)
            total_concat_size += feature_size

        #self.aux_encoder_proprio = nn.Sequential(nn.Linear(3, feature_size), nn.ReLU())
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size
        

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            #if key == "task_obs":
            #    print(observations[key])
            
            
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


def main():
    
    tensorboard_log_dir = "log_dir"
    model_log_dir = "model"


    scene_counter = 3

    use_discrete = True
    
    from src.discrete.SB3.ppo_mod_disc import PPO as PPO_MOD
    from src.discrete.igibson.igibson_env_MOD_DISCRETE import iGibsonEnv
    
    
    
    config_filename = os.path.join('./', 'config.yaml')
    env = iGibsonEnv(config_file=config_filename,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,)

    

    use_aux = True

    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,)
    
    deterministic_policy = False
    
        
        
    aux_bin_number= 12
    task_obs = env.observation_space['task_obs'].shape[0] -aux_bin_number
    model = PPO_MOD("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs,aux_pred_dim=aux_bin_number,proprio_dim=task_obs,cut_out_aux_head=aux_bin_number)
            
    model.set_parameters("model/6/saved/save_file_23",exact_match=True)# ResNet discrete collision penaltiy of -0.15


    for ep in range(700):        
        obs = env.reset()
        initial_geo_dist = env.task.initial_geodesic_length
        print(f"After reset:{initial_geo_dist}")
        agent_geo_dist_taken = 0
        curr_position = env.robots[0].get_position()[:2]
        steps_counter = 0
        ep_rew = 0
        while True:
            steps_counter += 1
            
            
            action, _states,aux_angle = model.predict(obs,deterministic=deterministic_policy)
            
            
            
            obs, rewards, dones, info = env.step({"action":action,"aux_angle":aux_angle[0]})
            
            
            if(dones):
            
                break
    
  
if __name__ == "__main__":
    main()

