import os
from typing import Callable

import igibson

from src.discrete.SB3.ppo_mod_disc import PPO as PPO_MOD
from src.discrete.igibson.igibson_env_MOD_DISCRETE import iGibsonEnv
from src.general.SB3.save_model_callback import SaveModel
import numpy as np
from torchvision import models
#import cv2
#from torchvision import models
#from VisTranNet import ViT
try:
    import gym
    import torch as th
    import torch.nn as nn
    
    from stable_baselines3.common.callbacks import CallbackList,EvalCallback
    
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor#,VecFrameStack,VecNormalize
    

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


"""
Example training code using stable-baselines3 PPO for PointNav task.
"""

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
            if key in ["proprioception", "task_obs",'goal_obs']:
                
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU())#

            elif key == "image":#["image","image_global","rgb","depth"]:
                #feature_size = 256
                n_input_channels = subspace.shape[0]  # channel last
                #print("INPut - ",key," -- ",n_input_channels,subspace.shape)
                n_input_channels = subspace.shape[0]
                
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                
                #cnn = models.resnet18()
                """
                if subspace.shape[1] == 256:
                    cnn = ViT(img_size=subspace.shape[1],depth=12,emb_size=256,n_classes=256)
                else:
                    cnn = ViT(img_size=subspace.shape[1],depth=6,emb_size=128,n_classes=256)
                """
                #num_ftrs = cnn.fc.in_features
                #cnn.fc = nn.Linear(num_ftrs, 128)
                
                test_tensor = th.zeros([n_input_channels, subspace.shape[1], subspace.shape[2]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key == "image_global":#["image","image_global","rgb","depth"]:
                feature_size = 256
                n_input_channels = subspace.shape[0]  # channel last
                
                n_input_channels = subspace.shape[0]
                """
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                """
                """
                if subspace.shape[1] == 256:
                    cnn = ViT(img_size=subspace.shape[1],depth=12,emb_size=256,n_classes=256)
                else:
                    cnn = ViT(img_size=subspace.shape[1],depth=6,emb_size=128,n_classes=256)
                """
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

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


def main():
    config_file = "config.yaml"
    tensorboard_log_dir = "log_dir"
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    model_log_dir = ""  
    for i in range(10000000):
        model_log_dir = './model/{}/'.format(i)
        if(os.path.exists(model_log_dir)):
            #counter.trial += 1
            continue
        else:
            break
    os.makedirs(model_log_dir, exist_ok=True)

    """
    model_log_dir2 = ""  
    for i in range(10000000):
        model_log_dir2 = './model2/{}/'.format(i)
        if(os.path.exists(model_log_dir2)):
            #counter.trial += 1
            continue
        else:
            break
    os.makedirs(model_log_dir2, exist_ok=True)
    """
    #SB3----------------------------------------------------------------------
    train_set = ['Merom_0_int', 'Benevolence_0_int', 'Pomaria_0_int','Wainscott_1_int', 'Rs_int', 'Ihlen_0_int','Beechwood_1_int', 'Ihlen_1_int','Merom_0_int', 'Benevolence_0_int', 'Pomaria_0_int','Wainscott_1_int', 'Rs_int', 'Ihlen_0_int','Beechwood_1_int', 'Ihlen_1_int']
    mix_sample = {'Merom_0_int':False, 'Benevolence_0_int':True, 'Pomaria_0_int':False,'Wainscott_1_int':False, 'Rs_int':True, 'Ihlen_0_int':False,'Beechwood_1_int':False, 'Ihlen_1_int':False}
    
    #val_set = ['Benevolence_1_int', 'Wainscott_0_int']
    #test_set = ['Pomaria_2_int', 'Benevolence_2_int', 'Beechwood_0_int','Pomaria_1_int', 'Merom_1_int']
    num_cpu_train = 2
    #num_cpu_eval = 2
    config_filename = os.path.join('./', 'config.yaml')
    def make_env(rank: int, seed: int = 91,data_set=[]) -> Callable:
        def _init() -> iGibsonEnv:
            env = iGibsonEnv(
                config_file=config_filename,
                scene_id=data_set[rank],
                mode="headless",
                action_timestep=1 / 10.0,
                physics_timestep=1 / 120.0,
                
                mix_sample=mix_sample[data_set[rank]]
            )
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    freqs = 256
    env = SubprocVecEnv([make_env(i,data_set=train_set) for i in range(num_cpu_train)])
    env = VecMonitor(env,filename=model_log_dir)
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor,)

    use_discrete = True
    aux_bin_number= 12
    task_obs = env.observation_space['task_obs'].shape[0] -aux_bin_number
    print("TASK OBS:",task_obs)
    if use_discrete:
        model = PPO_MOD("MultiInputPolicy",env,ent_coef=0.005,batch_size=64,clip_range=0.1,n_epochs=4,learning_rate=0.0001, verbose=1, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs\
            ,aux_pred_dim=aux_bin_number,proprio_dim=task_obs,cut_out_aux_head=aux_bin_number,deact_aux=False)
    else:
        model = PPO_MOD("MultiInputPolicy",env,ent_coef=0.005,batch_size=64,clip_range=0.1,n_epochs=4,learning_rate=0.0001, verbose=1, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs\
            ,aux_pred_dim=2,proprio_dim=task_obs,deact_aux=False)


    #model = PPO("MultiInputPolicy",env,ent_coef=0.005,batch_size=64,clip_range=0.1,n_epochs=4,learning_rate=0.0001, verbose=1, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs)
    #model.set_parameters("model/21/last_model",exact_match=True)#-> 
    #model.set_parameters("model/ResNet224x224/last_model",exact_match=True)#-> 20% aux, adapted pngs, kept max dist and no collision with cracker<-
    #model.set_parameters("model/18/last_model",exact_match=True)#-> 20% aux, adapted pngs, kept max dist and no collision with cracker<-
    print(model.policy)
   
    #evaluate env
    #eval_env = SubprocVecEnv([make_env(i,data_set=val_set,evaluate=True) for i in range(num_cpu_eval)])
    #eval_env = VecMonitor(eval_env)#,filename=model_log_dir)

    save_model_callback = SaveModel(check_freq=freqs, log_dir=model_log_dir)
    """
    eval_callback = EvalCallback(eval_env, n_eval_episodes=4,best_model_save_path=model_log_dir2,\
        log_path=model_log_dir2,\
        eval_freq=freqs*6,#32768
        deterministic=False,render=False)
    """
    #callback = CallbackList([save_model_callback, eval_callback])
    model.learn(11600000,callback=save_model_callback)



if __name__ == "__main__":
    main()
