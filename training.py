import os
import random
from typing import Callable

import numpy as np
from igibson.utils.utils import parse_config

from src.SB3.encoder import EgocentricEncoders
from src.SB3.ppo import PPO_AUX
from src.SB3.save_model_callback import SaveModel
from src.igibson.multi_object_env import MultiObjectEnv

try:

    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecMonitor
    from src.SB3.vec_env import VecEnvExt

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


def set_determinism_training(seed=0):
    set_random_seed(seed)

    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    th.backends.cudnn.enabled = False
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True


def main():
    config_file = "config.yaml"
    tensorboard_log_dir = "log_dir"
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    model_log_dir = ""
    for i in range(10000000):
        model_log_dir = './model/{}/'.format(i)
        if (os.path.exists(model_log_dir)):
            # counter.trial += 1
            continue
        else:
            break
    os.makedirs(model_log_dir, exist_ok=True)

    # SB3----------------------------------------------------------------------
    train_set = ['Merom_0_int', 'Benevolence_0_int', 'Pomaria_0_int', 'Wainscott_1_int', 'Rs_int', 'Ihlen_0_int',
                 'Beechwood_1_int', 'Ihlen_1_int', 'Merom_0_int', 'Benevolence_0_int', 'Pomaria_0_int',
                 'Wainscott_1_int', 'Rs_int', 'Ihlen_0_int', 'Beechwood_1_int', 'Ihlen_1_int']
    mix_sample = {'Merom_0_int': False, 'Benevolence_0_int': True, 'Pomaria_0_int': False, 'Wainscott_1_int': False,
                  'Rs_int': True, 'Ihlen_0_int': False, 'Beechwood_1_int': False, 'Ihlen_1_int': False}

    # val_set = ['Benevolence_1_int', 'Wainscott_0_int']
    # test_set = ['Pomaria_2_int', 'Benevolence_2_int', 'Beechwood_0_int','Pomaria_1_int', 'Merom_1_int']
    num_cpu_train = 8
    # num_cpu_eval = 2
    config_filename = os.path.join('./', 'config.yaml')

    def make_env(rank: int, seed: int = 0, data_set=[]) -> Callable:
        def _init() -> MultiObjectEnv:
            env = MultiObjectEnv(
                config_file=config_filename,
                scene_id=data_set[rank],
                mix_sample=mix_sample[data_set[rank]]
            )

            env.seed(seed + rank)

            return env

        set_random_seed(seed)
        # set_determinism_training(seed)

        return _init

    freqs = 2048
    env = VecEnvExt([make_env(i, data_set=train_set) for i in range(num_cpu_train)])
    env = VecMonitor(env, filename=model_log_dir)
    policy_kwargs = dict(features_extractor_class=EgocentricEncoders)

    config = parse_config(config_filename)
    aux_bin_number = config.get("num_bins", 12)
    task_obs = env.observation_space['task_obs'].shape[0] - aux_bin_number

    model = PPO_AUX("MultiInputPolicy", env, ent_coef=0.005, batch_size=64, clip_range=0.1, n_epochs=4,
                    learning_rate=0.0001, verbose=1, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs, aux_pred_dim=aux_bin_number, proprio_dim=task_obs)

    print(model.policy)

    save_model_callback = SaveModel(check_freq=freqs, log_dir=model_log_dir)

    model.learn(11600000, callback=save_model_callback)


if __name__ == "__main__":
    main()
