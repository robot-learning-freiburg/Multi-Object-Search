import os

import numpy as np

from src.SB3.encoder import EgocentricEncoders
from src.SB3.ppo import PPO_AUX
from src.igibson.multi_object_env import MultiObjectEnv

try:

    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


def main():
    tensorboard_log_dir = "log_dir"
    model_log_dir = "model"

    scene_counter = -1

    # test set
    scenes = ['Benevolence_1_int', 'Pomaria_2_int', 'Benevolence_2_int', 'Wainscott_0_int', 'Beechwood_0_int',
              'Pomaria_1_int', 'Merom_1_int']

    
    # test set
    scenes_succ = {'Benevolence_1_int': [], 'Pomaria_2_int': [], 'Benevolence_2_int': [], 'Wainscott_0_int': [],
                   'Beechwood_0_int': [], 'Pomaria_1_int': [], 'Merom_1_int': []}

    

    # test set
    scenes_succ_mean = {'Benevolence_1_int': 0, 'Pomaria_2_int': 0, 'Benevolence_2_int': 0, 'Wainscott_0_int': 0,
                        'Beechwood_0_int': 0, 'Pomaria_1_int': 0, 'Merom_1_int': 0}


    use_discrete = True

    config_filename = os.path.join('./', 'config.yaml')
    env = MultiObjectEnv(config_file=config_filename,
                         scene_id=scenes[scene_counter])

    env.seed(42)
    use_aux = True

    policy_kwargs = dict(features_extractor_class=EgocentricEncoders)

    deterministic_policy = False

    aux_bin_number = env.mapping.aux_bin_number
    task_obs = env.observation_space['task_obs'].shape[0] - aux_bin_number
    model = PPO_AUX("MultiInputPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir,
                    policy_kwargs=policy_kwargs, aux_pred_dim=aux_bin_number, proprio_dim=task_obs)

    
    #model.set_parameters("checkpoints/Fetch", exact_match=False)  # Depending on the robot selection (in config), select either Fetch or Locobot file

    model.set_parameters("checkpoints/LoCoBot", exact_match=False)  # Depending on the robot selection (in config), select either Fetch or Locobot file

    
    set_random_seed(42)

    print(model.policy)

    # heat_score = []
    succ_rate = []
    collission_rate = []
    SPL = []
    steps_taken = []
    for ep in range(700):
        if ep % 75 == 0:
            scene_counter += 1
            env.reload_model(scenes[scene_counter])
            env.last_scene_id = scenes[scene_counter]
            env.current_episode = ep
            
        acc_rew = []
        obs = env.reset()

        initial_geo_dist = env.task.initial_geodesic_length
        print(f"After reset:{initial_geo_dist}")
        agent_geo_dist_taken = 0
        curr_position = env.robots[0].get_position()[:2]
        steps_counter = 0
        ep_rew = 0
        while True:
            steps_counter += 1
            # action = np.random.uniform(-1, 1, 2)
            # action, _states = model.predict(obs)#,deterministic=True)

            action, _states, aux_angle = model.predict(obs)  # ,deterministic=deterministic_policy)

            obs, rewards, dones, info = env.step({"action": action, "aux_angle": aux_angle[0]})

            acc_rew.append(rewards)
            # -------------------------
            new_position = env.robots[0].get_position()[:2]
            _, geodesic_dist = env.scene.get_shortest_path(env.task.floor_num, curr_position, new_position,
                                                           entire_path=False)
            curr_position = new_position
            agent_geo_dist_taken += geodesic_dist
            # -------------------------

            ep_rew += rewards

            if (dones):
                # heat_score.append(env.current_heat_score)
                steps_taken.append(steps_counter)
                print("EPISODE:", ep)
                if (info['success']):
                    succ_rate.append(1)
                    scenes_succ[scenes[scene_counter]].append(1)
                    SPL.append(initial_geo_dist / max(initial_geo_dist, agent_geo_dist_taken))

                else:
                    scenes_succ[scenes[scene_counter]].append(0)
                    succ_rate.append(0)
                    SPL.append(0)

                collission_rate.append(env.collision_step)

                scenes_succ_mean[scenes[scene_counter]] = np.mean(np.array(scenes_succ[scenes[scene_counter]]))

                print("rew:", ep_rew, " Agent geo:", agent_geo_dist_taken, " SPL:", np.mean(np.array(SPL)),
                      "Num steps:", np.mean(np.array(steps_taken)), " curr succ rate:", np.mean(np.array(succ_rate)))
                # print("scene succ:",scenes_succ_mean, " heat score:",np.mean(np.array(heat_score))," col mean: ", np.mean(np.array(collission_rate)))
                print("scene succ:", scenes_succ_mean, " col mean: ", np.mean(np.array(collission_rate)))
                # if not info['success']:
                #    print("press key for continuing")
                #    input()

                break


if __name__ == "__main__":
    main()
