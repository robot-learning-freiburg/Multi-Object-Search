from igibson.reward_functions.potential_reward import PotentialReward


class PotentialRewardClipped(PotentialReward):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)
    """

    def __init__(self, config):
        super(PotentialRewardClipped, self).__init__(config)
        # specific value depending on the robot used.
        # The Fetch robot can move faster in the direction of the cloest object as the Locobot => Fetch clip value must the larger
        self.clip_value = config.get("pot_rew_clip_value", 0.2)

    def get_reward(self, task, env):
        """
        Reward is proportional to the potential difference between
        the current and previous timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        new_potential = task.get_potential(env)
        reward = self.potential - new_potential
        reward *= self.potential_reward_weight
        self.potential = new_potential
        return min(reward, self.clip_value)
