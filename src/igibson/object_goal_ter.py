from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition


class ObjectGoal(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(ObjectGoal, self).__init__(config)
        self.dist_tol = self.config.get("dist_tol", 0.5)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = True if task.wanted_objects.sum() == 0 else False
        # done = task.wanted_object == 0
        success = done
        return done, success
