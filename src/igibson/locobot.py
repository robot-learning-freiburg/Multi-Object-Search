from igibson.robots.locobot_robot import Locobot


class Locobot_DD(Locobot):
    def __init__(self, config):
        super(Locobot_DD, self).__init__(config)

    def apply_robot_action(self, action):
        # assume self.ordered_joints = [left_wheel, right_wheel]
        assert (
            action.shape[0] == 2 and len(self.ordered_joints) == 2
        ), "differential drive requires the first two joints to be two wheels"
        lin_vel, ang_vel = action
        if not hasattr(self, "wheel_axle_half") or not hasattr(self, "wheel_radius"):
            raise Exception(
                "Trying to use differential drive, but wheel_axle_half and wheel_radius are not specified."
            )
        left_wheel_ang_vel = (lin_vel - ang_vel * self.wheel_axle_half) / self.wheel_radius
        right_wheel_ang_vel = (lin_vel + ang_vel * self.wheel_axle_half) / self.wheel_radius

        self.ordered_joints[0].set_motor_velocity(
            self.velocity_coef * left_wheel_ang_vel * self.ordered_joints[0].max_velocity)
        self.ordered_joints[1].set_motor_velocity(
            self.velocity_coef * right_wheel_ang_vel * self.ordered_joints[1].max_velocity)
