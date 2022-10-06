import pybullet as p
from igibson.objects.ycb_object import YCBObject


class YCBObject_ID(YCBObject):
    def __init__(self, name, scale=1, **kwargs):
        super(YCBObject_ID, self).__init__(name, scale, **kwargs)

    def _load(self):
        body_id = super(YCBObject_ID, self)._load()
        self.bid = body_id[0]
        return body_id

    def reset(self):
        pass

    def force_wakeup(self):
        """
        Force wakeup sleeping objects
        """
        for joint_id in range(p.getNumJoints(self.get_body_id())):
            p.changeDynamics(self.get_body_id(), joint_id, activationState=p.ACTIVATION_STATE_WAKE_UP)
        p.changeDynamics(self.get_body_id(), -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
