from ur5_envs.gym_ur5 import gym_ur5
from ur5_envs import models
import pybullet as p
import numpy as np

if __name__ == "__main__":
    # env = TAP_Env(models.get_data_path(), True )
    env = gym_ur5(models.get_data_path(), disp=True)

    show_gui = True

    env.reset(show_gui)

    test_joints = [0, -2.183, -1.220, -0.260, 1.57, 0]

    i = 0
    while i < 20:
        print(i, p.getLinkState(env.robot.ur5, i))
        i+=1
    var = 0.3
    try:
        while True:
            p.stepSimulation()

            # env.robot.set_joints_state(test_joints)
            # import time
            # time.sleep(0.5)
            # # env.reset()
            env.key_event()
            if show_gui and env.robot.is_static:
                 env.robot.update_arm()
                 rgb, dep, seg = env.take_images('middle')

    except Exception as e:
        # env.close()
        print(e)