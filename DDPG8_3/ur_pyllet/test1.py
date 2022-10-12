import pybullet as p
import pybullet_data
from ur5_envs.task import TAP_Env
from ur5_envs import models
import ur5_envs.pb_ompl as pb_ompl

import gym

class UR5Demo():
    def __init__(self):
        self.obstacles = []

        # p.connect(p.GUI)
        # p.setGravity(0, 0, -9.8)
        # p.setTimeStep(1. / 240.)

        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.loadURDF("plane.urdf")

        # load robot
        useGUI = True
        self.env = TAP_Env(models.get_data_path(), useGUI)
        self.env.start(True)
        self.env.reset(True)
        robot_id = self.env.robot.ur5
        # robot_ee = self.env.robot.ee
        robot = pb_ompl.PbOMPLRobot(robot_id)
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("BITstar")

        # add obstacles
        self.add_obstacles()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_obstacles(self):
        # add box
        # self.add_box([1, 0, 0.7], [0.5, 0.5, 0.05])
        self.add_box([0.7, 0, 0.7], [0.5, 0.5, 0.05])

        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        self.obstacles.append(box_id)
        return box_id

    def demo(self):
        start = [-3.14, -1.57, 0, -1.57, -3.14, 0]
        goal = [-3.14, 0, 0, -1.57, -3.14, 0]

        self.robot.set_state(start)
        res, path = self.pb_ompl_interface.plan(goal)
        if res:
            while True:
                self.pb_ompl_interface.execute(path)
                # for joint in path:
                #     self.env.robot.movej(joint)
        return res, path

if __name__ == "__main__":

    ev = gym.make("")
    ev.step()

    useGUI = True
    env = TAP_Env(models.get_data_path(), useGUI)
    env.ompl_demo()
    # env = UR5Demo()

    # model_id = env.env.robot.ur5
    # link_name_to_index = {p.getBodyInfo(model_id)[0].decode('UTF-8'): -1, }
    #
    # for _id in range(p.getNumJoints(model_id)):
    #     _name = p.getJointInfo(model_id, _id)[12].decode('UTF-8')
    #     print(_id, _name)
    #
    # print("ur5 is: ", model_id)
    # print("info is: ", p.getLinkState(env.env.robot.ur5, 0))
    # print(p.getJointInfo(env.env.robot.ur5, 11))
    # print(p.getJointInfo(env.env.robot.ur5, 13))
    # print("ur5 is: ", env.env.robot.ur5)
    # print("ee is: ", env.env.robot.ee.base)
    # print("ee is: ", env.env.robot.ee.body)

    # env.demo()
    # useGUI = True
    # env = TAP_Env(models.get_data_path(), useGUI)
    # env.start(True)
    # env.reset(True)
    # #
    # # print(env.robot)
    # # # start to simulation
    # while True:
    #     p.stepSimulation()
    #
    #     if env.robot.is_static:
    #         env.robot.update_arm()
    #         rgb, dep, seg = env.take_images('middle')

    #     p_min, p_max = p.getAABB(env.robot.ur5)
    #     id_tuple = p.getOverlappingObjects(p_min, p_max)
    #     print(id_tuple)
    #
    #     if len(id_tuple) > 1:
    #         for id1, id2 in id_tuple:
    #             if id1 == env.robot.ur5:
    #                 continue
    #             else:
    #                 print("hit!!!!!", id1, id2)
    #
    #     env.key_event()