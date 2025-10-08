#!/usr/bin/python3

"""Script for testing URDFs in a simple pybullet enviroment."""

# Author: JCampbell9,
# Date: 6/17/2023

import pybullet as p
import time
import pybullet_data

import tkinter as tk

from tkinter.filedialog import askopenfile


class Vizulizer():
    """URDF Vizulizer."""

    def __init__(self, file_loc):
        """Initialize the Vizulizer class.

        Args:
            file_loc (str): path to the URDF that we want to be visualized
        """
        self.file_loc = file_loc
        
        


    def main(self):
        """Run the simulator."""           
        physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -9.81)
        LinkId = []
        cubeStartPos = [0, 0, 1]
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        boxId = p.loadURDF(self.file_loc, useFixedBase=1)

        robot = boxId

        p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=180, cameraPitch=-91, cameraTargetPosition=[0, 0.1, 0.1])

        for i in range(0, p.getNumJoints(robot)):
            
            p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, targetPosition=0, force=0)
            joint_info = p.getJointInfo(robot, i)
            linkName = joint_info[12].decode("ascii")

            if joint_info[2] == 4:
                LinkId.append("skip")
            else:
                LinkId.append(p.addUserDebugParameter(linkName, joint_info[8], joint_info[9], 0))



        while p.isConnected():

            p.stepSimulation()
            time.sleep(1. / 60.)

            for i in range(0, len(LinkId)):
                if LinkId[i] != "skip":
                    try:
                        linkPos = p.readUserDebugParameter(LinkId[i])
                        p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, targetPosition=linkPos)
                    except Exception as e:
                        print(f"Error setting joint {i}: {e}")


        p.disconnect()
    

if __name__ == '__main__':


    tk.Tk().withdraw()

    file = askopenfile()

    main_run = Vizulizer(file_loc=file.name)
    main_run.main()