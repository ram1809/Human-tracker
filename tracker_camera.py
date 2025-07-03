#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
#    Copyright (C) 2021 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

# \mainpage RoboComp::tracker_camera
#
# \section intro_sec Introduction
#
# Some information about the component...
#
# \section interface_sec Interface
#
# Descroption of the interface provided...
#
# \section install_sec Installation
#
# \subsection install1_ssec Software depencences
# Software dependences....
#
# \subsection install2_ssec Compile and install
# How to compile/install the component...
#
# \section guide_sec User guide
#
# \subsection config_ssec Configuration file
#
# <p>
# The configuration file...
# </p>
#
# \subsection execution_ssec Execution
#
# Just: "${PATH_TO_BINARY}/tracker_camera --Ice.Config=${PATH_TO_CONFIG_FILE}"
#
# \subsection running_ssec Once running
#
#
#

import sys
import traceback
import IceStorm
import time
import os
import copy
import argparse
from termcolor import colored
# Ctrl+c handling
import signal

from specificworker import *

if __name__ == '__main__':
    ic = Ice.initialize(sys.argv[1])
    status = 0
    mprx = {}
    parameters = {}
    for i in ic.getProperties():
        parameters[str(i)] = str(ic.getProperties().getProperty(i))


    # Remote object connection for Pose
    try:
        proxyString = ic.getProperties().getProperty('PoseProxy')
        print(proxyString)
        try:
            basePrx = ic.stringToProxy(proxyString)
            pose_proxy = RoboCompPose.PosePrx.uncheckedCast(basePrx)
            mprx["PoseProxy"] = pose_proxy
        except Ice.Exception:
            print('Cannot connect to the remote object (Pose)', proxyString)
            #traceback.print_exc()
            status = 1
    except Ice.Exception as e:
        print(e)
        print('Cannot get PoseProxy property.')
        status = 1

    if status == 0:
        worker = SpecificWorker(mprx)
        worker.setParams(parameters)
    else:
        print("Error getting required connections, check config file")
        sys.exit(-1)


    if ic:
        # try:
        ic.destroy()
        # except:
        #     traceback.print_exc()
        #     status = 1
