# xarm_bringup

## Overview

The `xarm_bringup` package provides tools and scripts for simulating and controlling the xArm robot in both simulated and real environments. All needed scripts you launch with this command:

```sh
    ros2 launch xarm_bringup lite6_cartesian_launch.py rviz:=false sim:=false
```
For the simulation environment, set `sim` to true. When `rviz` is set to true, the RViz tool will automatically launch, allowing for robot visualization and manual control through the GUI.

<div align='center'>

![Lite6 Physical Teleoperation](../media/rviz.gif) 
</div>


# Enable manual mode
If you want to physically move the robot joints, you need to follow these rules.

- Run commands:

   ## Set mode need to call twice time

      ```sh
         ros2 service call /xarm/set_mode xarm_msgs/srv/SetInt16 "{data: 2}"
      ```

      ```sh
         ros2 service call /xarm/set_mode xarm_msgs/srv/SetInt16 "{data: 2}"
      ```

      ```sh
         ros2 service call /xarm/set_state xarm_msgs/srv/SetInt16 "{data: 0}"
      ```

   **To return in normal mode run commands:**

      ```sh
         ros2 service call /xarm/set_mode xarm_msgs/srv/SetInt16 "{data: 0}"
      ```

      ```sh
         ros2 service call /xarm/set_state xarm_msgs/srv/SetInt16 "{data: 0}"
      ```
      - After this step need to run launch file again

