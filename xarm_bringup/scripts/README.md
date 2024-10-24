# Data recording

Inside this folder, you can find various scripts for data collection and robot teleoperation. This folder contains:

```
  .scripts/
    |-- episode_generator_picking        # Controls the simulated robot during data collection
    |-- episode_manager                  # Manages data saving in real environments
    |-- episode_recorder                 # Saves episodes during data collection
    |-- keyboard_teleop                  # Controls the robot via keyboard teleoperation
    |-- lite6_parallel_gripper_controller # Run this to open/close the gripper
    |-- save_parquet                     # Saves collected data in parquet format
    |-- sixd_speed_limiter               # Splits the path into smaller segments to prevent jerks
    |-- space_teleop                     # Controls the robot via space mouse teleoperation

```

>[!NOTE]
 We support two datasets in Parquet format, collected from both real and simulated environments. You can find these datasets inside the imitation/data folder. Currently, we support the collection of:
  - **Images**: RGB images sized 96x96 pixels
  - **Current Pose and Actions**: Vectors containing six values in the format [x,y,z,yaw,pitch,roll].

If you want to change the data format, you can do so directly during data collection in the episode_recorder script or while saving in parquet format.

## Usage

### **Simulated Data Collection**
  To collect data in a simulated environment, follow these steps:
  1. Run the script to save the episode data:
  ```sh
  ./episode_recorder --data_dir FILE_NAME
  ```
  2. Run the script to control the simulated robot:
  ```sh
  ./episode_generator_picking
  ```
    

### **Real  Data Collection**
  To collect data from a real robot, use the following steps:
  1. Start the episode manager to manage data saving:
  ```sh
  ./episode_manager
  ```
  2. Record the episode data:
  ```sh
  ./episode_recorder --data_dir FILE_NAME
  ```
  3. Choose a teleoperation method to control the robot:
  - Space Mouse Teleoperation:
  ```sh
  ./space_teleop
  ```
  - Keyboard Teleoperation:
  ```sh
  ./keyboard_teleop
  ```

  **Note:** 
  - Using the episode_manager, you can control when an episode starts, ends, and where breakpoints occur during an episode.


### **Data saving**
  To save the collected data in the proper format, run the following command:
  ```sh
  ./save_parquet --data_path DATA_PATH
  ```
  This will save your data in parquet format, ensuring efficient storage and compatibility for further processing. Saved file you need to move inside `imitation/data` folder.
