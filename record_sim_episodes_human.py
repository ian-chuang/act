import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

from constants import ( 
    MASTER_GRIPPER_POSITION_NORMALIZE_FN,
    MASTER_GRIPPER_JOINT_MID,
    DT,
    START_ARM_POSE,
    SIM_TASK_CONFIGS
)
from sim_env import make_sim_env
from robot_utils import move_arms, move_grippers, get_arm_gripper_positions, torque_off, torque_on
from webrtc_headset import WebRTCHeadset
import asyncio

import IPython
e = IPython.embed

async def opening_ceremony(master_bot_left, master_bot_right):
    """ Move all 4 robots to a pose where it is easy to start demonstration """
    # reboot gripper motors, and set operating modes for all motors
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot_left.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

    torque_on(master_bot_left)
    torque_on(master_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    await move_arms([master_bot_left, master_bot_right], [start_arm_qpos] * 4, move_time=1.5)
    # move grippers to starting position
    await move_grippers([master_bot_left, master_bot_right], [MASTER_GRIPPER_JOINT_MID] * 2, move_time=0.5)


    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')
    close_thresh = 0.0
    pressed = False
    while not pressed:
        gripper_pos_left = get_arm_gripper_positions(master_bot_left)
        gripper_pos_right = get_arm_gripper_positions(master_bot_right)
        if (gripper_pos_left < close_thresh) and (gripper_pos_right < close_thresh):
            pressed = True
        await asyncio.sleep(DT/10)
        await asyncio.sleep(0)

    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'Started!')

def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

async def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    import rospy
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    # start arms
    headset = WebRTCHeadset(asyncio.get_event_loop())
    await headset.run_offer()
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    task_name = args['task_name']
    episode_idx = args['episode_idx']
    dataset_dir = SIM_TASK_CONFIGS[task_name]['dataset_dir']
    num_episodes = SIM_TASK_CONFIGS[task_name]['num_episodes']
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    render_cam_name = 'angle'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    # setup the environment
    env = make_sim_env(task_name)

    i = episode_idx
    ts = env.reset()
    episode_replay = [ts]
    joint_traj = []

    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # ax = plt.subplot()
    # plt_img = ax.imshow(ts.observation['images']['angle'])
    # plt.ion()

    headset.send_image(np.concatenate([ts.observation['images']['left_eye'], ts.observation['images']['right_eye']], axis=1))

    await opening_ceremony(master_bot_left, master_bot_right)

    for t in tqdm(range(episode_len)):
        await asyncio.sleep(0)

        feedback = {
            'headOutOfSync': False,
            'leftOutOfSync': False,
            'rightOutOfSync': False,
            'ok': True,
            'info': f"Episode {i}, Timestep: {str(t).zfill(len(str(episode_len)))}/{episode_len}",
            'leftArmPosition': [0,0,0],
            'leftArmRotation': [0,0,0,1],
            'rightArmPosition': [0,0,0],  
            'rightArmRotation': [0,0,0,1],
            'middleArmPosition': [0,0,0],
            'middleArmRotation': [0,0,0,1],
        }
        headset.send_data(feedback)

        headset.send_image(np.concatenate([ts.observation['images']['left_eye'], ts.observation['images']['right_eye']], axis=1))
        await asyncio.sleep(0)

        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)

        await asyncio.sleep(0)

        episode_replay.append(ts)
        joint_traj.append(action)

        # plt_img.set_data(ts.observation['images']['angle'])
        # plt.pause(0.005)

        await asyncio.sleep(0.015)

        if rospy.is_shutdown():
            print('ROS shutdown: Failed to collect data')
            exit()

    """
    For each timestep:
    observations
    - images
        - each_cam_name     (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'

    action                  (14,)         'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
    # truncate here to be consistent
    joint_traj = joint_traj[:-1]
    episode_replay = episode_replay[:-1]

    # len(joint_traj) i.e. actions: max_timesteps
    # len(episode_replay) i.e. time steps: max_timesteps + 1
    max_timesteps = len(joint_traj)
    while joint_traj:
        action = joint_traj.pop(0)
        ts = episode_replay.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    # HDF5
    t0 = time.time()
    dataset_path = os.path.join(dataset_dir, f'episode_{i}')
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                        chunks=(1, 480, 640, 3), )
        # compression='gzip',compression_opts=2,)
        # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        qpos = obs.create_dataset('qpos', (max_timesteps, 14))
        qvel = obs.create_dataset('qvel', (max_timesteps, 14))
        action = root.create_dataset('action', (max_timesteps, 14))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='episode_idx', required=True)
    
    asyncio.run(main(vars(parser.parse_args())))

