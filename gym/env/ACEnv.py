import os
import time
import copy
import json
import traceback
import random
import numpy as np
import cv2
import wandb
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from gymnasium import spaces
from shapely.geometry import Point
from env import ac_controller
from utils.route_manager import Calculator

PREFIX_json = 'C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/python/DataParser/data_bucket/'
PREFIX_map = 'C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/python/DataParser/tracks/'
PREFIX_teleport = 'C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/teleport_vehicle/teleport_point.csv'
IS_VIS = False    # Visualization flag (Only for debugging)

class ACEnvSingle(gym.Env):
    """
    Gym environment for Assetto Corsa (AC) - Single player mode
    """

    def __init__(self, frame_exist, track_exist, **kwargs):
        super(ACEnvSingle, self).__init__()
        ##### 1. Metadata #####
        folder_list = [x for x in os.listdir(PREFIX_json) if os.path.isdir(os.path.join(PREFIX_json, x))]   # Select folders (neglect files)
        while True:
            try:
                with open(PREFIX_json + folder_list[-1] + '/' + 'exp.json', 'r') as f:
                     self.big_dict = json.load(f)
                     break

            except Exception as e:
                # print("[ACEnv] Error in __init__ : {}".format(traceback.format_exc()))
                pass
        
        ## Get vehicle data
        self.vehicle_count = self.big_dict['cars_count']
        self.vehicle_name_ego = self.big_dict['ego']['car_name']
        
        ## Change here if you want to add new cars (1 vehicle is ready now)
        if self.vehicle_name_ego == "ferrari_458_gt2":
            self.max_speed = 270                # [kmh]
            self.steer_scale = [-270, 270]      # [deg]
        else:
            self.max_speed = 300                # [kmh]
            self.steer_scale = [-360, 360]      # [deg]
        self.max_rpm = 10000                    # [-]
        
        ## Get track data (2 tracks are ready now)
        track_name = self.big_dict.get('track_name')
        if 'silverstone' in track_name:
            self.track_name = 'silverstone'
            self.lap_time_standard = (2*60 + 40) * 1000
        elif 'monza' in track_name:
            self.track_name = 'monza'
            self.lap_time_standard = (3*60 + 20) * 1000
        else:
            self.track_name = None
            print("[ACEnv] Error in __init__ : Invalid track name {}".format(track_name))
        
        self.track_data = np.genfromtxt(PREFIX_map + self.track_name +'/' + self.track_name + '_opt_smoothed_lua_full.csv', delimiter=',', skip_header=0)
        self.cc = Calculator(self.track_name, self.track_data)
        self.cc.calc_curvature_offline(5, 5, pfit_degree=2, is_smooth=True, filter_sigma=3, is_visualize=False)
        self.cc.calc_slope_offline(3, 3, pfit_degree=2, is_smooth=True, filter_sigma=5, is_visualize=False)
        self.cc.calc_bank_offline(is_smooth=True, filter_sigma=3, is_visualize=False)
        self.track_length = len(self.track_data)


        ##### 2. Define observation space (Customize here) #####
        self.observation_dict = spaces.Dict({
            "ego": spaces.Dict({
                "velocity_local": spaces.Box(low=-self.max_speed/3.6, high=self.max_speed/3.6, shape=(3,), dtype=float),
                "acceleration_local": spaces.Box(low=-9.8, high=9.8, shape=(3,), dtype=float),
                "track_progress": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                "collision_flag": spaces.Discrete(2, start=0),
            })
        })

        ## Use track info if track exists
        if track_exist:
            mode = kwargs["preview_kwargs"]["mode"]
            num_preview_sample = kwargs["preview_kwargs"]["num_sample"]
            if mode == "static":
                preview_length = preview_interval = kwargs["preview_kwargs"]["sample_interval"] * num_preview_sample
            elif mode == "dynamic":
                preview_length = self.max_speed * kwargs["preview_kwargs"]["factor"]
            
            ## If you want to add new observation, you can add like below.
            # self.observation_dict['ego']['preview_center_x'] = spaces.Box(low=-preview_length, high=preview_length, shape=(num_preview_sample,), dtype=float)
            self.observation_dict['ego']['preview_curvature'] = spaces.Box(low=min(self.cc.curvature_offline), high=max(self.cc.curvature_offline), shape=(num_preview_sample,), dtype=float)
            # self.observation_dict['ego']['preview_slope'] = spaces.Box(low=min(self.cc.slope_offline), high=max(self.cc.slope_offline), shape=(num_preview_samples,), dtype=float)
            # self.observation_dict['ego']['preview_bank'] = spaces.Box(low=min(self.cc.bank_offline), high=max(self.cc.bank_offline), shape=(num_preview_samples,), dtype=float)
            self.observation_dict['ego']['e_phi'] = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=float)
            
            ## 2D rangefinder
            range_max = kwargs["lidar_2d_kwargs"]["distance_max"]
            num_rays = kwargs["lidar_2d_kwargs"]["num_rays"]
            self.observation_dict['ego']['lidar_2d'] = spaces.Box(low=0, high=range_max, shape=(num_rays,), dtype=float)

        self.observation_dict['ego']['previous_steering_cmd'] = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        # self.observation_dict['ego']['previous_throttle_cmd'] = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)

        self.observation_space = spaces.utils.flatten_space(self.observation_dict)
        wandb.config.update({"observation_space":self.observation_dict})


        ##### 3. Define action space #####
        ## Unified
        action_dict = spaces.Dict({
            "steering": spaces.Box(low=-1, high=1, shape=(1,), dtype=float),
            "throttle_brake": spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        })

        ## Separated (Caution! brake-steering-throttle order.)
        # action_dict = spaces.Dict({
        #     "brake": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
        #     "steering": spaces.Box(low=-1, high=1, shape=(1,), dtype=float),
        #     "throttle": spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        # })

        self.action_len = len(action_dict)
        self.action_space = spaces.utils.flatten_space(action_dict)
        wandb.config.update({"action_space": action_dict})


        ##### 4. Define controller #####
        self.controller = ac_controller.ACController()


        ##### 5. Define Video Capture. (Not used in our work) #####
        # if frame_exist:
        #     self.cap = cv2.VideoCapture(0)
        #     if not self.cap.isOpened():
        #         print("Cannot open cam.")
        #         return
        #     self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        #     self.cap.set(cv2.CAP_PROP_XI_BUFFERS_QUEUE_SIZE, 1)

        #     self.frame_dim = [1080,1920 ,3]
        #     wandb.log({"log": "frame_dim" + str(self.frame_dim)},commit=False)


        ##### 6. Etc. #####
        self.reset_flag = True
        self.collision_flag = 0
        self.lap_end_flag = 0
        self.which_side = [0, 1, 0]         # Indicates where the COM of car located w.r.t road boundaries. [1, 0, 0]: Left outside / [0, 1, 0]: Inside / [0, 0, 1]: Right outside
        self.frame_exist = frame_exist
        self.track_exist = track_exist
        if track_exist:
            self.preview_kwargs = kwargs["preview_kwargs"]
            self.distance_kwargs = kwargs["distance_kwargs"]
            self.lidar_2d_kwargs = kwargs["lidar_2d_kwargs"]

        self.big_dict_pre = None
        self.obs_dict = None
        self.action_pre = [0., 0.]
        self.collision_impulse_pre = [0., 0., 0., 0.]
        self.ai_on_flag = False
        self.best_lap_time = np.inf
        self.stuck_step = 0                 # Accumulated time step when vehicle has stucked. [sec]
        self.is_slip = 0                    # Indicates whether the vehicle is slipping or not. [0: False, 1: True]
        self.step_time = 0                  # [sec]
        self.env_fps = 25                   # [hz]


        ##### 7. Visualization (Only for debugging) #####
        if IS_VIS:
            from collections import deque
            try:
                self.window_length = 200     # step
                self.ax_queue = deque([0]*self.window_length, maxlen=self.window_length)
                self.ay_queue = deque([0]*self.window_length, maxlen=self.window_length)
                self.az_queue = deque([0]*self.window_length, maxlen=self.window_length)
                self.vx_queue = deque([0]*self.window_length, maxlen=self.window_length)
                self.vy_queue = deque([0]*self.window_length, maxlen=self.window_length)
                self.vz_queue = deque([0]*self.window_length, maxlen=self.window_length)
                self.prev_steering_queue = deque([0]*self.window_length, maxlen=self.window_length)
                self.collision_flag_queue = deque([0]*self.window_length, maxlen=self.window_length)

                plt.ion()
                self.fig1 = plt.figure()
                self.fig1.set_size_inches(16, 9)
                suptit1 = r'$OBSV$'+' '+r'$Visualization$'
                self.fig1.suptitle(suptit1, fontsize=14)
                gs0 = gridspec.GridSpec(16, 9)              # (ny, nx)
                gs0.update(left=0.1, right=0.95, bottom=0.08, top=0.92, wspace=0.03, hspace=0.03)

                self.ax1 = plt.subplot(gs0[0:16, 0:4])      # [y0(top):y1(bottom), x0(left):x1(right)]
                self.ln11 = self.ax1.scatter([0], [0], c='red', s=120)
                self.ln12 = self.ax1.plot([], [], 'k--')
                self.ln13 = self.ax1.scatter([], [], c=[], cmap='seismic', vmin=-max(np.abs(min(self.cc.curvature_offline)), np.abs(max(self.cc.curvature_offline))), vmax=max(np.abs(min(self.cc.curvature_offline)), np.abs(max(self.cc.curvature_offline))))
                self.ln14 = self.ax1.plot([], [], 'k')
                self.ln15 = self.ax1.plot([], [], 'k')
                
                self.rays = [Line2D([], [], color='green', linewidth=1) for _ in range(num_rays)]
                for ray in self.rays:
                    self.ax1.add_line(ray)
                self.texts = [self.ax1.text(0, 0, '', color='black') for _ in range(num_rays)]
                
                self.e_phi_ray = Line2D([], [], color='r', linewidth=1)
                self.ax1.add_line(self.e_phi_ray)
                self.e_phi_text = self.ax1.text(0, 0, '', color='r')
                
                self.ax1.set_xlabel(r'$X [m]$')
                self.ax1.set_ylabel(r'$Z [m]$')
                self.ax1.invert_xaxis()
                # self.ax1.axis('equal')
                # self.ax1.set_xlim([-20, 20])
                # self.ax1.set_ylim([-150, 150])
                self.ax1.grid(True, alpha=0.6)

                self.ax2 = plt.subplot(gs0[0:3, 5:9])
                self.ln21 = self.ax2.plot([], [], 'ko-')
                self.ax2.set_xlabel(r'$Preview Length [m]$')
                self.ax2.set_ylabel(r'$Curvature [-]$')
                self.ax2.set_xlim([0, self.preview_kwargs['num_sample'] * self.preview_kwargs['sample_interval']])
                self.ax2.set_ylim([min(self.cc.curvature_offline), max(self.cc.curvature_offline)])
                self.ax2.grid(True, alpha=0.6)

                self.ax3 = plt.subplot(gs0[4:7, 5:9])
                self.ln31 = self.ax3.plot([], [], 'b', label='x')
                self.ln32 = self.ax3.plot([], [], 'r', label='y')
                self.ln33 = self.ax3.plot([], [], 'k', label='z')
                self.ax3.legend()
                self.ax3.set_ylabel(r'$Acceleration [m/s^2]$')
                self.ax3.set_xlim([0, self.window_length])
                self.ax3.set_ylim([-5, 5])
                self.ax3.grid(True, alpha=0.6)

                self.ax4 = plt.subplot(gs0[8:11, 5:9], sharex=self.ax3)
                self.ln41 = self.ax4.plot([], [], 'b', label='x')
                self.ln42 = self.ax4.plot([], [], 'r', label='y')
                self.ln43 = self.ax4.plot([], [], 'k', label='z')
                self.ax4.legend()
                self.ax4.set_ylabel(r'$Velocity [m/s]$')
                self.ax4.set_ylim([-10, 60])
                self.ax4.grid(True, alpha=0.6)

                self.ax5 = plt.subplot(gs0[12:14, 5:9], sharex=self.ax3)
                self.ln51 = self.ax5.plot([], [], 'k')
                self.ax5.set_ylabel(r'$SWA [rad]$')
                self.ax5.set_ylim([-np.pi, np.pi])
                self.ax5.grid(True, alpha=0.6)

                self.ax6 = plt.subplot(gs0[15:16, 5:9], sharex=self.ax3)
                self.ln61 = self.ax6.plot([], [], 'k')
                self.ax6.set_xlabel(r'$Step [-]$')
                self.ax6.set_ylabel(r'$Collision$' + ' ' + r'$Flag [-]$')
                self.ax6.set_ylim([0, 1])
                self.ax6.grid(True, alpha=0.6)

            except:
                print("INIT Not Fin,.")


    def _one_hot_encoding(self, value, n):
        """
        Encode discrete variables (range: N) to N-dimension one-hot vector.
        """
        return np.eye(n)[value].tolist()


    def _get_obs(self, absorb_wrapping=False):
        """
        Get observation data from json file.
        """
        _observation_dict = {}
        
        ## Add keys
        _observation_dict['ego'] = {}
        if self.vehicle_count >= 2:
            for cnt in range(1, self.vehicle_count):
                _observation_dict['surr_' + str(cnt)] = {}

        ## Read json file
        folder_list = [x for x in os.listdir(PREFIX_json) if os.path.isdir(os.path.join(PREFIX_json, x))]   # Select folders (neglect files)
        recent_folder = folder_list[-1] + '/'   # Folder which currently recording in progress
        while True:
            try:
                with open(PREFIX_json + recent_folder + 'exp.json', 'r') as f:
                    self.big_dict = json.load(f)
                    if self.big_dict['time_at_sampling']/1000 > self.step_time:
                            break
            except Exception as e:
                # print("[ACEnv] Error in _get_obs: {}".format(traceback.format_exc()))
                pass
        
        ##### Make observation dictionary #####
        ## [Ego] Direct info.
        _observation_dict['ego']['velocity_local'] = self.big_dict['ego']['velocity_local']                     # List (3,)
        if absorb_wrapping:
            _observation_dict['ego']['track_progress'] = [self.big_dict['ego']['track_progress'] + self.big_dict['ego']['lap_count'] - 1]   # Float
        else:
            _observation_dict['ego']['track_progress'] = [self.big_dict['ego']['track_progress']]
        _observation_dict['ego']['acceleration_local'] = self.big_dict['ego']['acceleration_local']             # List (3,)

        if self.action_len == 2:
            if self.big_dict_pre:
                _observation_dict['ego']['previous_steering_cmd'] = [self.big_dict_pre['ego']['steer_report'] / self.steer_scale[1]]
            else:
                 _observation_dict['ego']['previous_steering_cmd'] = [0.0]
        elif self.action_len == 3:
            _observation_dict['ego']['previous_steering_cmd'] = [self.big_dict_pre['ego']['steer_report'] / self.steer_scale[1]]
            # _observation_dict['ego']['previous_throttle_cmd'] = [self.action_pre[2]]
            # _observation_dict['ego']['previous_brake_cmd'] = [self.action_pre[0]]
     
        ## [Ego] Collision info.
        if self.collision_impulse_pre == self.big_dict['ego']['collision_impulse']:
            self.collision_flag = 0
        else:
            self.collision_flag = 1
            self.collision_impulse_pre = self.big_dict['ego']['collision_impulse']
        _observation_dict['ego']['collision_flag'] = self._one_hot_encoding(self.collision_flag, 2)
        
        ## [Ego] Track info.
        if self.track_exist:
            pos = self.big_dict['ego']['position_global']
            orientation = self.big_dict['ego']['orientation_global']
            speed = self.big_dict['ego']['speed']
            if self.reset_flag:
                self.reset_flag = False                
                self.idx_c = self.cc.find_nearest_idx_at_first(pos, 'center')
                self.idx_l, self.idx_r = self.idx_c, self.idx_c

            ####################### PREVIEW (Forward only) #########################
            self.idx_c, dist_c = self.cc.find_nearest_idx(pos, self.idx_c, which_lane='center')
            # self.idx_l, dist_l = self.cc.find_nearest_idx(pos, self.idx_l, which_lane='left')
            # self.idx_r, dist_r = self.cc.find_nearest_idx(pos, self.idx_r, which_lane='right')
            self.idx_l, self.idx_r = self.idx_c, self.idx_c     # For computation efficiency, we share center line index.

            preview_idxs_c, preview_x_c, preview_z_c = self.cc.find_result_idx(self.idx_c, which_lane='center', vel=speed, **self.preview_kwargs)
            curvature_preview, slope_preview, bank_preview = self.cc.find_preview(preview_idxs_c)
            local_preview_x_c, local_preview_z_c = self.cc.global_to_local(pos, orientation, preview_x_c, preview_z_c)

            _observation_dict['ego']['preview_curvature'] = curvature_preview.tolist()    # List (N,)
            # _observation_dict['ego']['preview_slope'] = slope_preview.tolist()
            # _observation_dict['ego']['preview_bank'] = bank_preview.tolist()
            _observation_dict['ego']['e_phi'] = [np.arctan2(local_preview_x_c[1]-local_preview_x_c[0], local_preview_z_c[1]-local_preview_z_c[0])]  # Standard: z-axis of local vehicle coord. + Counter-clockwise direction.
            
            ##################### DISTANCE (Forward + Backward) #####################
            # distance_idxs_c, distance_x_c, distance_z_c = self.cc.find_result_idx(self.idx_c, which_lane='center', **self.distance_kwargs)
            distance_idxs_l, distance_x_l, distance_z_l = self.cc.find_result_idx(self.idx_l, which_lane='left', **self.distance_kwargs)
            distance_idxs_r, distance_x_r, distance_z_r = self.cc.find_result_idx(self.idx_r, which_lane='right', **self.distance_kwargs)
            
            ## Not used in our work, but you can use like below.
            # distance_to_center_line = self.cc.find_distance(pos, distance_x_c, distance_z_c)
            # _observation_dict['ego']['distance_to_center'] = [dist_c]

            ############################# 2D Rangefinder #############################
            local_distance_x_l, local_distance_z_l = self.cc.global_to_local(pos, orientation, distance_x_l, distance_z_l)
            local_distance_x_r, local_distance_z_r = self.cc.global_to_local(pos, orientation, distance_x_r, distance_z_r)
            self.which_side, pcl_distance, _ = self.cc.lidar_2d(local_distance_x_l, local_distance_z_l, local_distance_x_r, local_distance_z_r, **self.lidar_2d_kwargs)
            _observation_dict['ego']['lidar_2d'] = pcl_distance

        ## --------- [Surr_1] Add if you need! (not our scope) --------- ##
        # TODO: Add surrounding vehicle info.

        ## Make observations into numpy array
        _observation_list = []
        for key, _ in self.observation_dict['ego'].items():
            _observation_list += _observation_dict['ego'][key]

        if self.vehicle_count >= 2:
            for cnt in range(1, self.vehicle_count):
                for key, _ in self.observation_dict['surr_' + str(cnt)].items():
                    _observation_list += _observation_dict['surr_' + str(cnt)][key]
        ret = np.array(_observation_list)
        
        if IS_VIS:  # Only for debug process
            try:
                self.ax_queue.append(_observation_dict['ego']['acceleration_local'][0])
                self.ay_queue.append(_observation_dict['ego']['acceleration_local'][1])
                self.az_queue.append(_observation_dict['ego']['acceleration_local'][2])
                self.vx_queue.append(_observation_dict['ego']['velocity_local'][0])
                self.vy_queue.append(_observation_dict['ego']['velocity_local'][1])
                self.vz_queue.append(_observation_dict['ego']['velocity_local'][2])
                self.prev_steering_queue.append(_observation_dict['ego']['previous_steering_cmd'][0])
                self.collision_flag_queue.append(self.collision_flag)

                self.ln12[0].set_data(local_preview_x_c, local_preview_z_c)
                self.ln13.set_offsets(np.array([local_preview_x_c, local_preview_z_c]).T)
                self.ln13.set_array(curvature_preview)
                self.ln14[0].set_data(local_distance_x_l, local_distance_z_l)
                self.ln15[0].set_data(local_distance_x_r, local_distance_z_r)

                angles_deg = np.linspace(self.lidar_2d_kwargs['roi_deg_min'], self.lidar_2d_kwargs['roi_deg_max'], self.lidar_2d_kwargs['num_rays'])
                ray_origin = Point(0, 0)
                for angle_deg, distance, ray, text in zip(angles_deg, pcl_distance, self.rays, self.texts):
                    ray_direction = np.array([np.cos(np.radians(angle_deg + 90)), np.sin(np.radians(angle_deg + 90))])
                    ray_end = ray_origin.x + ray_direction[0] * distance, ray_origin.y + ray_direction[1] * distance
                    ray.set_data([0, ray_end[0]], [0, ray_end[1]])
                    text.set_position(ray_end)
                    text.set_text(f'{distance:.2f}m')
                
                e_phi_direction = np.array([np.cos(_observation_dict['ego']['e_phi'][0] + np.pi/2), np.sin(_observation_dict['ego']['e_phi'][0] + np.pi/2)])
                e_phi_end = ray_origin.x + e_phi_direction[0] * 30, ray_origin.y + e_phi_direction[1] * 30
                self.e_phi_ray.set_data([0, e_phi_end[0]], [0, e_phi_end[1]])
                self.e_phi_text.set_position(e_phi_end)
                e_phi_deg = _observation_dict['ego']['e_phi'][0]*180/np.pi
                self.e_phi_text.set_text(f'{e_phi_deg:.2f}deg')
                self.ax1.set_xlim([20, -20])
                self.ax1.set_ylim([-150, 150])
                # self.ax1.invert_xaxis()
                
                xdata = np.arange(0, self.preview_kwargs['num_sample'] * self.preview_kwargs['sample_interval'], self.preview_kwargs['sample_interval'])
                self.ln21[0].set_data(xdata.tolist(), _observation_dict['ego']['preview_curvature'])

                xdata = np.arange(0, self.window_length, 1)
                self.ln31[0].set_data(xdata, self.ax_queue)
                self.ln31[0].set_data(xdata, np.array(self.ay_queue))
                self.ln31[0].set_data(xdata, np.array(self.az_queue))
                self.ln41[0].set_data(xdata, np.array(self.vx_queue))
                self.ln42[0].set_data(xdata, np.array(self.vy_queue))
                self.ln43[0].set_data(xdata, np.array(self.vz_queue))
                self.ln51[0].set_data(xdata, np.array(self.prev_steering_queue))
                self.ln61[0].set_data(xdata, np.array(self.collision_flag_queue))
                
                self.fig1.canvas.draw_idle()
                plt.pause(0.001)
            
            except Exception as e:
                print("_get_obs functione error")
                print(e)

        ## Normalization
        obs_limit_min = self.observation_space.low
        obs_limit_max = self.observation_space.high
        obs_mean  = (obs_limit_max + obs_limit_min) / 2
        ret = (ret - obs_mean)/(obs_limit_max - obs_limit_min)*2    # Normalization to [-1,1]
        
        return ret, _observation_dict


    def _get_frame(self):
        """
        Get frame data from captured image.
        """
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (int(self.frame_dim[1]), int(self.frame_dim[0])))
                break
            else:
                print("Cannot read frame from webcam.")

        ## Normalization
        frame = frame / 255.0
        
        return frame 


    def _get_info(self):
        """
        Get additional useful information from json file.
        """
        current_progress = self.big_dict['ego']['track_progress']

        if self.big_dict['ego']['speed'] < 5:   # [kph]
            self.stuck_step += 1
        elif (self.big_dict_pre is not None) and (self.big_dict['ego']['position_global'] == self.big_dict_pre['ego']['position_global']):
            self.stuck_step += 1
        elif current_progress == self.big_dict_pre['ego']['track_progress']:
            self.stuck_step += 1
        else:
            self.stuck_step = 0

        if any(ratio > 0.3 for ratio in self.big_dict['ego']['slip_ratio']):
            self.is_slip = 1
        else:
            self.is_slip = 0
        
        best_lap_time_updated = False
        if (self.big_dict['ego']['best_lap_time'] != 0) and (self.big_dict['ego']['best_lap_time'] < self.best_lap_time):
            self.best_lap_time = self.big_dict['ego']['best_lap_time']
            best_lap_time_updated = True
            print(" !!! Best lap time updated !!! ")
        
        return {"stuck_step": self.stuck_step, "is_slip": self.is_slip, "best_lap_time_updated": best_lap_time_updated}
        

    ######################## Define reward functions ########################
    def _get_ckpt_reward(self, obs_dict, action, info):
        """
        Reward function for passing a pre-defined checkpoint.
        """
        num_section = 10
        track_progress = self.big_dict['ego']['track_progress']
        pre_track_progress = self.big_dict_pre['ego']['track_progress']
        
        if track_progress < pre_track_progress - 0.9:
            track_progress = track_progress+1
        
        section_idx = track_progress // (1 / num_section)
        pre_section_idx = pre_track_progress // (1 / num_section)
        
        if section_idx != pre_section_idx:
            return 1
        else:
            return 0

    def _get_speed_reward(self, obs_dict, action, info):
        """
        Reward function for higher speed.
        """
        return self.big_dict['ego']['speed'] / 3.6  # [m/s]

    def _get_laptime_reward(self, obs_dict, action, info, power=1):
        """
        Reward function for faster lap time.
        """
        if (self.lap_end_flag==0) and (self.big_dict['ego']['track_progress'] < 0.05):
            self.lap_end_flag = 1
        if (self.lap_end_flag==1) and (self.big_dict['ego']['track_progress'] > 0.995):
            self.lap_end_flag = 0
            return max(3, np.sign(self.lap_time_standard - self.big_dict['ego']['current_lap_time']) * np.abs((self.lap_time_standard - self.big_dict['ego']['current_lap_time']) / 1000.0)** power)
        
        return 0
    
    def _get_best_laptime_reward(self, obs_dict, action, info):
        """
        Reward function for achieving best lap time.
        """
        if info["best_lap_time_updated"]:
            return 1
        else:
            return 0

    def _get_underpace_penalty(self, obs_dict, action, info):
        """
        Penalty function for underpace driving.
        """
        if info['stuck_step'] > 50:
            return -1
        if self.big_dict['ego']['speed'] < 30:
            return -1
        
        return 0
  
    def _get_time_penalty(self, obs_dict, action, info):
        """
        Penalty function for every step.
        """
        return -1

    def _get_tyre_off_track_penalty(self, obs_dict, action, info):
        """
        Penalty function for off-track driving.
        """
        if (self.big_dict['ego']['num_tyres_off_track'] >= 3) or ((self.big_dict['ego']['num_tyres_off_track'] >= 2) and ((self.which_side == [1, 0, 0]) or (self.which_side == [0, 0, 1]))):
            return -1
        
        return 0

    def _get_slip_penalty(self, obs_dict, action, info):
        """
        Penalty function for tire slip.
        """
        if info['is_slip']:
            return -1
        
        return 0

    def _get_collision_penalty(self, obs_dict, action, info):
        """
        Penalty function for collision.
        """
        return -self.collision_flag
    

    def _get_reward(self, obs_dict, action, info):
        """
        Get sum of weighted rewards from observation data. 
        """
        reward_weights_functions =[
            (3, self._get_ckpt_reward),
            (0, self._get_speed_reward),
            (1, self._get_laptime_reward),
            (0, self._get_best_laptime_reward),
            (0, self._get_underpace_penalty),
            (0, self._get_time_penalty),
            (5, self._get_tyre_off_track_penalty),
            (0, self._get_slip_penalty),
            (0, self._get_collision_penalty)
        ]
        reward = 0
        for weight, func in reward_weights_functions:
            weighted_reward = weight * func(obs_dict, action, info)
            # if weight != 0:
            #     wandb.log({"reward: "+func.__name__[5:]: weighted_reward}, commit=False)  # Optional, for logging
            reward += weighted_reward
        # wandb.log({"reward_at_each_step": reward},commit=False)
        
        return reward
    
    
    def _get_terminated(self, info):
        """
        Check whether the episode satisfies termination condition.
        """
        ## Lap end
        if self.big_dict['ego']['lap_count'] == 2:
            print("="*6 + " Lap end " + "="*6)
            print(" Best lap time  {} : {} : {} ".format(self.best_lap_time//60000, (self.best_lap_time%60000)//1000, (self.best_lap_time%60000)%1000))
            print(" Last lap time  {} : {} : {} ".format(self.big_dict['ego']['last_lap_time']//60000, (self.big_dict['ego']['last_lap_time']%60000)//1000, (self.big_dict['ego']['last_lap_time']%60000)%1000))
            wandb.log({"best_lap_time": self.best_lap_time},commit=False)
            return True
        
        ## Stuck end
        if info['stuck_step'] > 100:
            # print("="*6 + " Terminated! : stuck end " + "="*6)
            return True
        
        ## Tire-off-track end
        if self.big_dict['ego']['num_tyres_off_track'] >= 3:
            # print("="*6 + " Terminated! : tire_off_track end " + "="*6)
            return True
        elif ((self.big_dict['ego']['num_tyres_off_track'] >= 2) and ((self.which_side == [1, 0, 0]) or (self.which_side == [0, 0, 1]))):
            return True

        return False


    def step(self, action, return_dict=False, action_exist=True, absorb_wrapping=False):
        """
        Environment step function. Perform action and get observation, reward, termination, and info..
        """
        if not self.ai_on_flag and not action_exist:    # (Optional) Turn on built-in AI when action is not given. Not used in our work.
            self.controller.ai_on()
            self.ai_on_flag = True

        if action_exist:
            if self.ai_on_flag:
                self.controller.ai_off()
                self.ai_on_flag = False
            
            if self.action_len == 2:    # Unified
                if action[1] > 0:
                    self.controller.perform(action[1], 0, action[0])
                else:
                    self.controller.perform(0, -1 * action[1], action[0])
                    
            elif self.action_len == 3:  # Separated
                self.controller.perform(action[2], action[0], action[1])
        
        ## In algorithm loop (e.g. sac) where 'step' function called,
        ## there are collection stage, update stage, and test/save stage.
        ## 1. Collection stage : 'step' function is called periodically, just consider step_time.
        ## 2. Update stage : 'step' function is paused until update loop is done, consider update_time.
        ## 3. Test/Save stage : env will be reset, nothing to be considered.
        
        ## Sleep for maintaining the environment fps.
        sleep_time = 1/self.env_fps - (time.time() - self.step_time - self.update_time)
        self.update_time = 0
        time.sleep(max(0, sleep_time))
        self.step_time = time.time()
        
        obs, self.obs_dict = self._get_obs(absorb_wrapping=absorb_wrapping)
        if self.frame_exist:
            frame = self._get_frame()
        info = self._get_info()
        reward = self._get_reward(self.obs_dict, action, info)
        terminated = self._get_terminated(info)
        self.big_dict_pre = copy.deepcopy(self.big_dict)

        if return_dict:
            if self.frame_exist:
                return (obs, frame), reward, terminated, False, info, self.big_dict
            else:
                return obs, reward, terminated, False, info, self.big_dict
        else:
            if self.frame_exist:
                return (obs, frame), reward, terminated, False, info
            else:
                return obs, reward, terminated, False, info


    def _respawn_random(self, cnt):
        """
        Random respawn function.
        """
        ## Uniform probability
        # respawn_idx = np.random.randint(0, self.track_length-1)
        
        ## Or, you can use custom respawn points
        spawn_point_idx_custom = [0, 20, 730, 750, 1950, 3200]
        respawn_idx = random.sample(spawn_point_idx_custom, 1)[0]

        ## You also can give random offset to respawn point
        random_r = np.random.rand() * 5                 # Random radius offset from respawn point [m]
        random_theta = np.random.rand() * 2 * np.pi     # Random angle offset from respawn point [rad]

        respawn_x = self.track_data[respawn_idx, 0] + random_r * np.cos(random_theta)
        respawn_z = self.track_data[respawn_idx, 2] + random_r * np.sin(random_theta)
        respawn_dx = self.track_data[respawn_idx+1, 0] - self.track_data[respawn_idx, 0]
        respawn_dz = self.track_data[respawn_idx+1, 2] - self.track_data[respawn_idx, 2]
        with open(PREFIX_teleport, 'w') as f:
            f.write(str(respawn_x)+','+str(respawn_z)+','+str(respawn_dx)+','+str(respawn_dz)+','+str(cnt))
            
        return np.array([respawn_x, respawn_z])


    def reset(self, random_spawn=False, portion=1, return_dict=False, with_ai=False, absorb_wrapping=False):
        """
        Environment reset function.
        Reset the environment and return initial observations.
        """
        self.reset_flag = True
        self.lap_end_flag = 0
        self.big_dict_pre = None
        self.controller.reset_controller()
        self.controller.reset_car()
        if with_ai:
            self.controller.ai_on()
            self.ai_on_flag = True
        self.collision_flag = 0
        self.stuck_step = 0
        self.is_slip = 0
        self.update_time = 0
        cnt = 0               

        if random_spawn and (np.random.rand() < portion):
            while True:
                ## Try respawn 
                cnt += 1
                respawn_point = self._respawn_random(cnt)
                self.step_time = time.time()
                self.reset_flag = True
                
                ## Check whether the respawn point is valid
                obs, self.obs_dict = self._get_obs(absorb_wrapping=absorb_wrapping)
                position_global = np.array(self.big_dict['ego']['position_global'])[[0,2]]
                if np.linalg.norm(position_global - respawn_point) < 5:
                    break   # Sucess!
        else:
            self.reset_flag = True
            self.step_time = time.time()
            obs, _ = self._get_obs(absorb_wrapping=absorb_wrapping)

        self.big_dict_pre = copy.deepcopy(self.big_dict)
        info = self._get_info()
        
        if return_dict:
            return obs, self.big_dict
        if self.frame_exist:
            frame = self._get_frame()
            return (obs, frame)
        else:
            return obs
        

    def pause(self):
        """
        Pause the environment. (usually for update loop)
        """
        while True:
            self.pause_time = time.time()
            self.controller.pause()
            self.big_dict_pre = copy.deepcopy(self.big_dict)
            self._get_obs()
            if (self.big_dict_pre['time_at_sampling'] != self.big_dict['time_at_sampling']) and (self.big_dict_pre['ego']['current_lap_time'] == self.big_dict['ego']['current_lap_time']):
                break


    def resume(self):
        """
        Resume the environment.
        """
        while True:
            self.controller.resume()
            big_dict_pre = copy.deepcopy(self.big_dict)
            self._get_obs()
            if (big_dict_pre['time_at_sampling'] != self.big_dict['time_at_sampling']) and (big_dict_pre['ego']['current_lap_time'] != self.big_dict['ego']['current_lap_time']):
                break
        self.update_time = time.time() - self.pause_time


    def close(self):
        self.controller.reset_controller()
        self.cap.release()
        cv2.destroyAllWindows()