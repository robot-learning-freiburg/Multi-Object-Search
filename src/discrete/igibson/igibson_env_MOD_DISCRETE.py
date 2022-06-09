import argparse
import logging
import time
from collections import OrderedDict

import gym
import numpy as np
import pybullet as p
#from transforms3d.euler import euler2quat #<-- is set below

#from igibson.envs.env_base import BaseEnv
from src.general.igibson.env_base import BaseEnv
from igibson.external.pybullet_tools.utils import stable_z_on_aabb
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.bump_sensor import BumpSensor
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor

#from igibson.tasks.point_nav_random_task_MOD2 import PointNavRandomTask
from src.general.igibson.point_nav_random_task_MOD2 import PointNavRandomTask

from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.utils import quatToXYZW

#-----------------------------------------------------------
import cv2
#from functools import reduce 
#import time
from transforms3d.euler import euler2quat,euler2mat
from scipy.stats import circvar
from scipy.special import softmax
#from torchvision.transforms.functional import affine
#from PIL import Image #<--- only needed for saving map test
#from igibson.utils.utils import cartesian_to_polar

class iGibsonEnv(BaseEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
        evaluate=False,
        mix_sample=True
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        
        self.load_miscellaneous_FABI()
        self.eval = evaluate
        self.aux_bin_number = 12
        self.mix_sample = mix_sample
        super(iGibsonEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,

        )
        self.automatic_reset = automatic_reset
        self.ground_truth_map = self.config.get("use_ground_truth", False)
        #self.use_seperate_map = self.config.get("use_seperate_map", False)
        self.load_miscellaneous_FABI2()
        self.queue_for_task_resampling = []
        self.episode_counter = 0
        #self.aux_sr_rate = []
        #self.prev_aux_sr_rate = 0
        self.aux_episodic_prob = self.config.get('initial_aux_prob', 0.16)
        self.evaluate = self.config.get('evaluate',False)
        
        #self.cat_to_angle = {0:0.0,1:45.0,2:90.0,3:135.0,4:180.0,5:225.0,6:270.0,7:315.0}
        self.offset_for_cut = 150
        self.aux_points = []
        
        step_size = 360/self.aux_bin_number
        self.angle_to_bins = np.arange(step_size,360+step_size,step_size)#[45,90,135,180,225,270,315,360]
        #deg_steps = [0,45,90,135,180,225,270,315]
        #from degree 0 to 360 with step size 30 => 360/12 = 30 degree steps
        deg_steps = np.arange(0,360,step_size)
        for i,deg in enumerate(deg_steps):
            ax = self.pol2cart(0.5,np.deg2rad(deg))
            ax = np.array([ax[0],ax[1],0.0])
            self.aux_points.append(ax)
        
        self.SR_rate = []

    def load_miscellaneous_FABI2(self):
        #Images have BGR format
        self.floor = np.array([0, 255, 0])
        self.obstalce = np.array([255, 0, 0])
        self.trace = np.array([164, 0, 255])
        self.arrow = np.array([0, 128, 255])

        self.scene_reset_counter = 0
        self.aux_prob_counter = 0
        self.category = np.array([253, 253, 253])
        #self.category_cracker = np.array([64, 64, 64])
        #self.category_sugar = np.array([32, 152, 196])
        #self.category_pudding = np.array([12, 48, 96])

        self.categorie_array= [np.array([64, 64, 64]),np.array([32, 152, 196]),np.array([12, 48, 96]),np.array([102, 32, 77])\
        ,np.array([126, 55, 133]),np.array([140, 109, 84]),np.array([112, 101, 90]),np.array([155, 112, 101]),np.array([177, 155, 112]),\
        np.array([31, 177, 155]),np.array([51, 202, 177]),np.array([81, 214, 217])]

        self.category_found = np.array([249, 192, 203])

        #self.categorie_array= [np.array([64, 64, 64]),np.array([32, 152, 196]),np.array([12, 48, 96]),np.array([102, 32, 77]),np.array([126, 55, 71]),np.array([140, 71, 84])]
        self.category_picture = np.array([13, 66, 220])
        self.category_shelf = np.array([77, 12, 128])
        self.category_console_table = np.array([95, 190, 45])
        self.aux_pred =  np.array([192,25,79])
        self.depth_high = self.config.get('depth_high', 2.5)
        self.depth_low = self.config.get('depth_low', 0.5)
        self.aux_task = self.config.get('use_aux_task', False)
        self.show_map = self.config.get('show_map', False)
        self.target_on_map = self.config.get('target_on_map', False)
        self.history_length = self.config.get('history_length', 1)
        self.substitute_polar = self.config.get('substitute_polar', False)
        self.aux_on_map = self.config.get('aux_on_map', False)
        self.set_polar_dist_zero = self.config.get('set_polar_dist_zero', False)
        self.reset_agent_pos = self.config.get('reset_agent_pos', False)
        self.multiple_envs = self.config.get('multiple_envs', False)
        #self.grid_res = self.config.get('grid_res', 0.035)
        self.resample_task = self.config.get('resample_task', True)
        self.eval_heat_map = self.config.get('eval_heat_map', False)
        
        self.cam_fov = self.config.get('vertical_fov', 79.0)
        #self.aux_loss = args.aux
        self.initial_camera_pitch = 0.0
        
        
    def load_miscellaneous_map(self,scene_id):
        map_settings_size = self.map_settings[scene_id]['map_size']
        self.map_size = (map_settings_size,map_settings_size)#164 neu 142

        self.map_size_2 = (128,128)
        #self.proxy_task = args.proxy
        

        self.cut_out_size = (84,84)
        self.x_off1 = self.map_size[0] - self.cut_out_size[0]
        self.x_off2 = int(self.map_size[0] - (self.x_off1//2))
        self.x_off1 = self.x_off1 // 2

        self.y_off1 = self.map_size[1] - self.cut_out_size[1]
        self.y_off2 = int(self.map_size[1] - (self.y_off1//2))
        self.y_off1 = self.y_off1 // 2

        #greater map
        self.cut_out_size2 = (420,420)
        self.x_off11 = self.map_size[0] - self.cut_out_size2[0]
        self.x_off22 = int(self.map_size[0] - (self.x_off11//2))
        self.x_off11 = self.x_off11 // 2

        self.y_off11 = self.map_size[1] - self.cut_out_size2[1]
        self.y_off22 = int(self.map_size[1] - (self.y_off11//2))
        self.y_off11= self.y_off11 // 2

    def load_miscellaneous_FABI(self):
        # Removed it from other function because reset scene resets this parameter as well in load_miscellaneous_variables
        #self.episode_stats = {}
        #self.episode_stats['collision'] = []
        #self.episode_stats['succ_rate'] = []

        #self.rgb_input = False
        
        
        #self.num_cats = args.num_sem_categories + 4 
        #self.obs_space = 3
        #self.proprio_input_size = 3
        

        #self.grid_offset = np.array([2.5, 7.3,0.1]) <- old map
        
        self.grid_res = 0.033#self.config.get('grid_res', 0.035)
        self.map_settings = {}
        #Training
        #self.train_set = ['Rs_int','Benevolence_1_int','Benevolence_2_int','Beechwood_0_int','Beechwood_1_int','Wainscott_1_int','Pomaria_0_int','Pomaria_1_int','Pomaria_2_int','Ihlen_1_int']
        self.map_settings['Rs_int'] = {'grid_offset':np.array([6.0, 5.5,15.1]),'grid_spacing':np.array([self.grid_res,self.grid_res,0.1]),'offset':0,'object_dist':1.0,'doors':['door_54'],"map_size":450}
        self.map_settings['Benevolence_1_int'] = {'grid_offset':np.array([5.5, 10.5,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':1.4,'doors':['door_52'],"map_size":450}#1.2
        self.map_settings['Benevolence_2_int'] = {'grid_offset':np.array([5.5, 10.5,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':1.7,'doors':[],"map_size":450}#1.6
        self.map_settings['Beechwood_0_int'] = {'grid_offset':np.array([13.5, 8.5,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':2.5,'doors':['door_93','door_109'],"map_size":625}#2.5
        self.map_settings['Beechwood_1_int'] = {'grid_offset':np.array([11.5, 8.5,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':2.5,'doors':[],"map_size":600}
        self.map_settings['Wainscott_1_int'] = {'grid_offset':np.array([8.0, 8.0,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':2.5,'doors':[],"map_size":700} #<--- massive map
        self.map_settings['Pomaria_0_int'] = {'grid_offset':np.array([15.0, 6.5,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':2.0,'doors':['door_41','door_42'],"map_size":550}
        self.map_settings['Pomaria_1_int'] = {'grid_offset':np.array([15.0, 7.0,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':2.1,'doors':['door_65','door_70'],"map_size":550}
        self.map_settings['Pomaria_2_int'] = {'grid_offset':np.array([7.5, 7.5,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':1.24,'doors':[],"map_size":450}
        self.map_settings['Ihlen_1_int'] = {'grid_offset':np.array([7.0, 4.5,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':2.2,'doors':['door_86','door_91'],"map_size":500}
        
        
        #Validation
        self.map_settings['Ihlen_0_int'] = {'grid_offset':np.array([5.5, 3.0,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':2.5,'doors':['door_42'],"map_size":450}
        self.map_settings['Benevolence_0_int'] = {'grid_offset':np.array([5.5, 9.5,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':0.2,'doors':['door_9','door_12','door_13'],"map_size":450}
        self.map_settings['Merom_0_int'] = {'grid_offset':np.array([4.5, 3.5,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':1.6,'doors':['door_60'],"map_size":450}
        self.map_settings['Merom_1_int'] = {'grid_offset':np.array([10.0, 7.0,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':1.6,'doors':['door_74','door_93'],"map_size":650}

        self.map_settings['Wainscott_0_int'] = {'grid_offset':np.array([8.5, 8.0,15.1]),'grid_spacing':np.array([self.grid_res, self.grid_res,0.1]),'offset':0,'object_dist':1.6,'doors':['door_126','door_128','door_132','door_135'],"map_size":750} #<--- massive map
        
        

    def load_task_setup(self):
        """
        Load task setup
        """
        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config.get("discount_factor", 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)

        # task
        if self.config["task"] == "point_nav_fixed":
            self.task = PointNavFixedTask(self)
        elif self.config["task"] == "point_nav_random":
            self.task = PointNavRandomTask(self)
        elif self.config["task"] == "interactive_nav_random":
            self.task = InteractiveNavRandomTask(self)
        elif self.config["task"] == "dynamic_nav_random":
            self.task = DynamicNavRandomTask(self)
        elif self.config["task"] == "reaching_random":
            self.task = ReachingRandomTask(self)
        elif self.config["task"] == "room_rearrangement":
            self.task = RoomRearrangementTask(self)
        else:
            self.task = None


        self.task_2_object = self.config.get("task_2_object", False)
        self.task_3_object = self.config.get("task_3_object", False)
        self.numb_sem_categories = self.config.get("sem_categories", 1)
        self.last_scene_id = self.config.get("scene_id",'Rs_int')
        self.task.load_custom_objects(self)

        
    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config["output"]
        self.image_width = self.config.get("image_width", 128)
        self.image_height = self.config.get("image_height", 128)
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        if "task_obs" in self.output:
            observation_space["task_obs"] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=np.inf
            )
        if "rgb" in self.output:
            observation_space["rgb"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb")
        if "depth" in self.output:
            observation_space["depth"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            vision_modalities.append("depth")
        if "pc" in self.output:
            observation_space["pc"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("pc")
        if "optical_flow" in self.output:
            observation_space["optical_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2), low=-np.inf, high=np.inf
            )
            vision_modalities.append("optical_flow")
        if "scene_flow" in self.output:
            observation_space["scene_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("scene_flow")
        if "normal" in self.output:
            observation_space["normal"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("normal")
        if "seg" in self.output:
            observation_space["seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_CLASS_COUNT
            )
            vision_modalities.append("seg")
        if "ins_seg" in self.output:
            observation_space["ins_seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_INSTANCE_COUNT
            )
            vision_modalities.append("ins_seg")
        if "rgb_filled" in self.output:  # use filler
            observation_space["rgb_filled"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb_filled")
        if "highlight" in self.output:
            observation_space["highlight"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            vision_modalities.append("highlight")
        if "scan" in self.output:
            self.n_horizontal_rays = self.config.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan")
        if "occupancy_grid" in self.output:
            self.grid_resolution = self.config.get("grid_resolution", 128)
            self.occupancy_grid_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.grid_resolution, self.grid_resolution, 1)
            )
            observation_space["occupancy_grid"] = self.occupancy_grid_space
            scan_modalities.append("occupancy_grid")
        if "bump" in self.output:
            observation_space["bump"] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
            sensors["bump"] = BumpSensor(self)

        if len(vision_modalities) > 0:
            sensors["vision"] = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            sensors["scan_occ"] = ScanSensor(self, scan_modalities)

        task_ob = observation_space["task_obs"]
        observation_space = OrderedDict()
        self.last_action_obs = self.config.get('last_action_obs', False)
        self.use_rgb_depth = self.config.get('rgb_depth', False)
        self.downsample_size = self.config.get('global_map_size', 128)
        self.not_polar = self.config.get('not_polar', False)
        self.history_length_aux = self.config.get('history_length_aux', 10)
        self.numb_sem_categories = self.config.get("sem_categories", 1)
        """
        if self.not_polar:
            #angular,linear velocity[2,], collision flag[1,], last action [2,], number categories [n,], eventually [number_categories flag 
            #task_obs_dim = 4
            
            observation_space["task_obs"] = self.build_obs_space(shape=(self.task.task_obs_dim+self.numb_sem_categories+1,), low=-np.inf, high=np.inf)
        else:
            #polar dist and polar angle [2,],angular,linear velocity[2,], collision flag[1,], last action [2,], number categories [n,], eventually [number_categories flag 
        """    

        additional_stuff = self.aux_bin_number+(self.history_length_aux*self.aux_bin_number)
        observation_space["task_obs"] = self.build_obs_space(shape=(self.numb_sem_categories+1+2+1+1+2+additional_stuff+2,), low=-np.inf, high=np.inf)#<<<<<<<<<<<<<<<<<<<<<<

            
        #observation_space["task_obs"] = self.build_obs_space(shape=(self.task.task_obs_dim-2+self.numb_sem_categories+1+2+1,), low=-np.inf, high=np.inf)#--- Without auxilliary
        #observation_space['goal_obs'] = self.build_obs_space(shape=(self.numb_sem_categories,), low=0.0, high=1.0)

        observation_space['image'] = gym.spaces.Box(low=0,high=255,shape=(3,self.cut_out_size[0],self.cut_out_size[1]),dtype=np.uint8)# prev. use self.downsample_size 128 -> 64
        observation_space['image_global'] = gym.spaces.Box(low=0,high=255,shape=(3,self.downsample_size,self.downsample_size),dtype=np.uint8)
        """
        if self.use_rgb_depth :
            observation_space['rgb'] = gym.spaces.Box(low=0,high=1.0,shape=(3,128,128),dtype=np.float32)
            observation_space['depth'] = gym.spaces.Box(low=0,high=1.0,shape=(1,128,128),dtype=np.float32)
        """
        self.observation_space = gym.spaces.Dict(observation_space)
        #self.observation_space = gym.spaces.Box(low=0,high=255,shape=(3,self.downsample_size,self.downsample_size),dtype=np.uint8)
        self.sensors = sensors

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = self.robots[0].action_space

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = []

    def load(self):
        """
        Load environment
        """
        super(iGibsonEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def get_state(self, collision_links=[]):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if "task_obs" in self.output:
            state["task_obs"] = self.task.get_task_obs(self)
        if "vision" in self.sensors:
            vision_obs = self.sensors["vision"].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]
        if "scan_occ" in self.sensors:
            scan_obs = self.sensors["scan_occ"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "bump" in self.sensors:
            state["bump"] = self.sensors["bump"].get_obs(self)

        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class)

        :return: collision_links: collisions from last physics timestep
        """
        self.simulator_step()
        collision_links = list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        
        new_collision_links = []
        for item in collision_links:
           

            # ignore collision with body b
            
            if item[2] in self.collision_ignore_body_b_ids:
               continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:# or item[3] in [9,6]:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:# or (item[2] in [3]):
                continue

            if item[2] in self.task.remove_collision_links:
                continue
            
            #self.a_s.append(item[3])
            #self.b_s.append(item[2])
            #self.c_s.append(item[4])
            new_collision_links.append(item)

        #print("a:",np.unique(np.array(self.a_s)))
        #print("b:",np.unique(np.array(self.b_s)))
        #print("c:",np.unique(np.array(self.c_s)))
        #print("COL LINKS:",collision_links)
        return new_collision_links

    def populate_info(self, info):
        """
        Populate info dictionary with any useful information
        """
        info["episode_length"] = self.current_step
        info["collision_step"] = self.collision_step
        info["aux_episode"] = self.episodic_aux_prediction
        #info["map_id"] = self.last_scene_id
        #info["numb_objects"] = self.task.num_cat_in_episode

    def world2map(self,xy,tiny_global=False):
        #self.seg_map_size = int(height * self.seg_map_default_resolution / self.seg_map_resolution)
        #np.flip((xy / self.seg_map_resolution + self.seg_map_size / 2.0)).astype(np.int)
        #np.flip((xy / 0.1 + self.seg_map_size / 2.0)).astype(np.int)
        
        if(len(xy.shape) > 1):
            
            #arr =  []
            #for i in range(xy.shape[0]):
            #    arr.append([(xy[i,0]+self.grid_offset[0])/self.grid_spacing[0],(xy[i,1]+self.grid_offset[1])/self.grid_spacing[1]])
            
            gpoints = np.array([self.offset,self.offset,self.offset]) + np.round((xy + self.grid_offset) / self.grid_spacing)
            #arr = np.array(arr)
            #print(f"shapes: {xy.shape} and {gpoints.shape} arr: {arr.shape}")
            #return np.array(arr)
            
            return gpoints

        else:
            
            x = (xy[0]+self.grid_offset[0])/self.grid_spacing[0]
            y = (xy[1]+self.grid_offset[1])/self.grid_spacing[1]
        
            return [np.round(x)+self.offset,np.round(y)+self.offset]

    def get_contour_points(self,pos, origin, size=20):
        x, y, o = pos
    
        #spitze
        #pt5 = (int(x + 0.5*(size * np.cos(o))) + origin[0],
        #    int(y + 0.5*(size * np.sin(o))) + origin[1])

        #was at 0.4 -> 0.52 would be more accurate
        #vorne links
        pt1 = (int(x + 0.65*(size * np.cos(o-0.3))) + origin[0],
            int(y + 0.65*(size * np.sin(o-0.3))) + origin[1])
    
        #vorne rechts
        pt3 = (int(x + 0.65*(size * np.cos(o+0.3))) + origin[0],
            int(y + 0.65*(size * np.sin(o+0.3))) + origin[1])

        #hinten links
        pt2 = (int(x + 0.55*(size  * np.cos(o + np.pi +0.3))) + origin[0],
            int(y + 0.55*(size * np.sin(o + np.pi +0.3))) + origin[1])

        #hinten rechts
        pt4 = (int(x + 0.55*(size * np.cos(o - np.pi -0.3))) + origin[0],
            int(y + 0.55*(size * np.sin(o - np.pi -0.3))) + origin[1])

        return np.array([pt1,pt3, pt4, pt2])

    def draw_arrow(self,camera_yaw,tiny_global=False):
        """
        if(tiny_global):
            pos = ((self.map_size_2[1]//2),(self.map_size_2[0]//2),camera_yaw)#(agent_yaw - initial_yaw_orn))#)#)#)#)
            vis_image = np.ones((self.map_size_2[0], self.map_size_2[1], 3)).astype(np.uint8) * 255
        else:
        """


        #pos = ((self.map_size[1]//2),(self.map_size[0]//2),camera_yaw)#(agent_yaw - initial_yaw_orn))#)#)#)#)
        vis_image = np.ones((self.map_size[0], self.map_size[1], 3)).astype(np.uint8) * 255

        #agent_arrow = self.get_contour_points(pos, origin=(0, 0),size=13)
   
    
    
        
        color = (int(0.1 * 255),
        int(0.2 * 255),
        int(0.3 * 255))

    
        #cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

        cv2.circle(
            img=vis_image,
            center=(self.map_size[0] // 2, self.map_size[1] // 2),
            radius=int(6),
            color=color,#(255,255,255),
            thickness=-1,
        )

        mask = (vis_image == color[0])[:,:,0]
        return mask

    def pad_img_to_fit_bbox(self,img, x1, x2, y1, y2):
        left = np.abs(np.minimum(0, y1))
        right = np.maximum(y2 - img.shape[0], 0)
        top = np.abs(np.minimum(0, x1))
        bottom = np.maximum(x2 - img.shape[1], 0)
        img = np.pad(img, ((left, right), (top, bottom), (0, 0)), mode="constant")
        
        y1 += left
        y2 += left
        x1 += top
        x2 += top
        return img, x1, x2, y1, y2


    def crop_fn(self,img: np.ndarray, center, output_size):
        h, w = np.array(output_size, dtype=int)
        x = int(center[0] - w / 2)
        y = int(center[1] - h / 2)

        y1, y2, x1, x2 = y, y + h, x, x + w
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = self.pad_img_to_fit_bbox(img, x1, x2, y1, y2)
        return img[y1:y2, x1:x2]

    def affine4map(self,rot_agent,rot_camera,action=None,euler_mat=None,camera_translation=None):

        
        img_copy = self.global_map.copy()
        #img_copy2 = self.global_map_2.copy()
        #np.savetxt('data.csv', self.global_map, delimiter=',')
        """
        if self.target_on_map:
            #draw target position on map
            tar_world = self.world2map(self.task.target_pos)
            img_copy[int(tar_world[1])-4:int(tar_world[1])+4,int(tar_world[0])-4:int(tar_world[0])+4] = self.aux_pred
        """
        #if self.aux_on_map:

        #<<<<<<<<<<<<<<<<<<<
        
        if action is not None:
            #draw target position on map
            #self.aux_angle = np.arctan2(action['aux_angle'][0],action['aux_angle'][1])
            
            #y = np.exp(action['aux_angle'] - np.max(action['aux_angle']))
            self.aux_action =  softmax(action['aux_angle'])#y / np.sum(np.exp(action['aux_angle']))
            #print("aux softmax action",self.aux_action, "max:",np.argmax(self.aux_action))
            #print("sum:",np.sum(self.aux_action))
            #self.aux_angle = np.deg2rad(self.cat_to_angle[np.argmax(action['aux_angle'])])-np.pi
            #self.aux_dist = action['aux_dist'][0]
            p_ax = self.world2map(euler_mat.dot(np.array(self.aux_points).T).T + camera_translation)
            for i,p in enumerate(p_ax):
                
                img_copy[int(p[1])-2:int(p[1])+2,int(p[0])-2:int(p[0])+2,1::] = self.aux_pred[1::]
                img_copy[int(p[1])-2:int(p[1])+2,int(p[0])-2:int(p[0])+2,0] = self.aux_pred[0]*self.aux_action[i]

        

       
        


        pos = self.rob_pose#self.world2map(np.array([sim_pos[0],sim_pos[1]]))
        
        
        #40 is the additional offset to alleviate the cut off reagions while rotating
        cropped_map = self.crop_fn(img_copy, center=pos, output_size=(self.cut_out_size2[0]+self.offset_for_cut, self.cut_out_size2[1]+self.offset_for_cut))
        #when putting here, the circle can appear as rectangle on the global map.
        
        #cv2.circle(
        #    img=cropped_map,
        #    center=(self.cut_out_size2[0] // 2, self.cut_out_size2[1] // 2),
        #    radius=int(6),
        #    color=(int(self.arrow[0]),int(self.arrow[1]),int(self.arrow[2])),
        #    thickness=-1,
        #)
        

        w, h, _ = cropped_map.shape
        center = (h / 2, w / 2)
        M = cv2.getRotationMatrix2D(center, np.rad2deg(rot_agent) + 90.0, 1.0)
        ego_map = cv2.warpAffine(cropped_map, M, (h, w),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))  # flags=cv2.INTER_AREA,INTER_NEAREST

        #when putting the circle here, the global map has a sharper version in which the circle does not appear as rectangle in some cases
        cv2.circle(
            img=ego_map,
            center=((self.cut_out_size2[0]+self.offset_for_cut) // 2, (self.cut_out_size2[1]+self.offset_for_cut) // 2),
            radius=int(6),
            color=(int(self.arrow[0]),int(self.arrow[1]),int(self.arrow[2])),
            thickness=-1,
        )
        
        ego_map_local = self.crop_fn(ego_map, center=(ego_map.shape[0] / 2, ego_map.shape[1] / 2), output_size=(self.cut_out_size[0], self.cut_out_size[1]))
        ego_map = self.crop_fn(ego_map, center=(ego_map.shape[0] / 2, ego_map.shape[1] / 2), output_size=(self.cut_out_size2[0], self.cut_out_size2[1]))
        ego_map_global = cv2.resize(ego_map, (self.downsample_size, self.downsample_size), interpolation=cv2.INTER_NEAREST)

        

        return ego_map_local,ego_map_global

        

    def pointcloud(self,depth):
        
        #fy = fx = 0.5 / np.tan(fov * 0.5) # assume aspectRatio is one.
        depth = depth.squeeze()
        rows, cols = depth.shape

        px, py = (rows / 2, cols / 2)
        hfov = self.cam_fov / 360. * 2. * np.pi
        fx = rows / (2. * np.tan(hfov / 2.))

        vfov = 2. * np.arctan(np.tan(hfov / 2) * cols / cols)
        fy = cols / (2. * np.tan(vfov / 2.))

        
        
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 0)# & (depth < 255)
        z = np.where(valid, depth, 0.0)
        x = np.where(valid, z * (c - (rows/2)) / fx, 0)
        y = np.where(valid, z * (r - (cols/2)) / fy, 0)
        return np.dstack((z, -x, y))

    def draw_object_categories(self,objects_current_frame,point_cloud,euler_mat,camera_translation):
        

        #One could use np.flip
        for ob in objects_current_frame:
            #0 is outer space
            if(ob == 1020 or ob == 1275 or ob == 0):
                continue
            
            point_cloud_category = point_cloud[(self.seg_mask == ob).squeeze(),:]    
            #indices = np.argwhere(cat2[:,0] == 0.0)
            #cat2 = np.delete(cat2,indices[:,0],axis=0)
            point_cloud_category = euler_mat.dot(point_cloud_category.T).T + camera_translation
            point_cloud_category = self.world2map(point_cloud_category).astype(np.uint16)
            #255 is the ceiling
            if ob in self.task.ob_cats or ob == 1530:
                pass
            #sink
            elif(ob == 81600):
                self.global_map[point_cloud_category[:,1],point_cloud_category[:,0]] = self.category_picture
            #bed now door
            elif(ob == 31110):#6375): 6375 was originally the bed class
                self.global_map[point_cloud_category[:,1],point_cloud_category[:,0]] = self.category_shelf
            #sofa
            elif(ob == 82365):
                self.global_map[point_cloud_category[:,1],point_cloud_category[:,0]] = self.category_console_table
            else:
                self.global_map[point_cloud_category[:,1],point_cloud_category[:,0]] = self.category

            #semantic categories 
            if ob in self.task.ob_cats:
                ind = self.task.cats_to_ind[ob]#np.argwhere(self.task.ob_cats == ob)
                #if ind in self.task.uniq_indices and self.task.wanted_objects[ind] == 0:
                if self.task.wanted_objects[ind] == 0:
                    #pass
                    self.global_map[point_cloud_category[:,1],point_cloud_category[:,0]] = self.category_found
                else:
                    self.global_map[point_cloud_category[:,1],point_cloud_category[:,0]] = self.categorie_array[ind]

                

               
                
    
        

    def step(self, action):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        #self.check_target_changed()
        """
        eligible_for_gt = False
        if len(self.coll_track) > 125:# and self.reset_agent_pos:
            self.coll_track.pop(0)
            if np.mean(np.array(self.coll_track)) > 0.85:# and len(self.rob_track) > 5:
                #n_p = np.random.choice(np.arange(len(self.rob_track)-3))
                #n_p = self.rob_track[n_p]
                #self.robots[0].set_position([n_p[0],n_p[0],0.0])
                #self.coll_track = []
                #print("Hes Eligile")
                eligible_for_gt = True
                #print("NOW ELIGIBLE")
            #else:
            #    print("hes not eligible")

        """
        
        #if action is not None:
            #self.robots[0].apply_action(action)

            #if(self.aux_task):
        
        self.robots[0].apply_action(action['action'])#<<<<<<<<<<<<<<
        
        #self.robots[0].apply_action(action)#--- Without auxilliary

        self.current_step += 1
        collision_links = self.run_simulation()
        
        self.collision_links = collision_links
        c = int(len(collision_links) > 0)
        #print(c)
        #if len(collision_links) > 0:
        #    print("got a collison")
            #print("COLS:",collision_links)
            #print("collision:?",len(collision_links) > 0)
        self.collision_step += c

        
        self.coll_track.append(c)
        self.coll_track.pop(0)
        #print("added:",c)
        
        if len(self.prev_locations) >= self.history_length_aux:
            #print("popped")
            
            self.aux_angle_track.pop(0)#<<<<<<<<<<<<<<<<
            self.prev_locations.pop(0)
        

        
        state = self.get_state(collision_links)
        
        

        #-------------------------------------------------------------------------------------------
        
        
        
        self.sim_rob_position = self.robots[0].robot_body.get_position()
        

        self.camera = self.robots[0].parts['eyes'].get_pose()
        camera_translation = self.camera[:3]
        #print("offset:",self.sim_rob_position[0] - camera_translation[0],self.sim_rob_position[1]- camera_translation[1], self.sim_rob_position[2]- camera_translation[2])
        
        #depth_map = state['depth'].copy()
        self.seg_mask = (state['seg']*255).astype(int)
        
        #print("cam trans:",camera_translation)

        depth_map = (state['depth'] * (self.depth_high - self.depth_low) + self.depth_low)
        depth_map[state['depth'] > self.depth_high] = 0.0
        depth_map[state['depth'] < self.depth_low] = 0.0
        #masks = [self.seg_mask != 1275, self.seg_mask != 1020, self.seg_mask != 20400]
            
        #total_mask = reduce(np.logical_and, masks)
        #depth_map[total_mask] = 0.0

        camera_angles = p.getEulerFromQuaternion(self.camera[3:])
        #print("cam angles:",camera_angles)
        #change angle sign as it causes an error in the camera_frame -> world-frame
        euler_mat = euler2mat(camera_angles[0],-camera_angles[1],camera_angles[2])

        point_cloud = self.pointcloud(depth_map)#.reshape(-1,3)
        
        
        w = (self.seg_mask == 1020).squeeze()
        point_cloud_walls = point_cloud[w,:]

        point_cloud_walls = euler_mat.dot(point_cloud_walls.T).T + camera_translation
        point_cloud_walls = self.world2map(point_cloud_walls).astype(np.uint16)
        
        
        f = (self.seg_mask == 1275).squeeze()
        point_cloud_floor = point_cloud[f,:]

        point_cloud_floor = euler_mat.dot(point_cloud_floor.T).T + camera_translation
        #print("floor1",point_cloud_floor.shape)
        point_cloud_floor = self.world2map(point_cloud_floor).astype(np.uint16)
        #print("floor2",point_cloud_floor.shape)

        path_indices = self.global_map[...,2] == self.trace[2]
        
        try:
            self.global_map[point_cloud_floor[:,1],point_cloud_floor[:,0]] = self.obstalce  
            self.global_map[point_cloud_walls[:,1],point_cloud_walls[:,0]] = self.floor
            self.global_map[path_indices] = self.trace
                 
            
            
            objects = (np.unique(self.seg_mask)).astype(int)
            #Improve Performance: run if statements through ob in objects
            #print("current objects: ",objects)
            
            self.draw_object_categories(objects,point_cloud,euler_mat,camera_translation)
            

        except Exception as e:
            print("An exception occurred in self.globalMap line 638",e)
            print("SCENE:",self.last_scene_id)
            print("POS:",(self.camera[0],self.camera[1],camera_angles[2]))
            input()#causes an exception for multi env such that the whole process ends
        

        curr_sim_pose = self.camera[0],self.camera[1],camera_angles[2]
        #if self.current_step % 75 == 0 and self.reset_agent_pos:
        #    self.rob_track.append(curr_sim_pose)
        #print("pos:",curr_sim_pose)
        self.rob_pose = self.world2map(np.array([curr_sim_pose[0],curr_sim_pose[1]]))
        self.prev_locations.append(self.robots[0].robot_body.get_position()[:2])
        #print(f"rob pose:{self.rob_pose}")
        """
        if self.eval_heat_map:
            
            ind_floor = self.global_map[:,:,0] == self.obstalce[0]
            ind_walls = self.global_map[:,:,1] == self.floor[1]

            self.global_heat_map[ind_floor] = self.obstalce
            self.global_heat_map[ind_walls] = self.floor
            
            self.global_heat_map[int(self.rob_pose[1])-2:int(self.rob_pose[1])+2,int(self.rob_pose[0])-2:int(self.rob_pose[0])+2,2] += 5
            indices_too_big = self.global_heat_map[:,:,2] > 254.0
            self.global_heat_map[indices_too_big,2] = 254.0

            heat_indices = self.global_heat_map[:,:,2] > 100
            self.current_heat_score = self.global_heat_map[heat_indices,2].sum() / (self.map_size[0]**2)
        """
        self.global_map[int(self.rob_pose[1])-2:int(self.rob_pose[1])+2,int(self.rob_pose[0])-2:int(self.rob_pose[0])+2] = self.trace
        
        
        ego1,ego2 = self.affine4map(self.robots[0].get_rpy()[2],curr_sim_pose[2],action,euler_mat,camera_translation)#
        
       


        info = {}
        #t0 = time.time()
        reward, info = self.task.get_reward(self, collision_links, action, info)
        if reward > 9.0:
            ego1,ego2 = self.affine4map(self.robots[0].get_rpy()[2],curr_sim_pose[2],action,euler_mat,camera_translation)
        done, info = self.task.get_termination(self, collision_links, action, info)
        #task.step performs visualization, path length addition of last and new position and assigns new rob pos.
        self.task.step(self) 
        self.populate_info(info)



        task_o = state['task_obs'].copy()
        
        linear_vel, angular_vel = task_o[2:]
        if task_o[1] < 0:
            task_o[1] += 2*np.pi
        #print("ORIGINAL:",task_o[1])
        gt_bin = np.digitize([np.rad2deg(task_o[1])],self.angle_to_bins,right=True) 
        #print("GT bin:",gt_bin)
        gt_bin_vec = np.zeros(self.aux_bin_number)+np.random.uniform(0,0.1)

        gt_bin_vec[gt_bin] += 1.0
        gt_bin_vec = softmax(gt_bin_vec)
        #print("GT soft:",gt_bin_vec)
        #y = np.exp(gt_bin_vec - np.max(gt_bin_vec))
        #gt_final_vec =  y / np.sum(np.exp(gt_bin_vec))
        #print("GT vec:",gt_bin_vec, "SUM:",np.sum(gt_bin_vec))

        #task_o[1] = 
        #rgb = state['rgb'].copy()
        state = {}
        
        state['image'] = ego1.transpose(2,0,1)#self.policy_input.transpose(2,0,1)[:,self.x_off1:self.x_off2,self.y_off1:self.y_off2]#self.policy_input_history ##
        

        #self.image_local = state['image'].copy()
        
        state['image_global'] = ego2.transpose(2,0,1)#cv2.resize(self.policy_input[self.x_off11:self.x_off22,self.y_off11:self.y_off22,:], (self.downsample_size,self.downsample_size),interpolation=cv2.INTER_NEAREST).transpose(2,0,1)#self.policy_input_history_global###.transpose(1,2,0)
        
        
        

        
        #task obs has rho and phi, [0] -> rho and [1]-> phi which is the radius and distance
        #if(self.aux_task):
            #info['true_aux'] = state['task_obs'][:2].copy()#np.linalg.inv(euler_mat).dot(self.true_aux - camera_translation)#
        
        if self.episodic_aux_prediction:
            self.aux_angle_track.append(np.argmax(self.aux_action))#<<<<<<<<<<<<<<<<
        else:
            self.aux_angle_track.append(task_o[1])
        
        var0 = np.var(np.array(self.prev_locations)[:,0])
        var1 = np.var(np.array(self.prev_locations)[:,1])
        
        #if(self.aux_task):
        state['task_obs'] = np.concatenate([gt_bin_vec,np.array(self.prev_aux_predictions).flatten(),[linear_vel,angular_vel,circvar(self.aux_angle_track),var0,var1],np.array([int(len(collision_links) > 0)]),np.array([np.array(self.coll_track).sum()]),action['action'],self.task.wanted_objects])#,[self.task.num_cat_in_episode]])#<<<<<<<<<<<<<<<<
        
        #state['task_obs'][0] = 0.0
        #print(self.coll_track)
        #print("col flag :",np.array([int(len(collision_links) > 0)]), "col sum:",np.array([np.array(self.coll_track).sum()]), " coll mean:",np.maximum(0,np.mean([np.array(self.coll_track)])))
        #prev_aux_predictions has already [0.0]
        self.prev_aux_predictions.pop(0)
        self.prev_aux_predictions.append(self.aux_action)
        #state['task_obs'] = np.concatenate([task_o[2::],np.array([int(len(collision_links) > 0)]),np.array([np.array(self.coll_track).sum()]),action,self.task.wanted_objects])#--- Without auxilliary
        
        #state['task_obs'] = np.concatenate([task_o,np.array([int(len(collision_links) > 0)]),np.array([np.array(self.coll_track).sum()]),action['action'],self.task.wanted_objects])#,[self.task.num_cat_in_episode]])
        #state['goal_obs'] = self.task.wanted_objects
            
        #Ground truth information for PPO Auxilliary Lossfdd

        info['true_aux'] = gt_bin[0]#task_o[:2].copy()#<<<<<<<<<<<<<<<<

        #print("ground truth:",info['true_aux'], " GT state:",state['task_obs'][:2])

        #info['true_aux'][0] = 0.0
        #for visualiazing map
        #tmp_saved = task_o[:2].copy() 
        #set poar distance to zero in order to let the network ignore this observation input
        #state['task_obs'][0] = 0.0
                       

        #if((self.substitute_polar or np.random.uniform() < 0.6) and not eligible_for_gt):
            
            
        
        #print("argmax:",np.argmax(self.aux_action))
        if self.episodic_aux_prediction: #or self.eval:
            #print("before",state['task_obs'])
            state['task_obs'][:self.aux_bin_number] = self.aux_action
            #print("after",state['task_obs'])

            #state['task_obs'][0] = self.aux_dist #action['aux_dist'][0]#<<<<<<<<<<<<<<<<
            #state['task_obs'][1] = np.argmax(self.aux_action) #np.arctan2(action['aux_angle'][0],action['aux_angle'][1])  #action['aux'][0]*np.pi#<<<<<<<<<<<<<<<<
        #indicates that this sample is from the aux prediction and is not considered in the axu loss (simply set to zero)
        #info['true_aux'][0] = 20.0
                
            
        #add gaussian noise on the angle
        #if np.random.uniform() < 0.8:
        #    state['task_obs'][1] += np.random.normal(0,0.2,1)[0] # <- has been relatively small since a variance of 0.2 is not that big

            
        #else:
        #remove polar coordinates   
        #state['task_obs'] = np.concatenate([[task_o[2],task_o[3]],np.array([int(len(collision_links) > 0)]),action,self.task.wanted_objects])

            
            
    
        #if done and self.automatic_reset:
        #    info["last_observation"] = state
        #    state = self.reset()
        #print("map value:",ego1[42,42])
        
        
        if self.show_map:
            cv2.imshow("coarse",state['image'].transpose(1,2,0).astype(np.uint8))
            cv2.waitKey(1)

            cv2.imshow("global",state['image_global'].transpose(1,2,0).astype(np.uint8))
            cv2.waitKey(1)
            
            
            tmp = self.global_map.copy()
            
    
            w_ax = self.world2map(euler_mat.dot(self.aux_points[np.argmax(self.aux_action)].T).T + camera_translation)
            
            tmp[int(w_ax[1])-5:int(w_ax[1])+5,int(w_ax[0])-5:int(w_ax[0])+5] = self.aux_pred
            
               
                

                
                
                
            cv2.imshow("GLOBAL NO POLICY INPUT",tmp)
            cv2.waitKey(1)
            
        
        #if np.random.uniform() < 0.05:
        #cv2.imwrite('data/vid{}/fine_{}_{}'.format(self.current_episode,self.current_step,'.png'),state['image'].transpose(1,2,0).astype(np.uint8))
        #cv2.imwrite('data/vid{}/coarse_{}_{}'.format(self.current_episode,self.current_step,'.png'),state['image_global'].transpose(1,2,0).astype(np.uint8))
        #cv2.imwrite('data/vid{}/rgb_{}_{}'.format(self.current_episode,self.current_step,'.png'),cv2.cvtColor(rgb*255,cv2.COLOR_RGB2BGR))
        
        
        #cv2.imshow("RGB",cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        #cv2.waitKey(1)
        #    cv2.imwrite('data/test/pic_{}_{}_{}_{}'.format(self.current_step,self.task.wanted_objects,self.current_episode,'.png'),state['image_global'].transpose(1,2,0).astype(np.uint8))
        """
        if done:
            img_cpy = self.global_map.copy()
            open_ind = np.argwhere(self.task.wanted_objects == 1)
            for i in open_ind:
                print("ind:",i)
                tr_ps = self.task.target_pos_list[int(i)]
                tr_ps = self.world2map(tr_ps)
                img_cpy[int(tr_ps[1])-4:int(tr_ps[1])+4,int(tr_ps[0])-4:int(tr_ps[0])+4] = np.array([5, 128, 128])

            img_cpy[int(self.rob_pose[1])-4:int(self.rob_pose[1])+4,int(self.rob_pose[0])-4:int(self.rob_pose[0])+4] = self.arrow
            cv2.imwrite('data/{}/pic_HSR_{}_{}_{}_{}'.format(self.last_scene_id,str(info['success']),self.task.wanted_objects,self.current_episode,'.png'),img_cpy)
            #cv2.imwrite('data/{}/pic_heat_HSR_{}_{}_{}_{}'.format(self.last_scene_id,str(info['success']),self.task.wanted_objects,self.current_episode,'.png'),self.global_heat_map)
        """
        #Resampling task
        if done: #and self.episodic_aux_prediction:
            self.SR_rate.append(info['success'])
            

        if(self.episode_counter > 25 and done and not info['success'] and self.resample_task):# IT WAS 25 BEFORE RETRAIN
        #    self.task.create_target_position_heat_map(self)

            self.queue_for_task_resampling.append((self.task.initial_pos,self.task.initial_orn,self.task.target_pos_list,self.task.initial_wanted,self.last_scene_id))
            #self.task.queue.append((self.task.initial_pos,self.task.initial_orn,self.task.target_pos_list,self.task.initial_wanted,self.last_scene_id))
        
        #print("cats:",state['task_obs'][-6::])
        #print("HEHE:",state['task_obs'].shape)
        return state, reward, done, info
    
    def pol2cart(self,rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)

    def check_collision(self, body_id):
        """
        Check with the given body_id has any collision after one simulator step

        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        self.simulator_step()
        collisions = list(p.getContactPoints(bodyA=body_id))

        if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                logging.debug("bodyA:{}, bodyB:{}, linkA:{}, linkB:{}".format(item[1], item[2], item[3], item[4]))

        return len(collisions) == 0

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        is_robot = isinstance(obj, BaseRobot)
        body_id = obj.robot_ids[0] if is_robot else obj.get_body_id()
        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), "wxyz"))
        # compute stable z based on this orientation
        stable_z = stable_z_on_aabb(body_id, [pos, pos])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def test_valid_position(self, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.get_body_id()
        has_collision = self.check_collision(body_id)
        return has_collision

    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.get_body_id()

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_episode += 1
        self.episode_counter += 1
        self.current_step = 0
        self.collision_step = 0
        self.collision_links = []

    def randomize_domain(self):
        """
        Domain randomization
        Object randomization loads new object models with the same poses
        Texture randomization loads new materials and textures for the same object models
        """
        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

    def reset(self):
        """
        Reset episode
        """
        

        if len(self.SR_rate) > 32:
            self.SR_rate.pop(0)

        self.episodic_aux_prediction = np.random.uniform() < self.aux_episodic_prob
        print("is aux:",self.episodic_aux_prediction)
        if self.aux_prob_counter > 4 and self.episode_counter > 25 and np.mean(np.array(self.SR_rate)) > 0.5 and not self.evaluate:
            self.aux_episodic_prob = min(0.72,self.aux_episodic_prob+0.02)
            self.aux_prob_counter = 0
            print("------------------>>>>>>>> NEW EPISODIC PROB: ",self.aux_episodic_prob, " for env : ",self.last_scene_id)#, " with env succ-rate: ",succ_rate_mean)

        
        if self.scene_reset_counter > 4 and self.multiple_envs and self.mix_sample:
            scene_id = np.random.choice(['Rs_int','Wainscott_1_int','Benevolence_0_int','Beechwood_1_int'])
            
            #succ_rate_mean = np.mean(np.array(self.aux_sr_rate))
            #if succ_rate_mean >= self.prev_aux_sr_rate:
            #self.aux_episodic_prob = min(0.75,self.aux_episodic_prob+0.01)
            #print("------------------>>>>>> NEW EPISODIC PROB: ",self.aux_episodic_prob)#, " with env succ-rate: ",succ_rate_mean)
            #self.prev_aux_sr_rate = succ_rate_mean

            if self.last_scene_id != scene_id:
                print(f"SELECTED NEW SCENE ID FROM {self.last_scene_id} to {scene_id}")
                self.reload_model(scene_id)
                self.scene_reset_counter = 0
                self.last_scene_id = scene_id
        
        self.grid_offset = self.map_settings[self.last_scene_id]['grid_offset']
        self.grid_spacing = self.map_settings[self.last_scene_id]['grid_spacing']
        self.offset = self.map_settings[self.last_scene_id]['offset']
        self.task.object_distance = self.map_settings[self.last_scene_id]['object_dist']
        self.scene_reset_counter += 1
        self.aux_prob_counter += 1
        #self.rob_track = []
        self.coll_track = [0.0]*self.history_length_aux
        #self.agent_pos = []
        self.aux_angle_track = []
        self.randomize_domain()
        # move robot away from the scene
        self.robots[0].set_position([100.0, 100.0, 100.0])
        self.task.reset_scene(self)
        self.prev_locations = []
        #12 is the number
        self.prev_aux_predictions = [np.zeros(self.aux_bin_number)]*self.history_length_aux
        #reset_agent assumes target positions to be already sampled
        self.task.reset_agent(self)
        
        self.simulator.sync()
        state = self.get_state()
        self.reset_variables()

        #if self.eval_heat_map:
        #    self.global_heat_map = np.zeros((self.map_size[0],self.map_size[1], 3), dtype=np.uint8) * 255

        """
        
        if(len(self.episode_stats['collision']) > 30):
            self.episode_stats['collision'].pop(0)
            self.episode_stats['succ_rate'].pop(0)
        """
        self.global_map = np.zeros((self.map_size[0],self.map_size[1], 3), dtype=np.uint8) * 255#np.zeros((self.map_size[0],self.map_size[1],3))
        
        #load groundth truth map
        if(self.ground_truth_map):
            img = Image.open("map.png")
            img.load()
            self.global_map = np.asarray(img, dtype="uint8" )
        
        self.seg_mask = (state['seg']*255).astype(int)

        self.camera = self.robots[0].parts['eyes'].get_pose()
        camera_translation = self.camera[:3]
        depth_map = state['depth'].copy()
        depth_map = (depth_map * (self.depth_high - self.depth_low) + self.depth_low)
        depth_map[depth_map >=self.depth_high] = 0.0
        depth_map[depth_map <= self.depth_low] = 0.0

        #masks = [self.seg_mask != 1275, self.seg_mask != 1020, self.seg_mask != 20400]
        #total_mask = reduce(np.logical_and, masks)
        #depth_map[total_mask] = 0.0

        camera_angles = p.getEulerFromQuaternion(self.camera[3:])
        euler_mat = euler2mat(camera_angles[0],-camera_angles[1],camera_angles[2])

        point_cloud_t = self.pointcloud(depth_map)#.reshape(-1,3)#self.test_point(depth_map).reshape(3,420,420)#
        
        
        m1 = (self.seg_mask == 1020).squeeze()
        
        point_cloud_floor = point_cloud_t[m1,:]
        indices = np.argwhere(point_cloud_floor[:,0] == 0.0)
        point_cloud_floor = np.delete(point_cloud_floor,indices[:,0],axis=0)
        point_cloud_floor_trans = euler_mat.dot(point_cloud_floor.T).T + camera_translation
        point_cloud_floor_world = self.world2map(point_cloud_floor_trans).astype(np.uint16)
        #first channel for floor and obstacles
        try:
            

            #OPENCV USES BGR order instead of RGB
            self.global_map[point_cloud_floor_world[:,1],point_cloud_floor_world[:,0]] = self.floor
            #trajectory path
            #self.global_map[path_indices,1] = 0.1
            point_cloud_t[m1,:] = 0.0
            point_cloud_t2 = point_cloud_t.reshape(-1,3)
       
            indices = np.argwhere(point_cloud_t2[:,0] == 0.0)
            point_cloud_t2 = np.delete(point_cloud_t2,indices[:,0],axis=0)
        
            _trans = euler_mat.dot(point_cloud_t2.T).T + camera_translation
       
            _world = self.world2map(_trans).astype(np.uint16)
        
          
            self.global_map[_world[:,1],_world[:,0]] = self.obstalce
        except:
            print("global map drawing in reset")


        curr_sim_pose = self.camera[0],self.camera[1],camera_angles[2]#ut.get_sim_location(self)
        #self.rob_track.append(curr_sim_pose)
        #self.agent_pos.append(curr_sim_pose)
        self.rob_pose = self.world2map(np.array([curr_sim_pose[0],curr_sim_pose[1]]))
        #if self.eval_heat_map:
        #    self.global_heat_map[int(self.rob_pose[1])-2:int(self.rob_pose[1])+2,int(self.rob_pose[0])-2:int(self.rob_pose[0])+2,2] += 1
        
        self.global_map[int(self.rob_pose[1])-2:int(self.rob_pose[1])+2,int(self.rob_pose[0])-2:int(self.rob_pose[0])+2] = self.trace
        
        
        ego1,ego2 = self.affine4map(self.robots[0].get_rpy()[2],curr_sim_pose[2])
        #just for test purposes
        #self.prevv = self.policy_input
        #
        task_o = state['task_obs']
        #if(not self.use_rgb_depth):
        state = {}
        #else:
        
        
        #state = {}
        
        
        state['image'] = ego1.transpose(2,0,1)#[:,self.x_off1:self.x_off2,self.y_off1:self.y_off2]#self.policy_input_history
        state['image_global'] = ego2.transpose(2,0,1)#cv2.resize(self.policy_input[self.x_off11:self.x_off22,self.y_off11:self.y_off22,:], (self.downsample_size,self.downsample_size),interpolation=cv2.INTER_NEAREST).transpose(2,0,1)#self.policy_input_history_global#.transpose(1,2,0)#
        #state['goal_obs'] = self.task.wanted_objects
        gt_bin_vec = np.zeros(self.aux_bin_number)+(1/self.aux_bin_number)
        #<<<<<<<<<<<<<<<
        state['task_obs'] = np.concatenate([gt_bin_vec,np.array(self.prev_aux_predictions).flatten(),[task_o[2],task_o[3],0.0,0.0,0.0],np.array([0.0]),np.array([0.0]),np.array([0.0,0.0]),self.task.wanted_objects])#<<<<<<<<<<<<<<<<
        
        #if self.episodic_aux_prediction:
        #print("SHAPEZ:",state['task_obs'].shape)
        #state['task_obs'][0] = 0.0
        #state['task_obs'][1] = 0.0
        #print(state['task_obs'].shape)
        #state['task_obs'] = np.concatenate([task_o[2::],np.array([0.0]),np.array([0.0]),np.array([0.0,0.0]),self.task.wanted_objects]) #--- Without auxilliary
     
       
        return state


