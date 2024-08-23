import numpy as np
import cv2
from augement_policy import Policy, get_sub_policies
from FT_utils import img_loader, forward_point_cv2, eval_gt_pred, update_record
import torch
import gymnasium as gym
from gymnasium import spaces
import random
import pdb

class Prompt2Adapt(gym.Env):
    
    def __init__(self, ori_images_stack, gt_mask_stack, predictor, device, args):
        self.args = args
        self.observation_space = spaces.Box(0, 255, shape=(1024, 1024, 3), dtype=np.uint8)  #TODO: What is your observation space
        self.action_space = spaces.MultiDiscrete([16,10,16,10,16,10])
        self._image_stack = ori_images_stack #should be resized numpy image stack BGR 10 * 1024 * 1024 * 3  0-255
        self._gt_mask_stack = gt_mask_stack #should be resized numpy gt mask stack BGR 10 * 1024 * 1024  0-255
        self._random_int = np.random.randint(0, self._image_stack.shape[0])
        self._adapted_image = self._image_stack[self._random_int]
        self._predictor = predictor
        self._device = device
        self._policy_dict = []
        self._policy_dict_info = {}
        self._policy_dict_wrapper = ["step1", "step2", "step3"]
        #self.transform = transforms.Compose([transforms.Resize((1024,1024), interpolation = transforms.InterpolationMode.BICUBIC)])
        
    def _get_obs(self):
        self._adapted_image = self._image_stack[self._random_int]
        return self._adapted_image
    
    def _get_info(self):
        return self._policy_dict_info
    
    def reset(self, seed=None, options = None):
        # reset the seed self.np_random
        super().reset(seed=seed)
        
        self._random_int = np.random.randint(0, self._image_stack.shape[0])
        self._adapted_image = self._image_stack[self._random_int] # TODO reset adapted image to origin image
        self._policy_dict = []
        self._policy_dict_info = {}
        
        observation = self._get_obs()
        info = self._get_info()
        terminated = False
        
        return observation, info
    
    def step(self, action):
        
        #self._random_int = np.random.randint(0, self._image_stack.shape[0])
        policy_id_list = []
        magnitude_id_list = []
        #op_list = []
        #magnitude_list = []
        op_magnitude_list = []
        
        for i in range(self.args.subpolicy_num):
            policy_id_list.append(action[2 * i])
            magnitude_id_list.append(action[2 * i + 1])
        
        self._policy_dict = get_sub_policies(policy_id_list, magnitude_id_list, self.args)
        
        for i in range(self.args.subpolicy_num):
            op = self._policy_dict[i][0]['op']
            #op_list.append(op)
            magnitude = self._policy_dict[i][0]['magnitude']
            #magnitude_list.append(magnitude)
            op_magnitude_list.append(str(op) + " : " + str(magnitude))
            
        #if((len(op_list) != 3) | (len(magnitude_list) != 3)):
            #pdb.set_trace()
            
        self._policy_dict_info = dict(zip(self._policy_dict_wrapper, op_magnitude_list))
        
        self._adapted_image = self._image_stack[self._random_int]
        
        self._adapted_image = Policy(self.args, self._adapted_image, self._policy_dict)
        self._adapted_image = cv2.cvtColor(self._adapted_image, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(self._adapted_image)
        gt_mask = self._gt_mask_stack[self._random_int]
        input_point = forward_point_cv2(gt_mask)
        input_label = np.array([1])
        
        pred_mask, _, _ = self._predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=None,
        multimask_output=False,
        )
            
        pred_masks_tensor = torch.from_numpy(pred_mask).to(self._device).float()
        gt_masks_tensor = torch.from_numpy(gt_mask).float() / 255.0
        gt_masks_tensor = gt_masks_tensor.unsqueeze(0).to(self._device)
            
        IoU_cost = eval_gt_pred(gt_masks_tensor, pred_masks_tensor)
        
        terminated = True
        
        reward = IoU_cost
        
        reward = reward.cpu().numpy().item()
        
        #pdb.set_trace()
        
        observation = self._get_obs() #should be 10 adaptedd images
        info = self._get_info() #should be 3 policies with magnitude

        return observation, reward, terminated, False, info