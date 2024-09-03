import numpy as np
import cv2
from augement_policy import Policy, get_sub_policies
from FT_utils import img_loader, forward_point_cv2, eval_gt_pred, update_record_env
import torch
import gymnasium as gym
from gymnasium import spaces
import random
import pdb

class Prompt2Adapt(gym.Env):
    
    def __init__(self, ori_images_stack, gt_mask_stack, predictor, device, mode, args):
        self.args = args
        self.observation_space = spaces.Box(0, 255, shape=(10240, 1024, 3), dtype=np.uint8)  #TODO: What is your observation space, 10 1024*1024*3 images concatenate together
        self.action_space = spaces.MultiDiscrete([16,10,16,10,16,10])
        self._image_stack = ori_images_stack #should be resized numpy image stack BGR 10 * 1024 * 1024 * 3  0-255
        self._gt_mask_stack = gt_mask_stack #should be resized numpy gt mask stack BGR 10 * 1024 * 1024  0-255
        self._random_int = np.random.randint(0, self._image_stack.shape[0])
        self._adapted_image = self._image_stack.reshape(10*1024, 1024, 3)
        #self._original_image = self._image_stack.reshape(10*1024, 1024, 3)
        self._predictor = predictor
        self._device = device
        self._mode = mode
        self._policy_dict = []
        self._policy_dict_info = {}
        self._policy_dict_wrapper = ["step1", "step2", "step3"]
        self._best_policy_list = []
        self._best_mIoU_list = []
        
        #self.transform = transforms.Compose([transforms.Resize((1024,1024), interpolation = transforms.InterpolationMode.BICUBIC)])
        
    def _get_obs(self):
        #self._adapted_image = self._image_stack[self._random_int]
        return self._adapted_image
    
    def _get_info(self):
        return self._policy_dict_info
    
    def reset(self, seed=None, options = None):
        # reset the seed self.np_random
        super().reset(seed=seed)
        self._policy_dict = []
        self._policy_dict_info = {}
        
        self._adapted_image = self._image_stack.reshape(10*1024, 1024, 3)
        
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
        
        mIoU_list = []
        image_list = []

        for i in range(self._image_stack.shape[0]):
            image = self._image_stack[i]

            image = Policy(self.args, image, self._policy_dict)
            image_for_predictor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._predictor.set_image(image_for_predictor)
            
            image_list.append(image)
            
            gt_mask = self._gt_mask_stack[i]
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
            mIoU_list.append(IoU_cost.cpu().numpy().item())


        terminated = True

        reward = np.mean(mIoU_list)
        
        self._adapted_image = np.concatenate(image_list, axis = 0)
        '''
        
        if(self._mode == "All"):
            
            mIoU_list = []

            for i in range(self._image_stack.shape[0]):
                self._adapted_image = self._image_stack[i]

                self._adapted_image = Policy(self.args, self._adapted_image, self._policy_dict)
                self._adapted_image = cv2.cvtColor(self._adapted_image, cv2.COLOR_BGR2RGB)
                self._predictor.set_image(self._adapted_image)
                gt_mask = self._gt_mask_stack[i]
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
                mIoU_list.append(IoU_cost.cpu().numpy().item())


            terminated = True

            reward = np.mean(mIoU_list)
            
        elif(self._mode == "Single"):
            self._adapted_image = self._image_stack[self._random_int]
            #self._policy_dict = [{0: {'op': 'contrast_down', 'magnitude': 8}}, {0: {'op': 'saturation_down', 'magnitude': 8}}, {0: {'op': 'gaussianBlur', 'magnitude': 2}}]
            #just for test
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
        else:
            raise NameError("PleaseSetCorrectModeName")
        
        '''
           
        if(((len(self._best_mIoU_list) == self.args.save_policy_len) and (reward < self._best_mIoU_list[-1])) or ( reward in self._best_mIoU_list) or ( self._policy_dict in self._best_policy_list)):
            pass
        else:
            self._best_mIoU_list.append(reward)
            self._best_policy_list.append(self._policy_dict)
            self._best_mIoU_list, self._best_policy_list = update_record_env(self._best_mIoU_list, self._best_policy_list, self.args.save_policy_len)
        
        observation = self._get_obs() #should be 1 adaptedd images
        info = self._get_info() #should be 3 policies with magnitude

        return observation, reward, terminated, False, info