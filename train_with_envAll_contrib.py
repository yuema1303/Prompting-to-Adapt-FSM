import sys
import os
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import gymnasium 
import pdb
from stable_baselines3 import A2C, PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from controller import Controller
from augement_policy import Policy
from config import get_args
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from FT_utils import img_loader, forward_point_cv2, eval_gt_pred, update_record
from PromptToAdaptAll import Prompt2AdaptAll
from datetime import datetime
from logger import Logger
 





def main(args):
    
    '''
    filename = "log0819"
    log_file_name = f"{filename}.log"
    sys.stdout = Logger(str(log_file_name))
    sys.stderr = Logger(str(log_file_name))
    '''
    
    model_name = "RecurrentPPO_P2A_All_" + "lr_" + str(args.sb3_lr) + "_timesteps_" + str(args.total_timesteps)
    #log_file_name = f"{model_name}.log"
    #sys.stdout = Logger(str(log_file_name))
    #sys.stderr = Logger(str(log_file_name))
    
    sb3_log_path = "./sb3_log/" + model_name + "/"
    
    sb3_logger = configure(sb3_log_path, ["stdout", "csv", "tensorboard"])
    
    log_file_name = f"{model_name}.log"
    sys.stdout = Logger(str(log_file_name))
    sys.stderr = Logger(str(log_file_name))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    # SAM
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device)
    predictor = SamPredictor(sam) # predictor is set up

    # Controller
    #controller = Controller(args).to(device)
    #controller_optimizer = torch.optim.SGD(controller.parameters(), args.controller_lr, momentum=0.9)
    
    # load_controller
    #save_dict = "./save_dir/1203.pt.tar"
    #checkpoints = torch.load(save_dict)
    #controller.load_state_dict(checkpoints['controller_state'])

    #baseline = None
    #best_mIoU_record = []
    #best_p_dict = []
    #best_iter_record = []
    len_save = args.save_policy_len
    #best_p_val_dict = [0 for i in range(len_save)]
    
    transform = transforms.Compose([transforms.Resize((1024,1024), interpolation = transforms.InterpolationMode.BICUBIC, antialias = True)])
    
    # load dataset
    img_dir = args.img_dir
    label_dir = args.label_dir
    train_txt = args.train_txt
    val_txt = args.val_txt
    
    mode = "train"
    
    if mode == 'train':
        ann_file = open(train_txt, "r")
    elif mode == 'val':
        ann_file = open(val_txt, "r")
    else:
        ann_file = open(val_txt, "r")
    content = ann_file.read()
    imgs_list = content.splitlines()

    #pdb.set_trace()
    img_stack = np.zeros((10,1024,1024,3))
    gt_mask_stack = np.zeros((10,1024,1024))

    for i in range(len(imgs_list)):
        
        img_name = imgs_list[i].split(".")[0]
        img_path = os.path.join(img_dir, img_name + ".jpg")
        label_path = os.path.join(label_dir, img_name + ".jpg")
        #label = img_loader(label_path, "L")
        
        #gt_masks_tensor = torch.from_numpy(np.array(label)).float() / 255.0
        #gt_masks_tensor = gt_masks_tensor.unsqueeze(0).to(device)

        img = cv2.imread(img_path)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)
        img_resize = transform(img)
        img_stack[i] = img_resize.numpy().transpose(1,2,0)
        
        gt_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = torch.from_numpy(gt_mask)
        gt_mask_resize = transform(gt_mask.unsqueeze(0))
        gt_mask_stack[i] = gt_mask_resize.squeeze().numpy()
    
    # Till now, prepare predictor, training_set stack, (Validation_set stack), gt_mask stack should be prepared and env set up for just once
    
    
    env = Prompt2AdaptAll(ori_images_stack = img_stack.astype(np.uint8), 
                         gt_mask_stack = gt_mask_stack.astype(np.uint8), 
                         predictor = predictor, 
                         device = device, 
                         mode = "All",
                         args = args)
    
    #pdb.set_trace()

    
    #observation, info = env.reset(seed=42)
    
    #action = env.action_space.sample()
    
    #observation, reward, terminated, truncated, info = env.step(action)
    
    check_env(env)

    model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, n_steps = args.n_steps, learning_rate = args.sb3_lr)
    model.set_logger(sb3_logger)
    model.learn(total_timesteps=args.total_timesteps)
    vec_env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
    print(mean_reward)
    #model_name = "PPO_P2A_" + "lr_" + str(args.sb3_lr) + "_timesteps_" + str(args.total_timesteps)
    model.save(model_name)

    del model # remove to demonstrate saving and loading

    model = RecurrentPPO.load(model_name)

    obs, info = env.reset()
    
    rewards_list = []
    info_list = []
    for _ in range(100):
        obs, info = env.reset()
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        rewards_list.append(rewards)
        info_list.append(info)
    
    
    print("mean mIoU = ", np.mean(rewards))
    pdb.set_trace()
    
    ###################################
    
if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main(args)