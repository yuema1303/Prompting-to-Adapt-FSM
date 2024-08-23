import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from controller import Controller
from augement_policy import Policy
from config import get_args
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from FT_utils import img_loader, forward_point_cv2, eval_gt_pred, update_record
import cv2

def val_SAM(args, predictor, policy_dict, device, mode='train'):
    sum_IoU_cost = 0
    sum_imgs = 0
    
    # load dataset
    img_dir = args.img_dir
    label_dir = args.label_dir
    train_txt = args.train_txt
    val_txt = args.val_txt
    
    if mode == 'train':
        ann_file = open(train_txt, "r")
    elif mode == 'val':
        ann_file = open(val_txt, "r")
    else:
        ann_file = open(val_txt, "r")
    content = ann_file.read()
    imgs_list = content.splitlines()

    for i in range(len(imgs_list)):
        if i % 25 == 0 and i > 0:
            val_iou_temp = sum_IoU_cost / sum_imgs
            print('\rprocessing: {}/{}, val_iou: {}'.format(
                        i, len(imgs_list), val_iou_temp))
        sum_imgs += 1
        
        img_name = imgs_list[i].split(".")[0]
        img_path = os.path.join(img_dir, img_name + ".jpg")
        label_path = os.path.join(label_dir, img_name + ".jpg")
        label = img_loader(label_path, "L")
        gt_masks_tensor = torch.from_numpy(np.array(label)).float() / 255.0
        gt_masks_tensor = gt_masks_tensor.unsqueeze(0).to(device)

        img = cv2.imread(img_path)
        img = Policy(args, img, policy_dict)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        input_point = forward_point_cv2(mask)
        input_label = np.array([1])
        
        masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=None,
        multimask_output=False,
        )
        pred_masks_tensor = torch.from_numpy(masks).to(device).float()
        
        IoU_cost = eval_gt_pred(gt_masks_tensor, pred_masks_tensor)
        sum_IoU_cost += IoU_cost
    return sum_IoU_cost / sum_imgs


def train_controller(args, controller, optimizer, val_acc, baseline):
    controller.train()

    entropies, log_prob = controller.entropies, controller.log_probs
    # entropies, log_prob = torch.Tensor(np.array(entropies)).cuda(), torch.Tensor(np.array(log_prob)).cuda()
    # np_entropies = entropies.data.cpu().numpy()
    reward = val_acc + args.entropy_coeff * entropies

    if baseline is None:
        baseline = reward
    else:
        decay = args.baseline_decay
        baseline = decay * baseline + (1 - decay) * reward
        baseline = baseline.clone().detach()

    adv = reward - baseline
    #adv *= 1e-2
    # loss = -log_prob * get_variable(adv, args.cuda, requires_grad=False)
    loss = -log_prob * adv
    loss -= args.entropy_coeff * entropies
    loss = loss.sum()

    optimizer.zero_grad()
    loss.backward()

    if args.controller_grad_clip > 0:
        torch.nn.utils.clip_grad_norm(controller.parameters(), args.controller_grad_clip)
    optimizer.step()

    print('entropies: {}, log_prob: {}, reward: {}, loss: {}, baseline: {}'.format(entropies.item(), log_prob.item(), reward, loss, baseline))

    return baseline


def main(args):
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
    predictor = SamPredictor(sam)

    # Controller
    controller = Controller(args).to(device)
    controller_optimizer = torch.optim.SGD(controller.parameters(), args.controller_lr, momentum=0.9)
    
    # load_controller
    #save_dict = "./save_dir/1203.pt.tar"
    #checkpoints = torch.load(save_dict)
    #controller.load_state_dict(checkpoints['controller_state'])

    baseline = None
    best_mIoU_record = []
    best_p_dict = []
    best_iter_record = []
    len_save = args.save_policy_len
    best_p_val_dict = [0 for i in range(len_save)]

    # search
    for epoch in range(args.search_epochs):
        print('-'*50)
        print('{} th search'.format(epoch + 1))
        print('-'*50)

        # sample subpolicy
        print('*'*30)
        print('sample subpolicy')
        print('*'*30)
        controller.eval()
        policy_dict = controller.sample()
        for p in policy_dict:
            print(p)

        print('*' * 30)
        print('val SAM')
        print('*' * 30)
        val_iou = val_SAM(args, predictor, policy_dict, device, mode="train")
        
        best_mIoU_record.append(val_iou)
        best_p_dict.append(policy_dict)
        best_iter_record.append(epoch)

        best_mIoU_record, best_p_dict, best_iter_record = update_record(best_mIoU_record, best_p_dict, best_iter_record, args.save_policy_len)

        # train controller
        print('*' * 30)
        print('train controller')
        print('*' * 30)
        baseline = train_controller(args, controller, controller_optimizer, val_iou, baseline)

        print('*' * 30)
        print("best_mIoU_record: ", best_mIoU_record)
        print("Iter_record: ", best_iter_record)
        print("Policy_record: ", best_p_dict)
        print('*' * 30)
        
        # evaluate all best polcies
        if epoch != 0 and epoch % args.val_epoch == 0:
            for i in range(len(best_mIoU_record)):
                print("train_mIoU: ", best_mIoU_record[i])
                print("Iter: ", best_iter_record[i])
                print("Policy: ", best_p_dict[i])
                p = best_p_dict[i]
                val_iou = val_SAM(args, predictor, p, device, mode='val')
                best_p_val_dict[i] = val_iou
                print("val_mIoU: ", val_iou)
                print('*' * 30)
            print(best_p_val_dict)
            
        # save
        state = {
            'args': args,
            'best_mIoU_record': best_mIoU_record,
            'best_p_dict': best_p_dict,
            'baseline': baseline,
            'controller_state': controller.state_dict(),
            'policy_dict': policy_dict
        }
        # torch.save(state, './save_dir/{}.pt.tar'.format(epoch))
        
        
if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main(args)