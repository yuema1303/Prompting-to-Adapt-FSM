import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os
from typing import Any, Dict, List
from PIL import Image, ImageEnhance
import random

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

from pycocotools import mask as maskUtils
import torch.nn.functional as F
from pycocotools.coco import COCO

def annToRLE(segm, img_size):
    h, w = img_size
    rles = maskUtils.frPyObjects(segm, h, w)
    rle = maskUtils.merge(rles)
    return rle

def annToMask(segm, img_size):
    rle = annToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m

def targetToMask(target, img_size):
    masks_in_image = []
    for i in range(len(target)):
        if target[i]['iscrowd'] == 1:
            continue
        m = annToMask(target[i]["segmentation"], img_size)
        s = np.sum(m)
        m = torch.tensor(m).to("cuda")
        masks_in_image.append(m)
    return masks_in_image

def masks2tensor(masks):
    masks_tensor = []
    for i, mask_data in enumerate(masks):
        #mask = mask.to('cuda')
        mask = mask_data["segmentation"]
        mask = torch.from_numpy(mask).int().to('cuda')
        masks_tensor.append(mask)
    masks_tensor = torch.stack(masks_tensor, dim=0)
    return masks_tensor

def compute_mask_IoU_N2N(masks, target):
    assert target.shape[-2:] == masks.shape[-2:]
    A = masks.size(0)
    B = target.size(0)
    w, h = target.shape[-2:]
    #print(w, h)
    
    masks = masks.unsqueeze(1).expand(A, B, w, h)
    #print(masks)
    #print(masks.shape)
    
    target = target.broadcast_to(A, B, w, h)
    #print(target)
    #print(target.shape)
    
    temp = masks * target
    #print(temp)
    intersection = torch.einsum('abwh->ab', [temp])
    #print(intersection)
    
    union_temp = masks + target - temp
    union = torch.einsum('abwh->ab', [union_temp])
    if torch.isinf(union).any():
        print(">>>>>>>>>>>>>>>inf>>>>>>>>>>>>>>>>")
    #print(masks + target - temp)
    #print(union)
    
    IoU = intersection/union
    #print(IoU)
    #print(IoU.shape)
    
    return IoU

def eval_gt_pred(gt, pred):
    gt_down = gt
    pred_down = pred
    IoU = compute_mask_IoU_N2N(pred_down, gt_down)
    max_iou_per_column, _ = torch.max(IoU, 0)
    mean_iou = torch.mean(max_iou_per_column)
    return mean_iou

def vis_mask_img(img, masks, output_dir):
    cv2.imwrite(os.path.join(output_dir, "img.jpg"), img)
    
    for i, mask in enumerate(masks):
        mask = mask.to('cpu').numpy()
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(output_dir, filename), mask * 255)

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}_SAM.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def img_loader(path, mode):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert(mode)

def get_point_random(gt_mask):
    one_index = torch.where(gt_mask == 1)
    rand_index = random.randint(0, len(one_index[0]))
    rand_x = one_index[0][rand_index]
    rand_y = one_index[1][rand_index]
    
    input_point = np.array([[rand_y.numpy(),rand_x.numpy()]])
    
    return input_point
    
def forward_point_cv2(gt_path):
    mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    max_distance_position = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)

    center_x, center_y = max_distance_position
    
    input_point = np.array([[center_y,center_x]])
    return input_point

def update_record(best_mIoU_record, best_p_dict, best_iter_record, len_save):
    sort_best_mIoU_record = sorted(best_mIoU_record, reverse=True)
    sorted_index = [i[0] for i in sorted(enumerate(best_mIoU_record), key=lambda x:x[1], reverse=True)]
    
    sort_best_p_dict = [0 for i in range(len(sorted_index))]
    sort_best_iter_record = [0 for i in range(len(sorted_index))]
    
    for i in range(len(sorted_index)):
        sort_best_p_dict[i] = best_p_dict[sorted_index[i]]
        sort_best_iter_record[i] = best_iter_record[sorted_index[i]]

    if len(sort_best_mIoU_record) > len_save:
        best_mIoU_record = sort_best_mIoU_record[:len_save]
        best_p_dict = sort_best_p_dict[:len_save]
        best_iter_record = sort_best_iter_record[:len_save]
    else:
        best_mIoU_record = sort_best_mIoU_record
        best_p_dict = sort_best_p_dict
        best_iter_record = sort_best_iter_record
        
    return best_mIoU_record, best_p_dict, best_iter_record