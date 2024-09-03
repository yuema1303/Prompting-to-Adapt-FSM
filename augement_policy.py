from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

from IPTfunc import IPTdenoise, IPTderain

def Policy(args, img, policy_dict):
    if policy_dict == None:
        return img
    
    for i in range(args.subpolicy_num):
        op_1 = policy_dict[i][0]['op']
        magnitude_1 = policy_dict[i][0]['magnitude']
        #op_1 = "contrast_up"

        if op_1 == "brightness_up":
            img = brightness_up(img, magnitude_1)
        elif op_1 == "brightness_down":
            img = brightness_down(img, magnitude_1)
        elif op_1 == "contrast_up":
            img = contrast_up(img, magnitude_1)
        elif op_1 == "contrast_down":
            img = contrast_down(img, magnitude_1)
        elif op_1 == "saturation_up":
            img = saturation_up(img, magnitude_1)
        elif op_1 == "saturation_down":
            img = saturation_down(img, magnitude_1)
        elif op_1 == "boxFilter":
            img = boxFilter(img, magnitude_1)
        elif op_1 == "gaussianBlur":
            img = gaussianBlur(img, magnitude_1)
        elif op_1 == "logGray":
            img = logGray(img, magnitude_1)
        elif op_1 == "gamma_corrected":
            img = gamma_corrected(img, magnitude_1)
        elif op_1 == "meanBlur":
            img = meanBlur(img, magnitude_1)
        elif op_1 == "sharpen_lowpass":
            img = sharpen_lowpass(img, magnitude_1)
        elif op_1 == "sharpen_gaussian":
            img = sharpen_gaussian(img, magnitude_1)
        elif op_1 == "sharpen_lap":
            img = sharpen_lap(img, magnitude_1)
        elif op_1 == "bilateralFilter":
            img = bilateralFilter(img, magnitude_1)
        elif op_1 == "medianBlur":
            img = medianBlur(img, magnitude_1)
        elif op_1 == "IPTdenoise":
            img = IPTdenoise(img, magnitude_1) #magnitude here is noise level added before denoise
        elif op_1 == "IPTderain":
            img = IPTderain(img, magnitude_1) #useless magnitude

    return img


def brightness_up(img, magnitude):

    mag = np.linspace(0.0, 0.9, 10)[magnitude] 

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 

    brightness = 1 + mag 
    enhancer = ImageEnhance.Brightness(img_PIL)
    img_bright = enhancer.enhance(brightness)

    img_PIL = np.array(img_bright) 
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) 

    return img_cv2


def brightness_down(img, magnitude):

    mag = np.linspace(0.0, 0.9, 10)[magnitude] 

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 

    brightness = 1 - mag 
    enhancer = ImageEnhance.Brightness(img_PIL)
    img_bright = enhancer.enhance(brightness)

    img_PIL = np.array(img_bright) 
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) 

    return img_cv2


def contrast_up(img, magnitude): 

    mag = np.linspace(0.0, 0.9, 10)[magnitude] 

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    contrast = 1 + mag 
    enhancer = ImageEnhance.Contrast(img_PIL)
    img_contrast = enhancer.enhance(contrast)

    img_PIL = np.array(img_contrast)
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) 

    return img_cv2


def contrast_down(img, magnitude): 

    mag = np.linspace(0.0, 0.9, 10)[magnitude] 

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 

    contrast = 1 - mag 
    enhancer = ImageEnhance.Contrast(img_PIL)
    img_contrast = enhancer.enhance(contrast)

    img_PIL = np.array(img_contrast) 
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) 

    return img_cv2


def saturation_up(img, magnitude):

    mag = np.linspace(0.0, 0.9, 10)[magnitude] 

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 

    saturation = 1 + mag 
    enhancer = ImageEnhance.Color(img_PIL)
    img_saturation = enhancer.enhance(saturation)

    img_PIL = np.array(img_saturation) 
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR)

    return img_cv2


def saturation_down(img, magnitude): 

    mag = np.linspace(0.0, 0.9, 10)[magnitude]

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
    saturation = 1 - mag
    enhancer = ImageEnhance.Color(img_PIL)
    img_saturation = enhancer.enhance(saturation)

    img_PIL = np.array(img_saturation) 
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) 

    return img_cv2


def boxFilter(img, magnitude):

    size = magnitude + 1
    source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    result = cv2.boxFilter(source, -1, (size, size), normalize=1)
    source = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)

    return source


def gaussianBlur(img, magnitude): 

    mag = [1,3,5,7,9,11,13,15,17,19]
    size = mag[magnitude]
    source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    result = cv2.GaussianBlur(source, (size,size), 0)
    source = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)

    return source


def logGray(img, magnitude):
    mag = [27,30,32,35,37,40,42,45,47,50]
    c = mag[magnitude]
    output=log(c,img)

    return output


def gamma_corrected(img, magnitude): 
    mag = [-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5]
    gamma = mag[magnitude] + 1

    gamma_corrected = np.power(img / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)

    return gamma_corrected


def meanBlur(img, magnitude):
    kernel_size = (magnitude+1, magnitude+1) 

    blurred_image = cv2.blur(img, kernel_size)

    return blurred_image


def sharpen_lowpass(img, magnitude): 
    kernel_size = (3, 3)
    #blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    blurred_image = cv2.blur(img, kernel_size)
    #highpass_image = image - blurred_image
    highpass_image = cv2.subtract(img, blurred_image)
    mag = [-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4]
    alpha = 1 + mag[magnitude]

    sharpened_image = cv2.addWeighted(img, 1 + alpha, highpass_image, -alpha, 0)

    return sharpened_image


def sharpen_gaussian(img, magnitude): 
    kernel_size = (3, 3)
    blurred_image = cv2.GaussianBlur(img, kernel_size, 0)
    #blurred_image = cv2.blur(image, kernel_size)

    highpass_image = cv2.subtract(img, blurred_image)

    mag = [-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4]
    alpha = 1 + mag[magnitude]

    sharpened_image = cv2.addWeighted(img, 1 + alpha, highpass_image, -alpha, 0)

    return sharpened_image


def sharpen_lap(img, magnitude):
    laplacian_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=np.float32)
    sharpened_image = cv2.filter2D(img, -1, laplacian_kernel)

    return sharpened_image


def bilateralFilter(img, magnitude): 
    source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    mag = [15,25,35,45,55,65,75,85,95,105]

    size = mag[magnitude]

    result = cv2.bilateralFilter(source, 3, size, 75)

    source = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
    return source


def medianBlur(img, magnitude): 

    mag = [1,3,5,7,9,11,13,15,17,19]
    size = mag[magnitude]
    result = cv2.medianBlur(img, size)

    return result


def log(c,img): 
    output = c * np.log(1.0 + img)
    output=np.uint8(output+0.5)
    return output


def get_sub_policies(augment_id_list, magnitude_id_list, args):
    policies = []
    for n in range(args.subpolicy_num):    
        sub_policy = {}
        for i in range(args.op_num_pre_subpolicy): 
            policy = {}
            policy['op'] = args.augment_types[augment_id_list[n + i]]
            policy['magnitude'] = args.magnitude_types[magnitude_id_list[n + i]]
            sub_policy[i] = policy
        policies.append(sub_policy)
    return policies


if __name__ == '__main__':
    img = cv2.imread("rain.png")
    
    imgDenoise = IPTderain(img = img, magnitude = 0)
    
    cv2.imwrite("Derain.jpg", imgDenoise)