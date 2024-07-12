from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


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

    return img


def brightness_up(img, magnitude): # 1.增加亮度

    mag = np.linspace(0.0, 0.9, 10)[magnitude] # 设置强度范围

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转PIL格式

    brightness = 1 + mag # 亮度数据增强
    enhancer = ImageEnhance.Brightness(img_PIL)
    img_bright = enhancer.enhance(brightness)

    img_PIL = np.array(img_bright) #先转换为数组   H W C
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) # PIL转cv2

    return img_cv2


def brightness_down(img, magnitude): # 2.降低亮度

    mag = np.linspace(0.0, 0.9, 10)[magnitude] # 设置强度范围

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转PIL格式

    brightness = 1 - mag # 亮度数据增强
    enhancer = ImageEnhance.Brightness(img_PIL)
    img_bright = enhancer.enhance(brightness)

    img_PIL = np.array(img_bright) #先转换为数组   H W C
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) # PIL转cv2

    return img_cv2


def contrast_up(img, magnitude): # 3.提高对比度

    mag = np.linspace(0.0, 0.9, 10)[magnitude] # 设置强度范围

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转PIL格式

    contrast = 1 + mag 
    enhancer = ImageEnhance.Contrast(img_PIL)
    img_contrast = enhancer.enhance(contrast)

    img_PIL = np.array(img_contrast) #先转换为数组   H W C
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) # PIL转cv2

    return img_cv2


def contrast_down(img, magnitude): # 4.降低对比度

    mag = np.linspace(0.0, 0.9, 10)[magnitude] # 设置强度范围

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转PIL格式

    contrast = 1 - mag 
    enhancer = ImageEnhance.Contrast(img_PIL)
    img_contrast = enhancer.enhance(contrast)

    img_PIL = np.array(img_contrast) #先转换为数组   H W C
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) # PIL转cv2

    return img_cv2


def saturation_up(img, magnitude): # 5.提高饱和度

    mag = np.linspace(0.0, 0.9, 10)[magnitude] # 设置强度范围

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转PIL格式

    saturation = 1 + mag 
    enhancer = ImageEnhance.Color(img_PIL)
    img_saturation = enhancer.enhance(saturation)

    img_PIL = np.array(img_saturation) #先转换为数组   H W C
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) # PIL转cv2

    return img_cv2


def saturation_down(img, magnitude): # 6.降低饱和度

    mag = np.linspace(0.0, 0.9, 10)[magnitude] # 设置强度范围

    img_PIL = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转PIL格式
    saturation = 1 - mag
    enhancer = ImageEnhance.Color(img_PIL)
    img_saturation = enhancer.enhance(saturation)

    img_PIL = np.array(img_saturation) #先转换为数组   H W C
    img_cv2 = cv2.cvtColor(img_PIL,cv2.COLOR_RGB2BGR) # PIL转cv2

    return img_cv2


def boxFilter(img, magnitude): # 7.方框滤波

    size = magnitude + 1
    source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #方框滤波
    result = cv2.boxFilter(source, -1, (size, size), normalize=1)
    source = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)

    return source


def gaussianBlur(img, magnitude): # 8.高斯滤波

    mag = [1,3,5,7,9,11,13,15,17,19]
    size = mag[magnitude]
    source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #方框滤波
    result = cv2.GaussianBlur(source, (size,size), 0)
    source = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)

    return source


def logGray(img, magnitude): # 9.对数灰度变换
    mag = [27,30,32,35,37,40,42,45,47,50]
    c = mag[magnitude]
    output=log(c,img)

    return output


def gamma_corrected(img, magnitude): # 10.Gamma灰度变换
    mag = [-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5]
    gamma = mag[magnitude] + 1

    # 进行伽马变换
    gamma_corrected = np.power(img / 255.0, gamma) * 255.0
    gamma_corrected = gamma_corrected.astype(np.uint8)

    return gamma_corrected


def meanBlur(img, magnitude): # 11.均值滤波
    # 指定均值滤波的核大小
    kernel_size = (magnitude+1, magnitude+1)  # 这里使用一个5x5的核

    # 应用均值滤波
    blurred_image = cv2.blur(img, kernel_size)

    return blurred_image


def sharpen_lowpass(img, magnitude): # 12.空间域锐化低通滤波器
    kernel_size = (3, 3)
    #blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    blurred_image = cv2.blur(img, kernel_size)
    # 计算原始图像与低通滤波后的图像之差，得到高频细节
    #highpass_image = image - blurred_image
    highpass_image = cv2.subtract(img, blurred_image)
    # 调整增强系数（根据需求调整）
    mag = [-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4]
    alpha = 1 + mag[magnitude]

    # 将高频细节与原始图像相加，得到锐化后的图像
    sharpened_image = cv2.addWeighted(img, 1 + alpha, highpass_image, -alpha, 0)

    return sharpened_image


def sharpen_gaussian(img, magnitude): # 13.空间域锐化高斯核
    kernel_size = (3, 3)
    blurred_image = cv2.GaussianBlur(img, kernel_size, 0)
    #blurred_image = cv2.blur(image, kernel_size)

    # 计算原始图像与平滑后的图像之差，得到高频细节
    highpass_image = cv2.subtract(img, blurred_image)

    # 调整增强系数（根据需求调整）
    mag = [-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4]
    alpha = 1 + mag[magnitude]

    # 将高频细节与原始图像相加，得到锐化后的图像
    sharpened_image = cv2.addWeighted(img, 1 + alpha, highpass_image, -alpha, 0)

    return sharpened_image


def sharpen_lap(img, magnitude): # 14.空间域锐化拉普拉斯算子
    laplacian_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]], dtype=np.float32)
    # 应用拉普拉斯增强算子
    sharpened_image = cv2.filter2D(img, -1, laplacian_kernel)

    return sharpened_image


def bilateralFilter(img, magnitude): # 15.双边滤波
    source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    mag = [15,25,35,45,55,65,75,85,95,105]

    size = mag[magnitude]

    #双边滤波
    result = cv2.bilateralFilter(source, 3, size, 75)

    source = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
    return source


def medianBlur(img, magnitude): # 16.均值滤波

    mag = [1,3,5,7,9,11,13,15,17,19]
    size = mag[magnitude]
    #中值滤波
    result = cv2.medianBlur(img, size)

    return result


def log(c,img): # 灰度对数变换要用
    output = c * np.log(1.0 + img)
    output=np.uint8(output+0.5)
    return output