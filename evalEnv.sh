#!/bin/bash

# 指定Python解释器的路径，如果需要的话
# python_path=/path/to/your/python

# 使用指定的Python解释器（如果设置了python_path变量）或默认的Python解释器运行train.py
# python_path变量仅在Python不在PATH环境变量中时需要
# ${python_path:-python} 是一个默认值表达式，如果python_path未设置，则使用python
python eval_with_env.py --checkpoint ../segment-anything/checkpoints/sam_vit_h_4b8939.pth --model_type vit_h