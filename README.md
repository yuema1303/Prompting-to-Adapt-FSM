# Prompting-to-Adapt-FSM
## Installation
### Environment
This code was implemented with Python 3.10 and PyTorch 2.0.1. You can install all the requirements via:
```
pip install -r requirements.txt
```
### SAM Installation
Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
SAM Checkpoints:

Download [model-checkpoints](https://github.com/facebookresearch/segment-anything) from SAM official repository.
## Dataset
Set the path of images, masks, train-val splits in `config.py`. Take the [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) dataset as an example：
```
parser.add_argument('--img_dir', type=str, default='../../data/Medical/Kvasir-SEG/images/')
parser.add_argument('--label_dir', type=str, default='../../data/Medical/Kvasir-SEG/masks/')
parser.add_argument('--train_txt', type=str, default='../../data/Medical/Kvasir-SEG/Kavsir_train@1_10.txt')
parser.add_argument('--val_txt', type=str, default='../../data/Medical/Kvasir-SEG/Kavsir_val.txt')
```
! Please pay attention to the file format of the dataset and modify the code accordingly in `train.py`.
```
img_path = os.path.join(img_dir, img_name + ".jpg")
label_path = os.path.join(label_dir, img_name + ".jpg")
```
## Train
```
python train.py --checkpoint ../segment-anything/checkpoints/sam_vit_h_4b8939.pth --model_type vit_h
```
! Please note that the code evaluates searched policies every `args.val_epoch` during training.
