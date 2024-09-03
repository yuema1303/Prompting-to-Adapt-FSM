import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser('Prompting_to_Adapt')

    # dataset
    parser.add_argument('--img_dir', type=str, default='./dataset/Kvasir-SEG/images/')
    parser.add_argument('--label_dir', type=str, default='./dataset/Kvasir-SEG/masks/')
    parser.add_argument('--train_txt', type=str, default='./dataset/Kvasir-SEG/Kavsir_train@1_10.txt')
    parser.add_argument('--val_txt', type=str, default='./dataset/Kvasir-SEG/Kavsir_val.txt')
    
    # search space
    parser.add_argument('--augment_types', type=list, default=['brightness_up', 'brightness_down', 'contrast_up', 'contrast_down', 'saturation_up', 'saturation_down', 'boxFilter', 'gaussianBlur', 'logGray', 'gamma_corrected', 'meanBlur', 'sharpen_lowpass', 'sharpen_gaussian', 'sharpen_lap', 'bilateralFilter', 'medianBlur', 'IPTdenoise', 'IPTderain'
                                            ], help='all searched policies')
    parser.add_argument('--magnitude_types', type=list, default=range(10))
    parser.add_argument('--prob_types', type=list, default=range(11))
    parser.add_argument('--op_num_pre_subpolicy', type=int, default=1)
    parser.add_argument('--subpolicy_num', type=int, default=3)

    # controller
    parser.add_argument('--controller_hid_size', type=int, default=100)
    parser.add_argument('--controller_lr', type=float, default=3.5e-4)
    parser.add_argument('--softmax_temperature', type=float, default=5.)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--entropy_coeff', type=float, default=1e-5)
    parser.add_argument('--baseline_decay', type=float, default=0.95)
    parser.add_argument('--controller_grad_clip', type=float, default=0.)

    # training
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--search_epochs', type=int, default=1500)  # 1500
    parser.add_argument('--save_policy_len', type=int, default=20)  # 1500
    parser.add_argument('--val_epoch', type=int, default=50)  # 1500
    
    #stable baseline 3
    parser.add_argument('--sb3_lr', type=float, default=0.0005)
    parser.add_argument('--sb3_batchsize', type=int, default=2)
    parser.add_argument('--n_steps', type=int, default=2)
    parser.add_argument('--total_timesteps', type=int, default=1500)
    parser.add_argument('--sb3_checkpoint', type=str, default=None)

    # SAM
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )

    amg_settings = parser.add_argument_group("AMG Settings")

    amg_settings.add_argument(
        "--points-per-side",
        type=int,
        default=None,
        help="Generate masks by sampling a grid over the image with this many points to a side.",
    )

    amg_settings.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="How many input points to process simultaneously in one batch.",
    )

    amg_settings.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=None,
        help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-thresh",
        type=float,
        default=None,
        help="Exclude masks with a stability score lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-offset",
        type=float,
        default=None,
        help="Larger values perturb the mask more when measuring stability score.",
    )

    amg_settings.add_argument(
        "--box-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding a duplicate mask.",
    )

    amg_settings.add_argument(
        "--crop-n-layers",
        type=int,
        default=None,
        help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
        ),
    )

    amg_settings.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding duplicate masks across different crops.",
    )

    amg_settings.add_argument(
        "--crop-overlap-ratio",
        type=int,
        default=None,
        help="Larger numbers mean image crops will overlap more.",
    )

    amg_settings.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=None,
        help="The number of points-per-side in each layer of crop is reduced by this factor.",
    )

    amg_settings.add_argument(
        "--min-mask-region-area",
        type=int,
        default=None,
        help=(
            "Disconnected mask regions or holes with area smaller than this value "
            "in pixels are removed by postprocessing."
        ),
    )
    arguments = parser.parse_args()
    print(arguments)
    return argumentss

















