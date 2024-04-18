from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def visualize_pose_results(orig_img:Image.Image, skeleton_img:Image.Image, left_title:str = "Original image", right_title:str = "Pose"):
    """
    Helper function for pose estimationresults visualization

    Parameters:
       orig_img (Image.Image): original image
       skeleton_img (Image.Image): processed image with body keypoints
       left_title (str): title for the left image
       right_title (str): title for the right image
    Returns:
       fig (matplotlib.pyplot.Figure): matplotlib generated figure contains drawing result
    """
    orig_img = orig_img.resize(skeleton_img.size)
    im_w, im_h = orig_img.size
    is_horizontal = im_h <= im_w
    figsize = (20, 10) if is_horizontal else (10, 20)
    fig, axs = plt.subplots(2 if is_horizontal else 1, 1 if is_horizontal else 2, figsize=figsize, sharex='all', sharey='all')
    fig.patch.set_facecolor('white')
    list_axes = list(axs.flat)
    for a in list_axes:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.grid(False)
    list_axes[0].imshow(np.array(orig_img))
    list_axes[1].imshow(np.array(skeleton_img))
    list_axes[0].set_title(left_title, fontsize=15)
    list_axes[1].set_title(right_title, fontsize=15)
    fig.subplots_adjust(wspace=0.01 if is_horizontal else 0.00 , hspace=0.01 if is_horizontal else 0.1)
    fig.tight_layout()
    return fig

# pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
# img = load_image("pose.jpg")
# pose = pose_estimator(img)
# fig = visualize_pose_results(img, pose)
# plt.savefig('temp.png')