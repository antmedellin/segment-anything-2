import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import sys







# sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"

sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
# video_dir = "notebooks/videos/bedroom"
# video_dir = "../hsi_tracking/datasets/training/HSI-VIS-FalseColor/automobile/automobile"
video_dir = "../hsi_tracking/datasets/training/HSI-VIS-FalseColor/ball/ball"

results_output_dir = "notebooks/results"

# reference https://www.hsitracking.com/contest/ 


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
    
# Ensure results_output_dir exists
if not os.path.exists(results_output_dir):
    os.makedirs(results_output_dir)   

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
    
    
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)



# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# load the initialization box from the ground truth 
ground_truth_file = os.path.join(video_dir, "groundtruth_rect.txt")

# read the file 
with open(ground_truth_file, 'r') as f:
    lines = f.readlines()
    # box = [int(x) for x in lines[0].split(',')] # '463\t146\t13\t10\t\n'
    # print(lines)
# print(len(lines), lines[0])

# use tab as separator
box_base = [int(x) for x in lines[0].strip().split('\t')]
#  The bounding box is represented by the centre location and its height and width. 
# Convert box to a NumPy array of type np.float32
box_base = np.array(box_base, dtype=np.float32)


# box should be in fomrat [x0, y0, x1, y1] - origin is top left corner
# (x_min, y_min, x_max, y_max) 
# center_x, center_y, width, height = box_base
# x_min = center_x - width / 2
# y_min = center_y - height / 2
# x_max = center_x + width / 2
# y_max = center_y + height / 2

x_min, y_min, width, height = box_base
x_max = x_min + width
y_max = y_min + height
box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)




# # take a look the first video frame
# frame_idx = 0
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
# plt.show()

inference_state = predictor.init_state(video_path=video_dir)

predictor.reset_state(inference_state)


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
# points = np.array([[210, 350]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)

# box = np.array([300, 0, 500, 400], dtype=np.float32)



frame_idx , out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    # points=points,
    # labels=labels,
    box = box,
)


# print(frame_idx , out_obj_ids, out_mask_logits)

# # show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# # show_points(points, labels, plt.gca())
# show_box(box, plt.gca())
# show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# plt.show()


# Convert out_mask_logits to a binary mask
binary_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
# Squeeze the binary_mask to remove the extra dimension
binary_mask = np.squeeze(binary_mask)

# Find the coordinates of the non-zero elements in the mask
non_zero_coords = np.argwhere(binary_mask)

# Calculate the bounding box coordinates
y_min = np.min(non_zero_coords[:, 0])
x_min = np.min(non_zero_coords[:, 1])
y_max = np.max(non_zero_coords[:, 0])
x_max = np.max(non_zero_coords[:, 1])

# Create the bounding box
new_box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
print("reference box", box, "output box", new_box)

plt.imshow(binary_mask)
plt.show()
sys.exit(0)

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        
    plt.show()