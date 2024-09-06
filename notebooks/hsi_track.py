import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import sys
import matplotlib.patches as patches

def X2Cube(img,B=[4, 4],skip = [4, 4],bandNumber=16):
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//4, N//4,bandNumber )
    return DataCube

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

def is_within_box(coords, x_min, y_min, x_max, y_max):
    x, y = coords
    return x_min <= x <= x_max and y_min <= y <= y_max

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def get_median_rgb(image, mask):
    # Ensure the image is in RGB format
    # image = image.convert("RGB")
    image_np = np.array(image)
    
    # Extract the pixels in the mask
    masked_pixels = image_np[mask]
    
    # plt.imshow(masked_pixels)
    # plt.show()
    
    # number of mask pixels that are not zero and compare to mask non zero pixels
    # print("masked pixel size", masked_pixels.shape)
    # print("mask size", np.argwhere(mask).shape)
    
    # Calculate the median RGB values
    median_rgb = np.median(masked_pixels, axis=0)
    
    return median_rgb

# choose model to use

# can overload 24 gb vram, which causes it to be slow 
# sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"

# sam2_checkpoint = "checkpoints/sam2_hiera_base_plus.pt"
# model_cfg = "sam2_hiera_b+.yaml"



sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

# Define the threshold to exapand box by
threshold = 0 # expand box by this threshold
norm_threshold = 7 # spectral similarity allowed 

# choose video to track

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
# video_dir = "notebooks/videos/bedroom"
# video_dir = "../hsi_tracking/datasets/training/HSI-VIS-FalseColor/automobile/automobile"
# video_dir = "../hsi_tracking/datasets/training/HSI-VIS-FalseColor/ball/ball"

# results_output_dir = "notebooks/results/training/HSI-VIS-FalseColor"
# results_output_dir = "notebooks/results/training/HSI-NIR-FalseColor"
# results_output_dir = "notebooks/results/training/HSI-RedNIR-FalseColor"

# reference https://www.hsitracking.com/contest/ 

#print out all the directories in the datasets/training/HSI-VIS-FalseColor directory
# video_base_dir = "../hsi_tracking/datasets/training/HSI-VIS-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/training/HSI-NIR-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/training/HSI-RedNIR-FalseColor"


# results_output_dir = "notebooks/results/training/HSI-NIR-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/training/HSI-NIR-FalseColor"

# results_output_dir = "notebooks/results/training/HSI-VIS-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/training/HSI-VIS-FalseColor"

results_output_dir = "notebooks/results/training/HSI-RedNIR-FalseColor"
video_base_dir = "../hsi_tracking/datasets/training/HSI-RedNIR-FalseColor"
relative_path_to_hsi = "../../../HSI-RedNIR"

# results_output_dir = "notebooks/results/validation/HSI-NIR-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/validation/HSI-NIR-FalseColor"

# results_output_dir = "notebooks/results/validation/HSI-VIS-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/validation/HSI-VIS-FalseColor"
# relative_path_to_hsi = "../../../HSI-VIS"

# results_output_dir = "notebooks/results/validation/HSI-RedNIR-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/validation/HSI-RedNIR-FalseColor"


# results_output_dir = "notebooks/results/development"
# video_base_dir = "../hsi_tracking/datasets/validation/HSI-VIS-FalseColor/S_runner1"
# relative_path_to_hsi = "../../../HSI-VIS"


video_sub_dirs = os.listdir(video_base_dir)

# print(video_sub_dirs)



# start of code 
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
# Ensure results_output_dir exists
if not os.path.exists(results_output_dir):
    os.makedirs(results_output_dir)   


    
# initialize the predictor     
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
for current_video_sub_dir in video_sub_dirs:    
    
    
    
    
    # Get the base name of the results_output_dir
    base_name = os.path.basename(current_video_sub_dir)
    print(f"Processing {base_name}, directory number {video_sub_dirs.index(current_video_sub_dir)+1} of {len(video_sub_dirs)}")
    
    result_output_name = os.path.join(results_output_dir, base_name+".txt")
    
    


    video_dir = os.path.join(video_base_dir, current_video_sub_dir)
    # print(video_dir)
    
    # Check if the current path is a directory
    if not os.path.isdir(video_dir):
        print(f"Skipping {current_video_sub_dir} as it is not a directory.")
        continue
    
    # # # scan all the JPEG frame names in this directory and sort them by frame index
    # frame_names = [
    #     p for p in os.listdir(video_dir)
    #     if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    # ]
    # # print(f"Number of frames in video: {len(frame_names)}")#, first frame: {frame_names[0]}")
    
    # if len(frame_names) == 0:
    #     video_dir = os.path.join(video_base_dir, current_video_sub_dir,current_video_sub_dir)
    #     # print(video_dir)
    #     frame_names = [
    #         p for p in os.listdir(video_dir)
    #         if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    #     ]
    #     # print(f"Number of new frames in video: {len(frame_names)}")#, first frame: {frame_names[0]}")
    
    # Scan all the JPEG frame names in this directory and sort them by frame index
    try:
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
    except FileNotFoundError:
        print(f"Directory {video_dir} not found.")
        continue
    
    if len(frame_names) == 0:
        video_dir = os.path.join(video_base_dir, current_video_sub_dir, current_video_sub_dir)
        try:
            frame_names = [
                p for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
        except FileNotFoundError:
            print(f"Directory {video_dir} not found.")
            continue
        
        
    if len(frame_names) == 0:
        print(f"No frames found in {current_video_sub_dir}")
        continue
    # elif len(frame_names) > 800:
    #     # can cause memory issues on some gpus
    #     print(f"Skipping {current_video_sub_dir} as it has more than 800 frames.")
    #     continue
    
        # if longer than 800 frames, logic needs to be added to split it into smaller segments and process them separately
    
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


    # base_name = os.path.join(video_dir, frame_names[0])
    # test_image = Image.open(base_name)
    
    
    # # get the hyperspectral image path 
    # hsi_base_name = frame_names[0].split(".")[0]
    # hsi_image_path = os.path.join(video_dir, "../../..","HSI-VIS","S_runner1","S_runner1",hsi_base_name+".png")
    
    # # check to see if the file exists
    # if not os.path.exists(hsi_image_path):
    #     print(f"Could not find hyperspectral image {hsi_image_path}")
    #     continue
    # else:
    #     print(f"Found hyperspectral image {hsi_image_path}")
    #     # load image 
    #     hsi_image = Image.open(hsi_image_path)
    #     hsi_image_np = np.array(hsi_image)
    #     result = X2Cube(hsi_image_np)
    #     print(hsi_image.size, hsi_image.mode, result.size, result.shape)
    #     print(test_image.size)
        
        
    #     # Extract the 3rd, 6th, and 9th channels
    #     channel_1 = result[:, :, 4]
    #     channel_2 = result[:, :, 8]
    #     channel_3 = result[:, :, 12]
        
    #     # Stack the channels to form an RGB image
    #     rgb_image = np.stack([channel_1, channel_2, channel_3], axis=-1)
        
    #     # Display the RGB image
    #     plt.imshow(rgb_image)
    #     plt.title("RGB Image from Hyperspectral Channels ")
    #     plt.show()
        
    
    # print(base_name)
    # print(hsi_image_path)
    # print(frame_names[0])
    # plt.imshow(test_image)
    # plt.show()
    
    
    # sys.exit(0)


    # load the initialization box from the ground truth (assumes same for all video directories)
    try:
        # read the file 
        ground_truth_file = os.path.join(video_dir, "groundtruth_rect.txt")
        with open(ground_truth_file, 'r') as f:
            lines = f.readlines()
            # box = [int(x) for x in lines[0].split(',')] # '463\t146\t13\t10\t\n'
        # print(lines)
        # print(len(lines), lines[0])
    except:
        print(f"Could not find groundtruth_rect.txt in {current_video_sub_dir}")
        continue

   

    # use tab as separator and strip the newline character
    # box_base = [int(x) for x in lines[0].strip().split('\t')]
    box_base = [int(x) for x in lines[0].strip().split()]
    #  The bounding box is represented by the centre location and its height and width. 
    # Convert box to a NumPy array of type np.float32
    box_base = np.array(box_base, dtype=np.float32)


    # box should be in fomrat [x0, y0, x1, y1] - origin is top left corner
    # (x_min, y_min, x_max, y_max) 

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

    # Clear GPU memory
    torch.cuda.empty_cache()
    
    inference_state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=False,  offload_state_to_cpu=False,  async_loading_frames=False,)

    # below line is only needed when previous tracking is done and we want to reset the state
    predictor.reset_state(inference_state)


    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

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


    # below allows us to generate the box around the object
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

    # print("reference box", box, "output box", new_box)

    # plt.imshow(binary_mask)
    # plt.show()
    # sys.exit(0)

    # run propagation throughout the video and collect the results in a dict
    all_points_annotated = False
    max_reprompts = 10 
    current_reprompts = 0 
    reprompt_frame_idx = 0
    previous_success_index = 0
    while all_points_annotated is False:
        video_segments = {}  # video_segments contains the per-frame segmentation results
        output_track_boxes = []  # output_track_boxes contains the per-frame tracking results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            try: 
                result_mask = np.squeeze((out_mask_logits[0] > 0.0).cpu().numpy()) # squeeze to remove the extra dimension since only doing 1 object
                non_zero_coords = np.argwhere(result_mask)
                if non_zero_coords.size > 0:
                    y_min = np.min(non_zero_coords[:, 0])
                    x_min = np.min(non_zero_coords[:, 1])
                    y_max = np.max(non_zero_coords[:, 0])
                    x_max = np.max(non_zero_coords[:, 1])
                    result_box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                    output_track_boxes.append(result_box)
                    previous_mask = result_mask
                    all_points_annotated = True
                    previous_success_index = out_frame_idx
                else:
                    raise Exception("No object detected")
            except:
                
                
                
                # no object detected, append the previous result if available
                if output_track_boxes:
                    output_track_boxes.append(output_track_boxes[-1])
                else:
                    output_track_boxes.append([0, 0, 0, 0])
                # print(f"No object detected in frame {out_frame_idx}")
                
                
                if out_frame_idx <= reprompt_frame_idx:
                    all_points_annotated = False
                    continue
                else:
                    reprompt_frame_idx = out_frame_idx
                    # print("hi2", current_reprompts, max_reprompts)
                    if current_reprompts == max_reprompts:
                        continue
                    
                continue    
                # this is when we need to reinitialize the object by adding a new point or box
                
                # take the median spectral signature of the object from the first frame and the previous frame
                # in the frame where the object is not detected, 
                # use the closes spectral match pixel to add a new point 
                # validate the result by setting a threshold for spatial distance (in cases of occlusion)
                # validate the result by making sure the match is close enough
                
                
                # pass the first image, the previous image, current image, original object box, original object mask, previous object mask, previous object box
                # get workging on rgb first, then move to hyperspectral
                
                
                # Load images
                original_image = Image.open(os.path.join(video_dir, frame_names[0]))
                previous_image = Image.open(os.path.join(video_dir, frame_names[previous_success_index]))
                current_image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
                
                
                #hsi images
                # will need to modify below to match the hsi images directories 
                try:
                    
                    hsi_original_image = X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,frame_names[0].split(".")[0]+".png"))))
                    hsi_previous_image = X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,frame_names[previous_success_index].split(".")[0]+".png"))))
                    hsi_current_image =  X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,frame_names[out_frame_idx].split(".")[0]+".png"))))
                
                except:
                    hsi_original_image = X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,current_video_sub_dir,frame_names[0].split(".")[0]+".png"))))
                    hsi_previous_image = X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,current_video_sub_dir,frame_names[previous_success_index].split(".")[0]+".png"))))
                    hsi_current_image =  X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,current_video_sub_dir,frame_names[out_frame_idx].split(".")[0]+".png"))))
                

                # print("original object box:", box)
                # print("previous object box:", output_track_boxes[-1])
                # print("original object mask shape:", binary_mask.shape)
                # print("previous object mask shape:", previous_mask.shape)
                # print("original image size:", original_image.size)
                # print("previous image size:", previous_image.size)
                # print("current image size:", current_image.size)
                
                
                # Get the median RGB values for the original and previous images
                median_rgb_original = get_median_rgb(original_image, binary_mask)
                median_rgb_previous = get_median_rgb(previous_image, previous_mask)
                median_hsi_original = get_median_rgb(hsi_original_image, binary_mask)
                median_hsi_previous = get_median_rgb(hsi_previous_image, previous_mask)

                # print("Median RGB values in the original image:", median_rgb_original, median_hsi_original)
                # print("Median RGB values in the previous image:", median_rgb_previous, median_hsi_previous)
            
                
                # find the pixels in the current image that are closest to the median rgb values
                # find the closest pixel to the median rgb values
                current_image_np = np.array(current_image)
                
                
                

                # Check if the closest pixels are within the previous object box plus a threshold
                prev_box = output_track_boxes[-1]
                x_min, y_min, x_max, y_max = prev_box
                x_min -= threshold
                y_min -= threshold
                x_max += threshold
                y_max += threshold
                
                # make sure the box is within the image size
                x_min = int(max(x_min, 0))
                y_min = int(max(y_min, 0))
                x_max = int(min(x_max, current_image_np.shape[1]))
                y_max = int(min(y_max, current_image_np.shape[0]))
                

                # Extract the ROI from the current image
                # roi = current_image_np[y_min:y_max, x_min:x_max]
                roi_hsi = hsi_current_image[y_min:y_max, x_min:x_max]

                # Flatten the ROI for distance calculation
                # roi_flatten = roi.reshape(-1, 3)
                roi_hsi_flatten = roi_hsi.reshape(-1, 16)


                # flatten the image array
                # current_image_flatten = current_image_np.reshape(-1, 3)
                
                # calculate the distance between the median rgb values and the pixels in the current image
                # distance_original = np.linalg.norm(current_image_flatten - median_rgb_original, axis=1)
                # distance_original = np.linalg.norm(roi_flatten - median_rgb_original, axis=1)
                distance_original = np.linalg.norm(roi_hsi_flatten - median_hsi_original, axis=1)
                # find the index of the pixel with the smallest distance
                # closest_pixel_idx_original = np.argmin(distance_original)
                closest_pixel_idx_original = np.argmin(distance_original)
                # convert the index to 2D coordinates
                # closest_pixel_coords_orig = np.unravel_index(closest_pixel_idx_original, current_image_np.shape[:2])
                # print("Closest pixel coordinates in the original image:", closest_pixel_coords_orig, np.min(distance_original), "rgb pixel values current", current_image_np[closest_pixel_coords_orig[0], closest_pixel_coords_orig[1]], "mediam rgb original img", median_rgb_original) 
                # closest_pixel_coords_orig = np.unravel_index(closest_pixel_idx_original, roi.shape[:2])
                # closest_pixel_coords_orig = (closest_pixel_coords_orig[0] + y_min, closest_pixel_coords_orig[1] + x_min)
                
                closest_pixel_coords_orig = np.unravel_index(closest_pixel_idx_original, roi_hsi.shape[:2])
                closest_pixel_coords_orig = (closest_pixel_coords_orig[0] + y_min, closest_pixel_coords_orig[1] + x_min)

                print("Closest pixel coordinates in the original image within the bounding box:", closest_pixel_coords_orig, "difference amount", np.min(distance_original), "rgb pixel values current", current_image_np[closest_pixel_coords_orig[0], closest_pixel_coords_orig[1]], "mediam rgb original img", median_rgb_original)

                
                # find the pixels in the previous image that are closest to the median rgb values
                # distance_previous = np.linalg.norm(current_image_flatten - median_rgb_previous, axis=1)
                # distance_previous = np.linalg.norm(roi_flatten - median_rgb_previous, axis=1)
                distance_previous = np.linalg.norm(roi_hsi_flatten - median_hsi_previous, axis=1)
                closest_pixel_idx_previous = np.argmin(distance_previous)
                # closest_pixel_coords_previous = np.unravel_index(closest_pixel_idx_previous, current_image_np.shape[:2])
                # print("Closest pixel coordinates in the previous image:", closest_pixel_coords_previous,  np.min(distance_original), "rgb pixel values current", current_image_np[closest_pixel_coords_previous[0], closest_pixel_coords_previous[1]], "mediam rgb previous img", median_rgb_previous)
                
                closest_pixel_coords_previous = np.unravel_index(closest_pixel_idx_previous, roi_hsi_flatten.shape[:2])
                closest_pixel_coords_previous = (closest_pixel_coords_previous[0] + y_min, closest_pixel_coords_previous[1] + x_min)

                print("Closest pixel coordinates in the previous image within the bounding box:", closest_pixel_coords_previous, "difference amount", np.min(distance_previous), "rgb pixel values current", current_image_np[closest_pixel_coords_previous[0], closest_pixel_coords_previous[1]], "mediam rgb previous img", median_rgb_previous)


                # check to see if either of the closet pixels are within the previous object box plus a threshold
                # if they are, add a new point to the tracker
                
                
                if np.min(distance_original) < norm_threshold and np.min(distance_original ) < np.min(distance_previous):
                    print("Adding new point to the tracker. based on original image")
                    best_match_coords = closest_pixel_coords_orig
                elif np.min(distance_previous) < norm_threshold and np.min(distance_previous) < np.min(distance_original):
                    print("Adding new point to the tracker. based on previous image")
                    best_match_coords = closest_pixel_coords_previous
                else:
                    print("No valid point found within the threshold.")
                    # sys.exit()
                    continue
                    
                frame_idx , out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=out_frame_idx,
                    obj_id=ann_obj_id,
                    points= np.array([best_match_coords], dtype=np.float32),
                    labels= np.array([1], np.int32), # 1 is positive, 0 is negative 
                    # box = box,
                )

                
                
                # if is_within_box(closest_pixel_coords_previous, x_min, y_min, x_max, y_max):
                #     print("Adding new point to the tracker. based on previous image")
                # elif is_within_box(closest_pixel_coords_orig, x_min, y_min, x_max, y_max): 
                #     print("Adding new point to the tracker. based original image")   
                # else:
                #     print("No valid point found within the threshold.")

                # sys.exit()
                
                # plot original image, previous image, current image
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                ax[0].imshow(original_image)
                show_mask(binary_mask, ax[0], obj_id=ann_obj_id)  # Use show_mask function
                ax[0].set_title("Original Image")
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
                ax[0].add_patch(rect)

                ax[1].imshow(previous_image)
                show_mask(previous_mask, ax[1], obj_id=ann_obj_id)  # Use show_mask function
                ax[1].set_title("Previous Image")
                prev_box = output_track_boxes[-1]
                rect = patches.Rectangle((prev_box[0], prev_box[1]), prev_box[2] - prev_box[0], prev_box[3] - prev_box[1], linewidth=2, edgecolor='r', facecolor='none')
                ax[1].add_patch(rect)


                ax[2].imshow(current_image)
                ax[2].scatter(best_match_coords[1], best_match_coords[0], color='red', marker='x', s=100)  # Add marker for best match
                ax[2].set_title("Current Image")
                
                
                
                # plt.show()
                previous_mask = result_mask*0
                all_points_annotated = False
                current_reprompts+=1
                if current_reprompts < max_reprompts:
                    
                    break
                
                
                # sys.exit()
                
            

        # # render the segmentation results every few frames
        # vis_frame_stride = 30
        # plt.close("all")
        # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        #     plt.figure(figsize=(6, 4))
        #     plt.title(f"frame {out_frame_idx}")
        #     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                
        #     plt.show()
            



            
            
        # save results to result_output_name file
        if all_points_annotated:
            print(f"Saving results to {result_output_name}")
            with open(result_output_name, 'w') as f:
                for box in output_track_boxes:
                    x_min, y_min, x_max, y_max = box
                    f.write(f"{int(x_min)}\t{int(y_min)}\t{int(x_max-x_min)}\t{int(y_max-y_min)}\n")
            
print("Done!")