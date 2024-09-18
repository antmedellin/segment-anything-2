import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
import sys
import matplotlib.patches as patches
import math
from sklearn.cluster import SpectralClustering, DBSCAN
from matplotlib.colors import ListedColormap
from skimage.transform import resize


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
    
def show_mask_new(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)
    
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask_new(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
            
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

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
    
    
    
    
    # plt.imshow(mask)
    # # plt.imshow(masked_pixels)
    # plt.show()
    
    
    # Perform spectral clustering on these pixels
    n_clusters = 8  # Number of clusters
    # clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr', random_state=0)
    clustering = DBSCAN(eps=6, min_samples=5) # eps is max euclidean distance between samples 
    clusters = clustering.fit_predict(masked_pixels)
    
    # print(clusters)
    # print(masked_pixels.shape, clusters.shape)
    # print(image_np.shape, mask.shape)
    
    # Create an empty image to store the colored mask
    colored_mask = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)
    
    # Define a color map for the clusters
    # colors = plt.cm.get_cmap('tab10', n_clusters)
    base_cmap = plt.colormaps.get_cmap('tab10')
    colors = ListedColormap(base_cmap(np.linspace(0, 1, n_clusters)))
    
    # Color the mask pixels according to their cluster
    mask_indices = np.argwhere(mask)
    for idx, cluster in zip(mask_indices, clusters):
        if cluster == -1:
            continue  # Skip the -1 cluster
        cluster_color = (np.array(colors(cluster)[:3]) * 255).astype(int)
        colored_mask[idx[0], idx[1]] = cluster_color
    
    
    # Find the largest cluster
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    largest_cluster = max(cluster_sizes, key=lambda k: cluster_sizes[k] if k != -1 else -1)
    
    # Extract the pixels belonging to the largest cluster
    largest_cluster_indices = mask_indices[clusters == largest_cluster]
    largest_cluster_pixels = masked_pixels[clusters == largest_cluster]
    
    # Calculate the centroid of the largest cluster
    # centroid = np.mean(largest_cluster_indices, axis=0).astype(int)
    centroid_feature_space = np.mean(largest_cluster_pixels, axis=0).astype(int)
    # Find the pixel in the image space closest to the centroid in the feature space
    distances = np.linalg.norm(largest_cluster_pixels - centroid_feature_space, axis=1)
    closest_pixel_index = np.argmin(distances)
    centroid = largest_cluster_indices[closest_pixel_index]
    
    # print(f"Centroid of the largest cluster: {centroid}")
    centroid_pixel_values = image_np[centroid[0], centroid[1]]
    # print(f"Pixel values at the centroid: {centroid_pixel_values}")
    
    # # Display the original mask and the colored mask
    # plt.figure(figsize=(15, 7))
    # plt.subplot(1, 2, 1)
    # plt.title('Original Mask')
    # plt.imshow(mask, cmap='gray')
    
    # plt.subplot(1, 2, 2)
    # plt.title('Colored Mask by Clusters')
    # plt.imshow(colored_mask)
    # plt.scatter(centroid[1], centroid[0], color='red', marker='x')  # Mark the centroid
    # # plt.show()


    # number of mask pixels that are not zero and compare to mask non zero pixels
    # print("masked pixel size", masked_pixels.shape)
    # print("mask size", np.argwhere(mask).shape)
    
    # Calculate the median RGB values
    # median_rgb = np.median(masked_pixels, axis=0)
    median_rgb = np.mean(masked_pixels, axis=0)
    
    # return median_rgb
    return centroid_pixel_values, centroid


def spectral_angle_mapper(cur_pixel, ref_spec):
        # print(cur_pixel.shape, ref_spec.shape)
        assert cur_pixel.shape == ref_spec.shape
        dot_product = np.dot( cur_pixel, ref_spec)
        norm_spectral=np.linalg.norm(cur_pixel)
        norm_ref=np.linalg.norm(ref_spec)
        denom = norm_spectral * norm_ref
        if denom == 0:
            return 3.14
        alpha_rad=math.acos(dot_product / (denom)); 
        return alpha_rad*255/3.1416 
    
def spectral_similarity_analysis(test_spectrums, ref_spectrum):
    
    # Calculate the spectral angle between the test spectrum and the reference spectrum
    #replace with the algorithm that is desired to be used 
    # print(test_spectrums.shape, ref_spectrum.shape)
    
    # make sure the size of the arrays are compatible
    
    # print(test_spectrums.shape, ref_spectrum.shape)
    
    assert test_spectrums.shape[1] == ref_spectrum.shape[0]
    
    spectral_angles = []
    
    for i in range(test_spectrums.shape[0]):
        test_spectrum = test_spectrums[i]
        
        #perform spectral similarity analysis here
        result = spectral_angle_mapper(test_spectrum, ref_spectrum)
        
        spectral_angles.append(result)


    # print(spectral_angles)
    spectral_angles = np.array(spectral_angles)
    # return 1d array of spectral angles
    return spectral_angles

class ExtendedKalmanFilter:
    def __init__(self, dt, state_dim, measurement_dim):
        self.dt = dt  # Time step
        self.state_dim = state_dim  # Dimension of the state vector
        self.measurement_dim = measurement_dim  # Dimension of the measurement vector

        # State vector [x, y, vx, vy]
        self.x = np.zeros((state_dim, 1))

        # State covariance matrix
        self.P = np.eye(state_dim)

        # Process noise covariance matrix
        self.Q = np.eye(state_dim)

        # Measurement noise covariance matrix
        self.R = np.eye(measurement_dim)

        # State transition matrix
        self.F = np.eye(state_dim)
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt

        # Measurement matrix
        self.H = np.zeros((measurement_dim, state_dim))
        self.H[0, 0] = 1
        self.H[1, 1] = 1

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.F, self.x)

        # Predict the state covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Measurement residual
        y = z - np.dot(self.H, self.x)

        # Measurement residual covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update the state estimate
        self.x = self.x + np.dot(K, y)

        # Update the state covariance
        I = np.eye(self.state_dim)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def get_center_of_mask(mask):
    # Calculate the center of the mask
    indices = np.argwhere(mask)
    center = np.mean(indices, axis=0)
    return center
# choose model to use

# can overload 24 gb vram, which causes it to be slow 
# sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"

# sam2_checkpoint = "checkpoints/sam2_hiera_base_plus.pt"
# model_cfg = "sam2_hiera_b+.yaml"



sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

# Define the threshold to exapand box by
threshold = 5 # expand box by this threshold for each frame since previous success
norm_threshold = 15 # spectral similarity allowed 
min_obj_area = 50 



dt = 1.0  # Time step
state_dim = 4  # [x, y, vx, vy]
measurement_dim = 2  # [x, y]



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
# relative_path_to_hsi = "../../../HSI-NIR"

# results_output_dir = "notebooks/results/training/HSI-VIS-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/training/HSI-VIS-FalseColor"
# relative_path_to_hsi = "../../HSI-VIS"

# results_output_dir = "notebooks/results/training/HSI-RedNIR-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/training/HSI-RedNIR-FalseColor"
# relative_path_to_hsi = "../../HSI-RedNIR"

results_output_dir = "notebooks/results/validation/HSI-NIR-FalseColor"
video_base_dir = "../hsi_tracking/datasets/validation/HSI-NIR-FalseColor"
relative_path_to_hsi = "../../../HSI-NIR"

# results_output_dir = "notebooks/results/validation/HSI-VIS-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/validation/HSI-VIS-FalseColor"
# relative_path_to_hsi = "../../../HSI-VIS"

# results_output_dir = "notebooks/results/validation/HSI-RedNIR-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/validation/HSI-RedNIR-FalseColor"
# relative_path_to_hsi = "../../HSI-RedNIR"


# results_output_dir = "notebooks/results/development"
# # video_base_dir = "../hsi_tracking/datasets/validation/HSI-VIS-FalseColor/S_runner1"
# # video_base_dir = "../hsi_tracking/datasets/validation/HSI-VIS-FalseColor/oranges5"
# # video_base_dir = "../hsi_tracking/datasets/validation/HSI-VIS-FalseColor/droneshow2"
# video_base_dir = "../hsi_tracking/datasets/training/HSI-VIS-FalseColor/car6"
# relative_path_to_hsi = "../../../HSI-VIS"


# num_bands = 16 # 1, 25 (16 vis, 25 nir, 16 rednir (last column all zeros))
num_bands = 16

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

sam2_image = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)

# roi_mask_generator = SAM2AutomaticMaskGenerator(
#                         model = sam2_image,
#                         points_per_side=20,
#                         points_per_batch=128,
#                         pred_iou_thresh=0.60,
#                         stability_score_thresh=0.90,
#                         stability_score_offset=0.7,
#                         crop_n_layers=1,
#                         box_nms_thresh=0.60,
#                         crop_n_points_downscale_factor=2,
#                         min_mask_region_area=min_obj_area,
#                         use_m2m=False,
#                         multimask_output=False,
#                         )

# Initialize an empty set to store directories where no objects are detected
no_objects_detected_dirs = set()
    
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
        
        ekf = ExtendedKalmanFilter(dt, state_dim, measurement_dim)
        # Process noise covariance
        ekf.Q = np.diag([0.5, 0.5, 0.1, 0.1])
        # Measurement noise covariance
        ekf.R = np.diag([20,20])
        # Initial state [x, y, vx, vy]
        initial_center = get_center_of_mask(binary_mask)
        ekf.x[:2] = initial_center.reshape(-1, 1)
        
        video_segments = {}  # video_segments contains the per-frame segmentation results
        output_track_boxes = []  # output_track_boxes contains the per-frame tracking results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            try: 
                result_mask = np.squeeze((out_mask_logits[0] > 0.0).cpu().numpy()) # squeeze to remove the extra dimension since only doing 1 object
                # print(result_mask.shape, np.max(np.squeeze((out_mask_logits[0] ).cpu().numpy())), np.min(np.squeeze((out_mask_logits[0] ).cpu().numpy())))
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
                    
                    # Predicted center of the mask
                    ekf.predict()
                    predicted_center = ekf.x[:2].flatten()

                    new_center = get_center_of_mask(result_mask)
                    ekf.update(new_center.reshape(-1, 1))
                    # print(f"Predicted center: {predicted_center}, actual center: {new_center}")

    
                else:
                    raise Exception("No object detected")
                
                # if out_frame_idx >=2:
                #     raise Exception("testing")
                
            except:
                
                if out_frame_idx == len(frame_names) - 1:
                    all_points_annotated = True
                    break
                
                # no object detected, append the previous result if available
                if output_track_boxes:
                    output_track_boxes.append(output_track_boxes[-1])
                else:
                    output_track_boxes.append([0, 0, 0, 0])
                # print(f"No object detected in frame {out_frame_idx}")
                
                
                # Add the current video sub-directory to the set
                no_objects_detected_dirs.add(current_video_sub_dir)
                
                
                if out_frame_idx <= reprompt_frame_idx:
                    all_points_annotated = False
                    continue
                else:
                    reprompt_frame_idx = out_frame_idx
                    # print("hi2", current_reprompts, max_reprompts)
                    if current_reprompts == max_reprompts:
                        continue
                    
                all_points_annotated = False
                current_reprompts+=1
                    
                # print(f"Re-prompting for frame {out_frame_idx} and current reprompt frame {reprompt_frame_idx}")
                    
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
                    
                    hsi_original_image = X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,frame_names[0].split(".")[0]+".png"))) , bandNumber=num_bands)
                    hsi_previous_image = X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,frame_names[previous_success_index].split(".")[0]+".png"))) , bandNumber=num_bands)
                    hsi_current_image =  X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,frame_names[out_frame_idx].split(".")[0]+".png"))) , bandNumber=num_bands)
                
                except:
                    hsi_original_image = X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,current_video_sub_dir,frame_names[0].split(".")[0]+".png"))) , bandNumber=num_bands)
                    hsi_previous_image = X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,current_video_sub_dir,frame_names[previous_success_index].split(".")[0]+".png"))) , bandNumber=num_bands)
                    hsi_current_image =  X2Cube(np.array(Image.open(os.path.join(video_dir, relative_path_to_hsi,current_video_sub_dir,current_video_sub_dir,frame_names[out_frame_idx].split(".")[0]+".png"))) , bandNumber=num_bands)
                
                
                
                # resize the hsi images to match the rgb images using np.resize
                hsi_original_image = resize(hsi_original_image, (original_image.size[1], original_image.size[0], num_bands))
                hsi_previous_image = resize(hsi_previous_image, (previous_image.size[1], previous_image.size[0], num_bands))
                hsi_current_image = resize(hsi_current_image, (current_image.size[1], current_image.size[0], num_bands))
                
                
                # # create a figure that displays all three images as subplots, use the first channel layer of each image 
                # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                # ax[0].imshow(hsi_original_image[:, :, 0], cmap='gray')
                # ax[0].set_title("Original Image")
                # ax[1].imshow(hsi_previous_image[:, :, 0], cmap='gray')
                # ax[1].set_title("Previous Image")
                # ax[2].imshow(hsi_current_image[:, :, 0], cmap='gray')
                # ax[2].set_title("Current Image")
                
                
                
                
                
                
                
                

                # print("original object box:", box)
                # print("previous object box:", output_track_boxes[-1])
                # print("original object mask shape:", binary_mask.shape)
                # print("previous object mask shape:", previous_mask.shape)
                # print("original image size:", original_image.size)
                # print("previous image size:", previous_image.size)
                # print("current image size:", current_image.size)
                
                
                # print("shape of hsi images", hsi_original_image.shape, hsi_previous_image.shape, hsi_current_image.shape, "shape of masks", binary_mask.shape, previous_mask.shape, "shape of rgb images", original_image.size, previous_image.size, current_image.size)
                
                # Get the median RGB values for the original and previous images
                # median_rgb_original = get_median_rgb(original_image, binary_mask)
                # median_rgb_previous = get_median_rgb(previous_image, previous_mask)
                median_hsi_original, orig_coords = get_median_rgb(hsi_original_image, binary_mask)
                median_hsi_previous, prev_corrds = get_median_rgb(hsi_previous_image, previous_mask)

                # print(median_hsi_original.shape)
                # print("Median RGB values in the original image:", median_rgb_original, median_hsi_original)
                # print("Median RGB values in the previous image:", median_rgb_previous, median_hsi_previous)
                # print("output logits shape", out_mask_logits[0].shape)
                # print("hsi image shape", hsi_current_image.shape)            
                
                # find the pixels in the current image that are closest to the median rgb values
                # find the closest pixel to the median rgb values
                current_image_np = np.array(current_image)
                
                
                

                # Check if the closest pixels are within the previous object box plus a threshold
                prev_box = output_track_boxes[-1]
                x_min, y_min, x_max, y_max = prev_box
                # multiply threshold be number of frames since last success
                x_min -= threshold * (out_frame_idx - previous_success_index)
                y_min -= threshold * (out_frame_idx - previous_success_index)
                x_max += threshold * (out_frame_idx - previous_success_index)
                y_max += threshold * (out_frame_idx - previous_success_index)
                
                # make sure the box is within the image size
                x_min = int(max(x_min, 0))
                y_min = int(max(y_min, 0))
                x_max = int(min(x_max, current_image_np.shape[1]))
                y_max = int(min(y_max, current_image_np.shape[0]))
                

                # Extract the ROI from the current image
                roi_rgb = current_image_np[y_min:y_max, x_min:x_max]
                roi_hsi = hsi_current_image[y_min:y_max, x_min:x_max]
                roi_logits = np.squeeze((out_mask_logits[0]).cpu().numpy())[y_min:y_max, x_min:x_max]
                
                # print(roi_logits)
                # print(roi_hsi.shape, roi_logits.shape, np.max(roi_logits), np.min(roi_logits))
                
                # # print("ROI shape initial:", roi_hsi.shape)

                # # Flatten the ROI for distance calculation
                # # roi_flatten = roi.reshape(-1, 3)
                # roi_hsi_flatten = roi_hsi.reshape(-1, 16)


                # # flatten the image array
                # # current_image_flatten = current_image_np.reshape(-1, 3)
                
                # # calculate the distance between the median rgb values and the pixels in the current image
                # # distance_original = np.linalg.norm(current_image_flatten - median_rgb_original, axis=1)
                # # distance_original = np.linalg.norm(roi_flatten - median_rgb_original, axis=1)
                # # distance_original = np.linalg.norm(roi_hsi_flatten - median_hsi_original, axis=1)

                # #use spectral similarity instead of euclidean distance
                
                # # create a new function to take in mxc and 1xc and return mx1 which has the lowest dissimilarity
                
                # # print(distance_original) 
                # # print(len(distance_original), len(roi_hsi_flatten))
                
                # distance_original = spectral_similarity_analysis(roi_hsi_flatten, median_hsi_original)

                # # print("spectral analysis result", np.argmin(tmp_result),np.min(tmp_result),"lin alg norm result", np.argmin(distance_original),  np.min(distance_original))
                # # print(tmp_result.shape, distance_original.shape)
                
                
                # # sys.exit()
                # # find the index of the pixel with the smallest distance
                # # closest_pixel_idx_original = np.argmin(distance_original)
                # closest_pixel_idx_original = np.argmin(distance_original)
                # # convert the index to 2D coordinates
                # # closest_pixel_coords_orig = np.unravel_index(closest_pixel_idx_original, current_image_np.shape[:2])
                # # print("Closest pixel coordinates in the original image:", closest_pixel_coords_orig, np.min(distance_original), "rgb pixel values current", current_image_np[closest_pixel_coords_orig[0], closest_pixel_coords_orig[1]], "mediam rgb original img", median_rgb_original) 
                # # closest_pixel_coords_orig = np.unravel_index(closest_pixel_idx_original, roi.shape[:2])
                # # closest_pixel_coords_orig = (closest_pixel_coords_orig[0] + y_min, closest_pixel_coords_orig[1] + x_min)
                
                
                # # not sure if the unravel is correct right now
                # closest_pixel_coords_orig = np.unravel_index(closest_pixel_idx_original, roi_hsi.shape[:2])
                
                # # print("roi unravel shape", closest_pixel_coords_orig)
                # # sys.exit()
                
                # closest_pixel_coords_orig = (closest_pixel_coords_orig[0] + y_min, closest_pixel_coords_orig[1] + x_min)

                # print("Closest pixel coordinates in the original image within the bounding box:", closest_pixel_coords_orig, "difference amount", np.min(distance_original), "rgb pixel values current", current_image_np[closest_pixel_coords_orig[0], closest_pixel_coords_orig[1]])

                
                # # find the pixels in the previous image that are closest to the median rgb values
                # # distance_previous = np.linalg.norm(current_image_flatten - median_rgb_previous, axis=1)
                # # distance_previous = np.linalg.norm(roi_flatten - median_rgb_previous, axis=1)
                # # distance_previous = np.linalg.norm(roi_hsi_flatten - median_hsi_previous, axis=1)
                # # closest_pixel_coords_previous = np.unravel_index(closest_pixel_idx_previous, current_image_np.shape[:2])
                # # print("Closest pixel coordinates in the previous image:", closest_pixel_coords_previous,  np.min(distance_original), "rgb pixel values current", current_image_np[closest_pixel_coords_previous[0], closest_pixel_coords_previous[1]], "mediam rgb previous img", median_rgb_previous)
                # distance_previous = spectral_similarity_analysis(roi_hsi_flatten, median_hsi_previous)
                
                # closest_pixel_idx_previous = np.argmin(distance_previous)

                
                # closest_pixel_coords_previous = np.unravel_index(closest_pixel_idx_previous, roi_hsi.shape[:2])
                # closest_pixel_coords_previous = (closest_pixel_coords_previous[0] + y_min, closest_pixel_coords_previous[1] + x_min)

                # print("Closest pixel coordinates in the previous image within the bounding box:", closest_pixel_coords_previous, "difference amount", np.min(distance_previous), "rgb pixel values current", current_image_np[closest_pixel_coords_previous[0], closest_pixel_coords_previous[1]])

                """
                roi_image =  current_image_np[y_min:y_max, x_min:x_max]
                roi_mask_results = roi_mask_generator.generate(roi_image)
                
                # for each mask - cluster, compare the largest spectrally similar cluster centroid and use that as the point to add to the tracker 
                
                # spectral similarity analysis function for each cluster centroid - original image and previopus image 
                # reprompt with the closest centroid to the previous object centroid
                # largest to smallest mask 
                # biggest cluster centroid 

                # step 1 sort the masks by size 
                # step 2 get the centroid of the largest mask
                # step 3 spectral analysis until a match is found
                
                # sorting by area, can also sort by mask quality 
                sorted_anns = sorted(roi_mask_results, key=(lambda x: x['area']), reverse=True)
                # print("Number of masks", len(sorted_anns))
                # print(sorted_anns)
                # print("original image shape", hsi_original_image.shape, binary_mask.shape)
                added_point = False
                for ann in sorted_anns: 
                    
                    if ann['area'] < min_obj_area:
                        continue
                    
                    # print("mask segmentation shape", ann['segmentation'].shape)
                    ann_result_spectrum, centroid_coords = get_median_rgb(roi_hsi, ann['segmentation'])
                    # print(ann_result_spectrum.shape)
                    
                    # get corrdinates of where ann_result_spectrum is located in the image
                    
                    
                    distance_original_ann = spectral_angle_mapper(ann_result_spectrum, median_hsi_original)
                    distance_previous_ann = spectral_angle_mapper(ann_result_spectrum, median_hsi_previous)
                    # print("Spectral similarity analysis for mask", ann['segmentation'].shape,(distance_original_ann),(distance_previous_ann))
                    
                    if distance_original_ann < norm_threshold or distance_previous_ann < norm_threshold:

                        print("Adding new point to the tracker. based on mask with area", ann['area'])
                        # convert from roi coordinates to image coordinates
                        best_match_coords = (centroid_coords[0] + y_min, centroid_coords[1] + x_min)
                        added_point = True
                        # break from for loop
                        break
                    
                if added_point is False:
                    print("No valid point found within the threshold. scores were", distance_original_ann, distance_previous_ann)
                    # sys.exit()
                    continue
                
                # sys.exit()
                
                
                plt.figure()
                plt.imshow(roi_image)
                show_anns(roi_mask_results)
                # plt.show()
                # print(roi_image.shape, "displayed image")

                """



                # check to see if either of the closet pixels are within the previous object box plus a threshold
                # if they are, add a new point to the tracker
                
                
                # if np.min(distance_original) < norm_threshold and np.min(distance_original ) < np.min(distance_previous):
                #     print("Adding new point to the tracker. based on original image")
                #     best_match_coords = closest_pixel_coords_orig
                # elif np.min(distance_previous) < norm_threshold and np.min(distance_previous) < np.min(distance_original):
                #     print("Adding new point to the tracker. based on previous image")
                #     best_match_coords = closest_pixel_coords_previous
                # else:
                #     print("No valid point found within the threshold.")
                #     # sys.exit()
                #     continue
                
                ekf.predict()
                best_match_coords = ekf.x[:2].flatten()
                best_match_coords = np.round(best_match_coords).astype(int)
                
                
                
                
                
                # make sure the coordinates are within the image size
                best_match_coords[0] = int(max(best_match_coords[0], 0))
                best_match_coords[1] = int(max(best_match_coords[1], 0))
                best_match_coords[0] = int(min(best_match_coords[0], current_image_np.shape[0] - 1))
                best_match_coords[1] = int(min(best_match_coords[1], current_image_np.shape[1] - 1))
                
                
                
                """
                # get object mask for the corresponding object that is prompted with best_match_coords
                # cluster the results and if spectral match then add it 
                # do multi mask, pass each to see if it is a match
                image_predictor = SAM2ImagePredictor(sam2_image)
                image_predictor.set_image(current_image)
                input_point = np.array([[best_match_coords[1] , best_match_coords[0]]])
                input_label = np.array([1])
                
                # print(input_point, input_label, best_match_coords[0], best_match_coords[1])
                masks, scores, logits = image_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]
                
                # plt.figure(figsize=(10, 10))
                # plt.imshow(current_image)
                # show_points(input_point, input_label, plt.gca())
                # plt.axis('on')
                # plt.show()  
                # show_masks(current_image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
                add_new_point = False
                for mask in masks :
                    # print(mask.shape, hsi_current_image.shape)
                    mask = mask.astype(bool)
                    # print(np.min(mask), np.max(mask))
                    
                    object_spectra, centroid_coords = get_median_rgb(hsi_current_image, mask)
                    
                    similarity_result_orig = spectral_angle_mapper(object_spectra, median_hsi_original)
                    similarity_result_prev = spectral_angle_mapper(object_spectra, median_hsi_previous)
                    
                    # print(similarity_result_orig, similarity_result_prev)
                    if similarity_result_orig <= norm_threshold or similarity_result_prev <= norm_threshold:
                        print("Adding new point to the tracker. based on spectral similarity")
                        add_new_point = True
                        # show_masks(current_image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
                        break
                    
                if add_new_point is False:
                    print("No valid point found within the threshold.")
                    continue
                """
                
                
                
                
                frame_idx , out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=out_frame_idx,
                    obj_id=ann_obj_id,
                    points= np.array([best_match_coords], dtype=np.float32),

                    labels= np.array([1], np.int32), # 1 is positive, 0 is negative 
                    # box = box,
                )

                # sys.exit()

                
                # if is_within_box(closest_pixel_coords_previous, x_min, y_min, x_max, y_max):
                #     print("Adding new point to the tracker. based on previous image")
                # elif is_within_box(closest_pixel_coords_orig, x_min, y_min, x_max, y_max): 
                #     print("Adding new point to the tracker. based original image")   
                # else:
                #     print("No valid point found within the threshold.")

                # sys.exit()
                
                # plot original image, previous image, current image
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                ax[0].imshow(original_image)
                show_mask(binary_mask, ax[0], obj_id=ann_obj_id)  # Use show_mask function
                ax[0].set_title("Original Image")
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
                ax[0].add_patch(rect)
                ax[0].scatter(orig_coords[1], orig_coords[0], color='red', marker='x', s=100)


                ax[1].imshow(previous_image)
                show_mask(previous_mask, ax[1], obj_id=ann_obj_id)  # Use show_mask function
                ax[1].set_title("Previous Image")
                prev_box = output_track_boxes[-1]
                rect = patches.Rectangle((prev_box[0], prev_box[1]), prev_box[2] - prev_box[0], prev_box[3] - prev_box[1], linewidth=2, edgecolor='r', facecolor='none')
                ax[1].add_patch(rect)
                ax[1].scatter(prev_corrds[1], prev_corrds[0], color='red', marker='x', s=100)


                
                    
                ax[2].imshow(current_image)
                # plot y,x instead of 
                ax[2].scatter(best_match_coords[1], best_match_coords[0], color='red', marker='x', s=100)  # Add marker for best match
                # ax[2].scatter(predicted_center[1], predicted_center[0], color='blue', marker='x', s=100)
                # ax[2].scatter(350,169, color='green', marker='x', s=100)  # Add marker for best match
                ax[2].set_title("Current Image")
                
                # Add a rectangle for the ROI on the current frame
                # roi_rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='b', facecolor='none')
                # ax[2].add_patch(roi_rect)
                
                # Plot the median HSI values
                ax[3].plot(median_hsi_original, label='Spectrum of Cluster Centroid - Original Image')
                ax[3].plot(median_hsi_previous, label='Spectrum of Cluster Centroid - Previous Image')
                # Extract the pixel values at the best match coordinates from the current image
                best_match_pixel_values = hsi_current_image[best_match_coords[0], best_match_coords[1]]  
                ax[3].plot(best_match_pixel_values, label='Predicted Object Location', linestyle='--')    
                    
                # test_pixel = hsi_current_image[169,350]
                # ax[3].plot(test_pixel, label='Test Pixel Values', linestyle='--')


                ax[3].set_title("Median HSI Values")
                ax[3].legend()
                
                # plt.show()
                
                
                
               
               
                
                previous_mask = result_mask*0
                
                # if current_reprompts < max_reprompts:
                    
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
                
                # plt.show()
                
                
                
            
            
        # save results to result_output_name file
        if all_points_annotated:
            print(f"Saving results to {result_output_name}")
            with open(result_output_name, 'w') as f:
                for box in output_track_boxes:
                    x_min, y_min, x_max, y_max = box
                    f.write(f"{int(x_min)}\t{int(y_min)}\t{int(x_max-x_min)}\t{int(y_max-y_min)}\n")
            
print("Done!")
print(video_base_dir)
print("Number of directories with tracks lost:", len(no_objects_detected_dirs))
print("Directories with tracks lost:", no_objects_detected_dirs)
# plt.show()