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
from transformers import pipeline
# pip install -q -U transformers

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

class WidthHeightKalmanFilter:
    def __init__(self, dt, state_dim=2, measurement_dim=2):
        self.dt = dt  # Time step
        self.state_dim = state_dim  # Dimension of the state vector (width, height)
        self.measurement_dim = measurement_dim  # Dimension of the measurement vector (width, height)

        # State vector [width, height]
        self.x = np.zeros((state_dim, 1))

        # State covariance matrix
        self.P = np.eye(state_dim)

        # Process noise covariance matrix
        self.Q = np.eye(state_dim)

        # Measurement noise covariance matrix
        self.R = np.eye(measurement_dim)

        # State transition matrix (identity matrix since we are not modeling dynamics)
        self.F = np.eye(state_dim)

        # Measurement matrix (identity matrix since we are directly measuring width and height)
        self.H = np.eye(measurement_dim)

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

# choose model to use

# can overload 24 gb vram, which causes it to be slow 
# sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"

# sam2_checkpoint = "checkpoints/sam2_hiera_base_plus.pt"
# model_cfg = "sam2_hiera_b+.yaml"



sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

# Define the threshold to exapand box by
threshold = 3 # allowable change in box coordinates to prevent large changes in results, expand box by this threshold for each frame since previous success
norm_threshold = 15 # spectral similarity allowed 
min_obj_area = 50 


# kalman filter 
dt = 1.0  # Time step
state_dim = 4  # [x, y, vx, vy]
measurement_dim = 2  # [x, y]


# depth estimation
checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
with torch.autocast("cuda", dtype=torch.float16):
    pipe = pipeline("depth-estimation", model=checkpoint, device=device)

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

# results_output_dir = "notebooks/results/validation/HSI-NIR-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/validation/HSI-NIR-FalseColor"
# relative_path_to_hsi = "../../../HSI-NIR"

results_output_dir = "notebooks/results/validation/HSI-VIS-FalseColor"
video_base_dir = "../hsi_tracking/datasets/validation/HSI-VIS-FalseColor"
relative_path_to_hsi = "../../../HSI-VIS"

# results_output_dir = "notebooks/results/validation/HSI-RedNIR-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/validation/HSI-RedNIR-FalseColor"
# relative_path_to_hsi = "../../HSI-RedNIR"


# results_output_dir = "notebooks/results/development"
# # video_base_dir = "../hsi_tracking/datasets/validation/HSI-VIS-FalseColor/S_runner1"
# video_base_dir = "../hsi_tracking/datasets/validation/HSI-VIS-FalseColor/oranges5"
# # # video_base_dir = "../hsi_tracking/datasets/validation/HSI-VIS-FalseColor/droneshow2"
# # video_base_dir = "../hsi_tracking/datasets/training/HSI-VIS-FalseColor/car6"
# relative_path_to_hsi = "../../../HSI-VIS"
# # video_base_dir = "../hsi_tracking/datasets/validation/HSI-RedNIR-FalseColor/duck4"
# # relative_path_to_hsi = "../../HSI-RedNIR"

results_output_dir = "notebooks/results/ranking/HSI-NIR-FalseColor"
# video_base_dir = "../hsi_tracking/datasets/HSI-NIR-FalseColor"
video_base_dir = "../hsi_tracking/datasets/too_large"
relative_path_to_hsi = "../../../HSI-NIR"

# num_bands = 16 # 1, 25 (16 vis, 25 nir, 16 rednir (last column all zeros))
num_bands = 16

video_sub_dirs = os.listdir(video_base_dir)

# print(video_sub_dirs)



# start of code 

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
        # ground_truth_file = os.path.join(video_dir, "groundtruth_rect.txt")
        ground_truth_file = os.path.join(video_dir, "init_rect.txt")
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
    allowable_area_change = 0.15 # 5% change in area allowed
    while all_points_annotated is False:
        
        # keep track of the object width and height in the unoccluded frames
        object_ekf =  WidthHeightKalmanFilter(dt) # 2 states, each state is a 1d value
        # object_ekf.Q = np.diag([1,1])
        # object_ekf.R = np.diag([3,3])
        object_ekf.Q_width = np.diag([1, .5])
        object_ekf.Q_height = np.diag([1, .5])
        object_ekf.R_width = np.diag([10])
        object_ekf.R_height = np.diag([10])
        
        # get the width and height of the object using the bounding box 
        initial_width = box[2] - box[0]
        initial_height = box[3] - box[1]
        object_ekf.x[0] = initial_width
        object_ekf.x[1] = initial_height
        object_ekf.update(np.array([initial_width, initial_height]))
        # object_ekf.update(np.array([100,50]))
        # x[2] is the velocity in the width direction and x[3] is the velocity in the height direction
        # print(f"Initial width: {initial_width}, Initial height: {initial_height}")
        prev_depth = 0
        
        # keep track of center of modal mask, not center of object 
        modal_ekf = ExtendedKalmanFilter(dt, state_dim, measurement_dim)
        # Process noise covariance
        modal_ekf.Q = np.diag([1,1,.5,.5])
        # Measurement noise covariance
        modal_ekf.R = np.diag([10,10])
        # Initial state [x, y, vx, vy]
        initial_center = get_center_of_mask(binary_mask)
        # print(f"Initial center: {initial_center}")
        modal_ekf.x[:2] = initial_center.reshape(-1, 1)
        
        # Initialize the plot
        # fig_occ, ax_occ = plt.subplots(1, 4, figsize=(10, 5))


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
                    
                    
                    current_image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
                    current_image_np = np.array(current_image)
                    with torch.autocast("cuda", dtype=torch.float16):
                        predictions = pipe(current_image)
                        depth_prediction = predictions["depth"]
                    
                    
                    # below is for jumps in the object box size 
                    
                    frame_width = result_box[2] - result_box[0]
                    frame_height = result_box[3] - result_box[1]
                    
                    # get previous width and height
                    if out_frame_idx > 0:
                        prev_width = output_track_boxes[-1][2] - output_track_boxes[-1][0]
                        prev_height = output_track_boxes[-1][3] - output_track_boxes[-1][1]
                    else:
                        prev_width = initial_width
                        prev_height = initial_height
                    # prev_width = output_track_boxes[-1][2] - output_track_boxes[-1][0]
                    # prev_height = output_track_boxes[-1][3] - output_track_boxes[-1][1]
                    
                    # make sure that the mask is continuous (no discontinuities)
                    # print(np.sum(result_mask), frame_width * frame_height , "percentage of mask", np.sum(result_mask)/(frame_width * frame_height), "frame idx", out_frame_idx)
                    
                    object_ekf.predict()
                    predicted_width = object_ekf.x[0][0]
                    predicted_height = object_ekf.x[1][0]
                            
                    # if  np.sum(result_mask)/(frame_width * frame_height)  < 0.75 and out_frame_idx > 1:
                        
                    #     # Find the box that maximizes the logits score and is of the predicted width and height
                    #     max_score = -np.inf
                    #     best_box = result_box
                    #     for y in range(result_mask.shape[0] - int(predicted_height)):
                    #         for x in range(result_mask.shape[1] - int(predicted_width)):
                    #             candidate_box = result_mask[y:y + int(predicted_height), x:x + int(predicted_width)]
                    #             score = np.sum(out_mask_logits[0, y:y + int(predicted_height), x:x + int(predicted_width)].cpu().numpy())
                    #             if score > max_score:
                    #                 max_score = score
                    #                 best_box = [x, y, x + int(predicted_width), y + int(predicted_height)]

                    #     result_box = best_box
                    
                    
                    # if frame_width > 1.5 * prev_width or frame_height > 1.5 * prev_height:
                    if (frame_width > 3 * prev_width or frame_height > 3 * prev_height) and out_frame_idx > 1:
                    # if False:
                        # print(f"Previous width: {prev_width}, Previous height: {prev_height}")
                        # print(f"Current width: {frame_width}, Current height: {frame_height}")
                        # print(f"Previous box: {output_track_boxes[-1]}")
                        # print(f"Current box: {result_box}")
                        
                        # if out_frame_idx >1:
                        # print("Large jump in object size detected in frame", out_frame_idx)
                        # use the previous box
                        result_box = output_track_boxes[-1]
                        x_min, y_min, x_max, y_max = result_box
                        
                        
                        
                        # # Find the box that maximizes the logits score and is of the predicted width and height
                        # max_score = -np.inf
                        # best_box = result_box
                        # for y in range(result_mask.shape[0] - int(predicted_height)):
                        #     for x in range(result_mask.shape[1] - int(predicted_width)):
                        #         candidate_box = result_mask[y:y + int(predicted_height), x:x + int(predicted_width)]
                        #         score = np.sum(out_mask_logits[0, y:y + int(predicted_height), x:x + int(predicted_width)].cpu().numpy())
                        #         if score > max_score:
                        #             max_score = score
                        #             best_box = [x, y, x + int(predicted_width), y + int(predicted_height)]

                        # result_box = best_box
                        # x_min, y_min, x_max, y_max = result_box
                    
                    # print("starting results " , result_box)
                    
                    # below is for occlusions 
                    
                    
                    # Check if the closest pixels are within the previous object box plus a threshold
                    # prev_box = output_track_boxes[-1]
                    # x_min, y_min, x_max, y_max = prev_box
                    # multiply threshold be number of frames since last success
                    x_min -= threshold #* (out_frame_idx - previous_success_index)
                    y_min -= threshold #* (out_frame_idx - previous_success_index)
                    x_max += threshold #* (out_frame_idx - previous_success_index)
                    y_max += threshold #* (out_frame_idx - previous_success_index)
                    
                    # make sure the box is within the image size
                    x_min = int(max(x_min, 0))
                    y_min = int(max(y_min, 0))
                    x_max = int(min(x_max, current_image_np.shape[1]))
                    y_max = int(min(y_max, current_image_np.shape[0]))
                    
                    
                    depth_prediction_np = np.array(depth_prediction)
                    
                    # get max value in mask region 
                    mask_depth_full = depth_prediction_np * (result_mask) 
                    mask_depth_mask = mask_depth_full[y_min:y_max, x_min:x_max]
                    max_depth = np.max(mask_depth_mask)
                    # print(f"Max depth in roi: {max_depth}")
                    
                    cur_depth = max_depth
                    
                    # compare previous depth to current depth 
                    
                    # if cur_depth > prev_depth:
                    # add to kalman filter due to large depth change 
                    current_width = result_box[2] - result_box[0]
                    current_height = result_box[3] - result_box[1]
                    # object_ekf.update(np.array([current_width, current_height]))
                    
                    # prev_depth = cur_depth
                    # get max value in roi_depth that is not in the mask
                    mask_depth_full = depth_prediction_np * (-1+result_mask) * (-1)
                    mask_depth = mask_depth_full[y_min:y_max, x_min:x_max]
                    # mask_depth = mask_depth.flatten()
                    # mask_depth = mask_depth[mask_depth != 0]
                    max_mask_depth = np.max(mask_depth)
                    # print(f"Max depth not in mask: {max_mask_depth}")
                    
                    # larger values mean closer to the camera
                    # so if max_mask_depth is greater than max_depth, then likely that the object is occluded
                    if max_mask_depth > max_depth:
                    # if False:
                        # print("Object is possibly occluded in frame ", out_frame_idx)
                        
                        test_roi = current_image_np[y_min:y_max, x_min:x_max]
                        test_depth_roi = depth_prediction_np[y_min:y_max, x_min:x_max]
                        
                        
                        object_ekf.predict()
                        predicted_width = object_ekf.x[0][0]
                        predicted_height = object_ekf.x[1][0]
                        # print(f"Predicted width: {predicted_width}, height: {predicted_height}")
                        # print(f" kalman filter result {object_ekf.x}")
                        
                        # use the predicted width and height for amodal mask generation 
                        # find the top, bottom, left, or right side of the object that is closest in size to the predicted width and height and opposite side is occluded 
                        # from the center of the edge, propogate out the width and height of the object 
                        # use the predicted width and height to generate the amodal mask
                        
                        # if there is a full occlusion, then the object is not visible at all and the previous mask would be used. it would go into the exception case since non_zero_coords.size == 0 
                        
                        # if there is a partial occlusion, then the object is partially visible and the amodal mask would be generated
                        
                        actual_width = result_box[2] - result_box[0]
                        actual_height = result_box[3] - result_box[1]
                        
                        width_diff = actual_width - predicted_width
                        height_diff = actual_height - predicted_height

                        # only do this if the difference in width or height is greater than 2 pixels
                        # also make sure the depth hasnt changed too much
                        
                        # occluded only if the width and height are both less than the predicted width and height
                        # print(f"Predicted width: {object_ekf.x[0][0]}, height: {object_ekf.x[0][1]}, actual width: {current_width}, actual height: {current_height}")
                        
                        # this may be wrong here
                        if width_diff < -2 or height_diff < -2:
                            
                            # if width_diff < height_diff:
                            #     # closest_side = 'width'
                            #     closest_side = 'height'
                            # else:
                            #     # closest_side = 'height'
                            #     closest_side = 'width'
                            
                            
                            
                            
                                
                            # determine occluded side 
                            # use the location of the max value not in the mask to determine the occluded side
                            # take the side opposite the occluded side for propogation of the amodal mask 
                            
                            # Determine occluded side
                            max_mask_depth_coords = np.unravel_index(np.argmax(mask_depth, axis=None), mask_depth.shape)
                            max_y, max_x = max_mask_depth_coords
                            
                            # # Determine occluded side
                            # max_value = np.max(mask_depth)
                            # max_mask_depth_coords = np.argwhere(mask_depth == max_value)

                            # # Find the rightmost maximum value
                            # rightmost_max_coord = max_mask_depth_coords[np.argmax(max_mask_depth_coords[:, 1])]
                            # max_y, max_x = rightmost_max_coord
                            
                            
                            occluded_side_lateral = None
                            occluded_side_vertical = None

                            if width_diff < -2:
                                if max_x < (x_max - x_min) / 2:
                                    occluded_side_lateral = 'left'
                                    # occluded_side = 'bottom'
                                else:
                                    occluded_side_lateral = 'right'
                                    # occluded_side = 'top'
                            if height_diff < -2:
                                if max_y < (y_max - y_min) / 2:
                                    # occluded_side = 'left'
                                    occluded_side_vertical = 'top'
                                else:
                                    # occluded_side = 'right'
                                    occluded_side_vertical = 'bottom'
                            
                            
                            # [x_min, y_min, x_max, y_max]
                            # if closest_side == 'width':
                            if True:
                                new_y_max = y_max - threshold
                                new_y_min = y_min + threshold
                                new_x_min = x_min + threshold
                                new_x_max = x_max - threshold
                                
                                if occluded_side_vertical == 'top':
                                    # print("Top side is occluded")
                                    # take center location of bottom side and propogate out the width and height
                                    # new_y_min = y_min
                                    # new_y_max = y_min + predicted_height
                                    new_y_max = y_max
                                    new_y_min = y_max - predicted_height
                                    
                                    # new_x_min = x_min 
                                    # # new_x_max = x_max
                                    # new_x_max = x_min + predicted_width
                                # else:
                                elif occluded_side_vertical == 'bottom':
                                    # print("Bottom side is occluded")
                                    # new_y_max = y_max
                                    # new_y_min = y_max - predicted_height
                                    
                                    # new_x_min = x_min
                                    # new_x_max = x_max
                                    
                                    new_y_min = y_min
                                    new_y_max = y_min + predicted_height
                                    # new_x_min = x_min
                                    # # new_x_max = x_max
                                    # new_x_max = x_min + predicted_width
                                # else:
                                if occluded_side_lateral == 'left':
                                    # print("Left side is occluded")
                                    
                                    # new_y_min = y_min
                                    # # new_y_max = y_max
                                    # new_y_max = y_min + predicted_height
                                    
                                    new_x_max = x_max
                                    new_x_min = x_max - predicted_width
                                    
                                    # new_x_min = x_min
                                    # new_x_max = x_min + predicted_width
                                    
                                    
                                # else:
                                elif occluded_side_lateral == 'right':
                                    # print("Right side is occluded")
                                    
                                    # new_y_min = y_min
                                    # # new_y_max = y_max
                                    # new_y_max = y_min + predicted_height
                                    
                                    new_x_min = x_min
                                    new_x_max = x_min + predicted_width
                                    
                                    # new_x_max = x_max
                                    # new_x_min = x_max - predicted_width
                            
                            
                            # if the object box is not close to dimensions of the object then need to expand the box in the direction of the occluded side
                            
                            # if the other side is lacking in size, then expand the box in that direction nearest to the (corner is occluded)
                            
                            
                            # make sure the box is within the image size
                            new_x_min = int(max(new_x_min, 0))
                            new_y_min = int(max(new_y_min, 0))
                            new_x_max = int(min(new_x_max, current_image_np.shape[1]))
                            new_y_max = int(min(new_y_max, current_image_np.shape[0]))
                            
                            amodal_box = [new_x_min, new_y_min, new_x_max, new_y_max]
                            result_box = amodal_box
                            
                            
                            
                            # viz = True
                            viz = False
                            if viz:
                                
                                # plot both the roi and the depth roi
                                # fig, ax = plt.subplots(1, 4, figsize=(10, 5))
                                fig_occ, ax_occ = plt.subplots(1, 4, figsize=(10, 5))
                                ax_occ[0].imshow(test_roi)
                                ax_occ[0].set_title(f"ROI for frame {out_frame_idx}")
                                
                                ax_occ[1].imshow(test_depth_roi)
                                ax_occ[1].set_title("Depth ROI")
                                # plot an x for the max value not in the mask
                                ax_occ[1].scatter(max_x, max_y, color='red', marker='x')
                                
                                # plot the original image and the full current image with rectangle around mask region
                                ax_occ[2].imshow(current_image_np)
                                # ax_occ[2].set_title("Current Image")
                                # ax_occ[2].set_title(f"Current Image\nOccluded: {occluded_side}, Closest: {closest_side}")

                                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                                ax_occ[2].add_patch(rect)
                                
                                rect_amodal = patches.Rectangle((new_x_min, new_y_min), new_x_max - new_x_min, new_y_max - new_y_min, linewidth=2, edgecolor='g', facecolor='none')
                                ax_occ[2].add_patch(rect_amodal)
                                
                                box_base_gt = [int(x) for x in lines[out_frame_idx].strip().split()]
                                box_base_gt = np.array(box_base_gt, dtype=np.float32)
                                x_min_gt, y_min_gt, width_gt, height_gt = box_base_gt
                                x_max_gt = x_min_gt + width_gt
                                y_max_gt = y_min_gt + height_gt
                                rect_gt = patches.Rectangle((x_min_gt, y_min_gt), x_max_gt - x_min_gt, y_max_gt - y_min_gt, linewidth=2, edgecolor='b', facecolor='none')
                                ax_occ[2].add_patch(rect_gt)
                                
                                show_mask(result_mask, ax_occ[2], obj_id=ann_obj_id)  # Use show_mask function
                                
                                original_image = Image.open(os.path.join(video_dir, frame_names[0]))
                                ax_occ[3].imshow(original_image)
                                show_mask(binary_mask, ax_occ[3], obj_id=ann_obj_id)  # Use show_mask function
                                ax_occ[3].set_title("Original Image")
                                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
                                ax_occ[3].add_patch(rect)
                                # print(f"Predicted width: {object_ekf.x[0][0]}, height: {object_ekf.x[0][1]}, actual width: {current_width}, actual height: {current_height}")
                                # # print the box coordinates 
                                # print(f"Original box: {box}, Amodal box: {amodal_box}")
                                # print box width and height 
                            # print(f"Predicted width: {object_ekf.x[0][0]}, height: {object_ekf.x[0][1]}, actual width: {current_width}, actual height: {current_height}")
                            # print(f"amodal width: {result_box[2] - result_box[0]}, height: {result_box[3] - result_box[1]}")
                                
                              
                                # plt.show()
                                # plt.pause(1)
                                # plt.close(fig_occ)
                                
                        # # if the object is not occluded, then use the actual box
                        # elif width_diff > 0 and height_diff >0:    
                        #     current_width = result_box[2] - result_box[0]
                        #     current_height = result_box[3] - result_box[1]
                        #     object_ekf.update(np.array([current_width, current_height]))
                    else:
                        
                        
                        
                        
                        # print("Object is not occluded")
                        
                        
                        # if  np.sum(result_mask)/(frame_width * frame_height)  < 0.75:
                        
                        #     # Find the box that maximizes the logits score and is of the predicted width and height
                        #     max_score = -np.inf
                        #     best_box = result_box
                        #     for y in range(result_mask.shape[0] - int(predicted_height)):
                        #         for x in range(result_mask.shape[1] - int(predicted_width)):
                        #             candidate_box = result_mask[y:y + int(predicted_height), x:x + int(predicted_width)]
                        #             score = np.sum(out_mask_logits[0, y:y + int(predicted_height), x:x + int(predicted_width)].cpu().numpy())
                        #             if score > max_score:
                        #                 max_score = score
                        #                 best_box = [x, y, x + int(predicted_width), y + int(predicted_height)]

                        #     result_box = best_box
                        
                        
                        # kalman filter the height and width of the box 
                        current_width = result_box[2] - result_box[0]
                        current_height = result_box[3] - result_box[1]
                        
                        # # comment out for now but should propogate ekf properly 
                        # object_ekf.update(np.array([current_width, current_height]))
                        # object_ekf.predict()
                        # # print(f"Predicted width: {object_ekf.x[0][0]}, height: {object_ekf.x[0][1]}, actual width: {current_width}, actual height: {current_height}")
                        
                      
                    
                    # # plt.show()
                    # print("final results ", result_box)
                    if out_frame_idx == 0:
                        result_box = new_box
                        
                    output_track_boxes.append(result_box)
                    previous_mask = result_mask
                    all_points_annotated = True
                    previous_success_index = out_frame_idx
                    
                    # # Predicted center of the mask
                    modal_ekf.predict()
                    predicted_center = modal_ekf.x[:2].flatten()

                    # new_center = get_center_of_mask(result_mask)
                    
                    # get center of result_box 
                    new_center = np.array([(result_box[1] + result_box[3]) / 2, (result_box[0] + result_box[2]) / 2])
                    # new_center = np.array([(result_box[0] + result_box[2]) / 2, (result_box[1] + result_box[3]) / 2])
                    
                    modal_ekf.update(new_center.reshape(-1, 1))
                    # print(f"Predicted center: {predicted_center}, actual center: {new_center}, result box: {result_box}, frame idx: {out_frame_idx} ******")
                    # object_ekf.predict()
                    current_width = result_box[2] - result_box[0]
                    current_height = result_box[3] - result_box[1]
                    # print(f"Predicted width: {object_ekf.x[0][0]}, height: {object_ekf.x[0][1]}, actual width: {current_width}, actual height: {current_height}")
    
                else:
                    raise Exception("No object detected")
                
                # if out_frame_idx >=4:
                #     raise Exception("testing")
                
            except:
                
                
                
                # no object detected, append the previous result if available
                if output_track_boxes:
                    output_track_boxes.append(output_track_boxes[-1])
                    # output_track_boxes.append([0, 0, 0, 0])
                    
                    # # use the kalman filter to predict the next center of the mask
                    # modal_ekf.predict()
                    # predicted_center = modal_ekf.x[:2].flatten()
                    # # get difference between predicted center and previous center and shift the box by that amount
                    # # shift = predicted_center - new_center
                    
                    # # result_box[0] += shift[1]
                    # # result_box[1] += shift[0]
                    # # result_box[2] += shift[1]
                    # # result_box[3] += shift[0]
                    
                    # # given predicted center, propogate out the width and height of the object
                    # # use the predicted width and height to generate box
                    
                    # # Predict the width and height using the Kalman filter
                    # object_ekf.predict()
                    # predicted_width = object_ekf.x[0][0]
                    # predicted_height = object_ekf.x[1][0]

                    # # Calculate the new bounding box coordinates based on the predicted center and the predicted width and height
                    # x_min = int(predicted_center[1] - predicted_width / 2)
                    # y_min = int(predicted_center[0] - predicted_height / 2)
                    # x_max = int(predicted_center[1] + predicted_width / 2)
                    # y_max = int(predicted_center[0] + predicted_height / 2)

                    # result_box = [x_min, y_min, x_max, y_max]
                    
                    # current_image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
                    # current_image_np = np.array(current_image)
                    # # make sure the box is within the image size
                    
                    
                    # if result_box[0] > current_image_np.shape[0]:
                    #     result_box[0] = int(min(result_box[0], current_image_np.shape[0] - 1))
                    # if result_box[1] > current_image_np.shape[1]:
                    #     result_box[1] = int(min(result_box[1], current_image_np.shape[1] - 1))
                    # result_box[0] = int(max(result_box[0], 0))
                    # result_box[1] = int(max(result_box[1], 0))
                    
                    # if result_box[2] < 0:
                    #     result_box[2] = int(max(result_box[2], 0))
                    # if result_box[3] < 0:
                    #     result_box[3] = int(max(result_box[3], 0))
                
                    # result_box[2] = int(min(result_box[2],  current_image_np.shape[0] - 1))
                    # result_box[3] = int(min(result_box[3], current_image_np.shape[1] - 1))
                    
                    # # get previous center 
                    # previous_center = np.array([(output_track_boxes[-1][1] + output_track_boxes[-1][3]) / 2, (output_track_boxes[-1][0] + output_track_boxes[-1][2]) / 2])
                    
                    # # print(f"predicted center: {predicted_center}, previous center: {previous_center}, result box: {result_box}, frame idx: {out_frame_idx}, predicted width: {predicted_width}, predicted height: {predicted_height}")
                    
                    
                    # output_track_boxes.append(result_box)
                            
                else:
                    output_track_boxes.append([0, 0, 0, 0])
                # print(f"No object detected in frame {out_frame_idx}")
            
                # Add the current video sub-directory to the set
                no_objects_detected_dirs.add(current_video_sub_dir)
                
                if out_frame_idx == len(frame_names) - 1:
                    all_points_annotated = True
                    break
                
                continue 
            
            
            
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
                    
                
                # # Load images
                original_image = Image.open(os.path.join(video_dir, frame_names[0]))
                previous_image = Image.open(os.path.join(video_dir, frame_names[previous_success_index]))
                current_image = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
                
                # #depth estimation of image 
                # # brighter patches are closer to the camera 
                # # lower values are farther from the camera 
                # with torch.autocast("cuda", dtype=torch.float16):
                #     predictions = pipe(current_image)
                #     depth_prediction = predictions["depth"]
                
                # need to get the max value of the mask 
                # need to get the max value of region of interest that is not mask 
                # if there is a a higher value in the roi, then likely that the object is occluded
                # will need to perform amodal prediction for the full object 
                
                
                # for the bounding boxes that are sequential, make sure that there are no large jumps in the area of the bounding box
                # likely means that there is an outlier and result should be modified 
                
                
                
                # current_image_np = np.array(current_image)
                

                # # Check if the closest pixels are within the previous object box plus a threshold
                # prev_box = output_track_boxes[-1]
                # x_min, y_min, x_max, y_max = prev_box
                # # multiply threshold be number of frames since last success
                # x_min -= threshold #* (out_frame_idx - previous_success_index)
                # y_min -= threshold #* (out_frame_idx - previous_success_index)
                # x_max += threshold #* (out_frame_idx - previous_success_index)
                # y_max += threshold #* (out_frame_idx - previous_success_index)
                
                # # make sure the box is within the image size
                # x_min = int(max(x_min, 0))
                # y_min = int(max(y_min, 0))
                # x_max = int(min(x_max, current_image_np.shape[1]))
                # y_max = int(min(y_max, current_image_np.shape[0]))
                
                # roi_rgb = current_image_np[y_min:y_max, x_min:x_max]
                
                # depth_prediction_np = np.array(depth_prediction)
                # roi_depth = depth_prediction_np[y_min:y_max, x_min:x_max]
                
                # # get max value in mask region 
                # mask_depth_full = depth_prediction_np * (previous_mask) 
                # mask_depth_mask = mask_depth_full[y_min:y_max, x_min:x_max]
                # max_depth = np.max(mask_depth_mask)
                # # print(f"Max depth in roi: {max_depth}")
                
                # # get max value in roi_depth that is not in the mask
                # mask_depth_full = depth_prediction_np * (-1+previous_mask) * (-1)
                # mask_depth = mask_depth_full[y_min:y_max, x_min:x_max]
                # # mask_depth = mask_depth.flatten()
                # # mask_depth = mask_depth[mask_depth != 0]
                # max_mask_depth = np.max(mask_depth)
                # # print(f"Max depth not in mask: {max_mask_depth}")
                
                # # larger values mean closer to the camera
                # # so if max_mask_depth is greater than max_depth, then likely that the object is occluded
                # # if max_mask_depth > max_depth:
                # #     print("Object is possibly occluded")
                
                
                modal_ekf.predict()
                best_match_coords = modal_ekf.x[:2].flatten()
                best_match_coords = np.round(best_match_coords).astype(int)
                
                
                # make sure the coordinates are within the image size
                best_match_coords[0] = int(max(best_match_coords[0], 0))
                best_match_coords[1] = int(max(best_match_coords[1], 0))
                best_match_coords[0] = int(min(best_match_coords[0], current_image_np.shape[0] - 1))
                best_match_coords[1] = int(min(best_match_coords[1], current_image_np.shape[1] - 1))
                
                
                
                frame_idx , out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=out_frame_idx,
                    obj_id=ann_obj_id,
                    points= np.array([best_match_coords], dtype=np.float32),

                    labels= np.array([1], np.int32), # 1 is positive, 0 is negative 
                    # box = box,
                )
                
                viz_results = False
                if viz_results:
                    # plot original image, previous image, current image
                    fig, ax = plt.subplots(1, 6, figsize=(20, 5))
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
                    # box_base_gt = [int(x) for x in lines[out_frame_idx-1].strip().split()]
                    # box_base_gt = np.array(box_base_gt, dtype=np.float32)
                    # x_min_gt, y_min_gt, width_gt, height_gt = box_base_gt
                    # x_max_gt = x_min_gt + width_gt
                    # y_max_gt = y_min_gt + height_gt
                    # rect_gt = patches.Rectangle((x_min_gt, y_min_gt), x_max_gt - x_min_gt, y_max_gt - y_min_gt, linewidth=2, edgecolor='b', facecolor='none')
                    # ax[1].add_patch(rect_gt)

                        
                    ax[2].imshow(current_image)
                    # ax[2].scatter(best_match_coords[1], best_match_coords[0], color='red', marker='x', s=100)  # Add marker for best match
                    show_mask(previous_mask, ax[2], obj_id=ann_obj_id)  # Use show_mask function
                    ax[2].set_title(f"Current Image for frame {out_frame_idx}")
                    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                    ax[2].add_patch(rect)
                    
                    # box_base_gt = [int(x) for x in lines[out_frame_idx].strip().split()]
                    # box_base_gt = np.array(box_base_gt, dtype=np.float32)
                    # x_min_gt, y_min_gt, width_gt, height_gt = box_base_gt
                    # x_max_gt = x_min_gt + width_gt
                    # y_max_gt = y_min_gt + height_gt
                    # rect_gt = patches.Rectangle((x_min_gt, y_min_gt), x_max_gt - x_min_gt, y_max_gt - y_min_gt, linewidth=2, edgecolor='b', facecolor='none')
                    # ax[2].add_patch(rect_gt)
                                    
                                    
                
                    ax[3].imshow(depth_prediction)
                    ax[3].set_title("Current Depth Image")
                    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                    ax[3].add_patch(rect)
                    
                    ax[4].imshow(mask_depth_mask)
                    
                    ax[5].imshow(mask_depth)
                    
                    plt.show()
                
                
                previous_mask = result_mask*0
                
                
                break
                
         
        # find where lines of output_track_boxes are [0, 0, 0, 0]
        # interpolate the values between the previous non zero value and the next non zero value
        # print(output_track_boxes)

        # # Function to interpolate between two points
        # def interpolate_boxes(start_box, end_box, num_steps):
        #     interpolated_boxes = []
        #     for i in range(1, num_steps + 1):
        #         interpolated_box = start_box + (end_box - start_box) * (i / (num_steps + 1))
        #         interpolated_boxes.append(interpolated_box)
        #     return interpolated_boxes

        # # Convert output_track_boxes to a numpy array for easier manipulation
        # output_track_boxes = np.array(output_track_boxes)

        # # Find indices of zero-value boxes
        # zero_indices = np.where((output_track_boxes == [0, 0, 0, 0]).all(axis=1))[0]

        # # Iterate over zero-value indices and interpolate
        # for zero_idx in zero_indices:
        #     # Find the previous non-zero box
        #     prev_idx = zero_idx - 1
        #     while prev_idx >= 0 and (output_track_boxes[prev_idx] == [0, 0, 0, 0]).all():
        #         prev_idx -= 1
            
        #     # Find the next non-zero box
        #     next_idx = zero_idx + 1
        #     while next_idx < len(output_track_boxes) and (output_track_boxes[next_idx] == [0, 0, 0, 0]).all():
        #         next_idx += 1
            
        #     # If both previous and next non-zero boxes are found, interpolate
        #     if prev_idx >= 0 and next_idx < len(output_track_boxes):
        #         prev_box = output_track_boxes[prev_idx]
        #         next_box = output_track_boxes[next_idx]
        #         num_steps = next_idx - prev_idx - 1
        #         interpolated_boxes = interpolate_boxes(prev_box, next_box, num_steps)
                
        #         # Insert interpolated boxes into output_track_boxes
        #         output_track_boxes[prev_idx + 1:next_idx] = interpolated_boxes

        # # Handle trailing zero-value boxes by repeating the last non-zero box
        # last_non_zero_idx = len(output_track_boxes) - 1
        # while last_non_zero_idx >= 0 and (output_track_boxes[last_non_zero_idx] == [0, 0, 0, 0]).all():
        #     last_non_zero_idx -= 1

        # if last_non_zero_idx >= 0:
        #     last_non_zero_box = output_track_boxes[last_non_zero_idx]
        #     for zero_idx in zero_indices:
        #         if zero_idx > last_non_zero_idx:
        #             output_track_boxes[zero_idx] = last_non_zero_box
        
        # # Convert output_track_boxes back to a list of lists
        # output_track_boxes = output_track_boxes.tolist()
        
        # save results to result_output_name file
        # plt.show()
        if all_points_annotated:
            print(f"Saving results to {result_output_name}")
            with open(result_output_name, 'w') as f:
                for box in output_track_boxes:
                    x_min, y_min, x_max, y_max = box
                    f.write(f"{int(x_min)}\t{int(y_min)}\t{int(x_max-x_min)}\t{int(y_max-y_min)}\n")
        plt.show()   
print("Done!")
print(video_base_dir)
print("Number of directories with tracks lost:", len(no_objects_detected_dirs))
print("Directories with tracks lost:", no_objects_detected_dirs)
# plt.show()