#Read the masks and images from the specified folders
import os
import numpy as np
from PIL import Image
from scipy.ndimage import label
import matplotlib.pyplot as plt
mask_folder = r"C:\Users\anush\Documents\PostDoc\github code\MultichannelTextureAnalysis-main\filtered_masks"
os.makedirs(mask_folder, exist_ok=True)
image_folder = r"C:\Users\anush\Downloads\16 mei MSP Mapping 16 meter\tiles_output_NIR"
# List of specific tile numbers you want to extract
tile_numbers = [1489, 1490
]

#Convert to set for faster lookup
tile_set = {f"tile_{tile_num:05d}" for tile_num in tile_numbers}

# Get all mask files
all_filenames = os.listdir(mask_folder)
red_averages = []
white_averages = []
red_stds=[]
white_stds = []
red_areas = []
white_areas = []
saved_count = 0
for i,mask_file in enumerate(sorted(all_filenames)):
    print(f"Processing mask {i+1}/{len(all_filenames)}: {mask_file}")
    if mask_file.endswith("_mask.png"):
        base_name = mask_file.replace("_mask.png", "")
        if base_name in tile_set:
            image =  np.array(Image.open(os.path.join(image_folder,base_name + ".tif")))
            mask_path = os.path.join(mask_folder, mask_file)
            mask = np.array(Image.open(mask_path))
            # Display the mask
            # plt.figure(figsize=(6, 6))
            # plt.imshow(mask, cmap='gray')  # or cmap='viridis' if it's not binary
            # plt.title(f'Mask: {mask_file}')
            # plt.axis('off')
            # plt.show()

            # Remove alpha by taking only the first 3 channels (RGB)
            if mask.shape[2] == 4:  # Check if mask has an alpha channel
                # mask_rgb = mask[:, :, :3]

                # Now create the masks
                white_mask = np.all(mask == [255, 255, 255, 255], axis=-1)
                # Flexible red condition: R > 200, G < 100, B < 100
                red_mask = (mask[:, :, 0] > 200) & (mask[:, :, 1] < 100) & (mask[:, :, 2] < 100)
                
                white_labeled, num_white_regions  = label(white_mask)

                red_labeled, num_red_regions = label(red_mask)
                # Function to filter labeled regions by area threshold
                def filter_labels_by_area(labeled_mask, min_area):
                    filtered_mask = np.zeros_like(labeled_mask, dtype=bool)
                    for region_label in range(1, labeled_mask.max() + 1):
                        region = (labeled_mask == region_label)
                        if region.sum() > min_area:
                            filtered_mask[region] = True
                    return filtered_mask

                # Filter masks to keep only regions with area > 30 pixels
                white_mask_filtered = filter_labels_by_area(white_labeled, 30)
                red_mask_filtered = filter_labels_by_area(red_labeled, 30)

                # Re-label the filtered masks to get updated labeling and counts
                white_labeled, num_white_regions = label(white_mask_filtered)
                red_labeled, num_red_regions = label(red_mask_filtered)
                
                for region_label in range(1, num_white_regions + 1):
                    region_mask = (white_labeled == region_label)
                    masked_pixels = image[region_mask]
                    region_mask = (white_labeled == region_label)
                    area = region_mask.sum()  # number of pixels in this white region
                    white_areas.append(area)
                    avg_pixel = masked_pixels.mean(axis=0)
                    std_pixel= masked_pixels.std(axis=0)
                    white_stds.append(std_pixel)
                    white_averages.append(avg_pixel)

   
                for region_label in range(1, num_red_regions + 1):
                    region_mask = (red_labeled == region_label)
                    masked_pixels = image[region_mask]
                    region_mask = (red_labeled == region_label)
                    area = region_mask.sum()  # number of pixels in this red region
                    red_areas.append(area)
                    avg_pixel = masked_pixels.mean(axis=0)
                    std_pixel= masked_pixels.std(axis=0)
                    red_stds.append(std_pixel)
                    red_averages.append(avg_pixel)
print(red_averages)

red_areas = np.array(red_areas)
white_areas = np.array(white_areas)
mean_red_area = red_areas.mean()
std_red_area = red_areas.std()

mean_white_area = white_areas.mean()
std_white_area = white_areas.std()

print(f"Red areas: mean = {mean_red_area:.2f}, std = {std_red_area:.2f}")
print(f"White areas: mean = {mean_white_area:.2f}, std = {std_white_area:.2f}")
print(f"Mean red region area: {red_areas.mean():.2f} pixels")
print(f"Mean white region area: {white_areas.mean():.2f} pixels")
plt.hist(red_areas, bins=30, alpha=0.5, label='Red Areas')
plt.hist(white_areas, bins=30, alpha=0.5, label='White Areas')
plt.xlabel('Area (pixels)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of Region Areas')
plt.show()
# Convert to numpy arrays
red_averages = np.array(red_averages)
white_averages = np.array(white_averages)

red_stds = np.array(red_stds)
white_stds = np.array(white_stds)
# Compute means and standard deviations
mean_red = red_averages.mean()
std_red = red_stds.mean()

mean_white = white_averages.mean()
std_white = white_stds.mean()

# Pooled standard deviation
pooled_std = np.sqrt((std_red**2 + std_white**2) / 2)

# Cohen's d
cohens_d = abs(mean_red - mean_white) / pooled_std

print(f"Cohen's d: {cohens_d:.2f} standard deviations apart")

                 