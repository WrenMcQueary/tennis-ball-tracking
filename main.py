from skimage import io, color, feature
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


# I do not own the video used in this code.  You can find the video on YouTube at this URL:
# https://www.youtube.com/watch?v=GIRqYz_hnkQ

# This script inputs a series of frames from a video of a tennis ball bouncing several times off of a table, and outputs
# the tennis ball's vertical position, velocity, and acceleration as functions of time.

# Warren McQueary 6/2/20


# [v] Create a list of names of all images in the sequence.
list_of_image_names = []
for ii in range(134):
    list_of_image_names.append("scene" + str(ii+1).zfill(5) + ".png")   # Includes padding with 0s to the left as
    # needed.

# [v] Import the images and read them into an ImageCollection.
image_collection = io.imread_collection(list_of_image_names, False, None, )

# [v] Run blob detection and get the tennis ball centroid coordinates of each frame.
blob_record = []
for ii in tqdm(range(len(image_collection))):
    # Use a mask to make the entire image black except for the ball.
    my_image = image_collection[ii]

    my_image_grayscale = 255 * color.rgb2gray(my_image)
    tennis_ball_greener_than_red_mask = my_image[:, :, 1] > 1*my_image[:, :, 0]
    tennis_ball_greener_than_blue_mask = my_image[:, :, 1] > 1.5*my_image[:, :, 2]
    tennis_ball_bright_mask = my_image_grayscale > 20
    tennis_ball_combined_mask = np.logical_and(np.logical_and(tennis_ball_greener_than_red_mask,
                                                              tennis_ball_greener_than_blue_mask),
                                               tennis_ball_bright_mask)
    tennis_ball_combined_mask_inverted = np.logical_not(tennis_ball_combined_mask)
    my_image[tennis_ball_combined_mask] = [255, 255, 255]
    my_image[tennis_ball_combined_mask_inverted] = [0, 0, 0]
    my_image_ready_for_blob_detection = color.rgb2gray(my_image)

    # Find the largest blob for the frame and append to blob_record.
    blob_record.append(feature.blob_log(my_image_ready_for_blob_detection, max_sigma=30, min_sigma=3, num_sigma=10,
                                        threshold=0.1)[0])

# [v] From blob_record, make a simpler list of x and y positions (pixels), still with the origin in the top left and y
# increasing downward.
x_coords_pixels, y_coords_pixels = [], []
for frame in range(len(image_collection)):
    x_coords_pixels.append(blob_record[frame][1])
    y_coords_pixels.append(blob_record[frame][0])

# [v] From the framerate of the video and the number of frames, make a simple list of time values (seconds).
times = []
framerate = 25  # Hz
for ii in range(len(image_collection)):
    times.append(ii/framerate)

# [v] Convert the tennis ball centroid coordinates to a y position in each frame.
x_coords_meters = []
y_coords_meters = []
for ii in range(len(image_collection)):
    # Assumptions involved in the x conversion: The camera is perfectly aligned, each pixel is perfectly square.
    # x = 0 m at x = 233 pixels, where the ball first hits the table.
    x_coords_meters.append(0.003941 * (x_coords_pixels[ii] - 233))

    # Assumptions involved in the y conversion: The gray "ruler" is 3 ft = 0.9144 m long, so each pixel is 0.003941 m
    # long. y = 0 m at y = 270 pixels, where the ball first hits the table.
    y_coords_meters.append(-0.003941 * (y_coords_pixels[ii] - 270))

# [v] Create a list of velocities.
x_velocities_meters_per_second = []
y_velocities_meters_per_second = []
for ii in range(len(image_collection)):
    if ii == 0:
        x_velocities_meters_per_second.append(0)
        y_velocities_meters_per_second.append(0)
    else:
        x_velocities_meters_per_second.append((x_coords_meters[ii] - x_coords_meters[ii-1])*framerate)
        y_velocities_meters_per_second.append((y_coords_meters[ii] - y_coords_meters[ii-1])*framerate)

# [v] Create a list of accelerations.
x_accelerations_meters_per_second_squared = []
y_accelerations_meters_per_second_squared = []
for ii in range(len(image_collection)):
    if ii == 0:
        x_accelerations_meters_per_second_squared.append(0)
        y_accelerations_meters_per_second_squared.append(0)
    else:
        x_accelerations_meters_per_second_squared.append((x_velocities_meters_per_second[ii] -
                                                          x_velocities_meters_per_second[ii-1])*framerate)
        y_accelerations_meters_per_second_squared.append((y_velocities_meters_per_second[ii] -
                                                          y_velocities_meters_per_second[ii-1])*framerate)

# [v] Graph vertical position, velocity, and acceleration as functions of time.
# Create subplots.
fig, axs = plt.subplots(3, 2)
fig.suptitle("All Measurements vs Time (seconds)")

axs[0, 0].plot(times, x_coords_meters, "tab:blue")
axs[0, 0].set_title("Horizontal Position (m)")
axs[0, 0].grid()

axs[1, 0].plot(times, x_velocities_meters_per_second, "tab:red")
axs[1, 0].set_title("Horizontal Velocity (m/s)")
axs[1, 0].grid()

axs[2, 0].plot(times, x_accelerations_meters_per_second_squared, "tab:green")
axs[2, 0].set_title("Horizontal Acceleration (m/s**2)")
axs[2, 0].grid()

axs[0, 1].plot(times, y_coords_meters, "tab:blue")
axs[0, 1].set_title("Vertical Position (m)")
axs[0, 1].grid()

axs[1, 1].plot(times, y_velocities_meters_per_second, "tab:red")
axs[1, 1].set_title("Vertical Velocity (m/s)")
axs[1, 1].grid()

axs[2, 1].plot(times, y_accelerations_meters_per_second_squared, "tab:green")
axs[2, 1].set_title("Vertical Acceleration (m/s**2)")
axs[2, 1].grid()

plt.show()
