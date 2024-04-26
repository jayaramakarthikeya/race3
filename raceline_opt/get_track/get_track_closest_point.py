import json
import yaml
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.interpolate import CubicSpline


# image: black (occupied) and white (free) png image of the track
# yaml: yaml file following the ROS slam_toolbox format

MARGIN = 8
SPARSITY = 25
FLIP_SIDE = True
FLIP_DIRECTION = True
SAMPLE = 1000

def extract_boundaries(image_path, yaml_path,map_name):
    # TODO: Implement the logic to extract left and right boundaries from the image
    # using the provided yaml file for coordinate transformation
    
    # TODO: Implement the logic to extract the boundaries
    # read the image and yaml file
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    original_image = cv2.imread(f'map/{map_name}.pgm', cv2.IMREAD_GRAYSCALE)

    # extract the origin and resolution from the yaml file
    origin = yaml_data['origin']
    resolution = yaml_data['resolution']

    # # plot the image
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
        
    # threshold the image
    _, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # erode the white regions
    kernel = np.ones((3, 3), np.uint8)
    thresholded_image = cv2.erode(thresholded_image, kernel, iterations=MARGIN)

    # upsample the thresholded image
    blurred_image = cv2.pyrUp(thresholded_image)

    # Apply median blur multiple times
    for i in range(15):
        blurred_image = cv2.medianBlur(blurred_image, 7)

    # Downsample the blurred image
    blurred_image = cv2.pyrDown(blurred_image)

    # Threshold the blurred image
    _, blurred_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY)

    # # plot the image
    # plt.figure()
    # plt.imshow(blurred_image)
    # plt.show()

    # find contours in the image
    contours, hierarchy = cv2.findContours(blurred_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # squeeze the contours
    contours = [contour.squeeze() for contour in contours]
    # FLIP_SIDE the contours if needed
    if FLIP_SIDE:
        contours = contours[::-1]
    # FLIP_DIRECTION the contours if needed
    if FLIP_DIRECTION:
        contours = [contour[::-1, :] for contour in contours]
    # append the first point to the end to close the loop
    contours = [np.append(contour, [contour[0]], axis=0) for contour in contours]
    # interpolate the contours
    left_interpolator = CubicSpline(np.linspace(0, 1, len(contours[0])), contours[0], bc_type='periodic')
    right_interpolator = CubicSpline(np.linspace(0, 1, len(contours[1])), contours[1], bc_type='periodic')
    # sample the interpolators
    left_interpolated = left_interpolator(np.linspace(0, 1, SAMPLE))
    right_interpolated = right_interpolator(np.linspace(0, 1, SAMPLE))
    # overwrite the contours
    contours = [left_interpolated, right_interpolated]

    # plot the contours
    plt.figure()
    plt.imshow(original_image)
    for contour in contours:
        contour = contour.squeeze()
        plt.scatter(contour[:, 0], contour[:, 1], c=range(len(contour)))
    plt.savefig('contours.png')
    # plt.show()

    # iterate through the first contour
    x_left_world, y_left_world, x_right_world, y_right_world = [], [], [], []
    # init the left point
    left_index = 0
    left_point = contours[0][left_index]
    # get the right point which is the closest to the left point in the other contour
    right_index = np.argmin(np.linalg.norm(contours[1] - left_point, axis=1))
    right_point = contours[1][right_index]
    # record right start index
    start_right_index = right_index
    while True:
        # append the left and right points
        x_left_world.append(left_point[0] * resolution + origin[0])
        y_left_world.append((blurred_image.shape[0] - left_point[1]) * resolution + origin[1])
        x_right_world.append(right_point[0] * resolution + origin[0])
        y_right_world.append((blurred_image.shape[0] - right_point[1]) * resolution + origin[1])
        # increment the left index
        if left_index + SPARSITY >= len(contours[0]):
            break
        next_left_index = (left_index + SPARSITY)%len(contours[0])
        next_left_point = contours[0][next_left_index]
        next_right_index = np.argmin(np.linalg.norm(contours[1] - next_left_point, axis=1))
        if len(contours[1])/2 > (right_index-next_right_index)%len(contours[1]) > SPARSITY:
            right_index = (right_index-SPARSITY)%len(contours[1])
            right_point = contours[1][right_index]
            next_left_index = np.argmin(np.linalg.norm(contours[0] - right_point, axis=1))
            if len(contours[0])/2 < (next_left_index-left_index)%len(contours[0]) or (next_left_index-left_index)%len(contours[0]) == 0:
                left_index += 1
            else:
                left_index = next_left_index
            left_point = contours[0][left_index]
        elif (right_index-next_right_index)%len(contours[1]) > len(contours[1])/2 or (right_index-next_right_index)%len(contours[1]) == 0:
            left_index = next_left_index
            left_point = next_left_point
            right_index -= 1
            right_point = contours[1][right_index]
        else:
            left_index = next_left_index
            left_point = next_left_point
            right_index = next_right_index
            right_point = contours[1][right_index]

    # remove the last point
    x_left_world.pop()
    y_left_world.pop()
    x_right_world.pop()
    y_right_world.pop()        

    # append the first point to the end to close the loop
    x_left_world.append(x_left_world[0])
    y_left_world.append(y_left_world[0])
    x_right_world.append(x_right_world[0])
    y_right_world.append(y_right_world[0])

    # plot the points
    plt.figure()
    plt.scatter(x_left_world, y_left_world, c=range(len(x_left_world)))
    plt.scatter(x_right_world, y_right_world, c=range(len(x_right_world)))
    for i in range(len(x_left_world)):
        plt.plot([x_left_world[i], x_right_world[i]], [y_left_world[i], y_right_world[i]], c='b')
    plt.axis('equal')
    plt.savefig('points.png')
    # plt.show()

    # Export the boundaries as a JSON file
    boundaries = {
        'name': 'my_track',
        'left': {'x': x_left_world, 'y': y_left_world},
        'right': {'x': x_right_world, 'y': y_right_world}
    }
    
    with open('../dissertation-master/data/tracks/my_track.json', 'w') as file:
        json.dump(boundaries, file)

# Example usage
map_name = "sk2_0424"
image_path = f'map/{map_name}.png'
yaml_path = f'map/{map_name}.yaml'
extract_boundaries(image_path, yaml_path,map_name)