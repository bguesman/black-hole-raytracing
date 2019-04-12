# Script to generate (u,v) table that maps
# (u,v) coordinates in the film plane to
# (u,v) coordinates in the texture.

# NOTE: everything here is in relativistic units: G = 1, c = 1.

import numpy as np
import cv2
import sys
import pickle

NUM_ARGS = 7

# Main entry point for script.
def main():
  if len(sys.argv) < NUM_ARGS - 1:
    print("usage: python3 generate_filter.py <directory to write to> <width> <height> <camera r> <backdrop r> <mass>")
    exit()

  # Get the variables we'll need from the command line args.
  filepath = sys.argv[1]
  width = int(sys.argv[2])
  height = int(sys.argv[3])
  camera_r = float(sys.argv[4])
  backdrop_r = float(sys.argv[5])
  mass = float(sys.argv[6])
  schwarzschild_radius = calc_schwarzschild_radius(mass)

  # Print everything to the console.
  print("")
  print("writing table to " + filepath)
  print("width: %s" % width)
  print("height: %s" % height)
  print("camera radial distance: %s" % camera_r)
  print("backdrop radial distance: %s" % backdrop_r)
  print("mass: %s" % mass)
  print("schwarzschild radius: %s" % schwarzschild_radius)
  print("")

  mapping = construct_mapping(width, height, camera_r, backdrop_r, mass)
  savename = "width:%s_height:%s_cam_dist:%s_backdrop_dist:%s_mass:%s" % \
    (width, height, camera_r, backdrop_r, mass)
  with open(filepath + "/" + savename + ".pkl", 'wb') as f:
      pickle.dump(mapping, f, pickle.HIGHEST_PROTOCOL)

# Given the mass of the black hole, calculates the Schwarzschild radius.
def calc_schwarzschild_radius(mass):
    return 2 * mass;

# Constructs the mapping between the pixels of the input image and the
# pixels of the output image.
# @return: 2D array of x and y coordinates of old image to use when
# constructing new image. For instance, to look up which pixel in the
# old image to use when filling in pixel (i, j) in the new image, one
# would look in position (0, i * width + j) to find the x coordinate
# and position (1, i * width + j).
def construct_mapping(width, height, camera_r, backdrop_r, mass):
    filter = np.zeros((2, height * width))
    for i in range(height):
        for j in range(width):
            # Do something
            point = trace_ray(i, j, width, height,
                camera_r, backdrop_r, mass)
            # x-coordinate
            filter[0, i * width + j] = point[0]
            # y-coordinate
            filter[1, i * width + j] = point[1]
    return filter

# Traces a ray around the black hole to determine where in the original image
# to sample to determine the pixel value at location (i, j) in the filtered
# image.
# @return: tuple.
def trace_ray(i, j, width, height, camera_r, backdrop_r, mass):
    # Construct initial position and velocity.
    # p = make_photon_at_grid_point(i, j, width, height)

    # step the photon until it coincides with
    # the background plane (along z axis)
    # for ... check if intersect, if not step

    # Return the x, y position

    return ((height - 1) - i, j)


# Run main.
main()
