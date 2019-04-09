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
# @return: array of (u, v) points in filtered image to (u, v) points
# in original image, left to right from top to bottom.
def construct_mapping(width, height, camera_r, backdrop_r, mass):
    filter = np.zeros((2, height * width), dtype=int)
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
    # Currently flips image upside down.
    return ((height - 1) - i, j)


# Run main.
main()
