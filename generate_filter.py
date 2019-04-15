# Script to generate (u,v) table that maps
# (u,v) coordinates in the film plane to
# (u,v) coordinates in the texture.

# NOTE: everything here is in relativistic units: G = 1, c = 1.

import numpy as np
import cv2
import sys
import pickle

# import photon, make_photon_at_grid_pt from "Ray_Diff_Eqs.py"

NUM_ARGS = 7

# Main entry point for script.
def main():
  if len(sys.argv) < NUM_ARGS - 1:
    print("usage: python3 generate_filter.py <directory to write to> <width> <height> <height angle> <camera r> <backdrop r> <mass>")
    exit()

  # Get the variables we'll need from the command line args.
  filepath = sys.argv[1]
  width = int(sys.argv[2])
  height = int(sys.argv[3])
  height_angle = float(sys.argv[4])
  camera_r = float(sys.argv[5])
  backdrop_r = float(sys.argv[6])
  mass = float(sys.argv[7])
  schwarzschild_radius = calc_schwarzschild_radius(mass)

  # Print everything to the console.
  print("")
  print("writing table to " + filepath)
  print("width: %s" % width)
  print("height: %s" % height)
  print("height angle: %s" % height_angle)
  print("camera radial distance: %s" % camera_r)
  print("backdrop radial distance: %s" % backdrop_r)
  print("mass: %s" % mass)
  print("schwarzschild radius: %s" % schwarzschild_radius)
  print("")

  mapping = construct_mapping(width, height, height_angle, camera_r, backdrop_r, mass)
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
def construct_mapping(width, height, height_angle, camera_r, backdrop_r, mass):
    filter = np.zeros((2, height * width))
    # Construct the filter.
    for i in range(height):
        for j in range(width):
            if (i % 20 == 0 and j == 0):
                print("{:10.2f}".format(100 * i / height) + "% done")
            # Trace the ray.
            point = trace_ray(i, j, width, height, height_angle,
                camera_r, backdrop_r, mass)
            if (point[2]):
                # x-coordinate
                filter[0, i * width + j] = -10e100
                # y-coordinate
                filter[1, i * width + j] = -10e100
            else:
                # x-coordinate
                filter[0, i * width + j] = point[0]
                # y-coordinate
                filter[1, i * width + j] = point[1]
    # Scale and translate to the width and height of the image.
    print("{:10.2f}".format(100) + "% done")
    filter[0] /= np.amax(filter[0])
    filter[0] += np.amin((filter[0])[filter[0] != np.amin(filter[0])])
    filter[1] /= np.amax(filter[1])
    filter[1] += np.amin((filter[1])[filter[1] != np.amin(filter[1])])
    filter[0] *= height
    filter[1] *= width
    return filter

# Traces a ray around the black hole to determine where in the original image
# to sample to determine the pixel value at location (i, j) in the filtered
# image.
# @return: truple (x, y, whether or not ray fell in the black hole)
def trace_ray(i, j, width, height, height_angle, camera_r, backdrop_r, mass):
    # Construct initial position and velocity.

    # The film plane will be a distance of 1 away from the eye. We can
    # calculate the height of the film plane using the height angle.
    film_plane_to_eye = 1
    film_plane_height = tan(height_angle / 2) * film_plane_to_eye
    film_plane_width = (width/height) * film_plane_height

    # To calculate the film plane r coordinate, we subtract the distance from
    # the eye to the film plane from the eye point. This is opposite in sign
    # to the z-coordinate of the film plane.
    film_plane_r = camera_r - film_plane_to_eye

    # The position is the i, j coordinate, scaled to the size of the film
    # plane. The film plane is centered in the photon construction code.
    pt = np.array([i / film_plane_height, j / film_plane_width])
    p = make_photon_at_grid_point(pt, -camera_r, -film_plane_r,
                                    film_height, film_width)

    # step the photon until it coincides with
    # the background plane (along z axis)
    # for ... check if intersect, if not step
    schwarzschild_radius = calc_schwarzschild_radius(mass)
    epsilon = 0.01
    while (p.pos[2] < backdrop_r && p.sph_pos[1] > schwarzschild_radius + epsilon):
        p.step_rk4()

    # Return the x, y position, plus an indicator that tells us if the ray
    # was captured by the black hole.
    return (p.pos[2] < backdrop_r) ? (-1, -1, True) : (p.pos[1], p.pos[0], False)
    # return (i, j, i > 200 and i < 900 and j > 200 and j < 700)

# Run main.
main()
