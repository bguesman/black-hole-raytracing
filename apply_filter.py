import pickle
import sys
import cv2
import numpy as np
from scipy import interpolate
<<<<<<< HEAD
=======
from numba import jit
>>>>>>> a1db786128e10c112f67e475921c46c22b38b2a0

NUM_ARGS = 4

# Main entry point for script.
<<<<<<< HEAD
=======
# @jit
>>>>>>> a1db786128e10c112f67e475921c46c22b38b2a0
def main():
    if len(sys.argv) < NUM_ARGS - 1:
      print("usage: python3 apply_filter.py <filter file> <image> <image to write to>")
      exit()
    filter_file = sys.argv[1]
    image_file = sys.argv[2]
    to_write_to = sys.argv[3]
    with open(filter_file, "rb") as f:
        # Load the filter.
        filter = pickle.load(f)
        # Load the old image.
<<<<<<< HEAD
        image_old = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        # Create the new image.
        image_new = np.ones(image_old.shape, dtype="uint8") * 255

        # Create the interpolator. Here, the x coordinates
        # are a range from 0-width, the y coordinates are from
        # 0-height, and the z values are specified by the old image.
        width = image_old.shape[1]
        height = image_old.shape[0]
        f = interpolate.RectBivariateSpline(np.arange(height),
                                np.arange(width),
                                image_old)
=======
        image_old = cv2.imread(image_file)
        # Split into color channels.
        old_channels = [image_old[:,:,c] for c in range(3)]
        # Create the new image.
        image_new = np.ones(image_old.shape, dtype="uint8") * 255

        # Create the interpolators. Here, the x coordinates
        # are a range from 0-width, the y coordinates are from
        # 0-height, and the z values are specified by the old image.
        # Why create a separate interpolator for each color?
        # Because scipy's 2d interpolate function does not allow
        # for interpolation of vector fields.
        width = image_old.shape[1]
        height = image_old.shape[0]
        f = [interpolate.RectBivariateSpline(np.arange(height),
                                np.arange(width), old_channels[c])
                                for c in range(3)]
>>>>>>> a1db786128e10c112f67e475921c46c22b38b2a0

        # Now, we can use the interpolator together with the filter
        # lookup table to apply the filter and create the new image.
        for i in range(height):
            for j in range(width):
<<<<<<< HEAD
                image_new[i, j] = f(filter[0, i * width + j],
                                filter[1, i * width + j])
=======
                for c in range(3):
                    image_new[i, j, c] = f[c](filter[0, i * width + j],
                                    filter[1, i * width + j])

>>>>>>> a1db786128e10c112f67e475921c46c22b38b2a0
        cv2.imwrite(to_write_to, image_new)

# Run main.
main()
