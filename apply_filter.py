import pickle
import sys
import cv2
import numpy as np
from scipy import interpolate
from tqdm import tqdm

from numba import jit

NUM_ARGS = 4

# Pulled from the internet because scipy sucks.
def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

# Main entry point for script.
# @jit
# python apply_filter.py filters/width125_height125.pkl test-images/avolovich_photo_.jpg test-images/avolovich.jpg
def main():
    if len(sys.argv) < NUM_ARGS - 1:
      print("usage: python3 apply_filter.py <filter file> <image> <image to write to>")
      exit()
    filter_file = sys.argv[1]
    # print(sys.argv)
    image_file = sys.argv[2]
    to_write_to = sys.argv[3]
    with open(filter_file, "rb") as f:
        # Load the filter.
        filter = pickle.load(f)
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
        # f = [interpolate.RectBivariateSpline(np.arange(height),
        #                         np.arange(width), old_channels[c])
        #                         for c in range(3)]

        # row_mean = np.mean((filter[0])[filter[0] != np.amax(filter[0])])
        # col_mean = np.mean((filter[1])[filter[1] != np.amin(filter[1])])
        # print(row_mean)
        # print(col_mean)
        # filter_stats = np.transpose(filter)
        # filter_stats_temp = []
        # print(filter_stats.shape[0])
        # for i in range(filter_stats.shape[0]):
        #     if np.abs(filter[0, i] - row_mean) + np.abs(filter[1, i] - col_mean) < 1:
        #         filter_stats_temp.append([filter[0,i], filter[1,i]])
        # print(len(filter_stats_temp))
        # filter_stats = np.array(filter_stats_temp)
        # print(filter_stats.shape)
        # filter_stats = np.transpose(filter_stats)
        # print(filter_stats.shape)
        # row_min = np.amin(filter_stats[0])
        # print(row_min)
        # col_min = np.amin(filter_stats[1])
        # print(col_min)
        # filter_stats[0] -= row_min
        # filter[0] -= row_min
        # filter_stats[1] -= col_min
        # filter[1] -= col_min
        # row_max = np.amax(filter_stats[0])
        # col_max = np.amax(filter_stats[1])
        # print(row_max)
        # print(col_max)
        # filter[0] /= row_max
        # filter[1] /= col_max
        # filter[0] *= height
        # filter[1] *= width

        # Now, we can use the interpolator together with the filter
        # lookup table to apply the filter and create the new image.
        for i in tqdm(range(height)):
            for j in range(width):
                for c in range(3):
                    if (filter[1, i * width + j] < 0 or filter[1, i * width + j] >= width
                        or filter[0, i * width + j] < 0 or filter[0, i * width + j] >= height):
                        image_new[i, j, c] = 0
                    else:
                        xl = int((filter[1, i * width + j]))
                        xh = xl + 1
                        yl = int((filter[0, i * width + j]))
                        yh = yl + 1
                        chan = old_channels[c]
                        points = [(xl, yl, chan[yl, xl]), (xl, yh, chan[yh % height, xl]),
                                    (xh, yl, chan[yl, xh % width]), (xh, yh, chan[yh % height, xh % width])]
                        image_new[i, j, c] = bilinear_interpolation(filter[1, i * width + j],
                            filter[0, i * width + j], points)
        cv2.imwrite(to_write_to, image_new)

# Run main.
main()

# python3 apply_filter.py filters/width\:100_height\:56_cam_dist\:6.0_backdrop_dist\:1.0_mass\:0.1.pkl test-images/astronaut100.jpg out.p
# python3 generate_filter.py filters 100 56 45 6 1 .1

# FOR LONG GENERATE:
# python3 generate_filter.py filters 500 282 45 6 1 .1
