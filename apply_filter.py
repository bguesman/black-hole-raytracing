import pickle
import sys
import cv2
import numpy as np

NUM_ARGS = 4

# Main entry point for script.
def main():
    if len(sys.argv) < NUM_ARGS - 1:
      print("usage: python3 apply_filter.py <filter file> <image> <image to write to>")
      exit()
    filter_file = sys.argv[1]
    image_file = sys.argv[2]
    to_write_to = sys.argv[3]
    with open(filter_file, "rb") as f:
        filter = pickle.load(f)

        image_old = cv2.imread(image_file)

        image_new = np.ones(image_old.shape, dtype="uint8") * 255
        print("old shape is" + str(image_old.shape))

        for i in range(image_old.shape[0]):
            for j in range(image_old.shape[1]):
                index = filter[(i,j)]
                try:
                    image_new[i, j] = image_old[index]
                except:
                    # print(index)

        cv2.imwrite(to_write_to, image_new)


# Run main.
main()
