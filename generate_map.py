# Script to generate (u,v) table that maps
# (u,v) coordinates in the film plane to
# (u,v) coordinates in the texture.

import numpy as np
import cv2
import sys

NUM_ARGS = 2

# Main entry point for script.
def main():
  if len(sys.argv) < NUM_ARGS - 1:
    print("usage: python3 generate_map.py <w/h ratio> <file to write to>")
    exit()
  whRatio = sys.argv[1]
  filepath = sys.argv[2]
  print("writing table to " + filepath + " with w/h ratio " + whRatio)

# Run main.
main()
