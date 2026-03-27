# generate_rgb_image.py
import numpy as np
import cv2
import sys
# -------------------------------------------------
# 1. Image size
# -------------------------------------------------
WIDTH = 512
HEIGHT = 512
# -------------------------------------------------
# 2. Read RGB argument
# -------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python generate_rgb_image.py R,G,B")
    print("Example: python generate_rgb_image.py 255,0,0")
    sys.exit()

rgb_values = sys.argv[1].split(",")
R = int(rgb_values[0])
G = int(rgb_values[1])
B = int(rgb_values[2])
print("Requested color (R,G,B):", R, G, B)
# -------------------------------------------------
# 3. Create RGB image
# -------------------------------------------------
image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
image[:,:,0] = B
image[:,:,1] = G
image[:,:,2] = R
# -------------------------------------------------
# 4. Save image
# -------------------------------------------------
filename = f"image_{R}_{G}_{B}.png"
cv2.imwrite(filename, image)
print("Image saved:", filename)
print("Image size:", WIDTH, "x", HEIGHT)

