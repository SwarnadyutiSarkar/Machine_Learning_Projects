import cv2
import numpy as np

# Load and preprocess an image
image = cv2.imread('lane_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Apply Hough transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# Draw lines on the original image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Display the image with detected lanes
cv2.imshow('Lane Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
