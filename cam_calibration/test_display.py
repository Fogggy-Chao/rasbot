import cv2
import numpy as np

# Create a blank black image
img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.putText(img, "OpenCV Test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Test Window", img)
print("imshow called. Press any key in the window (if it appears) or Ctrl+C in terminal.")
key = cv2.waitKey(0)
print(f"waitKey returned: {key}")
cv2.destroyAllWindows()
print("Test finished.") 