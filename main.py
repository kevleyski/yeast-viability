# Pyrmont Brewery Raspbeery Pi Yeast Viability Counter
# Kevin Staunton-Lambert
# Copyright (c) Pyrmont Brewery 2007-2024

import cv2
import numpy as np

def detect_yeast_cells(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Separate foreground (yeast cells) from background
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # Find yeast cells
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Classify live vs dead cells
    for c in cnts:
        # Calculate moments
        M = cv2.moments(c)

        # Centre of mass
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Get area and perimeter of yeast cell
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)

            # Classify cell based on circularity and intensity
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            if circularity > 0.9 and cv2.pointPolygonTest(thresh, (cX, cY), False) == 1:
                # Circle Live cell as green
                cv2.drawContours(image, [c], -1, (0, 255, 0), 8)
            else:
                # Circle non-viable cell as red
                cv2.drawContours(image, [c], -1, (0, 0, 255), 8)

    # Show detected cells
    cv2.imshow("Yeast Cells", image)
    cv2.waitKey(0)

def run():
    # Load yeast cells photo/snapshot
    image = cv2.imread("yeast_cells.jpg")
    detect_yeast_cells(image.copy())
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
