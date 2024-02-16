import cv2
import numpy as np


def extract_polygonal_roi(frame, vertices):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), (255, 255, 255))
    roi = cv2.bitwise_and(frame, mask)
    return roi


# Define the vertices for the polygonal ROIי
vertices = [[449, 435], [295, 498], [786, 498], [786, 441]]

# Initialize video capture and video writer
cap = cv2.VideoCapture(r'C:\Users\User\my notebook\tryingss\How to deal with crosswalks on the road test.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    vertices = [[449, 435], [295, 498], [786, 498], [786, 441]]

    roi = extract_polygonal_roi(frame, vertices)
    # threshold on white/gray sidewalk stripes
    lower = (203, 203, 203)
    upper = (260, 260, 260)
    thresh = cv2.inRange(roi, lower, upper)
    thresh_gray = thresh

    contours, _ = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter contours based on area and aspect ratio
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10:  # Erase small contours
            cv2.fillPoly(thresh_gray, pts=[contour], color=0)
        else:
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if min(width, height) == 0: continue  # Prevent division by zero
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio < 1.5:  # Filter out contours that are not elongated
                cv2.fillPoly(thresh_gray, pts=[contour], color=0)
            else:
                filtered_contours.append(contour)

    # Close gaps between contours
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    # thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

    # Apply morphological erosion to remove shapes that cannot contain the rectangle
    thresh_gray = cv2.erode(thresh_gray, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (76, 76))
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Final contour processing
    for contour in contours:
        area = cv2.contourArea(contour)
        if 120 < area and area < 180:  # Ignore small contours
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 1)
    # Write the processed frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()