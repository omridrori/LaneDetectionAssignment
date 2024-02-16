import cv2
import numpy as np


# Function to detect lanes and change lanes within the specified region of interest (ROI)
def detect_lanes_in_roi(frame):
    # Grayscale
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray_scale, (5, 5), 0)

    # Canny
    canny = cv2.Canny(blur, 50, 150)

    # Define the ROI
    roi = np.array([[(530, 500), (800, 500), (800, 720), (500, 720)]],
                   np.int32)

    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, roi, 255)
    canny_mask = cv2.bitwise_and(canny, mask)

    # Hough transform
    lines = cv2.HoughLinesP(canny_mask, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=100)

    # Draw

    # Midpoint array to save results nd calculate mean
    left_midpoint = []
    right_midpoint = []
    # There are cases that the lines are not detected (avoid exceptions)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Calculate midpoint (the middle of a line that detected)
            midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            # Classify lines based on slope
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                # Only consider lines with slope not close to horizontal
                if abs(slope) > 0.1:  # not detect horizontal (with hyperparameter)
                    if slope < 0:
                        left_midpoint.append(midpoint)
                    else:
                        right_midpoint.append(midpoint)

    return frame, left_midpoint, right_midpoint

git commit -m "Added nili"

def main():
    # Initialize video
    capture = cv2.VideoCapture('night_highway.mp4')  # Replace 'night_highway.mp4' with your video file

    # VideoWriter - to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('night_lane_change_output.mp4', fourcc, 25.0, (int(capture.get(3)), int(capture.get(4))))

    # Previous midpoints (same but for the previous points)
    prev_midpoint_of_left = []
    prev_midpoint_of_right = []

    # Counter for lane change message display - it's a hyperparameter
    lane_change_display_frames_counter = 0
    # Detect if the change is from left or right
    left_change = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # Detect lanes in the specified ROI
        detected_frame, left_midpoint, right_midpoint = detect_lanes_in_roi(frame)

        # Detect lane change
        if prev_midpoint_of_left and left_midpoint:
            prev_left_x = np.mean([point[0] for point in prev_midpoint_of_left])
            current_left_x = np.mean([point[0] for point in left_midpoint])
            # 40 is a hyperparameter (the change in x-axis pixels)
            if abs(prev_left_x - current_left_x) > 40:
                left_change = 1
                # Set counter to display message for 70 frames (it's the value of the hypermarket)
                lane_change_display_frames_counter = 70

        if prev_midpoint_of_right and right_midpoint:
            prev_right_x = np.mean([point[0] for point in prev_midpoint_of_right])
            current_right_x = np.mean([point[0] for point in right_midpoint])
            # 40 is a hyperparameter (the change in x-axis pixels)
            if abs(prev_right_x - current_right_x) > 40:
                left_change = 0
                # Set counter to display message for 70 frames (it's the value of the hypermarket)
                lane_change_display_frames_counter = 70

        # Display lane changes
        if lane_change_display_frames_counter > 0:
            if left_change == 0:
                cv2.putText(detected_frame, 'Lane Change Detected From Right', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2,
                            cv2.LINE_AA)
            else:
                cv2.putText(detected_frame, 'Lane Change Detected From Left', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2,
                            cv2.LINE_AA)
            lane_change_display_frames_counter -= 1

        prev_midpoint_of_left = left_midpoint
        prev_midpoint_of_right = right_midpoint

        # Write to the output video file
        out.write(detected_frame)

    # Release video
    capture.release()
    out.release()
    cv2.destroyAllWindows()


if _name_ == "_main_":
    main()