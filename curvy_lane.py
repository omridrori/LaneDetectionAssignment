#Import necessary libraries
import cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correct(RGBimage, equalizeHist=False):  # 0.35

    vidsize = RGBimage.shape
    originalFile = RGBimage.copy()
    red = RGBimage[:, :, 2]
    green = RGBimage[:, :, 1]
    blue = RGBimage[:, :, 0]

    forLuminance = cv2.cvtColor(originalFile, cv2.COLOR_BGR2YUV)
    Y = forLuminance[:, :, 0]
    totalPix = vidsize[0] * vidsize[1]
    summ = np.sum(Y[:, :])
    Yaverage = np.divide(totalPix, summ)
    # Yclipped = np.clip(Yaverage,0,1)
    epsilon = 1.19209e-007
    correct_param = np.divide(-0.3, np.log10([Yaverage + epsilon]))
    correct_param = 0.7 - correct_param

    red = red / 255.0
    red = cv2.pow(red, correct_param[0])
    red = np.uint8(red * 255)
    if equalizeHist:
        red = cv2.equalizeHist(red)

    green = green / 255.0
    green = cv2.pow(green, correct_param[0])
    green = np.uint8(green * 255)
    if equalizeHist:
        green = cv2.equalizeHist(green)

    blue = blue / 255.0
    blue = cv2.pow(blue, correct_param[0])
    blue = np.uint8(blue * 255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)

    output = cv2.merge((blue, green, red))
    # print(correct_param)
    return output

def normalize_image_color_channels(image):

    channels = cv2.split(image)

    # Normalize each channel
    channels_norm = [cv2.normalize(chan, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) for chan in channels]

    # Merge the channels back together
    normalized_image = cv2.merge(channels_norm)
    return normalized_image

def histogram_equalization(image,clipLimit=7.0, tileSize=8):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a CLAHE object
    # You can experiment with the 'clipLimit' and 'tileGridSize' parameters
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileSize, tileSize))

    # Apply CLAHE to the grayscale image
    clahe_img = clahe.apply(gray)

    # If the input image is color, we should apply CLAHE on the L channel in the LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    l_clahe = clahe.apply(l)

    # Merge the CLAHE enhanced L channel back with A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))

    # Convert back to BGR color space
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return image_clahe


def extract_lines(image,low_white,high_white,lower_yellow,upper_yellow):
    equalized_image=gamma_correct(image,equalizeHist=False)
    #equalized_image= histogram_equalization(gamma_image)
    gray = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2HSV)

    # Convert the image to HLS colorspace
    hls = cv2.cvtColor(equalized_image, cv2.COLOR_RGB2HLS)

    # Extract saturation and lightness channels
    saturation = hls[:, :, 2]
    lightness = hls[:, :, 1]


    gblur = cv2.GaussianBlur(gray, (9,9), 0)
    mask = np.zeros_like(gblur)
    mask[(gblur >= low_white) & (gblur <= high_white)] = 1  # Set pixels within the range to 1

    # Apply the mask to the image
    white_areas = gblur * mask
    yellow_areas = cv2.inRange(hsv, np.array(lower_yellow), np.array(upper_yellow))
    bitwise_OR = cv2.bitwise_or(white_areas, yellow_areas)
    #bitwise_OR=white_areas
    return white_areas, yellow_areas, bitwise_OR



def apply_mask(image,polygonto):
    height, width = image.shape
    img = np.zeros_like(image)
    mask = cv2.fillPoly(img, pts=[polygonto], color =(255,255,255))
    masked_image = cv2.multiply(mask, image)
    return mask, masked_image


# Function that gives the left fit and right fit curves for the lanes in birdeye's view
def sliding_window(img):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 80
    margin = 70
    minpix = 50
    window_height = int(img.shape[0] / nwindows)
    y, x = img.nonzero()
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_indices = []
    right_lane_indices = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        good_left_indices = \
        ((y >= win_y_low) & (y < win_y_high) & (x >= win_xleft_low) & (x < win_xleft_high)).nonzero()[0]
        good_right_indices = \
        ((y >= win_y_low) & (y < win_y_high) & (x >= win_xright_low) & (x < win_xright_high)).nonzero()[0]
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)
        if len(good_left_indices) > minpix:
            leftx_current = int(np.mean(x[good_left_indices]))
        if len(good_right_indices) > minpix:
            rightx_current = int(np.mean(x[good_right_indices]))

    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    return left_lane_indices, right_lane_indices


def poly_fit(img, left_lane_indices, right_lane_indices):
    # Extract non-zero pixel locations
    y, x = img.nonzero()

    # Filter based on indices
    leftx = x[left_lane_indices]
    lefty = y[left_lane_indices]
    rightx = x[right_lane_indices]
    righty = y[right_lane_indices]

    # Check if any of the arrays are empty and handle accordingly
    if leftx.size > 0 and lefty.size > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = None  # or a default value

    if rightx.size > 0 and righty.size > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = None  # or a default value

    return left_fit, right_fit

# Function that give pixel location of points through which the curves of detected lanes passes
def get_pixel_location(img_shape, left_fit, right_fit):
    y = np.linspace(0, img_shape[0] - 1, img_shape[0])
    pts_left = np.array([np.transpose(np.vstack([np.polyval(left_fit, y), y]))]) if left_fit is not None else None
    pts_right = np.array([np.flipud(np.transpose(np.vstack([np.polyval(right_fit, y), y])))]) if right_fit is not None else None
    return pts_left, pts_right

def recognize_in_dark(frame,prev_left_fit,prev_right_fit):
    frame_equalized=histogram_equalization(frame)
    frame_equalized = normalize_image_color_channels(frame_equalized)

    polygon = np.array(([[421,623],[319,710],[899,710],[675,600]]),dtype=int)
    polygon_to_extract = np.array(([[340,700],[340,600],[710,600],[1024,700]]),dtype=int)

    white_areas, yellow_areas, bitwise_OR = extract_lines(frame_equalized,160,190,[3,11,175],[15,18,192])
    mask, masked_image = apply_mask(bitwise_OR,polygon_to_extract)


    left_lane_indices, right_lane_indices = sliding_window(masked_image)

    left_fit, right_fit = poly_fit(masked_image, left_lane_indices, right_lane_indices)

    # Check if there was no lane detection
    if left_fit is not None or right_fit is not None :
        if left_fit is not None or right_fit is not None:
            if left_fit is not None:
                left_fit = 0.01 * left_fit + 0.99 * prev_left_fit
                prev_left_fit = left_fit
            if right_fit is not None:
                right_fit = 0.01 * right_fit + 0.99 * prev_right_fit
                prev_right_fit = right_fit
        pts_left, pts_right = get_pixel_location(masked_image.shape, left_fit, right_fit)
        # Instead of fill_lane, directly draw lines or fill between lines based on what's detected
        img = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)
        if pts_left is not None:
            cv2.polylines(img, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=15)
        if pts_right is not None:
            cv2.polylines(img, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=15)
        detected_lane = img

        output = cv2.addWeighted(frame, 1, detected_lane, 1, 0)
        mask = np.zeros_like(frame)  # Assuming 'frame' is your image with the lines

        cv2.fillPoly(mask, [polygon], (255, 255, 255))
        frame_with_lines = output
        # Apply the mask to the image with the lines
        lanes_masked = cv2.bitwise_and(frame_with_lines, mask)

        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # Apply the inverted mask to the original image
        original_masked = cv2.bitwise_and(frame, mask_inv)

        # Combine the two images
        result = cv2.add(lanes_masked, original_masked)


    else:
        result = frame

    return result ,left_fit,right_fit




video = cv2.VideoCapture(r"C:\Users\User\my notebook\tryingss\curvey new.mp4")
out = cv2.VideoWriter(r'C:\Users\User\my notebook\tryingss\output5.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (1280, 720))
first_run = True
frame_count = 0
max_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

while True and frame_count<=max_frames:
    isTrue, frame = video.read()
    if isTrue == False:
        break
    frame_count += 1
    height, width, channels = frame.shape
    frame_equalized=histogram_equalization(frame)
    frame_equalized = normalize_image_color_channels(frame_equalized)

    polygon = np.array(([[421,623],[319,710],[899,710],[675,600]]),dtype=int)
    polygon_to_extract = np.array(([[340,700],[340,600],[710,600],[1024,700]]),dtype=int)
    white_areas, yellow_areas, bitwise_OR = extract_lines(frame_equalized,195,255,[15, 65, 235],[25, 100, 255])
    mask, masked_image = apply_mask(bitwise_OR,polygon_to_extract)


    left_lane_indices, right_lane_indices = sliding_window(masked_image)



    left_fit, right_fit = poly_fit(masked_image, left_lane_indices, right_lane_indices)
    if frame_count==1:
        prev_right_fit=right_fit
        prev_left_fit = left_fit
    # Check if there was no lane detection
    if left_fit is not None or right_fit is not None :
        if left_fit is not None:
            left_fit = 0.1*left_fit + 0.9*prev_left_fit
            prev_left_fit= left_fit
        if right_fit is not None:
            right_fit = 0.1*right_fit + 0.9*prev_right_fit
            prev_right_fit = right_fit
        pts_left, pts_right = get_pixel_location(masked_image.shape, left_fit, right_fit)
        # Instead of fill_lane, directly draw lines or fill between lines based on what's detected
        img = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)
        if pts_left is not None:
            cv2.polylines(img, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=15)
        if pts_right is not None:
            cv2.polylines(img, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=15)
        detected_lane = img

        output = cv2.addWeighted(frame, 1, detected_lane, 1, 0)
        mask = np.zeros_like(frame)  # Assuming 'frame' is your image with the lines

        cv2.fillPoly(mask, [polygon], (255, 255, 255))
        frame_with_lines = output
        # Apply the mask to the image with the lines
        lanes_masked = cv2.bitwise_and(frame_with_lines, mask)

        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # Apply the inverted mask to the original image
        original_masked = cv2.bitwise_and(frame, mask_inv)

        # Combine the two images
        result = cv2.add(lanes_masked, original_masked)

    else:
        result,prev_left_fit,prev_right_fit = recognize_in_dark(frame,prev_left_fit,prev_right_fit)






    # Write the output frame to the video
    out.write(result)





out.release()
print("Video output generated.\n")