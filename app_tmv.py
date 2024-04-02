import cv2
import os
import numpy as np
import csv
from fileinput import filename
from flask import *
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

from cv2 import dnn_superres
import sys

# bg
import torch
from torchvision import transforms
# end bg

# rembg
from rembg import remove
from PIL import Image
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from flask import Flask, send_file
from PIL import Image, ImageFilter
import io


app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html")

# Check Image is blury or not
def is_image_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# shapened image
def sharpen_image(image):
    kernel = np.array([[-1,-1,-1],
                       [-1,9,-1],
                       [-1,-1,-1]])
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel = kernel / np.sum(kernel)
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

@app.route('/shape', methods=['POST'])
def shape():
    image = cv2.imread("./SuperResolution/TSLS/new/WhatsApp Image 2024-03-19 at 12.57.25 PM.jpeg")
    sharpened_image = sharpen_image(image)
    # Save the sharpened image locally
    output_path = 'shapened_image1.png'
    cv2.imwrite(output_path, sharpened_image)

def removeBg(image):
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b = img_lab[:, :, 2]

    pixel_vals = b.flatten().astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    _, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((b.shape))
    _, binary_image = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked_image = cv2.bitwise_and(image, image, mask=binary_image)
    return masked_image

@app.route('/final_enhance_image', methods=["POST"])
def enhanced_image_super_resolution():
    # def main():
    folder_path = './SuperResolution/TSLS/new'
    enhance_folder = './SuperResolution/TSLS/TSLS_Output/'

    # folder_path = './SuperResolution/TSLS/TSLS_enhanced/TSLS_images'
    # enhance_folder = './SuperResolution/TSLS/TSLS_enhanced/TSLS_enhanced_images'

    # folder_path = './SuperResolution/TYLCV/TYLCV_enhanced/TYLCV_images'
    # enhance_folder = './SuperResolution/TYLCV/TYLCV_enhanced/TYLCV_enhanced_images'

    # folder_path = './SuperResolution/TTS/TTS_enhanced/TTS_images'
    # enhance_folder = './SuperResolution/TTS/TTS_enhanced/TTS_enhanced_images'

    if not os.path.exists(folder_path):
        return 'Folder not found11'

    files_processed = 0
    for filename in os.listdir(folder_path):
        # name = os.path.basename(filename)
        img_path = os.path.join(folder_path, filename)
        # image = cv2.imread(img_path)
        # image = removeBg(cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR))
        # convertedImage = removeBg(image)
        # convertedImage = super_resolution_upload(image)

        sr = dnn_superres.DnnSuperResImpl_create()
        path = 'EDSR_x4.pb'
        sr.readModel(path)
        sr.setModel('edsr', 4)
        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        image = removeBg(cv2.imread(img_path))
        upscaled = sr.upsample(image)
        # cv2.imwrite('upscaled_test.png', upscaled)

        output_path = os.path.join(enhance_folder, ('enhanced_' + filename))
        cv2.imwrite(output_path, upscaled)

        # Detect TMV spots
        # has_brown = detect_brown(convertedImage)
        # has_yellow = detect_yellow(convertedImage)

        # Save the image in the output folder
        # if has_brown or has_yellow:
        #     output_path = os.path.join(correct_folder, filename)
        #     cv2.imwrite(output_path, convertedImage)
        # else:
        #     output_path = os.path.join(incorrect_folder, filename)
        #     cv2.imwrite(output_path, convertedImage)

        files_processed += 1

    return f'Processed {files_processed} files '


@app.route('/bg-rem', methods=['POST'])
def bgrem():
    folder_path = 'Detection/Yellow_detection/input_T/image (2).JPG'
    correct_folder = './Detection/Yellow_detection/img/bg_image(2).jpg'

    with open(folder_path, 'rb') as i:
        input = i.read()
        output = remove(input, alpha_matting=True, alpha_matting_foreground_threshold=270,
                        alpha_matting_background_threshold=20, alpha_matting_erode_size=11, bgcolor=(0, 0, 0, 255))
        with open(correct_folder, 'wb') as o:
            o.write(output)

    return "background removed"

def bgremover(image):
    folder_path = 'image (8).JPG'
    # e_folder_path = './SuperResolution/TMV/TMV_enhanced/TMV_enhanced_images/'
    correct_folder = './Test/Output/bg_image(8).jpg'

    with open(folder_path, 'rb') as i:
        input = i.read()
        output = remove(input, alpha_matting=True, alpha_matting_foreground_threshold=270,
                        alpha_matting_background_threshold=20, alpha_matting_erode_size=11, bgcolor=(0, 0, 0, 255))
        with open(correct_folder, 'wb') as o:
            o.write(output)

    return "background removed"


# -------------------------------------TMV-----------------------------------------------


def detect_yellow1(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([22, 93, 0], dtype="uint8")
    upper_yellow = np.array([30, 255, 180], dtype="uint8")

    # Create a mask of the yellow areas
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) > 0:
        # At least one yellow spot was detected
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= w // 4
            y -= h // 4
            w += w // 2
            h += h // 2
            cv2.rectangle(image, (x, y), (x + w + 3, y + h + 3), (0, 255, 255), 1)
        return True
    else:
        # No brown spots were detected
        return False

def detect_brown1(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 100, 100], dtype="uint8")
    upper_yellow = np.array([20, 255, 255], dtype="uint8")

    # Create a mask of the yellow areas
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) > 0:
        # At least one yellow spot was detected
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= w // 4
            y -= h // 4
            w += w // 2
            h += h // 2
            cv2.rectangle(image, (x, y), (x + w + 3, y + h + 3), (0, 75, 150), 1)
        return True
    else:
        # No brown spots were detected
        return False


def detect_yellow2(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([22, 93, 0], dtype="uint8")
    upper_yellow = np.array([30, 255, 180], dtype="uint8")

    # Create a mask of the yellow areas
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list to store outermost contours
    outer_contours = []

    # Check if any contours were found
    if len(contours) > 0:
        # Process contour hierarchy to identify outermost contours
        for i in range(len(contours)):
            # Check if the contour has no parent and no child (outermost)
            if hierarchy[0][i][3] == -1 and hierarchy[0][i][2] == -1:
                outer_contours.append(contours[i])

        # Draw rectangles for outermost contours
        for contour in outer_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= w // 4
            y -= h // 4
            w += w // 2
            h += h // 2
            cv2.rectangle(image, (x, y), (x + w + 3, y + h + 3), (0, 255, 255), 1)

        return True
    else:
        # No yellow spots were detected
        return False
def detect_yellow_group(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([22, 93, 0], dtype="uint8")
    upper_yellow = np.array([30, 255, 180], dtype="uint8")

    # Create a mask of the yellow areas
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) > 0:
        # Combine all contours to get a bounding rectangle for the group
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Draw a bounding rectangle around the group of yellow areas
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
        return True
    else:
        # No yellow areas were detected
        return False
def detect_brown(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds of the brown color
    # lower_brown1 = np.array([0, 20, 20])
    # upper_brown1 = np.array([30, 255, 255])
    # lower_brown2 = np.array([151, 20, 20])
    # upper_brown2 = np.array([180, 255, 255])

    upper_brown1 = np.array([32, 58, 48])
    lower_brown1 = np.array([33, 47, 62])

    # Create masks for both brown color ranges and combine them
    mask = cv2.inRange(hsv, lower_brown1, upper_brown1)
    # mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    # mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) > 0:
        contour_list = []
        # At least one brown spot was detected
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= w//4
            y -= h//4
            w += w//2
            h += h//2
            cv2.rectangle(image, (x, y), (x+w+3 , y+h+3), (255, 0, 0), 1)
            contour_list.append([x, y, w, h])
        plt.imshow(image)
        return True
    else:
        # No brown spots were detected
        return False

def mark_curled(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        _, thresholded = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 6. Find the largest contour
        curled_contour = max(contours, key=cv2.contourArea)

        # 7. Draw a rectangle around the curled part of the leaf
        x, y, w, h = cv2.boundingRect(curled_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return True
    except Exception as e:
        print("An error occurred:", str(e))
        return False


def mark_curled1(img, threshold=80):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Analyze the orientation of detected lines
    curl_detected = False
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            print(abs(angle))
            if abs(angle) <= 200:
                # Draw the line on the image
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                curl_detected = True

    # Return true if curl detected and angle is greater than threshold
    return curl_detected



@app.route('/tmv-detect', methods=["POST"])
# def main():
def tmv():
    folder_path = './Detection/Yellow_detection/org_img'
    # e_folder_path = './SuperResolution/TMV/TMV_enhanced/TMV_enhanced_images/'
    correct_folder = './Detection/Yellow_detection/Y_detect'
    incorrect_folder = './Detection/Yellow_detection/N_Y_detect'

    if not os.path.exists(folder_path):
        return 'Folder not found'
    if not os.path.exists(correct_folder):
        return 'correct_folder Folder not found'
    if not os.path.exists(incorrect_folder):
        return 'incorrect_folder Folder not found'

    files_processed = 0
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)
        # convertedImage = sharpen_image(removeBg(image))
        convertedImage = removeBg(image)
        # image_np = np.fromfile(image, np.uint8)
        # image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # print(convertedImage)
        # sys.exit(1)
        # Detect TMV spots
        has_yellow = detect_yellow1(convertedImage)
        has_brown = detect_brown1(convertedImage)
        # has_curl = mark_curled1(image)

        # Save the image in the output folder
        if has_yellow:#if has_brown or has_yellow
            output_path = os.path.join(correct_folder, filename)
            cv2.imwrite(output_path, convertedImage)
        else:
            output_path = os.path.join(incorrect_folder, filename)
            cv2.imwrite(output_path, convertedImage)

        files_processed += 1

    return f'Processed {files_processed} files '


# ----------------------------TSLS------------------------------------------------------

# def detect_yellow(image):
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
#     lower_yellow = np.array([20, 100, 60])
#     upper_yellow = np.array([60, 255, 180])
#
#     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
#
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     if len(contours) > 0:
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             x -= w // 4
#             y -= h // 4
#             w += w // 2
#             h += h // 2
#         return True
#     else:
#         return False


def detect_gray(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([0, 0, 0])
    upper_gray = np.array([180, 50, 150])
    lower_tan = np.array([0, 20, 151])
    upper_tan = np.array([180, 50, 255])

    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    tan_mask = cv2.inRange(hsv, lower_tan, upper_tan)

    combined_mask = cv2.bitwise_or(gray_mask, tan_mask)

    contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= w // 4
            y -= h // 4
            w += w // 2
            h += h // 2
        return True
    else:
        return False


def detect_brown(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_brown1 = np.array([0, 20, 20])
    upper_brown1 = np.array([30, 255, 255])
    lower_brown2 = np.array([151, 20, 20])
    upper_brown2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= w // 4
            y -= h // 4
            w += w // 2
            h += h // 2
            cv2.rectangle(image, (x, y), (x + w + 3, y + h + 3), (255, 0, 0), 1)
        return True
    else:
        return False


@app.route('/tsls', methods=['POST'])
def tsls():
    # folder_path = r"TSLS/TSLS_images"
    # correct_folder = r"TSLS/TSLS_Output/correct"
    # incorrect_folder = r"TSLS/TSLS_Output/incorrect"

    # withoutEnhanced
    folder_path = "./SuperResolution/TSLS/TSLS_enhanced/TSLS_images"
    correct_folder = './SuperResolution/TSLS/TSLS_enhanced/withoutEnhancement/correct_folder'  # change
    incorrect_folder = './SuperResolution/TSLS/TSLS_enhanced/withoutEnhancement/incorrect_folder'  # change

    # withEnhanced
    # folder_path = './SuperResolution/TSLS/TSLS_enhanced/TSLS_enhanced_images/'
    # correct_folder = './SuperResolution/TSLS/TSLS_enhanced/withEnhancement/correct_folder'  # change
    # incorrect_folder = './SuperResolution/TSLS/TSLS_enhanced/withEnhancement/incorrect_folder'  # change

    files_processed = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG") or filename.endswith(".jpg"):
            print(f"checking: {filename}...")
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            convertedImage = removeBg(image)

            has_gray = detect_gray(convertedImage)
            has_brown = detect_brown(convertedImage)
            has_yellow = detect_yellow(convertedImage)

            if not os.path.exists(folder_path):
                return 'Folder not found'
            if not os.path.exists(correct_folder):
                return 'correct_folder Folder not found'
            if not os.path.exists(incorrect_folder):
                return 'incorrect_folder Folder not found'

            if has_brown:
                print(f"{filename} has brown spots")

            if has_gray:
                print(f"{filename} has gray spots")

            if has_yellow:
                print(f"{filename} has yellow spots")

            # Save the image in the output folder
            if has_gray and has_brown and has_yellow:
                output_path = os.path.join(correct_folder, filename)
                cv2.imwrite(output_path, convertedImage)
            else:
                output_path = os.path.join(incorrect_folder, filename)
                cv2.imwrite(output_path, convertedImage)

            files_processed += 1

    return f'Processed {files_processed} files '


# ----------------------------TYLCV------------------------------------------------------

def mark_curled(image):
    try:
        # 1. Convert the image to the LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # 2. Split the LAB image into channels
        l, a, b = cv2.split(lab)

        # 3. Apply adaptive thresholding on the 'a' channel
        _, thresholded = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Morphological operations to enhance the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

        # 5. Find contours in the thresholded image
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 6. Find the largest contour
        curled_contour = max(contours, key=cv2.contourArea)

        # 7. Draw a rectangle around the curled part of the leaf
        x, y, w, h = cv2.boundingRect(curled_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return True
    except Exception as e:
        print("An error occurred:", str(e))
        return False


def detect_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([44, 117, 82])
    upper_yellow = np.array([44, 255, 136])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= w // 4
            y -= h // 4
            w += w // 2
            h += h // 2
            cv2.rectangle(image, (x, y), (x + w + 3, y + h + 3), (0, 0, 255), 1)
        return True
    else:
        return False


@app.route('/tylcv', methods=['POST'])
def tylcv():
    # folder_path = r"TYLCV/TYLCV_images"
    # correct_folder = r"TYLCV/TYLCV_Output/correct"
    # incorrect_folder = r"TYLCV/TYLCV_Output/incorrect"

    # withoutEnhanced
    # folder_path = "./SuperResolution/TYLCV/TYLCV_enhanced/TYLCV_images"
    # correct_folder = './SuperResolution/TYLCV/TYLCV_enhanced/withoutEnhancement/correct_folder'  # change
    # incorrect_folder = './SuperResolution/TYLCV/TYLCV_enhanced/withoutEnhancement/incorrect_folder'  # change

    # withEnhanced
    folder_path = "./SuperResolution/TYLCV/TYLCV_enhanced/TYLCV_enhanced_images"
    correct_folder = './SuperResolution/TYLCV/TYLCV_enhanced/withEnhancement/correct_folder'  # change
    incorrect_folder = './SuperResolution/TYLCV/TYLCV_enhanced/withEnhancement/incorrect_folder'  # change

    if not os.path.exists(folder_path):
        return 'Folder not found'
    if not os.path.exists(correct_folder):
        return 'correct_folder Folder not found'
    if not os.path.exists(incorrect_folder):
        return 'incorrect_folder Folder not found'

    # withEnhanced
    # folder_path = './SuperResolution/TSLS/TSLS_enhanced/TSLS_enhanced_images/'
    # correct_folder = './SuperResolution/TSLS/TSLS_enhanced/withEnhancement/correct_folder'  # change
    # incorrect_folder = './SuperResolution/TSLS/TSLS_enhanced/withEnhancement/incorrect_folder'  # change

    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG") or filename.endswith(".jpg"):
            print(f"checking: {filename}...")
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            convertedImage = removeBg(img)

            has_curl = mark_curled(convertedImage)
            has_yellow = detect_yellow(convertedImage)

            if has_curl:
                print(f"{filename} has curls")

            if has_yellow:
                print(f"{filename} has yellow spots")

            # Save the image in the output folder
            if (has_curl and has_yellow):
                output_path = os.path.join(correct_folder, filename)
                cv2.imwrite(output_path, convertedImage)
            else:
                output_path = os.path.join(incorrect_folder, filename)
                cv2.imwrite(output_path, convertedImage)

    return 'tylcv detection completed'


# ----------------------------------------------TTS-----------------------------------------------
def detect_brown(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds of the brown color
    lower_brown1 = np.array([0, 20, 20])
    upper_brown1 = np.array([30, 255, 255])
    lower_brown2 = np.array([151, 20, 20])
    upper_brown2 = np.array([180, 255, 255])

    # Create masks for both brown color ranges and combine them
    mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) > 0:
        contour_list = []
        # At least one brown spot was detected
        for contour in contours:
            # print(contour)
            x, y, w, h = cv2.boundingRect(contour)
            x -= w // 4
            y -= h // 4
            w += w // 2
            h += h // 2
            cv2.rectangle(image, (x, y), (x + w + 3, y + h + 3), (0, 63, 123), 2)
            contour_list.append([x, y, w, h])
        # plt.imshow(image)
        return True, contour_list
    else:
        # No brown spots were detected
        # return False, contour_list
        return False, []


def create_csv(csv_file, csv_columns):
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()


@app.route('/tts', methods=['POST'])
def tts():
    # folder_path = "./SuperResolution/TTS/TTS_images"
    # correct_folder = "./SuperResolution/TTS/TTS_Output/correct"`
    # incorrect_folder = "./SuperResolution/TTS/TTS_Output/incorrect"

    # withEnhanced
    folder_path = './SuperResolution/TTS/TTS_enhanced/TTS_enhanced_images/'
    correct_folder = './SuperResolution/TTS/TTS_enhanced/withEnhancement/correct_folder'  # change
    incorrect_folder = './SuperResolution/TTS/TTS_enhanced/withEnhancement/incorrect_folder'  # change

    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG") or filename.endswith(".jpg"):
            print(f"checking: {filename}...")
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            convertedImage = removeBg(img)

            has_brown = detect_brown(convertedImage)

            if has_brown:
                print(f"{filename} has Brown spots")

            # Save the image in the output folder
            if (has_brown):
                output_path = os.path.join(correct_folder, filename)
                cv2.imwrite(output_path, convertedImage)
            else:
                output_path = os.path.join(incorrect_folder, filename)
                cv2.imwrite(output_path, convertedImage)

    return 'TTS detection completed'


# def tts():
#     # Get uploaded files
#     uploaded_files = request.files.getlist('files')
#
#     # Folder paths
#     folder_path = "/SuperResolution/TTS/TTS_images"
#     correct_folder = "/SuperResolution/TTS/TTS_Output/correct"
#     incorrect_folder = "/SuperResolution/TTS/TTS_Output/incorrect"
#     csv_file = "/SuperResolution/TTS/tts_contours.csv"
#     csv_columns = ["filename", "contours"]
#
#     if not os.path.exists(csv_file):
#         create_csv(csv_file, csv_columns)
#
#     with open(csv_file, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=csv_columns)
#         writer.writeheader()
#
#     for file in uploaded_files:
#         if file.filename == '':
#             continue
#         filename = file.filename
#         if filename.endswith(".JPG") or filename.endswith(".jpg"):
#             image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
#             has_tts, contour_list = detect_tts(image)
#             if contour_list:
#                 with open(csv_file, "a", newline="") as f:
#                     writer = csv.DictWriter(f, fieldnames=csv_columns)
#                     writer.writerow({"filename": filename, "contours": contour_list})
#             if has_tts:
#                 output_path = os.path.join(correct_folder, filename)
#                 cv2.imwrite(output_path, image)
#             else:
#                 output_path = os.path.join(incorrect_folder, filename)
#                 cv2.imwrite(output_path, image)
#     return jsonify({'message': 'Images processed successfully'})

# --------------------------------------------folder-----------------------------------------------
@app.route('/mkdir')
def mkdir():
    if not os.path.exists("TYLCV"):
        os.makedirs("TYLCV/TYLCV_images")
        os.makedirs("TYLCV/TYLCV_Output/correct")
        os.makedirs("TYLCV/TYLCV_Output/incorrect")

    if not os.path.exists("TTS"):
        os.makedirs("TTS/TTS_images")
        os.makedirs("TTS/TTS_Output/correct")
        os.makedirs("TTS/TTS_Output/incorrect")

    if not os.path.exists("TMV"):
        os.makedirs("TMV/TMV_images")
        os.makedirs("TMV/TMV_Output/correct")
        os.makedirs("TMV/TMV_Output/incorrect")

    if not os.path.exists("TSLS"):
        os.makedirs("TSLS/TSLS_images")
        os.makedirs("TSLS/TSLS_Output/correct")
        os.makedirs("TSLS/TSLS_Output/incorrect")
    return "Dir Created Successfully"
    # if not os.path.exists("Master"):
    #     os.makedirs("Master/input_images")
    #     os.makedirs("Master/output_images")
    #     os.makedirs("Master/test_images")
    #     os.makedirs("Master/TMV")
    #     os.makedirs("Master/TSLS")
    #     os.makedirs("Master/TTS")
    #     os.makedirs("Master/TYLCV")


# -----------------------------testing -----------------------------------------
@app.route('/detect-tmv', methods=['POST'])
def detect_tmv():
    if 'file' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('file')  # Get list of uploaded files

    output_paths = []

    for file in files:
        if file.filename == '':
            return redirect(request.url)

        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # blurry = is_immage_blurry(image)

        # if blurry:
        #     # Sharpen the image
        #     sharpened_image = sharpen_image(image)
        #     # Save the sharpened image locally
        #     output_path = f'shapened_image_{len(output_paths)+1}.png'
        #     cv2.imwrite(output_path, sharpened_image)
        #     output_paths.append(output_path)
        # else:
        #     return 'Image is not blurry'

        convertedImage = removeBg(image)

        # Detect TMV spots
        has_brown = detect_brown(convertedImage)
        has_yellow = detect_yellow(convertedImage)

        # Save the image in the output folder
        if has_brown or has_yellow:
            # output_path = os.path.join(correct_folder, filename)
            # print(f"{filename} has TMV")
            output_path = f'shapened_image_{len(output_paths) + 1}.png'
            cv2.imwrite(output_path, convertedImage)
            # return 'TMV detection completed {output_path}'
        else:
            # output_path = os.path.join(incorrect_folder, filename)
            output_path = f'shapened_image_{len(output_paths) + 1}.png'
            cv2.imwrite(output_path, convertedImage)
            # return 'failed to detect TMV {output_path}'

    # return ", ".join(output_paths)

    # for filename in os.listdir(folder_path):
    #     if filename.endswith(".JPG") or filename.endswith(".jpg"):
    #         print(f"checking: {filename}...")
    #         img_path = os.path.join(folder_path, filename)
    #         image = cv2.imread(img_path)

    #         # Perform background removal
    #         convertedImage = removeBg(image)

    #         # Detect TMV spots
    #         has_brown = detect_brown(convertedImage)
    #         has_yellow = detect_yellow(convertedImage)

    #         # Save the image in the output folder
    #         if has_brown or has_yellow:
    #             output_path = os.path.join(correct_folder, filename)
    #             print(f"{filename} has TMV")
    #             cv2.imwrite(output_path, convertedImage)
    #         else:
    #             output_path = os.path.join(incorrect_folder, filename)
    #             cv2.imwrite(output_path, convertedImage)

    # return 'TMV detection completed'


# @app.route('/ld', methods = ['POST'])
# def success():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
#         if file:
#             image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
#             kernel = np.array([[-1,-1,-1],
#                             [-1,9,-1],
#                             [-1,-1,-1]])
#             kernel = kernel / np.sum(kernel)
#             sharpened_image = cv2.filter2D(image, -1, kernel)
#             _, img_encoded = cv2.imencode('.png', sharpened_image)
#             return img_encoded
#             # file.save(file.filename)
#             # return img_encoded.tobytes()
#         # f.save(f.filename)


if __name__ == '__main__':
    app.run(debug=True, port=8081)


