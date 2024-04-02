import sys

import cv2
import numpy as np
import os
from flask import Flask, request, jsonify

app = Flask(__name__)
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
def detect_yellow(image,lower=[22, 93, 0],upper=[30, 255, 180]):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array(lower, dtype="uint8")
    upper_yellow = np.array(upper, dtype="uint8")

    # Create a mask of the yellow areas
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    yellow_detections = []
    if len(contours) > 0:
        # At least one yellow spot was detected
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= w // 4
            y -= h // 4
            w += w // 2
            h += h // 2
            yellow_detections.append((x, y, x + w + 3, y + h + 3))

    return yellow_detections
def remove_multiple_detections(detections):
    # Identify and remove overlapping or nested detections
    non_overlapping_detections = []
    for i, detection in enumerate(detections):
        is_inner_detection = False
        for other_detection in detections:
            if detection != other_detection and is_inside(detection, other_detection):
                # Check if the current detection is entirely contained within another detection
                is_inner_detection = True
                break
        if not is_inner_detection:
            non_overlapping_detections.append(detection)
    return non_overlapping_detections

def is_overlapping(detection1, detection2):
    # Check if two detections are overlapping
    x1, y1, x2, y2 = detection1
    x3, y3, x4, y4 = detection2
    return not (x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1)
def create_clusters(detections):
    clusters = []
    for detection in detections:
        inner_detection = False
        for other_detection in detections:
            if detection != other_detection and is_inside(detection, other_detection):
                inner_detection = True
                break
        if not inner_detection:
            clusters.append([detection])
    return clusters
def is_inside(inner, outer):
    # Check if inner detection is entirely contained within outer detection
    x1, y1, x2, y2 = inner
    x3, y3, x4, y4 = outer
    return x3 <= x1 and y3 <= y1 and x4 >= x2 and y4 >= y2

def merge_detections(image,lower,upper,border):
    detections = detect_yellow(image,lower,upper)
    non_overlapping_detections = remove_multiple_detections(detections)
    clusters = create_clusters(non_overlapping_detections)

    if clusters:  # if has_brown or has_yellow
        for cluster in clusters:
            for detection in cluster:
                x1, y1, x2, y2 = detection
                cv2.rectangle(image, (x1, y1), (x2, y2), border, 1)
    return clusters

def mark_curled(image):
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        _, thresholded = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        curled_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(curled_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return True, image

    except Exception as e:
        print("An error occurred:", str(e))
        return False, None

def detect_and_mark_curling(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 300, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            solidity = area / perimeter
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Define thresholds for curling detection
            area_threshold = 100  # Adjust as needed
            solidity_threshold = 0.8  # Adjust as needed
            aspect_ratio_threshold = 2.0  # Adjust as needed
            rect = cv2.minAreaRect(contour)
            angle = rect[2]

            # Check if the contour meets the curling criteria
            if area > area_threshold and solidity > solidity_threshold and aspect_ratio > aspect_ratio_threshold:
                return gray  # Curling detected

        # return False  # No curling detected

    except Exception as e:
        print("An error occurred:", str(e))
        return False

def compute_curvature(contour):
    # Ensure contour has at least three points
    curvature = []
    for i in range(len(contour)):
        # Get three neighboring points on the contour with wrap-around for the first and last points
        p1 = contour[(i - 1) % len(contour)]
        p2 = contour[i]
        p3 = contour[(i + 1) % len(contour)]

        # Calculate vectors between neighboring points
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)

        # Calculate cross product and its magnitude
        cross_product = np.cross(v1, v2)
        cross_mag = np.linalg.norm(cross_product)

        # Calculate magnitude of vectors
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)

        # Calculate curvature as the reciprocal of the radius of the circle passing through the three points
        if cross_mag == 0 or v1_mag == 0 or v2_mag == 0:
            curvature.append(float('inf'))
        else:
            curvature.append(2 * cross_mag / (v1_mag * v2_mag * (v1_mag + v2_mag)))

    return curvature

def detect_curls(contour, curvature, threshold):
    curl_points = []
    for i in range(len(curvature)):
        # Check if curvature exceeds the threshold
        if curvature[i] > threshold:
            curl_points.append(contour[i])

    return curl_points


# def detect_curl(image):
#     # Convert image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply Canny edge detection
#     edges = cv2.Canny(gray, 30, 150)
#
#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Analyze contours for curling
#     for contour in contours:
#         # Calculate properties of the contour
#         area = cv2.contourArea(contour)
#         perimeter = cv2.arcLength(contour, True)
#         solidity = area / perimeter
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = float(w) / h
#
#         # Define thresholds for curling detection
#         area_threshold = 100  # Adjust as needed
#         solidity_threshold = 0.8  # Adjust as needed
#         aspect_ratio_threshold = 2.0  # Adjust as needed
#
#         # Check if the contour meets the curling criteria
#         if area > area_threshold and solidity > solidity_threshold and aspect_ratio > aspect_ratio_threshold:
#             # If the contour meets the criteria, mark it as curl
#             cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
#
#     return image

def detect_leaf_curl(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    curled_regions = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # If the contour area is small, it might be noise, ignore it
        if area < 100:
            continue

        # Compute the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Compute aspect ratio to distinguish between long and narrow shapes
        aspect_ratio = float(w) / h

        # Check if aspect ratio is greater than a threshold (indicating curling)
        if aspect_ratio > 2:
            curled_regions.append((x, y, x + w, y + h))

    return curled_regions

@app.route('/detect-yellow', methods=["POST"])
def tmv():
    folder_path = './Detection/Yellow_detection/input_T'
    # e_folder_path = './SuperResolution/TMV/TMV_enhanced/TMV_enhanced_images/'
    correct_folder = './Detection/Yellow_detection/output_T'
    incorrect_folder = './Detection/Yellow_detection/Y_detect'

    if not os.path.exists(folder_path):
        return 'Folder not found'
    if not os.path.exists(correct_folder):
        return 'correct_folder Folder not found'
    if not os.path.exists(incorrect_folder):
        return 'incorrect_folder Folder not found'

    files_processed = 0
    for filename in os.listdir(folder_path):
        print(filename)
        img_path = os.path.join(folder_path, filename)
        image = cv2.imread(img_path)
        image = removeBg(image)
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        edges_detected=detect_leaf_curl(img)

        # if not edges_detected:
        #     print("yello",filename,"\n \n \n \n")

        output_path = os.path.join(correct_folder, filename)
        cv2.imwrite(output_path, edges_detected)
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # largest_contour = max(contours, key=cv2.contourArea)
        # curvature_threshold = 0.1
        # image_with_curls = detect_curls(image.copy(), largest_contour, curvature_threshold)
        #
        # output_path = os.path.join(correct_folder, filename)
        # cv2.imwrite(output_path, image_with_curls)

        # clusters = merge_detections(convertedImage,lower_yellow,upper_yellow,(0, 255, 255))
        # clusters = merge_detections(convertedImage, lower_brown, upper_brown,(0, 75, 150))

        # curling--testing
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # # Iterate over contours
        # for contour in contours:
        #     if len(contour) >= 3:
        #         # print(contour)
        #         # sys.exit()
        #         # Compute curvature along contour
        #         curvature = compute_curvature(contour)
        #         curl_points = detect_curls(contour, curvature, threshold=0.1)
        #         for point in curl_points:
        #             cv2.circle(convertedImage, tuple(point[0]), 5, (0, 0, 255), -1)
        #         # print(curvature)
        #         # sys.exit()
        #
        #         output_path = os.path.join(correct_folder, filename)
        #         cv2.imwrite(output_path, convertedImage)
        # curling end

        # if curl:
        #     output_path = os.path.join(correct_folder, filename)
        #     cv2.imwrite(output_path, convertedImage)
        # else:
        #     output_path = os.path.join(incorrect_folder, filename)
        #     cv2.imwrite(output_path, convertedImage)

        files_processed += 1

    return f'Processed {files_processed} files '
# working org
# def tmv():
#     folder_path = './Detection/Yellow_detection/org_img'
#     # e_folder_path = './SuperResolution/TMV/TMV_enhanced/TMV_enhanced_images/'
#     correct_folder = './Detection/Yellow_detection/Y_detect'
#     incorrect_folder = './Detection/Yellow_detection/N_Y_detect'
#
#     if not os.path.exists(folder_path):
#         return 'Folder not found'
#     if not os.path.exists(correct_folder):
#         return 'correct_folder Folder not found'
#     if not os.path.exists(incorrect_folder):
#         return 'incorrect_folder Folder not found'
#
#     files_processed = 0
#     for filename in os.listdir(folder_path):
#         img_path = os.path.join(folder_path, filename)
#         image = cv2.imread(img_path)
#         convertedImage = removeBg(image)
#         #org
#         # lower_yellow=[22, 93, 0];
#         # upper_yellow=[30, 255, 180];
#
#         lower_yellow = [22, 93, 0];
#         upper_yellow = [30, 255, 255];
#
#         lower_brown = [10, 100, 100];
#         upper_brown = [20, 255, 255];
#
#         clusters = merge_detections(convertedImage,lower_yellow,upper_yellow,(0, 255, 255))
#         clusters = merge_detections(convertedImage, lower_brown, upper_brown,(0, 75, 150))
#
#         if clusters:
#             output_path = os.path.join(correct_folder, filename)
#             cv2.imwrite(output_path, convertedImage)
#         else:
#             output_path = os.path.join(incorrect_folder, filename)
#             cv2.imwrite(output_path, convertedImage)
#
#         files_processed += 1
#
#     return f'Processed {files_processed} files '

# not sure
# def detect_yellow_route():
#     # Check if image file is provided
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'})
#
#     # Read the image file
#     image = request.files['image']
#     img_np = np.frombuffer(image.read(), np.uint8)
#     img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#
#     # Perform detection and clustering
#     clusters = merge_detections(img)
#     # print(clusters)
#     # sys.exit()
#
#     # Visualize clusters (optional)
#     for cluster in clusters:
#         for detection in cluster:
#             x1, y1, x2, y2 = detection
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
#
#     # Save the processed image locally
#     save_path = 'processed_image.png'
#     cv2.imwrite(save_path, img)
#
#     return jsonify({'processed_image_path': save_path})



if __name__ == "__main__":
    app.run(debug=True,port=8082)