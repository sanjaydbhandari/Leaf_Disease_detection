from flask import Flask, request, jsonify
import cv2
import numpy as np

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
        if area < 100 :
            continue
        print(area)

        # Compute the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Compute aspect ratio to distinguish between long and narrow shapes
        aspect_ratio = float(w) / h
        print(aspect_ratio)

        # Check if aspect ratio is greater than a threshold (indicating curling)
        if 0 < aspect_ratio < 3:
        # if 1.2< aspect_ratio < 2:
            curled_regions.append((x, y, x + w, y + h))

    return curled_regions

@app.route('/detect-yellow', methods=['POST'])
def detect_curl():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    # Read the image file
    image = request.files['image']
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = removeBg(img);
    # Detect leaf curl
    curled_regions = detect_leaf_curl(img)

    # Draw rectangles on the curled regions
    for region in curled_regions:
        x1, y1, x2, y2 = region
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite('processed_image.png', img)
    # Encode image to send as response
    _, encoded_img = cv2.imencode('.png', img)
    img_bytes = encoded_img.tobytes()

    return img_bytes

if __name__ == '__main__':
    app.run(debug=True,port=8082)