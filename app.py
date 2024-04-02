import cv2
import os
import numpy as np
import csv
from fileinput import filename 
from flask import *

from cv2 import dnn_superres
import sys

# bg
import torch
from torchvision import transforms
# end bg

# rembg
from rembg import remove
from PIL import Image
#

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
    # kernel = np.array([[-1,-1,-1],
    #                    [-1,9,-1],
    #                    [-1,-1,-1]])
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel = kernel / np.sum(kernel)
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


# remove background

import cv2
import numpy as np

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
    _, binary_image = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    masked_image = cv2.bitwise_and(image, image, mask=binary_image)
    return masked_image

# org testing rembg end
# @app.route('/bg-image', methods=['POST'])
# def bg():
#     # if 'file' not in request.files:
#     #     return redirect(request.url)
#     # file = request.files['file']
#     # if file.filename == '':
#     #     return redirect(request.url)
#     # if file:
#     #     deeplab_model = load_model()
#     #     foreground, bin_mask = remove_background(deeplab_model, file)
#     #     final_image = custom_background("background_file.jpg", foreground)
#     #     final_image.save("output_path.jpg")
#     #     return send_file("output_path.jpg", as_attachment=True)
#     image = cv2.imread('image.JPG')
#     convertedImage = removeBg(image)
#
#     output_path = "out.jpg"
#     cv2.imwrite(output_path, convertedImage)
#     return "completed"


# removebg testing
def load_model():
    model_script = torch.load('deeplabv3.py')
    model = model_script.MyModel()
    model.eval()
    return model

def make_transparent_foreground(pic, mask):
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    a = np.ones(mask.shape, dtype='uint8') * 255
    alpha_im = cv2.merge([b, g, r, a], 4)
    bg = np.zeros(alpha_im.shape)
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)
    return foreground

# def remove_background(model, input_file):
#     input_image = Image.open(input_file)
#     preprocess = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     input_tensor = preprocess(input_image)
#     input_batch = input_tensor.unsqueeze(0)
#
#     if torch.cuda.is_available():
#         input_batch = input_batch.to('cuda')
#         model.to('cuda')
#
#     with torch.no_grad():
#         output = model(input_batch)['out'][0]
#     output_predictions = output.argmax(0)
#     mask = output_predictions.byte().cpu().numpy()
#     background = np.zeros(mask.shape)
#     bin_mask = np.where(mask, 255, background).astype(np.uint8)
#     foreground = make_transparent_foreground(input_image, bin_mask)
#     return foreground, bin_mask
#
# def custom_background(background_file, foreground):
#     final_foreground = Image.fromarray(foreground)
#     background = Image.open(background_file)
#     x = (background.size[0]-final_foreground.size[0])/2 + 0.5
#     y = (background.size[1]-final_foreground.size[1])/2 + 0.5
#     box = (x, y, final_foreground.size[0] + x, final_foreground.size[1] + y)
#     crop = background.crop(box)
#     final_image = crop.copy()
#     paste_box = (0, final_image.size[1] - final_foreground.size[1], final_image.size[0], final_image.size[1])
#     final_image.paste(final_foreground, paste_box, mask=final_foreground)
#     return final_image



# testing rembg
# def apply_black_background(image):
#     # Create a new blank image with the same size as the input image and a black background
#     black_background = Image.new("RGB", image.size, (0, 0, 0))
#
#     # Paste the input image onto the black background image
#     black_background.paste(image, (0, 0), image)
#
#     return black_background
#
# def remove_background(input_image_path, output_image_path):
#     with open(input_image_path, "rb") as f:
#         input_image = Image.open(f).convert("RGBA")  # Open the input image and convert it to RGBA mode
#
#     output_image = remove(input_image)
#
#     # Apply black background to the output image
#     output_image_with_black_bg = apply_black_background(output_image)
#     # Save the image with black background to the output file
#     output_image_with_black_bg.save(output_image_path, "PNG")
#

    #     output_path = os.path.join(correct_folder, filename)
    #     cv2.imwrite(output_path, convertedImage)
    #
    #
    #     files_processed += 1
    #
    # return f'Processed {files_processed} files '
    #
    # input_path = 'image (575).JPG'
    # output_path = 'new_output_normal.png'
    #
    #
    #     image = cv2.imread('new_output_normal.png')
    #     convertedImage=removeBg(image)
    #     cv2.imwrite("new.jpg", convertedImage)





# @app.route('/bg-rem', methods = ['POST'])
# def bgrem_mul():
#     # folder_path = r"TSLS/TSLS_images"
#     # correct_folder = r"TSLS/TSLS_Output/correct"
#     # incorrect_folder = r"TSLS/TSLS_Output/incorrect"
#
#     # withoutEnhanced
#     folder_path = "./SuperResolution/TSLS/TSLS_enhanced/TSLS_images"
#     correct_folder = './SuperResolution/TSLS/TSLS_enhanced/withoutEnhancement/correct_folder'  # change
#     incorrect_folder = './SuperResolution/TSLS/TSLS_enhanced/withoutEnhancement/incorrect_folder'  # change
#
#     # withEnhanced
#     # folder_path = './SuperResolution/TSLS/TSLS_enhanced/TSLS_enhanced_images/'
#     # correct_folder = './SuperResolution/TSLS/TSLS_enhanced/withEnhancement/correct_folder'  # change
#     # incorrect_folder = './SuperResolution/TSLS/TSLS_enhanced/withEnhancement/incorrect_folder'  # change
#
#     files_processed=0
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".JPG") or filename.endswith(".jpg"):
#             print(f"checking: {filename}...")
#             img_path = os.path.join(folder_path, filename)
#             image = cv2.imread(img_path)
#             convertedImage = removeBg(image)
#
#             has_gray = detect_gray(convertedImage)
#             has_brown = detect_brown(convertedImage)
#             has_yellow = detect_yellow(convertedImage)
#
#             if not os.path.exists(folder_path):
#                 return 'Folder not found'
#             if not os.path.exists(correct_folder):
#                 return 'correct_folder Folder not found'
#             if not os.path.exists(incorrect_folder):
#                 return 'incorrect_folder Folder not found'
#
#             if has_brown:
#                 print(f"{filename} has brown spots")
#
#             if has_gray:
#                 print(f"{filename} has gray spots")
#
#             if has_yellow:
#                 print(f"{filename} has yellow spots")
#
#             # Save the image in the output folder
#             if has_gray and has_brown and has_yellow:
#                 output_path = os.path.join(correct_folder, filename)
#                 cv2.imwrite(output_path, convertedImage)
#             else:
#                 output_path = os.path.join(incorrect_folder, filename)
#                 cv2.imwrite(output_path, convertedImage)
#
#             files_processed += 1
#
#     return f'Processed {files_processed} files '



# file upload method
@app.route('/shapen-image1', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        input_image = Image.open('./TMV/TMV_images/image (1).JPG')
        # input_array = np.array(input_image)
        # output_array = rembg.remove(input_array)
        # output_image = Image.fromarray(output_array)
        # image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        sharpened_image = sharpen_image(input_image)
        # bgimage = removeBg(image)
        # blurred_image = cv2.medianBlur(sharpened_image, 3)

        # ---------start
        # blurred_image = cv2.GaussianBlur(sharpened_image, (3, 3), 0)
        # sharpened_image = sharpen_image(image)
        # Display the blurred image
        # cv2.imshow("Blurred Image", blurred_image)

        # ---------end
        # Save the sharpened image locally
        output_path = 'output_image.png'
        cv2.imwrite(output_path, sharpened_image)
        # Return the path to the saved image
        return output_path



@app.route('/final_enhance_image', methods=["POST"])
# def main():
def enhanced_image_super_resolution():
    folder_path = './SuperResolution/TSLS/TSLS_images/'
    enhance_folder = './SuperResolution/TSLS/TSLS_Output/correct'

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
    folder_path = 'image (8).JPG'
    # e_folder_path = './SuperResolution/TMV/TMV_enhanced/TMV_enhanced_images/'
    correct_folder = './Test/Output/bg_image(8).jpg'

    with open(folder_path, 'rb') as i:
        input = i.read()
        output = remove(input, alpha_matting=True, alpha_matting_foreground_threshold=270,alpha_matting_background_threshold=20, alpha_matting_erode_size=11,bgcolor=(0, 0, 0, 255))
        with open(correct_folder, 'wb') as o:
            o.write(output)

    return "background removed"

# -------------------------------------TMV-----------------------------------------------
def detect_yellow(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the yellow color
    lower_yellow = np.array([20, 100, 60])
    upper_yellow = np.array([60, 255, 180])

    # Create a mask of the yellow areas
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) > 0:
        # At least one yellow spot was detected
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= w//4
            y -= h//4
            w += w//2
            h += h//2
            cv2.rectangle(image, (x, y), (x+w+3 , y+h+3), (0, 0, 255), 1)
        return True
    else:
        # No brown spots were detected
        return False

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
    
@app.route('/tmv', methods=["POST"])
# def main():   
def tmv():
    folder_path = './TMV/TMV_images/'
    # e_folder_path = './SuperResolution/TMV/TMV_enhanced/TMV_enhanced_images/'
    correct_folder = './TMV/TMV_Output/correct'
    incorrect_folder = './TMV/TMV_Output/incorrect'

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

        # print(convertedImage)
        # sys.exit(1)
        # Detect TMV spots
        has_brown = detect_brown(convertedImage)
        has_yellow = detect_yellow(convertedImage)

        # Save the image in the output folder
        if has_brown or has_yellow:
            output_path = os.path.join(correct_folder, filename)
            cv2.imwrite(output_path, convertedImage)
        else:
            output_path = os.path.join(incorrect_folder, filename)
            cv2.imwrite(output_path, convertedImage)
            
        files_processed += 1

    return f'Processed {files_processed} files '

# ----------------------------TSLS------------------------------------------------------

def detect_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 60])
    upper_yellow = np.array([60, 255, 180])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= w//4
            y -= h//4
            w += w//2
            h += h//2
        return True
    else:
        return False

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
            x -= w//4
            y -= h//4
            w += w//2
            h += h//2
            cv2.rectangle(image, (x, y), (x+w+3 , y+h+3), (255, 0, 0), 1)
        return True
    else:
        return False

@app.route('/tsls', methods = ['POST'])   
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

    files_processed=0
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
            x -= w//4
            y -= h//4
            w += w//2
            h += h//2
            cv2.rectangle(image, (x, y), (x+w+3 , y+h+3), (0, 0, 255), 1)
        return True
    else:
        return False

@app.route('/tylcv', methods = ['POST'])   
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
            x -= w//4
            y -= h//4
            w += w//2
            h += h//2
            cv2.rectangle(image, (x, y), (x+w+3 , y+h+3), (0, 63, 123), 2)
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

@app.route('/tts', methods = ['POST'])
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
            output_path = f'shapened_image_{len(output_paths)+1}.png'
            cv2.imwrite(output_path, convertedImage)
            # return 'TMV detection completed {output_path}'
        else:
            # output_path = os.path.join(incorrect_folder, filename)
            output_path = f'shapened_image_{len(output_paths)+1}.png'
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


if __name__=='__main__':
   app.run(debug = True,port=8081)
   
   
