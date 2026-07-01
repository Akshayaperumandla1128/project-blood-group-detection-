import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from preprocessing import enhance_image, extract_minutiae_features
from model_utils import predict_blood_group
import time

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def identify_pattern_type(image_path):
    """Geometric heuristic to identify the core pattern."""
    img = cv2.imread(image_path, 0)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30)
    if circles is not None: return "Whorl"
    
    height, width = img.shape
    top_half_mean = np.mean(img[0:height//3, :])
    return "Arch" if top_half_mean < 230 else "Loop"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file: return jsonify({'status': 'Error'})

    # Unique filename with timestamp to prevent browser cache issues
    filename = f"{int(time.time())}_{file.filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    # 1. Processing
    enhanced_img = enhance_image(path)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], "enhanced_" + filename), enhanced_img)
    
    # 2. Logic
    pattern = identify_pattern_type(path) # Returns Whorl, Loop, or Arch
    minutiae = extract_minutiae_features(path)
    
    # 3. Final Result
    result = predict_blood_group(path, minutiae)
    result['pattern_type'] = pattern # Critical for your HTML display

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)