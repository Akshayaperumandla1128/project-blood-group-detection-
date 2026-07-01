import cv2
import numpy as np

def enhance_image(image_path):
    """Normalizes and binarizes the fingerprint for ridge detection."""
    img = cv2.imread(image_path, 0)
    if img is None: return None
    
    # Adaptive Histogram Equalization (CLAHE) for better ridge contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Adaptive Thresholding to create a clean black/white image
    enhanced = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    return enhanced

def extract_minutiae_features(image_path):
    """Calculates Crossing Number (CN) to find ridge endings and bifurcations."""
    img = cv2.imread(image_path, 0)
    if img is None: return np.array([0, 0])

    # Skeletonization
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    skeleton = cv2.ximgproc.thinning(binary)
    skeleton = (skeleton > 0).astype(int)
    
    endings, bifurcations = 0, 0
    rows, cols = skeleton.shape

    # 8-neighbor scan for Crossing Number
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 1:
                cells = [
                    skeleton[i-1, j-1], skeleton[i-1, j], skeleton[i-1, j+1],
                    skeleton[i, j+1],   skeleton[i+1, j+1], skeleton[i+1, j],
                    skeleton[i+1, j-1], skeleton[i, j-1],   skeleton[i-1, j-1]
                ]
                cn = 0.5 * sum(abs(cells[k] - cells[k+1]) for k in range(8))
                
                if cn == 1: endings += 1
                elif cn == 3: bifurcations += 1

    return np.array([endings, bifurcations])