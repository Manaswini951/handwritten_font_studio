import io
import os
import math
import random
import zipfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.fontBuilder import FontBuilder

# Page Config
st.set_page_config(
    page_title="Hand-Drawn Style Font & T-Shirt Studio",
    page_icon="✍️",
    layout="wide"
)

# ============================================================
# UTILITIES & IMAGE PROCESSING
# ============================================================

def preprocess_reference_image(pil_image, contrast=1.2, brightness=0, denoise_strength=5):
    arr = np.array(pil_image.convert("RGB"))
    if contrast != 1.0 or brightness != 0:
        arr = cv2.convertScaleAbs(arr, alpha=contrast, beta=brightness)
        
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if denoise_strength > 0:
        gray = cv2.fastNlMeansDenoising(gray, h=denoise_strength)
        
    bg_size = max(31, int(min(gray.shape) * 0.05) | 1)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, bg_size, 10
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    preview_cleaned = cv2.cvtColor(255 - cleaned, cv2.COLOR_GRAY2RGBA)
    return cleaned, Image.fromarray(preview_cleaned)

def detect_and_extract_glyphs(binary_mask, min_area=100, max_area_pct=0.85):
    h_img, w_img = binary_mask.shape
    max_area = h_img * w_img * max_area_pct
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    extracted_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        if min_area <= area <= max_area and w > 5 and h > 5:
            component_crop = binary_mask[y:y+h, x:x+w]
            rgba_crop = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_crop[component_crop > 0] = [0, 0, 0, 255]
            
            extracted_components.append({
                "id": i,
                "bbox": (x, y, w, h),
                "area": area,
                "aspect_ratio": float(w) / float(h),
                "crop_mask": component_crop,
                "preview_img": Image.fromarray(rgba_crop, "RGBA"),
                "classification": "Letter" if (w * h > 300) else "Decoration",
                "assigned_char": ""
            })
            
    extracted_components.sort(key=lambda c: (c["bbox"][1] // 50, c["bbox"][0]))
    return extracted_components

def extract_style_profile(components):
    if not components:
        return {
            "stroke_thickness": 0.50, "roughness": 0.30, "aspect_ratio": 0.80,
            "height_variation": 0.20, "baseline_wobble": 0.15, "angularity": 0.40
        }
        
    aspect_ratios = [c["aspect_ratio"] for c in components]
    heights = [c["bbox"][3] for c in components]
    
    mean_aspect = float(np.mean(aspect_ratios)) if aspect_ratios else 0.8
    height_std = float(np.std(heights) / (np.mean(heights) + 1e-5))
    
    stroke_thicknesses, roughness_scores = [], []
    for c in components:
        mask = c["crop_mask"]
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        max_dist = np.max(dist) if np.max(dist) > 0 else 1.0
        stroke_thicknesses.append(max_dist)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            perimeter = cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            hull_perimeter = cv2.arcLength(hull, True)
            if hull_perimeter > 0:
                roughness_scores.append(perimeter / hull_perimeter)

    return {
        "stroke_thickness": float(min(1.0, np.mean(stroke_thicknesses) / 15.0)),
        "roughness": float(min(1.0, (np.mean(roughness_scores) - 1.0) if roughness_scores else 0.3)),
        "aspect_ratio": float(np.clip(mean_aspect, 0.3, 2.0)),
        "height_variation": float(np.clip(height_std, 0.0, 1.0)),
        "baseline_wobble": float(np.clip(height_std * 0.8, 0.0, 1.0)),
        "angularity": 0.35
    }

# ============================================================
# VECTORIZATION & TTF COMPILER
# ============================================================

def mask_to_font_contours(binary_crop, target_size=1000):
    h, w = binary_crop.shape
    if h == 0 or w == 0:
        return []
    padded = np.pad(binary_crop, 10, mode="constant", constant_values=0)
    contours, hierarchy = cv2.findContours(padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return []

    scale = float(target_size) / max(h, w)
    vector_contours = []
    
    for idx, cnt in enumerate(contours):
        pts = cnt.squeeze()
        if pts.ndim != 2 or len(pts) < 3:
            continue
        scaled_pts = []
        for pt in pts:
            x_val = int((pt[0] - 10) * scale)
            y_val = int((h - (pt[1] - 10)) *
