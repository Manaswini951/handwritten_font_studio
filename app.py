import io
import os
import math
import random
import zipfile

import cv2
import numpy as np
import streamlit as st

from PIL import Image, ImageDraw, ImageFont

from fontTools.ttLib import TTFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hand-Drawn Font & T-Shirt Studio",
    page_icon="✍️",
    layout="wide"
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

UNITS_PER_EM = 1000
GLYPH_HEIGHT = 700
NUM_VARIATIONS = 50
PUA_START = 0xE000

CHARACTER_SET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
)

MIN_COMPONENT_AREA = 50
DEFAULT_PROXIMITY = 12


# ============================================================
# 50 VARIATION STYLE NAMES
# ============================================================

VARIATION_NAMES = [
    "Original", "Soft Bold", "Thin", "Extra Bold", "Narrow",
    "Wide", "Tall", "Short", "Right Slant", "Left Slant",
    "Wobbly", "Bouncy", "Dancing", "Rounded", "Soft Bubble",
    "Marker", "Brush", "Pencil", "Crayon", "Chalk",
    "Sketch", "Organic", "Playful", "Cartoon", "Comic",
    "Storybook", "Vintage", "Retro", "Fairy", "Magical",
    "Floral", "Leafy", "Vine", "Cloud", "Candy",
    "Jelly", "Puffy", "Wave", "Spiral", "Doodle",
    "Tiny", "Chunky", "Hand-Painted", "Fantasy", "Cute",
    "Messy", "Dynamic", "Quirky", "Random Mix", "Signature"
]


# ============================================================
# SESSION STATE
# ============================================================

if "components" not in st.session_state:
    st.session_state.components = []

if "generated_ttf" not in st.session_state:
    st.session_state.generated_ttf = None

if "font_name" not in st.session_state:
    st.session_state.font_name = "MyHandwriting50Var"

if "tshirt_variations" not in st.session_state:
    st.session_state.tshirt_variations = []

if "detected_preview" not in st.session_state:
    st.session_state.detected_preview = None

if "cleaned_preview" not in st.session_state:
    st.session_state.cleaned_preview = None

if "font_characters" not in st.session_state:
    st.session_state.font_characters = set()


# ============================================================
# IMAGE PREPROCESSING & REFINED DETECTION ENGINE
# ============================================================

def preprocess_reference_image(pil_image, contrast=1.1, denoise_strength=5):
    arr = np.array(pil_image.convert("RGB"))

    if contrast != 1.0:
        arr = cv2.convertScaleAbs(arr, alpha=contrast, beta=0)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    bg_dilated = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg_smooth = cv2.medianBlur(bg_dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg_smooth)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    if denoise_strength > 0:
        norm = cv2.fastNlMeansDenoising(norm, h=denoise_strength)

    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 41, 12
    )

    h, w = binary.shape
    border_y, border_x = int(h * 0.02), int(w * 0.02)
    binary[:border_y, :] = 0
    binary[-border_y:, :] = 0
    binary[:, :border_x] = 0
    binary[:, -border_x:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    preview = cv2.cvtColor(255 - cleaned, cv2.COLOR_GRAY2RGBA)
    return cleaned, Image.fromarray(preview)


def merge_nearby_stroke_boxes(boxes, max_distance=12):
    if not boxes:
        return []

    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        x1, y1, w1, h1, mask1 = boxes[i]
        curr_x1, curr_y1, curr_x2, curr_y2 = x1, y1, x1 + w1, y1 + h1
        masks_to_combine = [(x1, y1, mask1)]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                xj, yj, wj, hj, maskj = boxes[j]

                gap_x = max(0, max(curr_x1 - (xj + wj), xj - curr_x2))
                gap_y = max(0, max(curr_y1 - (yj + hj), yj - curr_y2))

                overlap_x = max(0, min(curr_x2, xj + wj) - max(curr_x1, xj))
                overlap_y = max(0, min(curr_y2, yj + hj) - max(curr_y1, yj))

                is_vertically_aligned = (gap_y <= max_distance * 1.5 and overlap_x > 0)
                is_horizontally_tight = (gap_x <= max_distance and overlap_y > 0.15 * min(curr_y2 - curr_y1, hj))

                new_w = max(curr_x2, xj + wj) - min(curr_x1, xj)
                max_glyph_width = max(w1, wj) * 2.5 + 40

                if (is_vertically_aligned or is_horizontally_tight) and new_w <= max_glyph_width:
                    curr_x1 = min(curr_x1, xj)
                    curr_y1 = min(curr_y1, yj)
                    curr_x2 = max(curr_x2, xj + wj)
                    curr_y2 = max(curr_y2, yj + hj)
                    masks_to_combine.append((xj, yj, maskj))
                    used[j] = True
                    changed = True

        mw = curr_x2 - curr_x1
        mh = curr_y2 - curr_y1
        combined_mask = np.zeros((mh, mw), dtype=np.uint8)

        for mx, my, mmask in masks_to_combine:
            ox = mx - curr_x1
            oy = my - curr_y1
            region = combined_mask[oy:oy + mmask.shape[0], ox:ox + mmask.shape[1]]
            combined_mask[oy:oy + mmask.shape[0], ox:ox + mmask.shape[1]] = cv2.bitwise_or(region, mmask)

        merged.append((curr_x1, curr_y1, mw, mh, combined_mask))

    return merged


def detect_and_extract_glyphs(binary_mask, min_area=MIN_COMPONENT_AREA, proximity_threshold=DEFAULT_PROXIMITY):
    h_img, w_img = binary_mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    raw_boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < min_area or w < 4 or h < 4:
            continue
        if w > w_img * 0.35 or h > h_img * 0.50:
            continue

        crop = binary_mask[y:y + h, x:x + w]
        raw_boxes.append((int(x), int(y), int(w), int(h), crop))

    merged_components = merge_nearby_stroke_boxes(raw_boxes, max_distance=proximity_threshold)

    extracted = []
    for idx, (x, y, w, h, crop_mask) in enumerate(merged_components):
        rgba_crop = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_crop[crop_mask > 0] = [0, 0, 0, 255]

        extracted.append({
            "id": idx + 1,
            "bbox": (x, y, w, h),
            "crop_mask": crop_mask,
            "preview_img": Image.fromarray(rgba_crop, "RGBA"),
            "assigned_char": ""
        })

    if extracted:
        heights = [item["bbox"][3] for item in extracted]
        avg_h = max(20, np.mean(heights))
        row_height = max(25, int(avg_h * 0.70))

        extracted.sort(key=lambda item: (
            item["bbox"][1] // row_height,
            item["bbox"][0]
        ))

        for i, item in enumerate(extracted):
            item["id"] = i + 1

    return extracted


# ============================================================
# SYNTHETIC GLYPH GENERATOR (FALLBACK FOR MISSING CHARS)
# ============================================================

def generate_fallback_bitmap(character, target_h=400, target_w=300):
    img = Image.new("L", (target_w, target_h), 0)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", int(target_h * 0.7))
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), character, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (target_w - tw) // 2
    y = (target_h - th) // 2 - bbox[1]

    draw.text((x, y), character, fill=255, font=font)

    arr = np.array(img, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    arr = cv2.dilate(arr, kernel, iterations=2)

    return arr


# ============================================================
# GLYPH PROCESSING & TRANSFORMATIONS
# ============================================================

def crop_to_content(mask, padding=6):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    x1 = max(0, xs.min() - padding)
    y1 = max(0, ys.min() - padding)
    x2 = min(mask.shape[1], xs.max() + padding + 1)
    y2 = min(mask.shape[0], ys.max() + padding + 1)
    return mask[y1:y2, x1:x2]


def normalize_glyph(glyph, target_height=GLYPH_HEIGHT):
    glyph = crop_to_content(glyph, padding=4)
    h, w = glyph.shape
    if h <= 0 or w <= 0:
        return glyph
    scale = target_height / float(h)
    new_w = max(1, int(w * scale))
    return cv2.resize(glyph, (new_w, target_height), interpolation=cv2.INTER_AREA)


def wave_distort(image, amplitude=3.0, wavelength=120.0, phase=0.0):
    h, w = image.shape
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        shift = amplitude * math.sin(2 * math.pi * y / wavelength + phase)
        map_x[y, :] = np.arange(w) - shift
        map_y[y, :] = y

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def apply_procedural_glyph_variation(binary_crop, variation_idx=1, base_seed=0):
    if binary_crop is None or binary_crop.size == 0:
        return binary_crop

    idx = variation_idx - 1
    rng = random.Random(base_seed * 1000 + idx)
    img = binary_crop.copy()

    if idx == 0:
        return img
    elif idx in [1, 3, 13, 14, 15, 16, 22, 23, 34, 35, 36, 41, 44]:
        k_size = 3 if idx in [1, 15] else (5 if idx in [3, 13, 16, 22, 34, 44] else 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        img = cv2.dilate(img, kernel, iterations=1)
    elif idx in [2, 17]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.erode(img, kernel, iterations=1)
    elif idx in [4, 5, 6, 7, 25, 27, 33, 40]:
        sx = 0.78 if idx == 4 else (1.22 if idx == 5 else (1.08 if idx == 25 else (0.92 if idx == 27 else (1.12 if idx == 33 else (0.72 if idx == 40 else 1.0)))))
        sy = 1.18 if idx == 6 else (0.86 if idx == 7 else (1.05 if idx == 25 else (1.06 if idx == 27 else (0.94 if idx == 33 else (0.80 if idx == 40 else 1.0)))))
        img = cv2.resize(img, (max(1, int(img.shape[1] * sx)), max(1, int(img.shape[0] * sy))), interpolation=cv2.INTER_AREA)
    elif idx in [8, 9]:
        shear = 0.16 if idx == 8 else -0.16
        h, w = img.shape
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (int(w + abs(shear * h)), h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    elif idx in [10, 11, 20, 21, 26, 29, 30, 31, 32, 37, 38, 39, 42, 45, 47]:
        amp = 5 if idx == 10 else (8 if idx in [11, 32, 38] else (3 if idx in [20, 45] else (10 if idx in [37, 47] else 6)))
        wl = 90 if idx == 10 else (150 if idx == 11 else (75 if idx == 21 else (180 if idx in [30, 37] else 110)))
        img = wave_distort(img, amplitude=amp, wavelength=wl)
    elif idx in [12, 24, 28, 43, 46]:
        angle = 4 if idx == 12 else (-3 if idx == 28 else rng.uniform(-7, 7))
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    elif idx in [48, 49]:
        shear = rng.uniform(-0.10, 0.10)
        h, w = img.shape
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (int(w + abs(shear * h)), h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        img = wave_distort(img, amplitude=rng.uniform(3, 7), wavelength=rng.uniform(80, 150))

    return crop_to_content(img, padding=4)


# ============================================================
# COMPILER & TTF BUILDER
# ============================================================

def bitmap_to_glyph(binary, target_height=GLYPH_HEIGHT):
    if binary is None or binary.size == 0:
        return None

    binary = crop_to_content(binary, padding=2)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return None

    pen = TTGlyphPen(None)
    height, width = binary.shape
    scale = target_height / max(height, 1)

    def convert_point(x, y):
        return x * scale, (height - y) * scale

    for i, contour in enumerate(contours):
        epsilon = 0.004 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            continue

        converted = [convert_point(float(pt[0]), float(pt[1])) for pt in points]
        if hierarchy[0][i][3] != -1:
            converted.reverse()

        pen.moveTo(converted[0])
        for point in converted[1:]:
            pen.lineTo(point)
        pen.closePath()

    return pen.glyph()


def glyph_advance_from_bitmap(bitmap):
    h, w = bitmap.shape
    return int(max(300, min(1400, w * 1.20)))


def get_pua_codepoint(character, variation_idx):
    if character not in CHARACTER_SET or not (1 <= variation_idx <= NUM_VARIATIONS):
        return None
    char_index = CHARACTER_SET.index(character)
    offset = char_index * NUM_VARIATIONS + (variation_idx - 1)
    return PUA_START + offset


def convert_text_to_variation(text, variation_idx, available_characters=None):
    if available_characters is None:
        available_characters = set(CHARACTER_SET)

    result = []
    for char in text:
        if char in [" ", "\n"]:
            result.append(char)
        elif char not in available_characters:
            result.append(char)
        else:
            cp = get_pua_codepoint(char, variation_idx)
            result.append(chr(cp) if cp else char)
    return "".join(result)


def compile_50_variant_ttf(mapped_components, font_name="MyHandwriting50Var"):
    glyph_order = [".notdef", "space"]
    cmap = {32: "space"}
    glyph_objects = {}
    metrics = {".notdef": (600, 0), "space": (300, 0)}
    available_characters = set()

    # .NOTDEF GLYPH
    notdef_pen = TTGlyphPen(None)
    notdef_pen.moveTo((100, 0))
    notdef_pen.lineTo((100, 700))
    notdef_pen.lineTo((500, 700))
    notdef_pen.lineTo((500, 0))
    notdef_pen.closePath()
    glyph_objects[".notdef"] = notdef_pen.glyph()

    # SPACE GLYPH (Explicitly defined empty pen outline)
    space_pen = TTGlyphPen(None)
    glyph_objects["space"] = space_pen.glyph()

    # Collect explicitly mapped characters
    character_map = {}
    for comp in mapped_components:
        character = comp.get("assigned_char", "").strip()
        if character and len(character) == 1 and character in CHARACTER_SET:
            character_map[character] = comp["crop_mask"]

    # Fallback Generation for missing letters/numbers
    for char in CHARACTER_SET:
        if char not in character_map:
            character_map[char] = generate_fallback_bitmap(char)

    # Compile all characters into TTF
    for character, crop_mask in character_map.items():
        available_characters.add(character)
        base = crop_to_content(crop_mask, padding=5)

        for variation_idx in range(1, NUM_VARIATIONS + 1):
            variant = apply_procedural_glyph_variation(base, variation_idx=variation_idx, base_seed=ord(character))
            normalized = normalize_glyph(variant, target_height=GLYPH_HEIGHT)
            glyph = bitmap_to_glyph(normalized, target_height=GLYPH_HEIGHT)

            if glyph is None:
                glyph = TTGlyphPen(None).glyph()
                advance = 500
            else:
                advance = glyph_advance_from_bitmap(normalized)

            if variation_idx == 1:
                normal_gname = f"char_{ord(character)}"
                if normal_gname not in glyph_order:
                    glyph_order.append(normal_gname)
                glyph_objects[normal_gname] = glyph
                metrics[normal_gname] = (advance, 0)
                cmap[ord(character)] = normal_gname

            variant_gname = f"char_{ord(character)}_v{variation_idx}"
            if variant_gname not in glyph_order:
                glyph_order.append(variant_gname)

            glyph_objects[variant_gname] = glyph
            metrics[variant_gname] = (advance, 0)
            pua_codepoint = get_pua_codepoint(character, variation_idx)
            cmap[pua_codepoint] = variant_gname

    fb = FontBuilder(unitsPerEm=UNITS_PER_EM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyph_objects)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({
        "familyName": font_name,
        "styleName": "50 Variations",
        "uniqueFontIdentifier": f"{font_name}-50Variations",
        "fullName": f"{font_name} 50 Variations",
        "psName": f"{font_name}-50Variations"
    })
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost()

    stream = io.BytesIO()
    fb.save(stream)
    font_bytes = stream.getvalue()

    TTFont(io.BytesIO(font_bytes)).close()
    return font_bytes, available_characters


# ============================================================
# CLIPPING MASK RENDERING ENGINE
# ============================================================

def render_text_mask(phrase, font_bytes, variation_idx, canvas_size=(3000, 3000), available_characters=None, font_scale=1.0, line_spacing=1.20):
    canvas_w, canvas_h = canvas_size
    mask = Image.new("L", canvas_size, 0)
    draw = ImageDraw.Draw(mask)

    variation_text = convert_text_to_variation(phrase, variation_idx, available_characters)
    clean_phrase = phrase.strip() or "TEXT"

    estimated_size = int((canvas_w / max(5, len(clean_phrase) * 0.50)) * font_scale)
    estimated_size = max(80, min(1500, estimated_size))

    font = ImageFont.truetype(io.BytesIO(font_bytes), size=estimated_size)
    lines = variation_text.split("\n")
    line_height = int(estimated_size * line_spacing)
    total_height = len(lines) * line_height
    current_y = int((canvas_h - total_height) * 0.35)

    for line_idx, line in enumerate(lines):
        if not line.strip():
            current_y += line_height
            continue

        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = int((canvas_w - text_w) / 2)

        random.seed(variation_idx * 100 + line_idx)
        wobble = random.randint(-10, 10)
        draw.text((x, current_y + wobble), line, font=font, fill=255)
        current_y += line_height

    return mask


def clip_artwork_inside_text(text_mask, artwork_img=None, canvas_size=(3000, 3000)):
    if artwork_img is None:
        artwork = Image.new("RGBA", canvas_size, (25, 25, 25, 255))
    else:
        artwork = artwork_img.convert("RGBA").resize(canvas_size, Image.Resampling.LANCZOS)

    art_arr = np.array(artwork, dtype=np.uint8)
    mask_arr = np.array(text_mask, dtype=np.uint8)

    art_alpha = art_arr[:, :, 3].astype(np.float32) / 255.0
    text_alpha = mask_arr.astype(np.float32) / 255.0

    final_alpha = (art_alpha * text_alpha * 255.0).clip(0, 255).astype(np.uint8)
    art_arr[:, :, 3] = final_alpha
    return Image.fromarray(art_arr, "RGBA")


def render_clipping_mask_design(phrase, font_bytes, artwork_img=None, canvas_size=(3000, 3000), variation_idx=1, available_characters=None, font_scale=1.0, line_spacing=1.20):
    text_mask = render_text_mask(phrase, font_bytes, variation_idx, canvas_size, available_characters, font_scale, line_spacing)
    return clip_artwork_inside_text(text_mask, artwork_img, canvas_size)


def generate_50_tshirt_variations(phrase, font_bytes, artwork_img=None, canvas_size=(3000, 3000), available_characters=None, font_scale=1.0, line_spacing=1.20):
    variations = []
    for variation_idx in range(1, NUM_VARIATIONS + 1):
        image = render_clipping_mask_design(phrase, font_bytes, artwork_img, canvas_size, variation_idx, available_characters, font_scale, line_spacing)
        variations.append({
            "id": variation_idx,
            "style_name": VARIATION_NAMES[variation_idx - 1],
            "image": image
        })
    return variations


def create_preview_gallery_grid(variations, cols=5, thumb_size=(280, 280)):
    rows = (len(variations) + cols - 1) // cols
    sheet_w = cols * thumb_size[0] + (cols + 1) * 20
    sheet_h = rows * thumb_size[1] + (rows + 1) * 55

    grid = Image.new("RGBA", (sheet_w, sheet_h), (242, 244, 247, 255))
    draw = ImageDraw.Draw(grid)

    for idx, variation in enumerate(variations):
        row, col = idx // cols, idx % cols
        x = col * thumb_size[0] + (col + 1) * 20
        y = row * thumb_size[1] + (row + 1) * 55

        draw.rounded_rectangle([x - 5, y - 5, x + thumb_size[0] + 5, y + thumb_size[1] + 28], radius=8, fill=(255, 255, 255, 255), outline=(205, 208, 212, 255), width=2)
        thumb = variation["image"].copy()
        thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)

        paste_x = x + (thumb_size[0] - thumb.width) // 2
        paste_y = y + (thumb_size[1] - thumb.height) // 2
        grid.alpha_composite(thumb, (paste_x, paste_y))
        draw.text((x + 8, y + thumb_size[1] + 6), f"#{variation['id']:02d} {variation['style_name']}", fill=(35, 35, 35))

    return grid


def package_zip_export(variations):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for variation in variations:
            image_buffer = io.BytesIO()
            variation["image"].save(image_buffer, format="PNG", dpi=(300, 300))
            filename = f"tshirt_design_{variation['id']:02d}_{variation['style_name'].replace(' ', '_')}.png"
            zf.writestr(filename, image_buffer.getvalue())
    return zip_buffer.getvalue()


def create_detection_preview(original_image, components):
    preview = np.array(original_image.convert("RGB")).copy()
    for component in components:
        x, y, w, h = component["bbox"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(preview, str(component["id"]), (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
    return Image.fromarray(preview)


# ============================================================
# STREAMLIT UI
# ============================================================

st.sidebar.title("✍️ Font Studio")
app_mode = st.sidebar.radio("Select Mode:", ["1. Extract & Build 50-Variant TTF", "2. T-Shirt Clipping Mask Studio"])

if app_mode == "1. Extract & Build 50-Variant TTF":
    st.title("🔤 Hand-Drawn Font Generator")
    ref_file = st.file_uploader("Upload Handwriting Sheet", type=["png", "jpg", "jpeg", "webp"], key="mode_a_file")

    if ref_file:
        pil_img = Image.open(ref_file).convert("RGBA")
        col1, col2, col3 = st.columns(3)
        with col1:
            contrast_val = st.slider("Contrast", 0.5, 3.0, 1.1, 0.1)
        with col2:
            denoise_val = st.slider("Denoise", 0, 15, 5, 1)
        with col3:
            proximity_val = st.slider("Stroke Clustering", 2, 25, DEFAULT_PROXIMITY, 1)

        binary_mask, cleaned_preview = preprocess_reference_image(pil_img, contrast=contrast_val, denoise_strength=denoise_val)
        st.image(cleaned_preview, caption="Cleaned Reference", use_container_width=True)

        if st.button("🔍 Detect Handwritten Glyphs", type="primary"):
            components = detect_and_extract_glyphs(binary_mask, proximity_threshold=proximity_val)
            st.session_state.components = components
            st.session_state.detected_preview = create_detection_preview(pil_img, components)
            st.success(f"Extracted {len(components)} character glyphs across all lines!")

    if st.session_state.detected_preview:
        st.subheader("🔢 Detected Glyphs Preview")
        st.image(st.session_state.detected_preview, use_container_width=True)

    if st.session_state.components:
        st.divider()
        st.header("📝 Character Mapping")

        for idx, component in enumerate(st.session_state.components):
            c1, c2, c3 = st.columns([1, 2, 5])
            with c1:
                st.image(component["preview_img"], width=80)
            with c2:
                st.markdown(f"### Glyph #{component['id']}")
            with c3:
                current_value = component.get("assigned_char", "")
                entered = st.text_input(f"Character for glyph #{component['id']}", value=current_value, max_chars=1, key=f"mapping_{component['id']}")
                component["assigned_char"] = entered

        font_name = st.text_input("Font Family Name", value=st.session_state.font_name)
        st.session_state.font_name = font_name

        if st.button("⚙️ BUILD ONE TTF WITH 50 VARIATIONS", type="primary", use_container_width=True):
            try:
                with st.spinner("Generating 50 variations per character (including fallbacks for unmapped characters) and compiling TTF..."):
                    ttf_bytes, available_characters = compile_50_variant_ttf(st.session_state.components, font_name=font_name)
                    st.session_state.generated_ttf = ttf_bytes
                    st.session_state.font_characters = available_characters
                st.success("🎉 Single TTF file built successfully with 50 variations embedded for ALL characters!")
            except Exception as e:
                st.error(f"Font compilation error: {e}")

        if st.session_state.generated_ttf:
            st.download_button(
                label="📥 Download ONE 50-Variant TTF",
                data=st.session_state.generated_ttf,
                file_name=f"{font_name}.ttf",
                mime="font/ttf",
                use_container_width=True
            )

else:
    st.title("👕 T-Shirt Clipping Mask Studio")
    if st.session_state.generated_ttf:
        st.success(f"Loaded Font: **{st.session_state.font_name}.ttf**")
    else:
        st.warning("No font loaded.")
        uploaded_ttf = st.file_uploader("Upload TTF File", type=["ttf"])
        if uploaded_ttf:
            st.session_state.generated_ttf = uploaded_ttf.getvalue()
            st.session_state.font_characters = set(CHARACTER_SET)

    if st.session_state.generated_ttf:
        tab_design, tab_gallery = st.tabs(["🎨 Create Design", "📦 Export Gallery"])

        with tab_design:
            left, right = st.columns([1, 1])
            with left:
                phrase = st.text_area("Phrase:", value="MAKE YOUR OWN PATH").strip()
                art_file = st.file_uploader("Upload Image to Clip inside Text:", type=["png", "jpg", "jpeg", "webp"], key="art_upload")
                artwork_pil = Image.open(art_file).convert("RGBA") if art_file else None

            with right:
                variation_idx = st.selectbox("Font Style:", options=list(range(1, NUM_VARIATIONS + 1)), format_func=lambda x: f"#{x:02d} — {VARIATION_NAMES[x - 1]}")
                resolution = st.selectbox("Resolution:", ["3000 × 3000", "4000 × 4000", "4500 × 5400"])
                canvas_size = (4500, 5400) if resolution == "4500 × 5400" else (3000, 3000)

                if st.button("🚀 GENERATE ALL 50 DESIGNS", type="primary", use_container_width=True):
                    with st.spinner("Generating 50 clipping mask variations..."):
                        st.session_state.tshirt_variations = generate_50_tshirt_variations(
                            phrase.upper(), st.session_state.generated_ttf, artwork_pil, canvas_size, st.session_state.font_characters
                        )
                    st.success("All 50 T-shirt designs generated!")

        with tab_gallery:
            if st.session_state.tshirt_variations:
                gallery = create_preview_gallery_grid(st.session_state.tshirt_variations)
                st.image(gallery, use_container_width=True)
                st.download_button("📦 DOWNLOAD ALL 50 PNG DESIGNS (ZIP)", package_zip_export(st.session_state.tshirt_variations), "50_tshirt_designs.zip", "application/zip", use_container_width=True)
