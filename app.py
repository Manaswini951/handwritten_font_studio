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
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.fontBuilder import FontBuilder

# Page Configuration
st.set_page_config(
    page_title="Hand-Drawn Style Font & T-Shirt Studio",
    page_icon="✍️",
    layout="wide"
)

# ============================================================
# IMAGE PROCESSING & GLYPH EXTRACTION
# ============================================================

def preprocess_reference_image(pil_image, contrast=1.1, denoise_strength=5):
    arr = np.array(pil_image.convert("RGB"))
    if contrast != 1.0:
        arr = cv2.convertScaleAbs(arr, alpha=contrast, beta=0)
        
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Shadow removal via illumination estimation
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
    
    # Crop outer margins to eliminate border noise
    h, w = binary.shape
    border_y, border_x = int(h * 0.04), int(w * 0.04)
    binary[:border_y, :] = 0
    binary[-border_y:, :] = 0
    binary[:, :border_x] = 0
    binary[:, -border_x:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    preview_cleaned = cv2.cvtColor(255 - cleaned, cv2.COLOR_GRAY2RGBA)
    return cleaned, Image.fromarray(preview_cleaned)

def merge_nearby_stroke_boxes(boxes, max_distance=18):
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
                
                if not (xj > curr_x2 + max_distance or 
                        xj + wj < curr_x1 - max_distance or 
                        yj > curr_y2 + max_distance or 
                        yj + hj < curr_y1 - max_distance):
                    
                    curr_x1 = min(curr_x1, xj)
                    curr_y1 = min(curr_y1, yj)
                    curr_x2 = max(curr_x2, xj + wj)
                    curr_y2 = max(curr_y2, yj + hj)
                    masks_to_combine.append((xj, yj, maskj))
                    used[j] = True
                    changed = True
                    
        mw, mh = curr_x2 - curr_x1, curr_y2 - curr_y1
        combined_mask = np.zeros((mh, mw), dtype=np.uint8)
        for mx, my, mmask in masks_to_combine:
            ox, oy = mx - curr_x1, my - curr_y1
            combined_mask[oy:oy+mmask.shape[0], ox:ox+mmask.shape[1]] = cv2.bitwise_or(
                combined_mask[oy:oy+mmask.shape[0], ox:ox+mmask.shape[1]], mmask
            )
            
        merged.append((curr_x1, curr_y1, mw, mh, combined_mask))
        
    return merged

def detect_and_extract_glyphs(binary_mask, min_area=250, proximity_threshold=18):
    h_img, w_img = binary_mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    raw_boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 10 or h < 10 or w > w_img * 0.45 or h > h_img * 0.45:
            continue
        crop = binary_mask[y:y+h, x:x+w]
        raw_boxes.append((x, y, w, h, crop))

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
        avg_h = np.mean([b["bbox"][3] for b in extracted])
        row_height = max(40, int(avg_h * 0.8))
        extracted.sort(key=lambda b: (b["bbox"][1] // row_height, b["bbox"][0]))

    return extracted

# ============================================================
# PROCEDURAL VARIATION & TTF COMPILER (50 ALTERNATES)
# ============================================================

def apply_procedural_glyph_variation(binary_crop, seed=0):
    random.seed(seed)
    h, w = binary_crop.shape
    if h == 0 or w == 0:
        return binary_crop

    # 1. Stroke thickness variation
    thick_factor = random.choice([-1, 0, 1, 2])
    if thick_factor > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thick_factor * 2 + 1, thick_factor * 2 + 1))
        mod_crop = cv2.dilate(binary_crop, k, iterations=1)
    elif thick_factor < 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mod_crop = cv2.erode(binary_crop, k, iterations=1)
    else:
        mod_crop = binary_crop.copy()

    # 2. Shear / Slant variation
    shear = random.uniform(-0.15, 0.15)
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    nw = int(w + abs(shear * h))
    mod_crop = cv2.warpAffine(mod_crop, M, (nw, h))

    return mod_crop

def bitmap_to_glyph(binary, target_height=700):
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None

    pen = TTGlyphPen(None)
    height, width = binary.shape
    scale = target_height / max(height, 1)

    def convert_point(x, y):
        return x * scale, (height - y) * scale

    for i, contour in enumerate(contours):
        is_hole = hierarchy[0][i][3] != -1
        epsilon = 0.008 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)

        if len(points) < 3:
            continue

        converted = [convert_point(float(pt[0]), float(pt[1])) for pt in points]
        if is_hole:
            converted = converted[::-1]

        pen.moveTo(converted[0])
        for pt in converted[1:]:
            pen.lineTo(pt)
        pen.closePath()

    return pen.glyph()

def compile_50_variant_ttf(mapped_components, font_name="MyHandwriting", units_per_em=1000):
    glyph_order = [".notdef", "space"]
    cmap = {32: "space"}
    glyph_objects = {}

    notdef_pen = TTGlyphPen(None)
    notdef_pen.moveTo((100, 0))
    notdef_pen.lineTo((100, 700))
    notdef_pen.lineTo((500, 700))
    notdef_pen.lineTo((500, 0))
    notdef_pen.closePath()
    glyph_objects[".notdef"] = notdef_pen.glyph()
    glyph_objects["space"] = TTGlyphPen(None).glyph()

    metrics = {".notdef": (600, 0), "space": (300, 0)}

    default_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    auto_idx = 0

    for comp in mapped_components:
        char = comp.get("assigned_char", "").strip()
        if not char and auto_idx < len(default_alphabet):
            char = default_alphabet[auto_idx]
            auto_idx += 1

        if char:
            ys, xs = np.where(comp["crop_mask"] > 0)
            if len(xs) > 0:
                base_crop = comp["crop_mask"][ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            else:
                base_crop = comp["crop_mask"]

            # Base Character Glyphs
            main_gname = f"glyph_{ord(char)}"
            if main_gname not in glyph_order:
                glyph_order.append(main_gname)
                cmap[ord(char)] = main_gname

            # Generate 50 procedural variations per letter stored inside single TTF
            for var_idx in range(50):
                variant_gname = f"glyph_{ord(char)}_var{var_idx+1}"
                if variant_gname not in glyph_order:
                    glyph_order.append(variant_gname)

                mod_crop = apply_procedural_glyph_variation(base_crop, seed=var_idx + ord(char))
                h, w = mod_crop.shape
                new_w = max(1, int(w * (700 / max(1, h))))
                normalized = cv2.resize(mod_crop, (new_w, 700), interpolation=cv2.INTER_AREA)

                glyph = bitmap_to_glyph(normalized)
                if glyph is not None:
                    glyph_objects[variant_gname] = glyph
                    advance = int(max(350, min(1200, normalized.shape[1] * 1.15)))
                    metrics[variant_gname] = (advance, 0)
                    if var_idx == 0:
                        glyph_objects[main_gname] = glyph
                        metrics[main_gname] = (advance, 0)

    if len(glyph_order) <= 2:
        raise ValueError("No characters detected to build font.")

    fb = FontBuilder(units_per_em, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyph_objects)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({
        "familyName": font_name,
        "styleName": "Regular",
        "uniqueFontIdentifier": f"{font_name}-50Vars-Regular",
        "fullName": f"{font_name} 50-Variation Regular"
    })
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost()

    font_stream = io.BytesIO()
    fb.save(font_stream)
    font_bytes = font_stream.getvalue()
    TTFont(io.BytesIO(font_bytes))
    return font_bytes

# ============================================================
# CLIPPING MASK ENGINE (USING COMPILED TTF)
# ============================================================

def render_clipping_mask_design(phrase, font_bytes, artwork_img=None, canvas_size=(3000, 3000), variation_idx=1):
    w_canvas, h_canvas = canvas_size
    text_mask = Image.new("L", (w_canvas, h_canvas), 0)
    draw_mask = ImageDraw.Draw(text_mask)

    # Load font directly from compiled memory stream
    font_stream = io.BytesIO(font_bytes)
    font = ImageFont.truetype(font_stream, size=int(w_canvas / max(4, len(phrase) * 0.45)))

    words = phrase.strip().split() or ["TEXT"]
    curr_y = int(h_canvas * 0.2)

    for word in words:
        bbox = font.getbbox(word)
        tw = bbox[2] - bbox[0]
        tx = (w_canvas - tw) // 2
        
        # Apply style variations per word layout
        random.seed(variation_idx + len(word))
        y_wobble = int(random.uniform(-15, 15))
        draw_mask.text((tx, curr_y + y_wobble), word, font=font, fill=255)
        curr_y += int(font.size * 1.25)

    # Apply Artwork Clipping Mask
    if artwork_img is not None:
        art_rgba = artwork_img.convert("RGBA").resize(canvas_size, Image.Resampling.LANCZOS)
    else:
        art_rgba = Image.new("RGBA", canvas_size, (20, 20, 20, 255))

    art_arr = np.array(art_rgba, dtype=np.uint8)
    mask_arr = np.array(text_mask, dtype=np.uint8)

    art_alpha = (art_arr[:, :, 3].astype(np.float32) / 255.0)
    text_alpha = mask_arr.astype(np.float32) / 255.0
    
    final_alpha = (art_alpha * text_alpha * 255.0).clip(0, 255).astype(np.uint8)
    art_arr[:, :, 3] = final_alpha

    return Image.fromarray(art_arr, "RGBA")

def generate_50_tshirt_variations(phrase, font_bytes, artwork_img=None, canvas_size=(3000, 3000)):
    variations = []
    for var_idx in range(1, 51):
        img = render_clipping_mask_design(phrase, font_bytes, artwork_img, canvas_size, variation_idx=var_idx)
        variations.append({
            "id": var_idx,
            "style_name": f"Variant #{var_idx:02d}",
            "image": img
        })
    return variations

def create_preview_gallery_grid(variations, cols=5, thumb_size=(300, 300)):
    rows = (len(variations) + cols - 1) // cols
    sheet_w, sheet_h = cols * thumb_size[0] + (cols + 1) * 20, rows * thumb_size[1] + (rows + 1) * 40
    grid_img = Image.new("RGBA", (sheet_w, sheet_h), (240, 242, 245, 255))
    draw = ImageDraw.Draw(grid_img)
    
    for idx, var in enumerate(variations):
        r, c = idx // cols, idx % cols
        x, y = c * thumb_size[0] + (c + 1) * 20, r * thumb_size[1] + (r + 1) * 40
        
        draw.rectangle([x-5, y-5, x+thumb_size[0]+5, y+thumb_size[1]+25], fill=(255, 255, 255, 255), outline=(200, 205, 210), width=2)
        thumb = var["image"].copy()
        thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        
        grid_img.alpha_composite(thumb, (x + (thumb_size[0] - thumb.width) // 2, y + (thumb_size[1] - thumb.height) // 2))
        draw.text((x + 10, y + thumb_size[1] + 5), f"#{var['id']:02d} {var['style_name']}", fill=(40, 40, 40))
        
    return grid_img

def package_zip_export(variations):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for var in variations:
            img_buf = io.BytesIO()
            var["image"].save(img_buf, format="PNG", dpi=(300, 300))
            zf.writestr(f"tshirt_design_{var['id']:03d}.png", img_buf.getvalue())
    return zip_buf.getvalue()

# ============================================================
# STREAMLIT UI
# ============================================================

if "components" not in st.session_state:
    st.session_state.components = []
if "generated_ttf" not in st.session_state:
    st.session_state.generated_ttf = None
if "tshirt_variations" not in st.session_state:
    st.session_state.tshirt_variations = []

st.sidebar.title("✍️ Navigation")
app_mode = st.sidebar.radio("Select Mode:", ["1. Extract & Build 50-Variant .TTF", "2. T-Shirt Clipping Mask Studio"])

if app_mode == "1. Extract & Build 50-Variant .TTF":
    st.title("🔤 Step 1 — Extract Glyphs & Build Multi-Variant TTF")
    st.write("Upload your handwriting sheet to extract characters and compile a single `.ttf` file containing 50 procedural style variations per character.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Reference Image Upload")
        ref_file = st.file_uploader("Upload Alphabet Sheet:", type=["png", "jpg", "jpeg", "webp"], key="mode_a_file")
        if ref_file:
            pil_img = Image.open(ref_file).convert("RGBA")
            contrast_val = st.slider("Contrast Enhancement:", 0.5, 3.0, 1.1, 0.1)
            proximity_val = st.slider("Stroke Clustering Distance:", 5, 40, 18)
            
            binary_mask, cleaned_preview = preprocess_reference_image(pil_img, contrast=contrast_val)
            st.image(cleaned_preview, caption="Cleaned Threshold Mask", use_container_width=True)
            
            if st.button("🔍 Detect & Extract Glyphs"):
                st.session_state.components = detect_and_extract_glyphs(binary_mask, proximity_threshold=proximity_val)
                st.success(f"Detected {len(st.session_state.components)} glyph components!")

    with col2:
        st.subheader("Character Mapping & TTF Compilation")
        if st.session_state.components:
            st.write(f"Detected **{len(st.session_state.components)}** character glyphs:")
            for idx, comp in enumerate(st.session_state.components):
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.image(comp["preview_img"], width=65)
                with c2:
                    comp["assigned_char"] = st.text_input(f"Character #{idx+1}", value=comp["assigned_char"], key=f"char_{idx}", max_chars=1)

            font_name_input = st.text_input("Font Family Name:", "MyHandwriting50Var")
            if st.button("⚙️ Build 50-Variant .TTF File"):
                try:
                    ttf_bytes = compile_50_variant_ttf(st.session_state.components, font_name=font_name_input)
                    st.session_state.generated_ttf = ttf_bytes
                    st.success("TrueType Font with 50 variations compiled successfully!")
                except Exception as e:
                    st.error(f"Compilation Error: {str(e)}")

            if st.session_state.generated_ttf:
                st.download_button(
                    label="📥 Download 50-Variant .TTF Font",
                    data=st.session_state.generated_ttf,
                    file_name=f"{font_name_input}.ttf",
                    mime="font/ttf"
                )

else:
    st.title("👕 Step 2 — T-Shirt Clipping Mask Studio")
    st.write("Render phrases using your compiled multi-variant `.ttf` font file and clip an uploaded image pattern inside the letters.")

    if not st.session_state.generated_ttf:
        st.warning("⚠️ Please compile a `.ttf` font in Step 1 first, or upload an existing `.ttf` file below.")
        uploaded_ttf = st.file_uploader("Upload External TTF Font File:", type=["ttf"])
        if uploaded_ttf:
            st.session_state.generated_ttf = uploaded_ttf.getvalue()

    tab_tshirt, tab_export = st.tabs(["1. Clipping Mask Studio", "2. Export Gallery"])

    with tab_tshirt:
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            tshirt_phrase = st.text_area("Enter Phrase:", value="MAKE YOUR OWN PATH").upper()
            art_file = st.file_uploader("Upload Image/Pattern for Clipping Mask:", type=["png", "jpg", "jpeg", "webp"], key="art_upload")
            artwork_pil = Image.open(art_file).convert("RGBA") if art_file else None

        with col_t2:
            res_choice = st.selectbox("Resolution:", ["3000 x 3000 px", "4000 x 4000 px", "4500 x 5400 px"])
            canvas_dim = (4500, 5400) if "4500" in res_choice else (3000, 3000)

            if st.button("🚀 GENERATE 50 CLIPPED T-SHIRT DESIGNS", type="primary", use_container_width=True):
                if st.session_state.generated_ttf:
                    with st.spinner("Generating 50 designs using compiled `.ttf` font..."):
                        st.session_state.tshirt_variations = generate_50_tshirt_variations(
                            tshirt_phrase, st.session_state.generated_ttf, artwork_pil, canvas_dim
                        )
                    st.success("Generated 50 unique T-shirt clipping mask compositions!")
                else:
                    st.error("Missing `.ttf` font binary!")

    with tab_export:
        if st.session_state.tshirt_variations:
            st.image(create_preview_gallery_grid(st.session_state.tshirt_variations), use_container_width=True)
            sel_idx = st.selectbox("Select Design:", options=list(range(len(st.session_state.tshirt_variations))), format_func=lambda i: st.session_state.tshirt_variations[i]["style_name"])
            selected_var = st.session_state.tshirt_variations[sel_idx]
            
            buf = io.BytesIO()
            selected_var["image"].save(buf, format="PNG", dpi=(300, 300))
            st.download_button(f"📥 Download Design #{selected_var['id']:02d}", buf.getvalue(), f"tshirt_design_{selected_var['id']:02d}.png", "image/png")
            st.download_button("📦 Download All 50 Designs (ZIP)", package_zip_export(st.session_state.tshirt_variations), "handwritten_clipping_mask_collection.zip", "application/zip", use_container_width=True)
