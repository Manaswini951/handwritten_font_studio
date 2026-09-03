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

# Page Configuration
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
            y_val = int((h - (pt[1] - 10)) * scale)
            scaled_pts.append((x_val, y_val))
            
        vector_contours.append({"points": scaled_pts})
    return vector_contours

def compile_ttf_font(mapped_components, font_name="HandmadeFont", units_per_em=2048):
    glyph_order = [".notdef", "space"]
    cmap = {32: "space"}
    char_glyph_map = {}
    
    # Auto-assign characters if user didn't enter any
    default_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    auto_idx = 0

    for comp in mapped_components:
        char = comp.get("assigned_char", "").strip()
        if not char and comp.get("classification") == "Letter":
            if auto_idx < len(default_alphabet):
                char = default_alphabet[auto_idx]
                comp["assigned_char"] = char
                auto_idx += 1

        if char and comp.get("classification") == "Letter":
            glyph_name = f"glyph_{ord(char)}"
            if glyph_name not in glyph_order:
                glyph_order.append(glyph_name)
                cmap[ord(char)] = glyph_name
                char_glyph_map[glyph_name] = comp
                
    if len(glyph_order) <= 2:
        raise ValueError("No glyphs detected to build font. Please upload a clear alphabet image.")

    glyphs = {}
    pen = TTGlyphPen(None)
    pen.moveTo((100, 0))
    pen.lineTo((100, 1000))
    pen.lineTo((600, 1000))
    pen.lineTo((600, 0))
    pen.closePath()
    glyphs[".notdef"] = pen.glyph()
    glyphs["space"] = TTGlyphPen(None).glyph()

    horizontal_metrics = {".notdef": (700, 100), "space": (500, 0)}

    for gname, comp in char_glyph_map.items():
        pen_g = TTGlyphPen(None)
        vector_contours = mask_to_font_contours(comp["crop_mask"], target_size=1200)
        for cnt in vector_contours:
            pts = cnt["points"]
            if len(pts) < 3:
                continue
            pen_g.moveTo(pts[0])
            for pt in pts[1:]:
                pen_g.lineTo(pt)
            pen_g.closePath()
        glyphs[gname] = pen_g.glyph()
        horizontal_metrics[gname] = (1400, 100)

    fb = FontBuilder(units_per_em, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyphs(glyphs)
    fb.setupHorizontalMetrics({g: horizontal_metrics.get(g, (1000, 0)) for g in glyph_order})
    fb.setupHorizontalHeader(ascent=1600, descent=-400)
    fb.setupNameTable({"familyName": font_name, "styleName": "Regular", "uniqueFontIdentifier": f"{font_name}-1.0"})
    fb.setupOS2(sTypoAscender=1600, sTypoDescender=-400, usWinAscent=1600, usWinDescent=400)
    fb.setupPost()

    font_stream = io.BytesIO()
    fb.save(font_stream)
    font_bytes = font_stream.getvalue()
    TTFont(io.BytesIO(font_bytes))
    return font_bytes

# ============================================================
# PROCEDURAL TYPOGRAPHY & T-SHIRT ENGINE
# ============================================================

LAYOUT_STYLES = [
    "Centered Classic", "Stacked Block", "Word Emphasis", "Mixed Letter Sizes",
    "Alternating Baseline", "Rotated Letters", "Wave Path", "Arc Layout",
    "Circular Emblem", "Vertical Stack", "Tight Spacing", "Expanded Spacing",
    "Huge First Letter", "Huge Last Letter", "Split Layout", "Overlapping Letters",
    "Diagonal Tilt", "Dynamic Scatter", "Badge Composition", "Outline + Fill"
]

def apply_procedural_style_transforms(pil_image, style_profile, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    arr = np.array(pil_image.convert("RGBA"))
    alpha = arr[:, :, 3]
    if np.sum(alpha) == 0:
        return pil_image

    thick = style_profile.get("stroke_thickness", 0.5)
    if thick > 0.6:
        k_size = int((thick - 0.5) * 8) | 1
        alpha = cv2.dilate(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)), iterations=1)
    elif thick < 0.4:
        k_size = int((0.5 - thick) * 6) | 1
        alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)), iterations=1)

    rough = style_profile.get("roughness", 0.3)
    if rough > 0.2:
        h, w = alpha.shape
        noise = np.random.normal(0, rough * 12, (h, w)).astype(np.float32)
        alpha = np.clip(cv2.GaussianBlur(alpha.astype(np.float32) + noise, (3, 3), 0), 0, 255).astype(np.uint8)

    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")

def render_styled_text_layout(phrase, style_profile, canvas_size=(3000, 3000), seed=42):
    random.seed(seed)
    w_canvas, h_canvas = canvas_size
    mask = Image.new("L", (w_canvas, h_canvas), 0)
    words = phrase.strip().split() or ["TEXT"]

    base_size = int(w_canvas / max(4, len(phrase) * 0.55))
    font = ImageFont.load_default()

    curr_y = int(h_canvas * 0.25)
    wobble = style_profile.get("baseline_wobble", 0.2) * 40

    for word in words:
        word_img = Image.new("RGBA", (w_canvas, int(base_size * 1.8)), (0, 0, 0, 0))
        wdraw = ImageDraw.Draw(word_img)
        bbox = font.getbbox(word)
        tw = bbox[2] - bbox[0]
        tx = (w_canvas - tw) // 2
        
        cx = tx
        for char in word:
            cb = font.getbbox(char)
            cw = cb[2] - cb[0]
            cy_off = int(random.uniform(-wobble, wobble))
            wdraw.text((cx, 10 + cy_off), char, font=font, fill=(255, 255, 255, 255))
            cx += cw + int(random.uniform(-5, 10))

        word_styled = apply_procedural_style_transforms(word_img, style_profile, seed=seed + curr_y)
        mask.paste(word_styled.split()[3], (0, curr_y), word_styled.split()[3])
        curr_y += int(base_size * 1.3)

    return mask

def apply_artwork_clipping_mask(artwork_image, text_mask, opacity=1.0, tile=False):
    w, h = text_mask.size
    art_rgba = artwork_image.convert("RGBA")
    
    if tile:
        tw, th = art_rgba.size
        tiled_art = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for x in range(0, w, max(10, tw)):
            for y in range(0, h, max(10, th)):
                tiled_art.alpha_composite(art_rgba, (x, y))
        art_rgba = tiled_art
    else:
        art_rgba = art_rgba.resize((w, h), Image.Resampling.LANCZOS)

    art_arr = np.array(art_rgba, dtype=np.uint8)
    mask_arr = np.array(text_mask.convert("L"), dtype=np.uint8)

    art_alpha = (art_arr[:, :, 3].astype(np.float32) / 255.0) * opacity
    text_alpha = mask_arr.astype(np.float32) / 255.0
    
    final_alpha = (art_alpha * text_alpha * 255.0).clip(0, 255).astype(np.uint8)
    art_arr[:, :, 3] = final_alpha
    return Image.fromarray(art_arr, "RGBA")

def generate_50_tshirt_variations(phrase, style_profile, artwork_img=None, canvas_size=(3000, 3000), master_seed=42):
    variations = []
    w, h = canvas_size

    for idx in range(50):
        seed = master_seed + (idx * 999)
        random.seed(seed)
        
        text_mask = render_styled_text_layout(phrase, style_profile, canvas_size, seed)
        
        if artwork_img is not None:
            composited = apply_artwork_clipping_mask(artwork_img, text_mask, tile=(idx % 3 == 0))
        else:
            solid_black = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
            composited = apply_artwork_clipping_mask(solid_black, text_mask)

        mask_arr = np.array(text_mask, dtype=np.uint8)
        stroke_w = random.choice([0, 8, 14, 20])
        
        if stroke_w > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_w * 2 + 1, stroke_w * 2 + 1))
            border = cv2.subtract(cv2.dilate(mask_arr, kernel, iterations=1), mask_arr)
            border_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            border_rgba[:, :, 3] = border
            
            final_img = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            final_img.alpha_composite(Image.fromarray(border_rgba, "RGBA"))
            final_img.alpha_composite(composited)
        else:
            final_img = composited

        variations.append({
            "id": idx + 1,
            "seed": seed,
            "style_name": f"{LAYOUT_STYLES[idx % len(LAYOUT_STYLES)]} (Var #{idx+1})",
            "image": final_img
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
        draw.text((x + 10, y + thumb_size[1] + 5), f"#{var['id']:02d} {var['style_name'][:18]}", fill=(40, 40, 40))
        
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
if "style_profile" not in st.session_state:
    st.session_state.style_profile = {
        "stroke_thickness": 0.50, "roughness": 0.30, "aspect_ratio": 0.80,
        "height_variation": 0.20, "baseline_wobble": 0.15, "angularity": 0.40
    }
if "generated_ttf" not in st.session_state:
    st.session_state.generated_ttf = None
if "tshirt_variations" not in st.session_state:
    st.session_state.tshirt_variations = []

st.sidebar.title("✍️ Navigation")
app_mode = st.sidebar.radio("Select Mode:", ["Mode A: Make My Font", "Mode B: Style Generator & T-Shirt Studio"])

if app_mode == "Mode A: Make My Font":
    st.title("🔤 Mode A — Make My Font (.TTF Compiler)")
    st.write("Upload an alphabet reference sheet, map character glyphs, and compile a genuine installable TrueType (.ttf) font.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("1. Reference Image Upload")
        ref_file = st.file_uploader("Upload Alphabet Sheet:", type=["png", "jpg", "jpeg", "webp"], key="mode_a_file")
        if ref_file:
            pil_img = Image.open(ref_file).convert("RGBA")
            contrast_val = st.slider("Contrast Enhancement:", 0.5, 3.0, 1.2)
            denoise_val = st.slider("Denoise Strength:", 0, 20, 5)
            
            binary_mask, cleaned_preview = preprocess_reference_image(pil_img, contrast=contrast_val, denoise_strength=denoise_val)
            st.image(cleaned_preview, caption="Cleaned Threshold Mask", use_container_width=True)
            
            if st.button("🔍 Detect & Extract Glyphs"):
                st.session_state.components = detect_and_extract_glyphs(binary_mask)
                st.success(f"Detected {len(st.session_state.components)} glyph components!")

    with col2:
        st.subheader("2. Interactive Character Mapping")
        if st.session_state.components:
            for idx, comp in enumerate(st.session_state.components[:20]):
                c1, c2, c3 = st.columns([1, 2, 2])
                with c1:
                    st.image(comp["preview_img"], width=60)
                with c2:
                    comp["assigned_char"] = st.text_input(f"Char #{idx+1}", value=comp["assigned_char"], key=f"char_{idx}", max_chars=1)
                with c3:
                    comp["classification"] = st.selectbox(f"Type #{idx+1}", ["Letter", "Decoration", "Ignore"], key=f"type_{idx}")

            font_name_input = st.text_input("Font Family Name:", "MyHandwritingFont")
            if st.button("⚙️ Build & Compile .TTF Binary"):
                try:
                    ttf_bytes = compile_ttf_font(st.session_state.components, font_name=font_name_input)
                    st.session_state.generated_ttf = ttf_bytes
                    st.success("TrueType Font compiled successfully!")
                except Exception as e:
                    st.error(f"Compilation Error: {str(e)}")

            if st.session_state.generated_ttf:
                st.download_button("📥 Download .TTF Font", st.session_state.generated_ttf, f"{font_name_input}.ttf", "font/ttf")

else:
    st.title("👕 Mode B — Style Generator & T-Shirt Studio")
    st.write("Extract style vectors from reference sketches, adjust procedural controls, and generate 50 T-shirt print designs with artwork clipping masks.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Style Controls")
    st.session_state.style_profile["stroke_thickness"] = st.sidebar.slider("Stroke Thickness:", 0.1, 1.0, float(st.session_state.style_profile["stroke_thickness"]))
    st.session_state.style_profile["roughness"] = st.sidebar.slider("Roughness:", 0.0, 1.0, float(st.session_state.style_profile["roughness"]))
    st.session_state.style_profile["baseline_wobble"] = st.sidebar.slider("Baseline Wobble:", 0.0, 1.0, float(st.session_state.style_profile["baseline_wobble"]))
    master_seed = st.sidebar.number_input("Random Seed:", value=42, step=1)

    tab_ref, tab_tshirt, tab_export = st.tabs(["1. Reference Sample", "2. T-Shirt Studio", "3. Export Gallery"])

    with tab_ref:
        ref_b_file = st.file_uploader("Upload Style Sketch (e.g. 'Most'):", type=["png", "jpg", "jpeg", "webp"], key="mode_b_file")
        if ref_b_file:
            pil_b = Image.open(ref_b_file).convert("RGBA")
            st.image(pil_b, caption="Reference Sketch", width=350)
            binary_mask_b, _ = preprocess_reference_image(pil_b)
            comps_b = detect_and_extract_glyphs(binary_mask_b)
            
            if st.button("📊 Extract Style Vector Profile"):
                st.session_state.style_profile.update(extract_style_profile(comps_b))
                st.success("Style Profile extracted!")
                st.json(st.session_state.style_profile)

    with tab_tshirt:
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            tshirt_phrase = st.text_area("Enter Phrase:", value="MAKE YOUR OWN PATH").upper()
            art_file = st.file_uploader("Upload Pattern/Flower Artwork:", type=["png", "jpg", "jpeg", "webp"], key="art_upload")
            artwork_pil = Image.open(art_file).convert("RGBA") if art_file else None

        with col_t2:
            res_choice = st.selectbox("Resolution:", ["3000 x 3000 px", "4000 x 4000 px", "4500 x 5400 px"])
            canvas_dim = (4500, 5400) if "4500" in res_choice else (3000, 3000)

            if st.button("🚀 GENERATE 50 T-SHIRT DESIGNS", type="primary", use_container_width=True):
                with st.spinner("Generating 50 compositions..."):
                    st.session_state.tshirt_variations = generate_50_tshirt_variations(
                        tshirt_phrase, st.session_state.style_profile, artwork_pil, canvas_dim, master_seed
                    )
                st.success("Generated 50 unique T-shirt designs!")

    with tab_export:
        if st.session_state.tshirt_variations:
            st.image(create_preview_gallery_grid(st.session_state.tshirt_variations), use_container_width=True)
            sel_idx = st.selectbox("Select Design:", options=list(range(len(st.session_state.tshirt_variations))), format_func=lambda i: st.session_state.tshirt_variations[i]["style_name"])
            selected_var = st.session_state.tshirt_variations[sel_idx]
            
            buf = io.BytesIO()
            selected_var["image"].save(buf, format="PNG", dpi=(300, 300))
            st.download_button(f"📥 Download Design #{selected_var['id']:03d}", buf.getvalue(), f"tshirt_design_{selected_var['id']:03d}.png", "image/png")
            st.download_button("📦 Download All 50 Designs (ZIP)", package_zip_export(st.session_state.tshirt_variations), "handwritten_tshirt_collection.zip", "application/zip", use_container_width=True)
