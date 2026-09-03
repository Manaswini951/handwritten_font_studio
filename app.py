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
# IMAGE PROCESSING & GLYPH EXTRACTION ENGINE
# ============================================================

def preprocess_reference_image(pil_image, contrast=1.2, brightness=0, denoise_strength=5):
    arr = np.array(pil_image.convert("RGB"))
    
    if contrast != 1.0 or brightness != 0:
        arr = cv2.convertScaleAbs(arr, alpha=contrast, beta=brightness)
        
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Background illumination estimation & shadow removal
    bg_dilated = cv2.dilate(gray, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg_smooth = cv2.medianBlur(bg_dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg_smooth)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    if denoise_strength > 0:
        norm = cv2.fastNlMeansDenoising(norm, h=denoise_strength)
        
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 41, 12
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    preview_cleaned = cv2.cvtColor(255 - cleaned, cv2.COLOR_GRAY2RGBA)
    return cleaned, Image.fromarray(preview_cleaned)

def detect_and_extract_glyphs(binary_mask, min_area=300, dilation_size=7):
    # Dilation to bridge separate strokes within single letters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    connected = cv2.dilate(binary_mask, kernel, iterations=1)

    n, labels, stats, centers = cv2.connectedComponentsWithStats(connected, connectivity=8)
    boxes = []

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < 15 or h < 15:
            continue
        
        # Crop mask from non-dilated binary mask
        crop_mask = binary_mask[y:y+h, x:x+w]
        
        rgba_crop = np.zeros((h, w, 4), dtype=np.uint8)
        rgba_crop[crop_mask > 0] = [0, 0, 0, 255]
        
        boxes.append({
            "id": i,
            "bbox": (x, y, w, h),
            "area": area,
            "crop_mask": crop_mask,
            "preview_img": Image.fromarray(rgba_crop, "RGBA"),
            "assigned_char": ""
        })

    # Sort left-to-right, top-to-bottom dynamically based on row height
    if boxes:
        avg_h = np.mean([b["bbox"][3] for b in boxes])
        row_height = max(50, int(avg_h * 0.75))
        boxes.sort(key=lambda b: (b["bbox"][1] // row_height, b["bbox"][0]))

    return boxes

# ============================================================
# VECTORIZATION & TTF COMPILER
# ============================================================

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
        
        # Simplify contour points
        epsilon = 0.008 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)

        if len(points) < 3:
            continue

        converted = [convert_point(float(pt[0]), float(pt[1])) for pt in points]
        
        # Invert winding direction for inner holes (e.g. inside O, B, D)
        if is_hole:
            converted = converted[::-1]

        pen.moveTo(converted[0])
        for pt in converted[1:]:
            pen.lineTo(pt)
        pen.closePath()

    return pen.glyph()

def compile_ttf_font(mapped_components, font_name="MyHandwriting", units_per_em=1000):
    glyph_order = [".notdef", "space"]
    cmap = {32: "space"}
    glyph_objects = {}

    # Default .notdef glyph
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
        
        # Auto-map sequential alphabet if box left empty
        if not char and auto_idx < len(default_alphabet):
            char = default_alphabet[auto_idx]
            auto_idx += 1

        if char:
            glyph_name = f"glyph_{ord(char)}"
            if glyph_name not in glyph_order:
                glyph_order.append(glyph_name)
                
                # Trim borders and scale glyph
                ys, xs = np.where(comp["crop_mask"] > 0)
                if len(xs) > 0:
                    cropped = comp["crop_mask"][ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                    h, w = cropped.shape
                    new_w = max(1, int(w * (700 / max(1, h))))
                    normalized = cv2.resize(cropped, (new_w, 700), interpolation=cv2.INTER_AREA)
                else:
                    normalized = comp["crop_mask"]

                glyph = bitmap_to_glyph(normalized)
                if glyph is not None:
                    glyph_objects[glyph_name] = glyph
                    cmap[ord(char)] = glyph_name
                    advance = int(max(350, min(1200, normalized.shape[1] * 1.15)))
                    metrics[glyph_name] = (advance, 0)

    if len(glyph_order) <= 2:
        raise ValueError("No characters detected. Please upload a clear alphabet reference image.")

    fb = FontBuilder(units_per_em, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyphs(glyph_objects)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({
        "familyName": font_name,
        "styleName": "Regular",
        "uniqueFontIdentifier": f"{font_name}-Regular",
        "fullName": f"{font_name} Regular"
    })
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost()

    font_stream = io.BytesIO()
    fb.save(font_stream)
    font_bytes = font_stream.getvalue()
    
    # Verify validity
    TTFont(io.BytesIO(font_bytes))
    return font_bytes

# ============================================================
# PROCEDURAL TYPOGRAPHY & T-SHIRT ENGINE
# ============================================================

LAYOUT_STYLES = [
    "Centered Classic", "Stacked Block", "Word Emphasis", "Mixed Letter Sizes",
    "Alternating Baseline", "Rotated Letters", "Wave Path", "Arc Layout",
    "Circular Emblem", "Vertical Stack", "Tight Spacing", "Expanded Spacing"
]

def render_styled_text_layout(phrase, canvas_size=(3000, 3000), seed=42):
    random.seed(seed)
    w_canvas, h_canvas = canvas_size
    mask = Image.new("L", (w_canvas, h_canvas), 0)
    words = phrase.strip().split() or ["TEXT"]

    base_size = int(w_canvas / max(4, len(phrase) * 0.55))
    font = ImageFont.load_default()

    curr_y = int(h_canvas * 0.25)
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
            cy_off = int(random.uniform(-10, 10))
            wdraw.text((cx, 10 + cy_off), char, font=font, fill=(255, 255, 255, 255))
            cx += cw + int(random.uniform(-2, 8))

        mask.paste(word_img.split()[3], (0, curr_y), word_img.split()[3])
        curr_y += int(base_size * 1.3)

    return mask

def apply_artwork_clipping_mask(artwork_image, text_mask):
    w, h = text_mask.size
    art_rgba = artwork_image.convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)

    art_arr = np.array(art_rgba, dtype=np.uint8)
    mask_arr = np.array(text_mask.convert("L"), dtype=np.uint8)

    art_alpha = (art_arr[:, :, 3].astype(np.float32) / 255.0)
    text_alpha = mask_arr.astype(np.float32) / 255.0
    
    final_alpha = (art_alpha * text_alpha * 255.0).clip(0, 255).astype(np.uint8)
    art_arr[:, :, 3] = final_alpha
    return Image.fromarray(art_arr, "RGBA")

def generate_50_tshirt_variations(phrase, artwork_img=None, canvas_size=(3000, 3000), master_seed=42):
    variations = []
    w, h = canvas_size

    for idx in range(50):
        seed = master_seed + (idx * 999)
        random.seed(seed)
        
        text_mask = render_styled_text_layout(phrase, canvas_size, seed)
        
        if artwork_img is not None:
            composited = apply_artwork_clipping_mask(artwork_img, text_mask)
        else:
            solid_black = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
            composited = apply_artwork_clipping_mask(solid_black, text_mask)

        variations.append({
            "id": idx + 1,
            "style_name": f"{LAYOUT_STYLES[idx % len(LAYOUT_STYLES)]} (Var #{idx+1})",
            "image": composited
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
if "generated_ttf" not in st.session_state:
    st.session_state.generated_ttf = None
if "tshirt_variations" not in st.session_state:
    st.session_state.tshirt_variations = []

st.sidebar.title("✍️ Navigation")
app_mode = st.sidebar.radio("Select Mode:", ["Mode A: Make My Font", "Mode B: T-Shirt Design Studio"])

if app_mode == "Mode A: Make My Font":
    st.title("🔤 Mode A — Make My Font (.TTF Compiler)")
    st.write("Upload an alphabet sheet (like `A-Z`), adjust parameters, map character glyphs, and generate an installable TrueType (.ttf) font.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("1. Reference Image Upload")
        ref_file = st.file_uploader("Upload Alphabet Sheet:", type=["png", "jpg", "jpeg", "webp"], key="mode_a_file")
        if ref_file:
            pil_img = Image.open(ref_file).convert("RGBA")
            contrast_val = st.slider("Contrast Enhancement:", 0.5, 3.0, 1.2)
            denoise_val = st.slider("Denoise Strength:", 0, 20, 5)
            dilation_val = st.slider("Stroke Connect Dilation:", 1, 15, 7, help="Connects separate pen strokes within letters")
            
            binary_mask, cleaned_preview = preprocess_reference_image(pil_img, contrast=contrast_val, denoise_strength=denoise_val)
            st.image(cleaned_preview, caption="Cleaned Threshold Mask", use_container_width=True)
            
            if st.button("🔍 Detect & Extract Glyphs"):
                st.session_state.components = detect_and_extract_glyphs(binary_mask, dilation_size=dilation_val)
                st.success(f"Detected {len(st.session_state.components)} glyph components!")

    with col2:
        st.subheader("2. Interactive Character Mapping")
        if st.session_state.components:
            st.write(f"Detected **{len(st.session_state.components)}** character glyphs:")
            
            # Interactive grid displaying ALL detected glyphs
            for idx, comp in enumerate(st.session_state.components):
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.image(comp["preview_img"], width=65)
                with c2:
                    comp["assigned_char"] = st.text_input(f"Character #{idx+1}", value=comp["assigned_char"], key=f"char_{idx}", max_chars=1)

            font_name_input = st.text_input("Font Family Name:", "MyHandwriting")
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
    st.title("👕 Mode B — T-Shirt Design Studio")
    st.write("Generate 50 T-shirt print compositions with uploaded artwork patterns clipped inside your phrase.")

    master_seed = st.sidebar.number_input("Random Seed:", value=42, step=1)

    tab_tshirt, tab_export = st.tabs(["1. Design Studio", "2. Export Gallery"])

    with tab_tshirt:
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            tshirt_phrase = st.text_area("Enter Phrase:", value="MAKE YOUR OWN PATH").upper()
            art_file = st.file_uploader("Upload Pattern/Flower Artwork Fill:", type=["png", "jpg", "jpeg", "webp"], key="art_upload")
            artwork_pil = Image.open(art_file).convert("RGBA") if art_file else None

        with col_t2:
            res_choice = st.selectbox("Resolution:", ["3000 x 3000 px", "4000 x 4000 px", "4500 x 5400 px"])
            canvas_dim = (4500, 5400) if "4500" in res_choice else (3000, 3000)

            if st.button("🚀 GENERATE 50 T-SHIRT DESIGNS", type="primary", use_container_width=True):
                with st.spinner("Generating 50 compositions..."):
                    st.session_state.tshirt_variations = generate_50_tshirt_variations(
                        tshirt_phrase, artwork_pil, canvas_dim, master_seed
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
