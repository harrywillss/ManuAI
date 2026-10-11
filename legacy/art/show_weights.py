from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter 

# === Setup ===
weights_path = "art/all_weights.txt"
mask_path = "art/manuai_mask2.png"  # silhouette image
output_path = "art/manuai_weights_fantail2.png"

# === Poster settings ===
img_width, img_height = 4961, 7016  # A2 at 300 DPI
padding = 256  # pixels
usable_width = img_width - 2 * padding
usable_height = img_height - 2 * padding
font_size = 18
line_height = font_size + 4
font_path = "/Library/Fonts/Arial.ttf"

# === Load weights ===
with open(weights_path, "r") as f:
    weights = f.read().replace('\n', ' ')

# === Create image canvas ===
base_img = Image.new('L', (img_width, img_height), color=255)  # 'L' mode (grayscale)
draw = ImageDraw.Draw(base_img)
font = ImageFont.truetype(font_path, font_size)

# # === Draw text ===
x, y = padding, padding
line_height = font_size + 4
max_width = img_width - padding

for char in weights:
    bbox = font.getbbox(char)
    char_width = bbox[2] - bbox[0]

    if x + char_width > padding + usable_width:
        x = padding
        y += line_height
        if y + line_height > padding + usable_height:
            break
    draw.text((x, y), char, font=font, fill=0)  # black text on white
    x += char_width

# === Load and resize bird silhouette mask ===
bird_mask = Image.open(mask_path).convert("L")  # grayscale
resized_bird = bird_mask.resize((usable_width, usable_height), resample=Image.Resampling.LANCZOS)
bird_mask_padded = Image.new("L", (img_width, img_height), color=0)
bird_mask_padded.paste(resized_bird, (padding, padding))
bird_mask = bird_mask_padded

# Threshold the mask to make it binary (black/white)
threshold = 128
bird_mask = bird_mask.point(lambda p: 255 if p > threshold else 0)

# Invert: bird = white, background = black
inverted_mask = Image.eval(bird_mask, lambda p: 255 - p)

# === Apply mask ===
final_img = Image.composite(base_img, Image.new("L", base_img.size, 255), inverted_mask)

# === Convert to RGB for saving ===
final_rgb = final_img.convert("RGB")
final_rgb.save(output_path)
print(f"Poster saved to {output_path}")