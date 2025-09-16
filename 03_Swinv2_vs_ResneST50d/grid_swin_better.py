# grid_from_swin_better.py
import pandas as pd, math
from PIL import Image, ImageOps, ImageDraw, ImageFont

IN_CSV = "swin_better_than_resnet.csv"  # needs columns: file,f1_swin,f1_resnet,delta
OUT_IMG = "swin_better_grid.jpg"
TOP_N = 16        # how many images to show
COLS = 4          # grid columns
CELL = 256        # image tile size (px)
PAD = 8

def load_font(sz=16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=sz)
    except:
        return ImageFont.load_default()

def load_image(path, size=(CELL, CELL)):
    try:
        img = Image.open(path).convert("RGB")
        return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)
    except Exception:
        return Image.new("RGB", size, (80,80,80))

def annotate(img, text, font):
    draw = ImageDraw.Draw(img)
    strip = int(img.height * 0.24)
    overlay = Image.new("RGBA", (img.width, strip), (0,0,0,160))
    img = img.convert("RGBA"); img.paste(overlay, (0, img.height-strip), overlay); img = img.convert("RGB")
    y = img.height - strip + 6
    for line in text.split("\n"):
        draw.text((8, y), line, font=font, fill=(255,255,255))
        y += font.size + 2
    return img

def main():
    df = pd.read_csv(IN_CSV).head(TOP_N)
    if df.empty:
        print("No rows in", IN_CSV); return

    rows = math.ceil(len(df) / COLS)
    W = COLS*CELL + (COLS+1)*PAD
    H = rows*CELL + (rows+1)*PAD
    canvas = Image.new("RGB", (W, H), (20,20,20))
    font = load_font(16)

    for i, r in enumerate(df.itertuples(index=False)):
        img = load_image(r.file)
        cap = f"ΔF1={r.delta:.2f}\nSwin={r.f1_swin:.2f}  Res={r.f1_resnet:.2f}"
        img = annotate(img, cap, font)
        c, rr = i % COLS, i // COLS
        canvas.paste(img, (PAD + c*(CELL+PAD), PAD + rr*(CELL+PAD)))

    canvas.save(OUT_IMG, quality=95)
    print("[✓] Saved", OUT_IMG)

if __name__ == "__main__":
    main()
