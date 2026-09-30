"""
육안 확인용 비교 시트 생성.

1) 힌트 10자 원본 vs 같은 글자 생성본을 나란히 (상단: 원본, 하단: 생성)
2) 임의 문장을 생성 폰트 글리프로 렌더링

원본 힌트(data/style/*.png)와 생성본(output/images/<hex>.png)을 셀 격자로 배치.
"""
import os
import argparse
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 새 TTF로 직접 렌더링할 임의 문장 (모두 KS X 1001 상용 2350자 내 글자)
TTF_SENTENCES = [
    "다람쥐 헌 쳇바퀴에 타고파",
    "나는 오늘 학교에 가서 친구를 만났다",
]


def _load_cell(path, cell, pad=6):
    img = Image.open(path).convert("L").resize((cell - 2 * pad, cell - 2 * pad),
                                               Image.LANCZOS)
    canvas = Image.new("L", (cell, cell), 255)
    canvas.paste(img, (pad, pad))
    return canvas


def build_sheet(images_dir, style_dir, out_path, cell=120, ttf_path=None):
    hint_files = sorted(f for f in os.listdir(style_dir) if f.lower().endswith(".png"))
    hints = [(chr(int(os.path.splitext(f)[0], 16)),
              os.path.join(style_dir, f)) for f in hint_files]

    n = len(hints)
    label_h = 34
    header_h = 30
    # 상단: 원본 행, 하단: 생성 행
    sheet_w = n * cell + 4
    sheet_h = header_h + label_h + cell + label_h + cell + 4 + 40 + 4 * cell
    sheet = Image.new("L", (sheet_w, sheet_h), 255)
    draw = ImageDraw.Draw(sheet)
    try:
        lab_font = ImageFont.truetype("C:\\Windows\\Fonts\\malgun.ttf", 20)
        big_font = ImageFont.truetype("C:\\Windows\\Fonts\\malgun.ttf", 24)
    except Exception:
        lab_font = big_font = ImageFont.load_default()

    y = 6
    draw.text((6, y), "원본 손글씨 힌트 (위) vs 생성 글리프 (아래)", fill=0, font=big_font)
    y += header_h

    # 원본 행 라벨 + 셀
    draw.text((6, y), "원본:", fill=0, font=lab_font)
    y += label_h
    for i, (ch, p) in enumerate(hints):
        sheet.paste(_load_cell(p, cell), (i * cell + 2, y))
    y += cell

    # 생성 행 라벨 + 셀
    draw.text((6, y), "생성:", fill=0, font=lab_font)
    y += label_h
    for i, (ch, p) in enumerate(hints):
        gp = os.path.join(images_dir, f"{ord(ch):04X}.png")
        if os.path.exists(gp):
            sheet.paste(_load_cell(gp, cell), (i * cell + 2, y))
    y += cell + 4

    # 문장 렌더링 (빌드된 새 TTF로 직접 렌더링)
    if ttf_path is None:
        ttf_path = os.path.join(REPO_ROOT, "output", "Font", "MyHandWriting.ttf")
    draw.text((6, y), "문장 렌더링 (MyHandWriting.ttf):", fill=0, font=big_font)
    y += 44
    try:
        hand_font = ImageFont.truetype(ttf_path, 72)
        for sent in TTF_SENTENCES:
            draw.text((10, y), sent, fill=0, font=hand_font)
            y += 100
    except Exception as e:
        draw.text((10, y), f"TTF 렌더링 실패: {e}", fill=0, font=lab_font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)
    print(f"[비교시트] 저장: {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=os.path.join(REPO_ROOT, "output", "images"))
    ap.add_argument("--style-dir", default=os.path.join(REPO_ROOT, "data", "style"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "output", "eval",
                                                  "comparison_sheet.png"))
    a = ap.parse_args()
    build_sheet(a.images_dir, a.style_dir, a.out)
