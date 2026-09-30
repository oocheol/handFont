"""
글자 단위 반복 개선 도구 (human-in-the-loop).

샘플이 10자뿐이라 디퓨전 생성이 확률적으로 애매한 글자가 계속 나온다.
이 스크립트는 "이상한 글자를 발견 → 후보 뽑아 보고 → 고른 것 설치"
사이클을 두 명령으로 줄인다.

사용법:
  # 1) 후보 시트 생성 (글자당 20개, output/fix/<hex>_NN.png + sheet_<hex>.png)
  py -3 src/postprocess/fix_glyphs.py candidates 새효화고

  # 2) 고른 후보 설치 (두께 정규화 + 기울기 적용 + SVG + TTF 재빌드까지)
  py -3 src/postprocess/fix_glyphs.py install 새=03 효=11 고=00

  # (선택) 자동 선택: 영역검사+성분수+잉크비 복합 점수 최상 후보 자동 설치
  py -3 src/postprocess/fix_glyphs.py auto 새효화고
"""
import os
import sys
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "model"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "eval"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "postprocess"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "font_builder"))

import numpy as np
import cv2

FIX_DIR = os.path.join(REPO_ROOT, "output", "fix")
IMAGES_DIR = os.path.join(REPO_ROOT, "output", "images")
SVGS_DIR = os.path.join(REPO_ROOT, "output", "svgs")
TTF_PATH = os.path.join(REPO_ROOT, "output", "Font", "MyHandWriting.ttf")
FFPYTHON = "C:\\Program Files\\FontForgeBuilds\\bin\\ffpython.exe"

TARGET_THICKNESS = 10.13
SLANT_DELTA = 0.115
N_CANDIDATES = 20


def _shear(img, delta=SLANT_DELTA):
    m = (img < 128).astype(np.uint8)
    mo = cv2.moments(m, binaryImage=True)
    if mo['mu02'] == 0 or mo['m00'] == 0:
        return img
    h, w = img.shape
    cy = mo['m01'] / mo['m00']
    M = np.float32([[1, delta, -delta * cy], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    return out


def generate_candidates(chars, seed_base=None):
    """글자당 N_CANDIDATES개 후보 + 육안 선택용 시트 생성."""
    import torch
    from PIL import Image, ImageFont
    import torchvision.transforms as transforms
    from generator import (load_fontdiffuser_pipeline, make_content_image,
                           get_best_style_char, postprocess_glyph)

    if seed_base is None:
        import random
        seed_base = random.randint(0, 10**6)  # 매 실행 새로운 시드군

    device = 'cuda:0'
    pipe, args = load_fontdiffuser_pipeline(os.path.join(REPO_ROOT, 'weights'), device)
    style_dir = os.path.join(REPO_ROOT, 'data', 'style')
    style_hints = {}
    for f in os.listdir(style_dir):
        if f.lower().endswith('.png'):
            style_hints[chr(int(os.path.splitext(f)[0], 16))] = \
                Image.open(os.path.join(style_dir, f)).convert('RGB')

    font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 100)
    ct = transforms.Compose([
        transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    st = transforms.Compose([
        transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    cache = {c: st(img)[None, :].to(device) for c, img in style_hints.items()}

    os.makedirs(FIX_DIR, exist_ok=True)
    for char in chars:
        hexn = f"{ord(char):04X}"
        content = ct(make_content_image(char, font, size=128))[None, :].to(device)
        style = cache[get_best_style_char(char, style_hints.keys())]
        sheet = Image.new('L', (128 * 10, 128 * ((N_CANDIDATES + 9) // 10)), 255)
        for k in range(N_CANDIDATES):
            torch.manual_seed(seed_base + ord(char) * 41 + k * 6151)
            with torch.no_grad():
                raw = pipe.generate(
                    content_images=content, style_images=style, batch_size=1,
                    order=args.order, num_inference_step=args.num_inference_steps,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    t_start=args.t_start, t_end=args.t_end,
                    dm_size=args.content_image_size, return_tensor=True)
            gray = ((raw[0, 0] * 0.299 + raw[0, 1] * 0.587 + raw[0, 2] * 0.114)
                    .cpu().numpy() * 255).astype(np.uint8)
            cand = postprocess_glyph(gray)
            cv2.imwrite(os.path.join(FIX_DIR, f"{hexn}_{k:02d}.png"), cand)
            from PIL import Image as PImage
            sheet.paste(PImage.fromarray(cand).resize((128, 128)),
                        ((k % 10) * 128, (k // 10) * 128))
        sheet_path = os.path.join(FIX_DIR, f"sheet_{hexn}.png")
        sheet.save(sheet_path)
        print(f"'{char}' 후보 {N_CANDIDATES}개 -> {sheet_path}")


def install(picks, rebuild=True):
    """picks: {char: index}. 정규화+기울기 적용 후 설치, SVG/TTF 갱신."""
    from normalize_stroke_weight import normalize_glyph
    from vectorize import png_to_svg_pure_python

    for char, idx in picks.items():
        hexn = f"{ord(char):04X}"
        src = os.path.join(FIX_DIR, f"{hexn}_{int(idx):02d}.png")
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[오류] 후보 없음: {src}")
            continue
        img, b, a = normalize_glyph(img, TARGET_THICKNESS, max_steps=10)
        img = _shear(img)
        out_png = os.path.join(IMAGES_DIR, f"{hexn}.png")
        cv2.imwrite(out_png, img)
        png_to_svg_pure_python(out_png, os.path.join(SVGS_DIR, f"{hexn}.svg"))
        print(f"'{char}' 설치 (두께 {b:.1f}->{a:.1f})")

    if rebuild and picks:
        rebuild_ttf()


def rebuild_ttf():
    """TTF 재빌드. 설치된 폰트로 파일이 잠긴 경우 새 경로 빌드 후 교체."""
    tmp = TTF_PATH.replace(".ttf", "_new.ttf")
    r = subprocess.run([FFPYTHON, os.path.join(REPO_ROOT, "src", "font_builder", "build_font.py"),
                        SVGS_DIR, tmp], capture_output=True, text=True)
    if os.path.exists(tmp):
        try:
            os.replace(tmp, TTF_PATH)
            print(f"[완료] TTF 재빌드: {TTF_PATH}")
        except PermissionError:
            print(f"[경고] {TTF_PATH} 잠김 - 새 빌드는 {tmp}에 있음. 폰트 삭제 후 교체 필요.")
    else:
        print("[오류] 빌드 실패:", r.stdout[-500:] if r.stdout else r.stderr[-500:])


def auto_pick(chars):
    """component_check + 성분 수 + 잉크비 복합 점수로 최상 후보 자동 설치."""
    from component_check import missing_regions, template_grid
    from PIL import Image, ImageDraw, ImageFont

    def tmpl_stats(char):
        font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 200)
        img = Image.new('L', (256, 256), 255)
        d = ImageDraw.Draw(img)
        l, t, r, b = d.textbbox((0, 0), char, font=font)
        d.text(((256 - (r - l)) // 2 - l, (256 - (b - t)) // 2 - t), char, fill=0, font=font)
        m = (np.array(img) < 128).astype(np.uint8)
        n, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        return n - 1, m.mean()

    picks = {}
    for char in chars:
        hexn = f"{ord(char):04X}"
        t_cc, t_ink = tmpl_stats(char)
        best = None
        for k in range(N_CANDIDATES):
            p = os.path.join(FIX_DIR, f"{hexn}_{k:02d}.png")
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            m = (img < 128).astype(np.uint8)
            n, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            score = (len(missing_regions(char, img)) * 10
                     + abs((n - 1) - t_cc) * 2
                     + abs(m.mean() - t_ink * 0.7) * 10)
            if best is None or score < best[0]:
                best = (score, k)
        if best:
            picks[char] = best[1]
            print(f"'{char}' 자동 선택: #{best[1]:02d} (점수 {best[0]:.2f})")
    install(picks)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "candidates":
        generate_candidates(list(sys.argv[2]))
    elif cmd == "install":
        picks = {}
        for a in sys.argv[2:]:
            c, i = a.split("=")
            picks[c] = int(i)
        install(picks)
    elif cmd == "auto":
        generate_candidates(list(sys.argv[2]))
        auto_pick(list(sys.argv[2]))
    else:
        print(__doc__)
