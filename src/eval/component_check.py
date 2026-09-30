"""
템플릿 대비 영역별 잉크 검사 — 자모/획 통째 소실 감지.

배경: 디퓨전 생성이 확률적이라 ㅏ의 짧은 가로획 같은 미세 획이 실행에 따라
소실될 수 있음 ('다'→'디', '나'→'니'). CC 수 비교는 손글씨의 자연스러운
자모 연결(정상)과 소실을 구분하지 못하고, OCR은 이 손글씨체를 신뢰성 있게
읽지 못함(실측 '가'→'커'). 대신:

  글자를 잉크 bbox로 정규화한 뒤 4x4 격자로 나눠, 시스템 폰트 템플릿에
  유의미한 잉크가 있는 칸이 생성본에서 거의 비어 있으면 "영역 소실"로 판정.

손글씨의 위치 왜곡/이어쓰기에는 관대하고(칸 단위 비교 + 여유 임계),
가로획 하나가 통째로 없어진 경우는 해당 칸이 비므로 확실히 잡는다.
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

GRID = 4
TMPL_SIGNIFICANT = 0.06   # 템플릿 칸 잉크 비율이 이 이상이면 "여기에 획이 있어야 함"
GEN_EMPTY = 0.012         # 생성본 칸 잉크 비율이 이 미만이면 "비어 있음"

_font_cache = {}


def _get_font(size=200):
    if size not in _font_cache:
        for fp in ("C:\\Windows\\Fonts\\malgun.ttf", "C:\\Windows\\Fonts\\batang.ttc"):
            if os.path.exists(fp):
                _font_cache[size] = ImageFont.truetype(fp, size)
                break
    return _font_cache[size]


def _bbox_normalized_mask(gray_or_mask, size=128):
    """잉크 bbox로 크롭 후 size x size로 리사이즈한 잉크 마스크(bool) 반환."""
    mask = gray_or_mask < 128 if gray_or_mask.dtype == np.uint8 else gray_or_mask
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return resized > 60


def _grid_ink(mask, grid=GRID):
    """정규화 마스크의 grid x grid 칸별 잉크 비율."""
    size = mask.shape[0]
    cell = size // grid
    out = np.zeros((grid, grid))
    for r in range(grid):
        for c in range(grid):
            block = mask[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            out[r, c] = block.mean()
    return out


def template_grid(char):
    font = _get_font()
    img = Image.new('L', (256, 256), 255)
    draw = ImageDraw.Draw(img)
    l, t, r, b = draw.textbbox((0, 0), char, font=font)
    draw.text(((256 - (r - l)) // 2 - l, (256 - (b - t)) // 2 - t), char, fill=0, font=font)
    mask = _bbox_normalized_mask(np.array(img))
    return _grid_ink(mask) if mask is not None else None


def missing_regions(char, gen_gray):
    """
    생성 글리프에서 소실된 격자 칸 목록 반환. 빈 리스트면 통과.
    gen_gray: 흰 배경/검은 잉크 그레이스케일 (np.uint8)
    """
    tmpl = template_grid(char)
    gen_mask = _bbox_normalized_mask(gen_gray)
    if tmpl is None:
        return []
    if gen_mask is None:
        return [("ALL", 0.0)]
    gen = _grid_ink(gen_mask)

    missing = []
    for r in range(GRID):
        for c in range(GRID):
            if tmpl[r, c] >= TMPL_SIGNIFICANT and gen[r, c] < GEN_EMPTY:
                missing.append(((r, c), round(float(tmpl[r, c]), 3)))

    # 판정 보정: 칸 1개 소실 + 템플릿 잉크가 옅은 칸(<0.15)은 필기 위치
    # 왜곡에 의한 오탐일 가능성이 높아 통과시킨다 (실측: 정상 '가'가
    # (3,2)=0.094 단일 칸으로 오탐되던 케이스).
    if len(missing) == 1 and missing[0][1] < 0.15:
        return []
    return missing


def check_directory(images_dir):
    """디렉토리 전체 검사, 소실 글리프 dict {char: missing_cells} 반환."""
    bad = {}
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith('.png'))
    for f in files:
        try:
            char = chr(int(os.path.splitext(f)[0], 16))
        except ValueError:
            continue
        img = cv2.imread(os.path.join(images_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        m = missing_regions(char, img)
        if m:
            bad[char] = m
    return bad


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "output/images"
    bad = check_directory(d)
    print(f"영역 소실 글리프: {len(bad)}개")
    print("".join(sorted(bad.keys())))
