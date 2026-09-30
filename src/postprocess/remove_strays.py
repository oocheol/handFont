"""
고립 잡티(stray speck) 제거 — 파이프라인 정식 단계.

실행 순서: generator.py → repair_missing_strokes.py → **remove_strays.py**
           → normalize_stroke_weight.py → vectorize.py → build_font.py

규칙 (2026-07-07 전/후 비교 시트 검증으로 확정):
  단순 "본체에서 먼 소형 성분 제거"는 ㅊ의 꼭지점('옻','촛')이나 ㅋ의 안쪽
  획('캘')처럼 정당하게 분리된 획까지 절단함을 확인. 따라서:
  - 생성 글리프의 성분 수가 시스템 폰트 템플릿의 기대 성분 수를 초과할 때만
  - 초과분 개수만큼만
  - "본체(최대 성분)에서 가장 멀리 떨어진 소형 성분"부터 제거한다.
  성분 수가 기대치 이하면 아무것도 제거하지 않는다 (정당한 획 보호).
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REL_MAX = 0.08   # 전체 잉크 대비 이 비율 이상인 성분은 잡티 후보에서 제외
MIN_DIST = 18    # 본체와의 최단거리(px)가 이 이하인 성분은 제거하지 않음

_font = None


def _template_cc(char):
    global _font
    if _font is None:
        _font = ImageFont.truetype("C:\\Windows\\Fonts\\malgun.ttf", 200)
    img = Image.new('L', (256, 256), 255)
    d = ImageDraw.Draw(img)
    l, t, r, b = d.textbbox((0, 0), char, font=_font)
    d.text(((256 - (r - l)) // 2 - l, (256 - (b - t)) // 2 - t), char, fill=0, font=_font)
    m = (np.array(img) < 128).astype(np.uint8)
    n, _, _, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return n - 1


def remove_strays(img, char):
    """단일 글리프에서 잡티 제거. (결과 이미지, 제거 수) 반환."""
    ink = (img < 128).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 2:
        return img, 0
    excess = (n - 1) - _template_cc(char)
    if excess <= 0:
        return img, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    total = areas.sum()
    main_lbl = 1 + int(np.argmax(areas))
    main_mask = (labels == main_lbl).astype(np.uint8)
    dist = cv2.distanceTransform(1 - main_mask, cv2.DIST_L2, 5)

    cands = []
    for lbl in range(1, n):
        if lbl == main_lbl:
            continue
        if stats[lbl, cv2.CC_STAT_AREA] >= total * REL_MAX:
            continue
        d = dist[labels == lbl].min()
        if d > MIN_DIST:
            cands.append((d, lbl))
    cands.sort(reverse=True)

    out = img.copy()
    removed = 0
    for d, lbl in cands[:excess]:
        out[labels == lbl] = 255
        removed += 1
    return out, removed


def clean_directory(images_dir=None):
    if images_dir is None:
        images_dir = os.path.join(REPO_ROOT, "output", "images")
    changed = []
    for f in sorted(os.listdir(images_dir)):
        if not f.endswith('.png'):
            continue
        try:
            char = chr(int(f[:4], 16))
        except ValueError:
            continue
        if not (0xAC00 <= ord(char) <= 0xD7A3):
            continue  # 문장부호 등은 건드리지 않음
        p = os.path.join(images_dir, f)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        out, removed = remove_strays(img, char)
        if removed:
            cv2.imwrite(p, out)
            changed.append(char)
    print(f"[잡티 제거] {len(changed)}자 정리: {''.join(changed)}")
    return changed


if __name__ == "__main__":
    clean_directory()
