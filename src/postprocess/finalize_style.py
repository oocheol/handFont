"""
스타일 마무리 단계: 기울기 적용 + 힌트 원본 재설치.

실행 순서(전체 파이프라인):
  generate(_bestofn) → repair_missing_strokes → remove_strays
  → normalize_stroke_weight → **finalize_style** → vectorize → build

1. 기울기: 원본 필체 평균 전단계수 0.18 vs 생성본 0.065 실측 → 글자별
   고유 기울기에 +0.115 평행이동 (개성 보존, 전체 경향만 원본에 맞춤).
2. 힌트 10자(매/실/효/소/이/새/별/조/영/화)는 디퓨전 근사본 대신
   원본 손글씨(data/style)를 직접 사용 — 두께만 통일하고 기울기는
   원본 그대로(진짜 필체이므로 전단 미적용).
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from normalize_stroke_weight import normalize_glyph

SLANT_DELTA = 0.115
TARGET_THICKNESS = None  # None이면 현재 분포 중앙값 사용


def apply_slant(img, delta=SLANT_DELTA):
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


def _median_thickness(images_dir):
    ts = []
    for f in sorted(os.listdir(images_dir)):
        if not f.endswith('.png'):
            continue
        img = cv2.imread(os.path.join(images_dir, f), cv2.IMREAD_GRAYSCALE)
        m = (img < 128).astype(np.uint8)
        if m.sum() < 20:
            continue
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        ts.append(dist[m > 0].mean() * 2)
    return float(np.median(ts)) if ts else 10.0


def finalize(images_dir=None, style_dir=None):
    if images_dir is None:
        images_dir = os.path.join(REPO_ROOT, "output", "images")
    if style_dir is None:
        style_dir = os.path.join(REPO_ROOT, "data", "style")

    target = TARGET_THICKNESS or _median_thickness(images_dir)
    print(f"[마무리] 목표 두께 {target:.2f}px")

    # 1. 전체 기울기 적용 (문장부호 등 비한글 코드포인트는 순수 도형이므로 제외)
    n = 0
    for f in sorted(os.listdir(images_dir)):
        if not f.endswith('.png'):
            continue
        try:
            if not (0xAC00 <= int(f[:4], 16) <= 0xD7A3):
                continue
        except ValueError:
            continue
        p = os.path.join(images_dir, f)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(p, apply_slant(img))
        n += 1
    print(f"[마무리] 기울기 +{SLANT_DELTA} 적용: {n}자")

    # 2. 힌트 원본 재설치 (전단 미적용 - 원본 기울기 보존)
    installed = []
    for f in sorted(os.listdir(style_dir)):
        if not f.endswith('.png'):
            continue
        hexn = f[:4]
        img = cv2.imread(os.path.join(style_dir, f), cv2.IMREAD_GRAYSCALE)
        up = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        up = cv2.GaussianBlur(up, (3, 3), 0)
        _, up = cv2.threshold(up, 127, 255, cv2.THRESH_BINARY)
        out, _, _ = normalize_glyph(up, target, max_steps=10)
        cv2.imwrite(os.path.join(images_dir, f"{hexn}.png"), out)
        installed.append(chr(int(hexn, 16)))
    print(f"[마무리] 힌트 원본 재설치: {''.join(installed)}")


if __name__ == "__main__":
    finalize()
