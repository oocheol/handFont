"""
생성된 글리프들의 획 굵기를 일정 범위로 정규화.

원인: 스타일 힌트 10장(매/실/효/소/이/새/별/조/영/화)의 실제 펜 압력/굵기가
서로 2배 가까이 차이 남 (매≈7px 두께 vs 화≈3.3px 두께, distanceTransform
기준 실측). 단일 참조 스타일 선택(get_best_style_char) 특성상 어떤 힌트를
쓰느냐에 따라 생성된 글자의 획 굵기가 그대로 따라가서, 한 문장 안에서 어떤
글자는 굵고 어떤 글자는 가늘게 보이는 얼룩덜룩한 결과가 나온다
(output/images 2,350자 실측: 두께 4.8~13.7px, 중앙값 8.1px, 표준편차 2.5px).

이 스크립트는 새로 추론을 돌리지 않고, 이미 생성된 PNG들의 획 두께만
목표값 쪽으로 침식/팽창시켜 시각적 일관성을 맞춘다.
"""
import os
import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def measure_thickness(mask):
    """이진 잉크 마스크(1=잉크)의 평균 획 두께를 distance transform으로 추정."""
    ink_px = int(mask.sum())
    if ink_px < 20:
        return 0.0, ink_px
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return float(dist[mask > 0].mean() * 2), ink_px


def normalize_glyph(gray_img, target_thickness, tolerance=0.5, max_steps=6):
    """
    글리프 그레이스케일 이미지(흰 배경/검은 잉크)의 획 두께를 target_thickness
    쪽으로 침식(굵음→얇게) 또는 팽창(얇음→굵게)시켜 조정.

    v2: 큰 커널 1회 → 3x3 소폭 반복으로 변경. 이전 방식은 굵은 글자에
    큰 커널 침식을 걸다 미세 획이 사라지면 통째로 롤백해 아예 조정을 못 받는
    글자가 남았고(실측 8.0~13.9px, 1.7배 편차), 반복 방식은 "성분이 사라지기
    직전 단계"까지는 조정을 유지할 수 있어 잔여 편차가 훨씬 작다.
    """
    mask = (gray_img < 128).astype(np.uint8)
    thickness, ink_px = measure_thickness(mask)
    if ink_px < 20:
        return gray_img, thickness, thickness  # 빈 글리프는 손대지 않음

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    n_orig = cv2.connectedComponents(mask, connectivity=8)[0]
    cur = mask
    cur_t = thickness

    for _ in range(max_steps):
        delta = target_thickness - cur_t
        if abs(delta) <= tolerance:
            break
        if delta > 0:
            step = cv2.dilate(cur, k3, iterations=1)
        else:
            step = cv2.erode(cur, k3, iterations=1)
            # 침식 가드: 이 단계에서 미세 획(성분)이 사라지면 직전 단계에서 멈춤
            if cv2.connectedComponents(step, connectivity=8)[0] < n_orig:
                break
        step_t, step_ink = measure_thickness(step)
        if step_ink < 20:
            break
        # 목표를 지나쳐 반대편으로 더 멀어지면 직전 단계가 최선
        if abs(target_thickness - step_t) >= abs(target_thickness - cur_t):
            break
        cur, cur_t = step, step_t

    result = np.where(cur > 0, 0, 255).astype(np.uint8)
    return result, thickness, cur_t


def _is_hangul_file(fname):
    try:
        return 0xAC00 <= int(os.path.splitext(fname)[0], 16) <= 0xD7A3
    except ValueError:
        return False


def normalize_directory(images_dir, target_thickness=None, tolerance=0.6):
    # 문장부호(.,!?:;) 등은 순수 도형이라 손글씨 두께 정규화(침식/팽창) 대상이
    # 아님 - 서로 떨어진 여러 성분(쉼표=원+꼬리)이 팽창으로 뭉개지는 결함이
    # 실측됨(2026-07). 한글 음절 코드포인트(AC00-D7A3)만 처리한다.
    files = sorted(f for f in os.listdir(images_dir)
                   if f.lower().endswith('.png') and _is_hangul_file(f))

    # 목표 두께가 없으면 현재 분포의 중앙값을 사용
    if target_thickness is None:
        samples = []
        for f in files:
            img = cv2.imread(os.path.join(images_dir, f), cv2.IMREAD_GRAYSCALE)
            mask = (img < 128).astype(np.uint8)
            t, ink_px = measure_thickness(mask)
            if ink_px >= 20:
                samples.append(t)
        target_thickness = float(np.median(samples)) if samples else 6.0
        print(f"[정규화] 목표 두께(중앙값)를 자동 산출: {target_thickness:.2f}px")

    changed = 0
    before_vals, after_vals = [], []
    for f in files:
        path = os.path.join(images_dir, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        result, before, after = normalize_glyph(img, target_thickness, tolerance)
        if before > 0:
            before_vals.append(before)
            after_vals.append(after)
        if not np.array_equal(result, img):
            cv2.imwrite(path, result)
            changed += 1

    print(f"[정규화] 총 {len(files)}자 중 {changed}자 두께 조정")
    if before_vals:
        print(f"[정규화] 조정 전 std={np.std(before_vals):.2f}, "
              f"조정 후 std={np.std(after_vals):.2f}")
    return target_thickness, changed


if __name__ == "__main__":
    normalize_directory(os.path.join(REPO_ROOT, "output", "images"))
