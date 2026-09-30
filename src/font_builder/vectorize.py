import os
import cv2
import numpy as np


def _contour_to_bezier_path(cnt, epsilon_ratio=0.006):
    """
    픽셀 외곽선(계단 형태의 폴리라인)을 부드러운 3차 베지어 SVG 경로로 변환.

    이전 버전은 외곽선의 모든 픽셀 점을 M/L(직선)로만 이었기 때문에 확대 시
    손글씨 특유의 곡선이 각지고 계단처럼 보였다. 두 단계로 개선한다:
    1. cv2.approxPolyDP로 폴리곤을 단순화해 대표 정점만 남긴다
       (perimeter 비례 epsilon → 글자 크기에 무관하게 일정한 단순화 강도).
    2. 단순화된 정점들을 Catmull-Rom 스플라인으로 보간해 3차 베지어 제어점을
       계산한다 — 모든 원래 정점을 정확히 지나가는 매끄러운 곡선이 된다.
    """
    peri = cv2.arcLength(cnt, True)
    epsilon = max(1.0, epsilon_ratio * peri)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    pts = approx.reshape(-1, 2).astype(np.float64)

    if len(pts) < 3:
        return None

    n = len(pts)
    d = f"M {pts[0][0]:.2f} {pts[0][1]:.2f} "
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        # Catmull-Rom -> Bezier 제어점 변환 (표준 1/6 계수)
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        d += f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p2[0]:.2f} {p2[1]:.2f} "
    d += "Z"
    return d


def png_to_svg_pure_python(png_path, svg_path):
    """
    외부 프로그램(potrace) 의존성 없이, OpenCV의 외곽선 탐지와 Catmull-Rom
    베지어 피팅을 활용하여 PNG 이미지를 매끄러운 벡터 SVG 파일로 변환합니다.
    """
    try:
        img = cv2.imread(png_path)
        if img is None:
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # RETR_CCOMP로 내부 구멍(ㅁ, ㅇ 등)까지 모두 검출
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        path_segments = []
        if hierarchy is not None:
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                d = _contour_to_bezier_path(cnt)
                if d:
                    path_segments.append(d)

        if not path_segments:
            return False

        full_path_d = " ".join(path_segments)

        h, w = gray.shape
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(f'<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n')
            f.write(f'  <path d="{full_path_d}" fill="black" fill-rule="evenodd" stroke="none" />\n')
            f.write('</svg>\n')

        return True
    except Exception as e:
        print(f"[오류] 변환 실패 ({png_path}): {e}")
        return False


def batch_vectorize(input_dir, output_dir):
    """
    지정된 폴더 내의 모든 PNG 파일을 SVG로 일괄 변환합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    success_count = 0

    print(f"총 {len(files)}개의 글자 이미지 벡터화를 시작합니다...")
    for idx, filename in enumerate(files):
        png_path = os.path.join(input_dir, filename)
        svg_filename = filename.replace('.png', '.svg')
        svg_path = os.path.join(output_dir, svg_filename)

        if png_to_svg_pure_python(png_path, svg_path):
            success_count += 1

        if (idx + 1) % 500 == 0:
            print(f" -> 벡터화 진행도: {idx + 1}/{len(files)} 완료...")

    print(f"벡터화 완료! 성공: {success_count}/{len(files)}")
    return success_count


if __name__ == "__main__":
    batch_vectorize("output/images", "output/svgs")
