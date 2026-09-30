"""
스타일 힌트 이미지 재구축 (v2).

v1의 한계(사용자 피드백으로 확인): 원본 crop을 그대로 Otsu 이진화만 해서
- 저해상도·블러 원본에서 획이 점으로 조각남 ('이'의 ㅇ, '새'의 ㅅ)
- 복사기 블러 헤일로가 잉크로 잡혀 획이 뭉개진 블롭이 됨 ('매','실','효')
- 끊긴 획을 복원하는 단계가 아예 없었음

v2 추출 파이프라인 (글자당 모든 후보에 적용 후 최고 점수 선택):
1. 작업 해상도로 업스케일 (짧은 변 512px) — 저해상도 원본의 형태 정보 보존
2. median blur로 JPEG/복사기 점 노이즈 제거
3. 이중 threshold: Otsu / Otsu*0.82(엄격) 두 버전 생성
   → 블러 헤일로가 낀 원본은 Otsu가 블롭을 만들므로, 획 두께가 건전한
     범위(캔버스 대비 3~9%)에 더 가까운 버전을 채택
4. closing(잉크=255 극성)으로 끊긴 획 연결 → opening으로 잔여 점 제거
5. 상대 기준 CC 필터 (총 잉크의 1% 미만 & 절대 소형은 노이즈)
6. 채점: 파편화(끊김) + 블롭성(두께 과다) + 노이즈 비율 종합 벌점
7. 잉크 bbox 중심 정렬 + 14% 패딩 → 128px 저장

출력: data/style/<유니코드 hex>.png (128x128, 흰 배경/검은 잉크) 문자당 1장.
"""
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ORIGINAL_DIR = os.path.join(REPO_ROOT, "data", "original")
STYLE_DIR = os.path.join(REPO_ROOT, "data", "style")

WORK_SIZE = 512      # 처리용 업스케일 크기 (짧은 변 기준)
PAD_RATIO = 0.14
OUT_SIZE = 128
# 건전한 획 두께 범위: 업스케일 캔버스(512) 대비 비율
# LO 0.03 -> 0.045 상향: 얇은 힌트('화' 0.033)가 그대로 통과되면 생성 단계에서
# ㅏ의 짧은 가로획 같은 미세 획이 소실되는 회귀가 관측됨(A/B '가' 비교)
THICKNESS_LO = 0.045  # 이보다 얇으면 생성시 미세 획 소실 위험 -> 팽창 보정
THICKNESS_HI = 0.09   # 이보다 굵으면 블롭


def _measure_thickness(ink_mask):
    """잉크 마스크(255=잉크)의 평균 획 두께(px)."""
    m = (ink_mask > 0).astype(np.uint8)
    if m.sum() < 20:
        return 0.0
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return float(dist[m > 0].mean() * 2)


def _binarize_dual(gray):
    """Otsu와 엄격(Otsu*0.82) 두 가지 이진화 후보 반환 (잉크=255)."""
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results = []
    for tval in (otsu_val, otsu_val * 0.82):
        _, mask = cv2.threshold(gray, tval, 255, cv2.THRESH_BINARY_INV)
        results.append(mask)
    return results


def _repair_strokes(ink_mask, open_size=3, max_close=31, target_cc=3):
    """
    끊긴 획을 적응적으로 복원. 원본 필기 자체가 점선처럼 끊긴 경우('이'의 ㅇ,
    '새'의 ㅅ)가 있어 고정 커널로는 부족 — 주요 성분 수가 target_cc 이하로
    떨어지거나 상한에 닿을 때까지 closing 커널을 키운다. 커진 커널이 만드는
    모서리 뭉침은 마지막 opening으로 완화.
    """
    ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    best = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, ok)

    for close_size in range(7, max_close + 1, 4):
        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        closed = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, ck)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ok)
        cleaned, _ = _cc_filter(opened)
        _, n_cc = _fragmentation(cleaned)
        best = opened
        if 0 < n_cc <= target_cc:
            break
    return best


def _normalize_thickness(ink_mask, canvas, lo=THICKNESS_LO, hi=THICKNESS_HI):
    """획 두께가 건전 범위를 벗어나면 팽창/침식으로 범위 안까지 보정."""
    t = _measure_thickness(ink_mask) / canvas
    for _ in range(4):
        if t < lo:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            ink_mask = cv2.dilate(ink_mask, k, iterations=1)
        elif t > hi:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            ink_mask = cv2.erode(ink_mask, k, iterations=1)
        else:
            break
        t = _measure_thickness(ink_mask) / canvas
    return ink_mask


def _cc_filter(ink_mask, rel_ratio=0.01, abs_min=40):
    """총 잉크량 대비 비율 + 절대 크기 기준으로 소형 노이즈 성분 제거."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    if num_labels <= 1:
        return ink_mask, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = areas.sum()
    min_area = max(abs_min, total * rel_ratio)
    cleaned = np.zeros_like(ink_mask)
    removed = 0
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == lbl] = 255
        else:
            removed += 1
    return cleaned, removed


def _fragmentation(ink_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    if num_labels <= 1:
        return 1.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return 1.0 - areas.max() / areas.sum(), num_labels - 1


def expected_components(char):
    """
    글자의 '올바른' 연결요소 수를 시스템 폰트로 렌더링해 산출.
    예: '이'=2(ㅇ+ㅣ), '소'=2(ㅅ+ㅗ), '매'=2 — 이 수를 초과하는 성분은
    끊긴 획이므로 closing을 그 수에 도달할 때까지 키운다.
    """
    for fp in ("C:\\Windows\\Fonts\\malgun.ttf", "C:\\Windows\\Fonts\\batang.ttc"):
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 200)
            break
    else:
        return 3  # 폰트 없으면 보수적 기본값
    img = Image.new('L', (256, 256), 255)
    draw = ImageDraw.Draw(img)
    l, t, r, b = draw.textbbox((0, 0), char, font=font)
    draw.text(((256 - (r - l)) // 2 - l, (256 - (b - t)) // 2 - t), char, fill=0, font=font)
    mask = (np.array(img) < 128).astype(np.uint8)
    n, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return max(1, n - 1)


def extract_glyph(bgr_img, target_cc=3):
    """
    단일 크롭 이미지에서 정제된 잉크 마스크(잉크=255, WORK 캔버스)를 추출하고
    품질 벌점을 함께 반환. 벌점이 낮을수록 좋음.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # 1. 업스케일 (짧은 변 -> WORK_SIZE)
    h, w = gray.shape
    scale = WORK_SIZE / min(h, w)
    gray = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_CUBIC)

    # 2. 점 노이즈 제거
    gray = cv2.medianBlur(gray, 5)

    # 3. 이중 threshold 후 획 두께가 건전 범위에 가까운 버전 채택
    canvas = min(gray.shape)
    best = None
    for mask in _binarize_dual(gray):
        repaired = _repair_strokes(mask, target_cc=target_cc)
        cleaned, _ = _cc_filter(repaired)
        cleaned = _normalize_thickness(cleaned, canvas)
        t = _measure_thickness(cleaned) / canvas  # 캔버스 대비 두께 비율
        # 건전 범위 중심(0.06)에서 벗어난 정도
        thickness_penalty = abs(t - (THICKNESS_LO + THICKNESS_HI) / 2)
        if t < THICKNESS_LO:
            thickness_penalty += (THICKNESS_LO - t) * 8   # 끊김 위험 가중 벌점
        if t > THICKNESS_HI:
            thickness_penalty += (t - THICKNESS_HI) * 8   # 블롭 가중 벌점
        frag, n_cc = _fragmentation(cleaned)
        score = thickness_penalty + frag * 0.8
        if best is None or score < best[0]:
            best = (score, cleaned, t, frag, n_cc)
    return best  # (penalty, mask, thickness_ratio, fragmentation, n_cc)


def _crop_pad_resize(ink_mask):
    """잉크 bbox 크롭 -> 패딩 정방형 -> 128px, 흰 배경/검은 잉크로 반환."""
    ys, xs = np.where(ink_mask > 0)
    if len(xs) == 0:
        return None
    ink = ink_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    pad = int(max(h, w) * PAD_RATIO)
    dim = max(h, w) + pad * 2
    canvas = np.zeros((dim, dim), dtype=np.uint8)
    canvas[(dim - h) // 2:(dim - h) // 2 + h, (dim - w) // 2:(dim - w) // 2 + w] = ink

    resized = cv2.resize(canvas, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    _, final = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(final)


def group_char_files(original_dir):
    files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    groups = {}
    for f in files:
        name = os.path.splitext(f)[0]
        if re.match(r'^\d', name):   # '1(매실효소)' 같은 전체 스캔 제외
            continue
        label = re.sub(r'[\d\s]+', '', name)
        if not label or label == 'ㅏ':  # 단독 모음은 완성형 힌트가 아님
            continue
        groups.setdefault(label, []).append(os.path.join(original_dir, f))
    return groups


def refine_all(original_dir=ORIGINAL_DIR, style_dir=STYLE_DIR):
    os.makedirs(style_dir, exist_ok=True)
    groups = group_char_files(original_dir)
    print(f"[힌트 재구축 v2] 원본 고유 글자: {sorted(groups.keys())}")

    saved = {}
    for char, paths in sorted(groups.items()):
        target_cc = expected_components(char)
        candidates = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            result = extract_glyph(img, target_cc=target_cc)
            if result is None:
                continue
            penalty, mask, t, frag, n_cc = result
            candidates.append((penalty, p, mask, t, frag, n_cc))

        if not candidates:
            print(f"   [경고] '{char}' 유효 후보 없음")
            continue
        candidates.sort(key=lambda c: c[0])
        penalty, best_path, mask, t, frag, n_cc = candidates[0]

        final_img = _crop_pad_resize(mask)
        if final_img is None:
            print(f"   [경고] '{char}' 잉크 없음 ({best_path})")
            continue

        hex_name = f"{ord(char):04X}"
        cv2.imwrite(os.path.join(style_dir, f"{hex_name}.png"), final_img)
        saved[char] = best_path
        print(f"   -> '{char}' <- {os.path.basename(best_path)} "
              f"(벌점 {penalty:.3f}, 두께비 {t:.3f}, 성분 {n_cc}개/기대 {target_cc}개, "
              f"후보 {len(candidates)}개 중 최선)")

    print(f"[완료] {len(saved)}개 힌트 생성: {sorted(saved.keys())}")
    return saved


if __name__ == "__main__":
    refine_all()
