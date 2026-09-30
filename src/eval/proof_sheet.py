"""
생성된 글리프 품질 평가 하네스.

두 가지 산출물을 만든다:
1. 프루프시트: 표본 글자들을 격자로 나열한 PNG (육안 검수용)
2. 결함 리포트: 시스템 폰트(맑은고딕) 템플릿과 비교해 아래 신호로
   "의심 글리프"를 자동 탐지한 CSV/JSON
   - ink_ratio: 잉크 픽셀 비율이 템플릿 대비 너무 적음(획 소실) / 너무 많음(블롭·노이즈)
   - fragmentation: 연결요소(Connected Component) 수가 템플릿보다 훨씬 많음
     (획 끊김 또는 잔여 노이즈 점을 뜻함)

이 스크립트 자체는 파이프라인을 바꾸지 않는다. 매 개선 단계 전후로 실행해
"의심 글리프 수"를 숫자로 비교하기 위한 도구다.
"""
import os
import json
import csv
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FONT_CANDIDATES = [
    "C:\\Windows\\Fonts\\malgun.ttf",
    "C:\\Windows\\Fonts\\batang.ttc",
    "C:\\Windows\\Fonts\\gulim.ttc",
]

# 눈으로 훑기 좋은 대표 표본: 자주 쓰이는 음절 + 모음 그룹 커버 + 받침 유무 커버
SAMPLE_CHARS = list("가나다라마바사아자차카타파하"
                     "고노도로모보소오조초코토포호"
                     "구누두루무부수우주추쿠투푸후"
                     "그느드르므브스으즈츠크트프흐"
                     "기니디리미비시이지치키티피히"
                     "너덜멈넘든능당많너널넣떻렇겠"
                     "강경공관광국권근금기끝나난")


def _template_stats(char, font, size=384):
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    try:
        l, t, r, b = draw.textbbox((0, 0), char, font=font)
        w, h = r - l, b - t
        x = (size - w) // 2 - l
        y = (size - h) // 2 - t
        draw.text((x, y), char, fill=0, font=font)
    except Exception:
        draw.text((size // 4, size // 4), char, fill=0, font=font)
    arr = np.array(img)
    return _ink_stats(arr)


def _ink_stats(gray_arr):
    """흰 배경(255)/검은 잉크(0) 그레이스케일 배열에서 통계 산출."""
    ink_mask = (gray_arr < 128).astype(np.uint8)
    total = gray_arr.size
    ink_pixels = int(ink_mask.sum())
    ink_ratio = ink_pixels / total

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
    num_components = max(0, num_labels - 1)  # 배경 라벨 제외

    largest = 0
    if num_components > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(areas.max())
    fragmentation = 1.0 - (largest / ink_pixels) if ink_pixels > 0 else 1.0

    return {
        "ink_ratio": ink_ratio,
        "num_components": num_components,
        "fragmentation": fragmentation,
    }


def analyze_images(images_dir, chars=None, font_path=None):
    """output/images의 PNG들을 시스템 폰트 템플릿과 비교해 결함 신호 산출."""
    if font_path is None:
        font_path = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
    if font_path is None:
        raise RuntimeError("시스템 한글 폰트를 찾을 수 없습니다.")
    font = ImageFont.truetype(font_path, 300)

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith('.png'))
    if chars is not None:
        want_hex = {f"{ord(c):04X}" for c in chars}
        files = [f for f in files if os.path.splitext(f)[0].upper() in want_hex]

    results = []
    template_cache = {}

    for fname in files:
        hex_name = os.path.splitext(fname)[0]
        try:
            char = chr(int(hex_name, 16))
        except ValueError:
            continue

        gen_img = cv2.imread(os.path.join(images_dir, fname), cv2.IMREAD_GRAYSCALE)
        if gen_img is None:
            continue
        gen_stats = _ink_stats(gen_img)

        if char not in template_cache:
            template_cache[char] = _template_stats(char, font)
        tmpl_stats = template_cache[char]

        suspect_reasons = []
        if tmpl_stats["ink_ratio"] > 0:
            ratio = gen_stats["ink_ratio"] / tmpl_stats["ink_ratio"]
            if ratio < 0.3:
                suspect_reasons.append("ink_too_low(stroke_loss)")
            elif ratio > 2.5:
                suspect_reasons.append("ink_too_high(blob_or_noise)")
        elif gen_stats["ink_ratio"] > 0:
            suspect_reasons.append("ink_present_but_template_empty")

        if gen_stats["num_components"] == 0:
            suspect_reasons.append("blank_glyph")
        elif gen_stats["num_components"] > tmpl_stats["num_components"] + 2:
            suspect_reasons.append("fragmentation_or_noise")
        # (2026-07 제거) "성분 수 < 템플릿"을 결함으로 보던 규칙 삭제.
        # 인쇄체 템플릿은 자모가 항상 분리되어 있지만, 커서브 손글씨는
        # 자연스럽게 이어써서 성분이 합쳐지는 게 정상 특성이다(예: '각'이
        # 인쇄체 3조각(ㄱㅏㄱ) vs 손글씨 2조각은 결함이 아니라 손글씨다움).
        # 실제 "획 소실"은 component_check.missing_regions(템플릿 잉크
        # 영역 기준 격자 비교)가 더 정확히 잡아낸다.

        if (gen_stats["fragmentation"] > 0.5
                and gen_stats["num_components"] > tmpl_stats["num_components"]):
            # 성분 수가 템플릿보다 많으면서 파편화도 심한 경우만 "끊김"으로 간주.
            # 성분 수가 템플릿과 같으면 자모가 원래 여러 조각인 정상 구조일 뿐임
            # (예: '강'=ㄱ+ㅏ+ㅇ 3조각은 정상, 끊긴 게 아님).
            suspect_reasons.append("main_stroke_broken")

        results.append({
            "char": char,
            "hex": hex_name,
            "gen_ink_ratio": round(gen_stats["ink_ratio"], 4),
            "tmpl_ink_ratio": round(tmpl_stats["ink_ratio"], 4),
            "gen_components": gen_stats["num_components"],
            "tmpl_components": tmpl_stats["num_components"],
            "fragmentation": round(gen_stats["fragmentation"], 3),
            "suspect": len(suspect_reasons) > 0,
            "reasons": ";".join(suspect_reasons),
        })

    return results


def write_report(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "defect_report.csv")
    json_path = os.path.join(out_dir, "defect_report.json")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    n_total = len(results)
    n_suspect = sum(1 for r in results if r["suspect"])
    summary = {
        "total_glyphs_checked": n_total,
        "suspect_count": n_suspect,
        "suspect_ratio": round(n_suspect / n_total, 4) if n_total else 0,
        "suspects": [r["char"] for r in results if r["suspect"]],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, csv_path, json_path


def make_proof_sheet(images_dir, chars, out_path, cols=16, cell=64):
    files = []
    for c in chars:
        hex_name = f"{ord(c):04X}"
        p = os.path.join(images_dir, f"{hex_name}.png")
        if os.path.exists(p):
            files.append((c, p))

    if not files:
        print("[경고] 프루프시트에 넣을 이미지가 없습니다.")
        return None

    rows = (len(files) + cols - 1) // cols
    sheet = Image.new('L', (cols * cell, rows * cell), color=255)

    for idx, (c, p) in enumerate(files):
        img = Image.open(p).convert('L').resize((cell - 4, cell - 4), Image.LANCZOS)
        x = (idx % cols) * cell + 2
        y = (idx // cols) * cell + 2
        sheet.paste(img, (x, y))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="글리프 품질 평가 하네스")
    parser.add_argument("--images-dir", default=os.path.join(REPO_ROOT, "output", "images"))
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "output", "eval"))
    parser.add_argument("--sample-only", action="store_true",
                         help="전체 대신 SAMPLE_CHARS 표본만 분석 (빠른 A/B 비교용)")
    args = parser.parse_args()

    chars = SAMPLE_CHARS if args.sample_only else None
    results = analyze_images(args.images_dir, chars=chars)

    if not results:
        print("[오류] 분석할 이미지가 없습니다.")
        return

    summary, csv_path, json_path = write_report(results, args.out_dir)
    proof_path = make_proof_sheet(args.images_dir, SAMPLE_CHARS,
                                   os.path.join(args.out_dir, "proof_sample.png"))

    print(f"[완료] 검사 글리프 수: {summary['total_glyphs_checked']}")
    print(f"[완료] 의심 글리프 수: {summary['suspect_count']} ({summary['suspect_ratio']*100:.1f}%)")
    print(f"[완료] CSV: {csv_path}")
    print(f"[완료] JSON: {json_path}")
    if proof_path:
        print(f"[완료] 프루프시트: {proof_path}")


if __name__ == "__main__":
    main()
