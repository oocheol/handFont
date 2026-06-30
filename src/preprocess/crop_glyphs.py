import os
import cv2
import numpy as np

def clean_and_extract_glyphs(image_path, output_dir, file_index):
    """
    OpenCV를 사용하여 이미지에서 테두리선과 점 노이즈를 지우고
    순수 글씨만 탐지하여 128x128 크기로 저장합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 이미지 로드 및 그레이스케일 변환
    img = cv2.imread(image_path)
    if img is None:
        print(f"[오류] 이미지를 로드할 수 없습니다: {image_path}")
        return False
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 2. 이진화 (Otsu's Thresholding)
    # 글씨가 흰색(255), 배경이 검은색(0)이 되도록 반전(INV)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 연결 성분 분석 (Connected Components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    # 유효한 글씨 글리프만 그릴 마스크 이미지 생성
    clean_mask = np.zeros_like(thresh)

    # 이미지 경계선 근처(테두리 방지 마진)
    border_margin = 8

    glyphs_info = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # A. 너무 작은 노이즈 점 제거 (면적이 100픽셀 이하인 경우)
        if area < 100:
            continue

        # B. 너무 거대하거나 이미지 거의 전체를 덮는 테두리 제거
        if width > w * 0.95 or height > h * 0.95:
            continue

        # C. 외곽 경계선에 닿아 있는 성분 제거 (상자 테두리선 필터링)
        if x <= border_margin or y <= border_margin or (x + width) >= (w - border_margin) or (y + height) >= (h - border_margin):
            # 테두리선일 확률이 높으므로 스킵
            continue

        # D. 가로 또는 세로로 너무 길고 얇은 선 제거 (밑줄 등)
        aspect_ratio = width / float(height)
        if (aspect_ratio > 10.0 and height < 20) or (aspect_ratio < 0.1 and width < 20):
            continue

        # 통과된 성분은 유효한 글씨 후보로 마스크에 추가
        clean_mask[labels == i] = 255
        glyphs_info.append((x, y, width, height))

    # 4. 정제된 마스크에서 글자 영역 탐지
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가로축(X) 좌표 기준으로 정렬하여 읽는 순서대로 글자를 배열
    contours_with_box = []
    for cnt in contours:
        x, y, width, height = cv2.boundingRect(cnt)
        if width * height > 150: # 최소 크기 한 번 더 확인
            contours_with_box.append((x, y, width, height))
            
    contours_with_box = sorted(contours_with_box, key=lambda b: b[0])

    saved_count = 0
    for idx, (x, y, width, height) in enumerate(contours_with_box):
        # 글자 주변에 여백(Padding) 추가
        pad = int(max(width, height) * 0.15)
        
        # 바운딩 박스 좌표 계산
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + width + pad)
        y2 = min(h, y + height + pad)
        
        # 글자 이미지 추출
        glyph_crop = clean_mask[y1:y2, x1:x2]
        
        # 정방형 만들기
        gh, gw = glyph_crop.shape
        max_dim = max(gh, gw)
        square_glyph = np.zeros((max_dim, max_dim), dtype=np.uint8)
        
        # 중앙 배치
        dy = (max_dim - gh) // 2
        dx = (max_dim - gw) // 2
        square_glyph[dy:dy+gh, dx:dx+gw] = glyph_crop

        # 최종 128x128 리사이즈 및 색상 반전 (배경 흰색 255, 글자 검은색 0)
        resized_glyph = cv2.resize(square_glyph, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        final_glyph = cv2.bitwise_not(resized_glyph)

        output_name = f"style_{file_index}_{idx}.png"
        cv2.imwrite(os.path.join(output_dir, output_name), final_glyph)
        saved_count += 1

    print(f"[성공] 이미지 {file_index} 완료: {saved_count}개 글씨 추출됨.")
    return True

def run_pipeline(raw_dir, style_dir):
    if not os.path.exists(raw_dir):
        print(f"[정보] 스캔 폴더 {raw_dir} 가 존재하지 않아 생성합니다.")
        os.makedirs(raw_dir)
        return
        
    raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not raw_files:
        print("[경고] 전처리할 원본 스캔 이미지 파일이 data/raw/ 폴더에 없습니다.")
        return

    for idx, filename in enumerate(raw_files):
        img_path = os.path.join(raw_dir, filename)
        clean_and_extract_glyphs(img_path, style_dir, idx)

if __name__ == "__main__":
    # 실행 경로
    run_pipeline("data/raw", "data/style")
