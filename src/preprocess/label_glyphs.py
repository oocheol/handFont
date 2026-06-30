import os
import cv2
import numpy as np
import re

# 각 파일(순서대로 정렬됨)에 대응하는 텍스트 목록 정의 (공백 제외)
TARGET_TEXTS = [
    "매실효소",         # 이미지 0
    "이새별이새별",     # 이미지 1 (공백 제거)
    "매실",             # 이미지 2
    "조영화",           # 이미지 3
    "조영화",           # 이미지 4
    "매실"              # 이미지 5
]

def label_raw_images(raw_dir, output_style_dir):
    if not os.path.exists(output_style_dir):
        os.makedirs(output_style_dir)

    # 1. raw 디렉토리 내의 이미지들을 파일명 안의 숫자 크기 순으로 정렬하여 정확한 인덱스 매칭 보장
    raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    def extract_number(filename):
        nums = re.findall(r'\d+', filename)
        return int(nums[0]) if nums else 0

    raw_files = sorted(raw_files, key=extract_number)
    
    print(f"정렬된 이미지 리스트: {raw_files}")

    for idx, filename in enumerate(raw_files):
        if idx >= len(TARGET_TEXTS):
            break
            
        img_path = os.path.join(raw_dir, filename)
        target_text = TARGET_TEXTS[idx]
        num_chars_needed = len(target_text)
        
        print(f"\n[라벨링 시작] 파일: {filename} -> 목표 텍스트: {target_text} ({num_chars_needed}글자)")
        
        # 이미지 전처리
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 이진화
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 가로선 및 세로선(테두리선) 감지 및 제거하여 글자 뭉침 방지
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
        detect_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        detect_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
        thresh = cv2.subtract(thresh, detect_h)
        thresh = cv2.subtract(thresh, detect_v)
        
        # 연결 성분
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        
        # 유효 성분 수집
        border_margin = 8
        candidates = []
        
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # 노이즈 필터링
            if area < 120 or width > w * 0.95 or height > h * 0.95:
                continue
                
            # 이미지 가로/세로 외곽 경계선 근처(5% 이내)에 있는 작은 찌꺼기 성분 제거
            if (x < w * 0.08 or (x + width) > w * 0.82) and area < 1500:
                continue
                
            # C. 외곽 경계선에 닿아 있는 성분 제거 (상자 테두리선 필터링)
            is_touching_border = (x <= border_margin or y <= border_margin or 
                                  (x + width) >= (w - border_margin) or 
                                  (y + height) >= (h - border_margin))
            if is_touching_border:
                # 테두리선처럼 아주 길고 얇거나 면적이 극단적으로 크면 지움
                aspect_ratio = width / float(height)
                if aspect_ratio > 3.0 or aspect_ratio < 0.33 or area > (w * h * 0.15):
                    continue
            
            candidates.append((x, y, width, height, i))
            
        # 목표 글자 수에 맞추기 위해 면적(Area)이 큰 상위 N개 후보군을 먼저 정렬
        # 면적이 너무 작은 쓰레기 값을 쳐내기 위해 크기순 정렬
        candidates = sorted(candidates, key=lambda c: stats[c[4], cv2.CC_STAT_AREA], reverse=True)
        
        # 필요한 글자 수만큼 상위 면적 객체만 선택
        valid_candidates = candidates[:num_chars_needed]
        
        # 다시 읽는 순서인 가로 좌표(x) 기준으로 최종 정렬
        valid_candidates = sorted(valid_candidates, key=lambda c: c[0])
        
        # 글자 수 매칭 실패 검사
        if len(valid_candidates) < num_chars_needed:
            print(f"[경고] 탐지된 유효 글자 수가 부족합니다 (검출: {len(valid_candidates)} / 필요: {num_chars_needed}). 임계치를 조정해 전부 가져옵니다.")
            # 필터 조건을 강제로 낮춤
            valid_candidates = sorted(candidates, key=lambda c: c[0])[:num_chars_needed]
            
        # 라벨 매핑 및 저장
        for char_idx, (x, y, width, height, label_idx) in enumerate(valid_candidates):
            if char_idx >= len(target_text):
                break
                
            char = target_text[char_idx]
            
            # 글자 영역 마스크 추출
            glyph_mask = np.zeros_like(thresh)
            glyph_mask[labels == label_idx] = 255
            
            # 바운딩 박스 크롭
            pad = int(max(width, height) * 0.15)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + width + pad)
            y2 = min(h, y + height + pad)
            
            glyph_crop = glyph_mask[y1:y2, x1:x2]
            
            # 정방형화
            gh, gw = glyph_crop.shape
            max_dim = max(gh, gw)
            square_glyph = np.zeros((max_dim, max_dim), dtype=np.uint8)
            dy = (max_dim - gh) // 2
            dx = (max_dim - gw) // 2
            square_glyph[dy:dy+gh, dx:dx+gw] = glyph_crop
            
            # 리사이즈 및 반전
            resized = cv2.resize(square_glyph, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            final_img = cv2.bitwise_not(resized)
            
            # 유니코드 16진수 문자열로 파일명 지정 (AI 모델 템플릿 대응 및 특수문자 방지)
            unicode_hex = f"{ord(char):04X}"
            output_name = f"{unicode_hex}.png" # 예: '매' -> 'B9E4.png'
            
            out_path = os.path.join(output_style_dir, output_name)
            cv2.imwrite(out_path, final_img)
            print(f" -> 글자 '{char}' 저장 완료: {output_name}")

if __name__ == "__main__":
    label_raw_images("data/raw", "data/style")
