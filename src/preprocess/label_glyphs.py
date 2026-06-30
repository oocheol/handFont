import os
import cv2
import numpy as np
import re
import easyocr

# 파일 이름의 번호를 기준으로 매핑할 타겟 텍스트 (단어 목록으로 분리)
TARGET_TEXTS_MAP = {
    1: ["매실효소"],
    2: ["이새별", "이새별"],  # 두 단어
    3: ["매실"],
    4: ["조영화"],
    5: ["조영화"],
    6: ["매실"]
}

def label_raw_images_with_craft(raw_dir, output_style_dir):
    if not os.path.exists(output_style_dir):
        os.makedirs(output_style_dir)

    print("[정보] CRAFT 텍스트 탐지기 (EasyOCR)를 로드하는 중...")
    reader = easyocr.Reader(['ko'], gpu=False)
    
    raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    def extract_number(filename):
        nums = re.findall(r'\d+', filename)
        return int(nums[0]) if nums else 0

    raw_files = sorted(raw_files, key=extract_number)
    print(f"정렬된 원본 파일 목록: {raw_files}")

    for filename in raw_files:
        file_num = extract_number(filename)
        if file_num not in TARGET_TEXTS_MAP:
            continue
            
        target_words = TARGET_TEXTS_MAP[file_num]
        img_path = os.path.join(raw_dir, filename)
        
        print(f"\n[CRAFT 전처리 시작] {filename} -> 목표: '{' '.join(target_words)}'")
        
        # 3. 이미지 로드 및 전처리
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 가로선 및 세로선 제거
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
        detect_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        detect_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
        thresh = cv2.subtract(thresh, detect_h)
        thresh = cv2.subtract(thresh, detect_v)

        # 2. EasyOCR (CRAFT)의 detect()를 사용하여 텍스트 박스 좌표만 가져옴 (OCR 텍스트 인식 실패 방지)
        # detect() 반환: (horizontal_list, free_list)
        horizontal_list, free_list = reader.detect(img_path)
        
        raw_boxes = []
        # horizontal_list 파싱
        for box in horizontal_list:
            if not box:
                continue
            if len(box) == 4 and not isinstance(box[0], (list, np.ndarray)):
                raw_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            elif len(box) == 4:
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                raw_boxes.append((int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))))
                
        # free_list 파싱
        for box in free_list:
            if not box or len(box) < 4:
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            raw_boxes.append((int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))))
        
        # 찌꺼기 테두리 상자 방지를 위해 노이즈 박스 필터링 (높이가 너무 작거나 외곽인 것 제거)
        valid_boxes = []
        for xmin, xmax, ymin, ymax in raw_boxes:
            box_w = xmax - xmin
            box_h = ymax - ymin
            
            # 너무 극단적인 크기 제거
            if box_h < 20 or box_w < 20 or box_h > h * 0.9 or box_w > w * 0.9:
                continue
            # 외곽 찌꺼기 제거
            if (xmin < w * 0.05 or xmax > w * 0.95) and box_h < 40:
                continue
                
            valid_boxes.append((xmin, xmax, ymin, ymax))
            
        # xmin 기준 가로 정렬
        valid_boxes = sorted(valid_boxes, key=lambda b: b[0])
        
        # 만약 탐지된 박스가 전혀 없다면 이미지 중앙 영역을 통째로 하나의 큰 박스로 지정하여 예외 처리
        if not valid_boxes:
            print("[경고] CRAFT가 박스를 찾지 못했습니다. 기하학적 대체 박스를 생성합니다.")
            # 글자가 존재할 확률이 높은 중앙 영역을 바운딩 박스로 지정
            # Connected Components의 중심 영역 탐색
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
            valid_stats = []
            for i in range(1, num_labels):
                s = stats[i]
                area = s[cv2.CC_STAT_AREA]
                left = s[cv2.CC_STAT_LEFT]
                width = s[cv2.CC_STAT_WIDTH]
                height = s[cv2.CC_STAT_HEIGHT]
                
                # 최소 면적 기준
                if area < 300:
                    continue
                # 글자는 일정한 세로 높이를 지니므로 가로 테두리선(두께가 얇음) 제거
                if height < 50:
                    continue
                # 이미지 좌우 끝단(10% 및 80%)에 걸쳐 있고 면적이 크지 않은 테두리 성분 제거
                if (left < w * 0.1 or (left + width) > w * 0.8) and area < 1500:
                    continue
                valid_stats.append(s)
                
            if valid_stats:
                xmin = min(s[cv2.CC_STAT_LEFT] for s in valid_stats)
                xmax = max(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH] for s in valid_stats)
                ymin = min(s[cv2.CC_STAT_TOP] for s in valid_stats)
                ymax = max(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT] for s in valid_stats)
                valid_boxes = [(xmin, xmax, ymin, ymax)]
            else:
                valid_boxes = [(int(w*0.1), int(w*0.9), int(h*0.1), int(h*0.9))]

        print(f" -> 필터링 후 유효 텍스트 영역 수: {len(valid_boxes)}개 (목표 단어 수: {len(target_words)}개)")

        # 단어 수와 박스 수가 맞지 않을 때 병합 또는 분배
        # 간단한 매칭: 단어 수에 맞게 valid_boxes를 조합하거나 쪼갬
        if len(valid_boxes) < len(target_words):
            # 박스 수가 단어 수보다 작을 경우, 가장 큰 박스를 쪼개거나 
            # 단순히 전체 탐지 영역을 단어 글자 수의 누적 비율로 나눕니다.
            # 여기서는 편의를 위해 박스들을 하나의 거대한 영역으로 묶은 후 글자 단위 분할을 적용
            all_xmin = min(b[0] for b in valid_boxes)
            all_xmax = max(b[1] for b in valid_boxes)
            all_ymin = min(b[2] for b in valid_boxes)
            all_ymax = max(b[3] for b in valid_boxes)
            
            # 단어들을 하나의 문자열로 합쳐서 등분 분할 처리
            combined_text = "".join(target_words)
            word_len = len(combined_text)
            char_w = (all_xmax - all_xmin) / word_len
            
            for idx, target_char in enumerate(combined_text):
                cx1 = int(all_xmin + idx * char_w)
                cx2 = int(all_xmin + (idx + 1) * char_w)
                save_char_glyph(thresh, cx1, cx2, all_ymin, all_ymax, target_char, output_style_dir)
                
        else:
            # 박스 수가 충분하거나 단어 수와 같을 때, 각 단어 박스를 해당 단어 글자 수만큼 등분
            for w_idx, target_word in enumerate(target_words):
                if w_idx >= len(valid_boxes):
                    break
                xmin, xmax, ymin, ymax = valid_boxes[w_idx]
                word_len = len(target_word)
                char_w = (xmax - xmin) / word_len
                
                for idx, target_char in enumerate(target_word):
                    cx1 = int(xmin + idx * char_w)
                    cx2 = int(xmin + (idx + 1) * char_w)
                    save_char_glyph(thresh, cx1, cx2, ymin, ymax, target_char, output_style_dir)

    print("\n[성공] CRAFT 기반 10글자 고품질 전처리 완료!")

def save_char_glyph(thresh, x1, x2, y1, y2, target_char, output_style_dir):
    h, w = thresh.shape
    pad = int(max(x2-x1, y2-y1) * 0.15)
    
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    
    glyph_crop = thresh[cy1:cy2, cx1:cx2]
    
    # 노이즈를 지우고 순수 자소만 남김
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(glyph_crop)
    cleaned_glyph = np.zeros_like(glyph_crop)
    if num_labels > 1:
        for l in range(1, num_labels):
            area = stats[l, cv2.CC_STAT_AREA]
            if area > 15:
                cleaned_glyph[labels == l] = 255
    else:
        cleaned_glyph = glyph_crop
        
    gh, gw = cleaned_glyph.shape
    max_dim = max(gh, gw, 1)
    square_glyph = np.zeros((max_dim, max_dim), dtype=np.uint8)
    
    dy = (max_dim - gh) // 2
    dx = (max_dim - gw) // 2
    square_glyph[dy:dy+gh, dx:dx+gw] = cleaned_glyph
    
    resized = cv2.resize(square_glyph, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    final_img = cv2.bitwise_not(resized)
    
    unicode_hex = f"{ord(target_char):04X}"
    output_name = f"{unicode_hex}.png"
    out_path = os.path.join(output_style_dir, output_name)
    
    cv2.imwrite(out_path, final_img)
    print(f"   -> 글자 '{target_char}' 저장 완료: {output_name}")

if __name__ == "__main__":
    label_raw_images_with_craft("data/raw", "data/style")
