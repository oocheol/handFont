import os
import cv2
import numpy as np
import re

def reprocess_user_labeled_images(raw_dir="data/raw", output_style_dir="data/style"):
    """
    사용자가 직접 자른 data/raw/*.JPG 이미지들을 읽어서
    파일명의 한글 자소를 유니코드 파일명으로 맵핑하고,
    적응형 이진화와 경계선 스무딩을 가해 data/style/ 에 저장합니다.
    """
    if not os.path.exists(output_style_dir):
        os.makedirs(output_style_dir)

    raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not raw_files:
        print("[오류] data/raw 폴더에 원본 손글씨 파일이 없습니다.")
        return False

    print(f"사용자 라벨링된 원본 파일 목록 발견: {len(raw_files)}개")

    # 글자별 이미지 그룹화 (동일 자소 중복 처리용)
    char_groups = {}
    for filename in raw_files:
        name_part = os.path.splitext(filename)[0]
        # 숫자 및 공백 제거하여 순수 한글 자소만 획득 (예: '별1' -> '별')
        char_label = re.sub(r'[\d\s]+', '', name_part)
        if not char_label:
            continue
            
        if char_label not in char_groups:
            char_groups[char_label] = []
        char_groups[char_label].append(os.path.join(raw_dir, filename))

    print(f"-> 파싱된 고유 한글 자소 수: {len(char_groups)}개 ({list(char_groups.keys())})")

    for char_label, paths in char_groups.items():
        # 'ㅏ' 모음만 단독으로 들어온 경우 완성형 '아' 로 매핑하여 모델에 힌트 제공
        target_char = '아' if char_label == 'ㅏ' else char_label
        unicode_hex = f"{ord(target_char):04X}"
        output_name = f"{unicode_hex}.png"
        out_path = os.path.join(output_style_dir, output_name)
        
        # 동일 자소의 스캔본이 여러 개일 경우, 이들의 이진화 마스크를 OR 병합하여
        # 혹시 모를 획 유실(얇아짐)을 원천 방지하고 가장 가득 찬 획을 합성해냅니다.
        merged_mask = None
        
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 적응형 이진화 (조명 불균형 완벽 방어)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 81, 13
            )
            
            # 획 구멍 메우기 (MORPH_CLOSE)
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
            
            # 외곽 테두리 찌꺼기 방지 (여백 근처의 극소 노이즈 제거)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
            cleaned = np.zeros_like(thresh)
            if num_labels > 1:
                for l in range(1, num_labels):
                    area = stats[l, cv2.CC_STAT_AREA]
                    # 자소 성분 보존을 위해 노이즈 필터링 완화 (area > 3)
                    if area > 3:
                        cleaned[labels == l] = 255
            else:
                cleaned = thresh
                
            if merged_mask is None:
                merged_mask = cleaned
            else:
                # 크기가 다를 경우 리사이즈 후 병합
                if cleaned.shape != merged_mask.shape:
                    cleaned = cv2.resize(cleaned, (merged_mask.shape[1], merged_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                # 픽셀 OR 연산으로 획 병합 (유실 차단)
                merged_mask = cv2.bitwise_or(merged_mask, cleaned)
                
        if merged_mask is None:
            continue

        # 정방형(Square)으로 캔버스 확장
        gh, gw = merged_mask.shape
        max_dim = max(gh, gw, 1)
        square_glyph = np.zeros((max_dim, max_dim), dtype=np.uint8)
        
        dy = (max_dim - gh) // 2
        dx = (max_dim - gw) // 2
        square_glyph[dy:dy+gh, dx:dx+gw] = merged_mask
        
        # 128x128 크기로 LANCZOS 스케일링
        resized = cv2.resize(square_glyph, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        
        # 가우시안 앤티앨리어싱 스무딩으로 벡터 엣지 미학 극대화
        smoothed = cv2.GaussianBlur(resized, (3, 3), 0)
        _, final_thresh = cv2.threshold(smoothed, 100, 255, cv2.THRESH_BINARY)
        
        # 색상 반전 (배경 흰색, 글씨 검은색)
        final_img = cv2.bitwise_not(final_thresh)
        
        cv2.imwrite(out_path, final_img)
        print(f"   -> 글자 '{char_label}' 정밀 추출 및 저장 완료: {output_name} (스캔본 {len(paths)}개 병합)")



    print("[성공] 사용자 직접 크롭본 기반 10글자 고도화 전처리 완료!")
    return True

if __name__ == "__main__":
    reprocess_user_labeled_images()
