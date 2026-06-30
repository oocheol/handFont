import os
import cv2
import numpy as np

def png_to_svg_pure_python(png_path, svg_path):
    """
    외부 프로그램(potrace) 의존성 없이, OpenCV의 외곽선 탐지 기능을 활용하여
    PNG 이미지를 완벽한 벡터 SVG 파일로 변환합니다.
    """
    try:
        # 1. 이미지 로드 및 그레이스케일 변환
        img = cv2.imread(png_path)
        if img is None:
            return False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. 이진화 (글씨가 255(흰색), 배경이 0(검은색)이 되도록 처리)
        # OpenCV findContours는 흰색 성분의 외곽선을 탐색하므로 글자가 흰색이어야 합니다.
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 3. 외곽선 찾기
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        
        # 4. SVG 패스 데이터 생성
        svg_paths = []
        for cnt in contours:
            if len(cnt) < 3:
                continue
                
            path_data = []
            for i, pt in enumerate(cnt):
                x, y = pt[0][0], pt[0][1]
                if i == 0:
                    path_data.append(f"M {x} {y}")
                else:
                    path_data.append(f"L {x} {y}")
            path_data.append("Z")
            svg_paths.append(" ".join(path_data))
            
        # 5. SVG 파일 쓰기
        h, w = gray.shape
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(f'<?xml version="1.0" encoding="utf-8"?>\n')
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n')
            for path in svg_paths:
                f.write(f'  <path d="{path}" fill="black" stroke="none" />\n')
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
    # 배치 실행 테스트
    batch_vectorize("output/images", "output/svgs")
