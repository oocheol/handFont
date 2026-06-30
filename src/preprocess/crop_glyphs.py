import os
from PIL import Image, ImageOps, ImageFilter

def preprocess_and_crop(image_path, output_dir, grid_size=(1, 1)):
    """
    스캔 이미지에서 글자를 분할하여 잘라내는 기본 전처리 함수.
    grid_size: (rows, cols) 형태의 격자 정보. 격자가 명확하다면 격자 분할을 시도하고,
               그렇지 않으면 윤곽선(Bounding Box) 검출을 시도합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    img = Image.open(image_path).convert('L')
    
    # 이진화 (Thresholding)
    # 배경을 흰색(255), 글씨를 검은색(0)으로 변환
    threshold = 127
    img_bin = img.point(lambda p: 255 if p > threshold else 0)
    
    # 반전 (배경 0, 글씨 255) - 경계 탐색에 편리
    img_inv = ImageOps.invert(img_bin)
    
    width, height = img.size
    rows, cols = grid_size
    
    # 격자형 분할이 지정된 경우
    if rows > 1 or cols > 1:
        cell_w = width // cols
        cell_h = height // rows
        count = 0
        for r in range(rows):
            for c in range(cols):
                left = c * cell_w
                top = r * cell_h
                right = left + cell_w
                bottom = top + cell_h
                
                # 해당 셀 크롭
                crop_img = img_bin.crop((left, top, right, bottom))
                
                # 여백 제거 (Auto-crop)
                crop_inv = ImageOps.invert(crop_img)
                bbox = crop_inv.getbbox()
                if bbox:
                    # 마진을 약간 둠
                    w_box, h_box = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    margin = int(max(w_box, h_box) * 0.1)
                    
                    left_m = max(0, bbox[0] - margin)
                    top_m = max(0, bbox[1] - margin)
                    right_m = min(cell_w, bbox[2] + margin)
                    bottom_m = min(cell_h, bbox[3] + margin)
                    
                    final_crop = crop_img.crop((left_m, top_m, right_m, bottom_m))
                else:
                    final_crop = crop_img
                
                # 128x128 크기로 정규화하여 저장
                final_crop = final_crop.resize((128, 128), Image.Resampling.LANCZOS)
                
                output_filename = f"char_{r}_{c}.png"
                final_crop.save(os.path.join(output_dir, output_filename))
                count += 1
        print(f"Successfully cropped {count} characters to {output_dir}")
    else:
        # 단일 이미지인 경우 자동 크롭 및 128x128 리사이즈만 수행
        bbox = img_inv.getbbox()
        if bbox:
            final_crop = img_bin.crop(bbox)
        else:
            final_crop = img_bin
            
        final_crop = final_crop.resize((128, 128), Image.Resampling.LANCZOS)
        name = os.path.splitext(os.path.basename(image_path))[0]
        final_crop.save(os.path.join(output_dir, f"{name}_processed.png"))
        print(f"Processed single image and saved to {output_dir}")

if __name__ == "__main__":
    # 임시 테스트용 진입점
    print("Preprocess module loaded.")
