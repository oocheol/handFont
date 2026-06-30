import os
import subprocess
from PIL import Image

def png_to_svg(png_path, svg_path):
    """
    PNG 이미지를 Potrace를 사용하여 SVG 벡터 이미지로 변환합니다.
    Potrace는 BMP 형식만 입력으로 받으므로 내부적으로 변환 과정을 거칩니다.
    """
    # 1. 임시 BMP 파일명 정의
    bmp_path = png_path.replace('.png', '_temp.bmp')
    
    try:
        # 2. PNG를 흑백 1비트 BMP로 변환 및 저장
        img = Image.open(png_path).convert('1')
        img.save(bmp_path)
        
        # 3. Potrace CLI 호출 (potrace가 환경 변수에 있거나 실행 경로에 존재해야 함)
        # -s: SVG 출력 포맷 지정
        subprocess.run(['potrace', '-s', '-o', svg_path, bmp_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        print("[오류] 시스템에서 'potrace' 명령을 찾을 수 없습니다. Potrace가 설치되어 있고 PATH 환경 변수에 등록되어 있는지 확인하십시오.")
        return False
    except Exception as e:
        print(f"[오류] 변환 실패 ({png_path}): {e}")
        return False
    finally:
        # 임시 BMP 파일 정리
        if os.path.exists(bmp_path):
            os.remove(bmp_path)

def batch_vectorize(input_dir, output_dir):
    """
    지정된 폴더 내의 모든 PNG 파일을 SVG로 일괄 변환합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    success_count = 0
    
    print(f"총 {len(files)}개의 이미지 벡터화를 시작합니다...")
    for idx, filename in enumerate(files):
        png_path = os.path.join(input_dir, filename)
        svg_filename = filename.replace('.png', '.svg')
        svg_path = os.path.join(output_dir, svg_filename)
        
        if png_to_svg(png_path, svg_path):
            success_count += 1
            
        if (idx + 1) % 100 == 0:
            print(f"진행 상황: {idx + 1}/{len(files)} 완료...")
            
    print(f"벡터화 완료! 성공: {success_count}/{len(files)}")

if __name__ == "__main__":
    # 테스트 실행
    print("Vectorization module initialized.")
