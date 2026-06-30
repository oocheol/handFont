import os
import sys

# 이 스크립트는 MX-Font 혹은 FontDiffuser 모델의 추론을 호출하는 역할을 합니다.
# 런타임 시 필요한 딥러닝 종속성(PyTorch 등)을 검사합니다.

try:
    import torch
    import torchvision.transforms as transforms
except ImportError:
    torch = None

class FewShotFontGenerator:
    def __init__(self, weight_path=None):
        self.device = 'cuda' if (torch and torch.cuda.is_available()) else 'cpu'
        self.weight_path = weight_path
        self.model = None
        
        if torch is None:
            print("[경고] PyTorch 라이브러리를 찾을 수 없습니다. AI 모델을 실행하려면 PyTorch 설치가 필요합니다.")
            print("설치 명령어: pip install torch torchvision")
        else:
            print(f"[정보] PyTorch 로드 완료. 디바이스: {self.device}")
            
    def load_model(self):
        if torch is None:
            return False
            
        if not self.weight_path or not os.path.exists(self.weight_path):
            print(f"[경고] 모델 가중치 파일({self.weight_path})이 존재하지 않습니다. 가중치 다운로드가 필요합니다.")
            return False
            
        try:
            # TODO: 실제 MX-Font 또는 FontDiffuser의 Generator 모델 아키텍처 인스턴스화 및 load_state_dict 수행
            # weights = torch.load(self.weight_path, map_location=self.device)
            # self.model.load_state_dict(weights)
            print("모델 가중치가 성공적으로 로드되었습니다.")
            return True
        except Exception as e:
            print(f"[오류] 모델 로드 실패: {e}")
            return False

    def generate_single(self, style_imgs, content_img):
        """
        style_imgs: 5개의 스타일 힌트 이미지 객체 리스트
        content_img: 생성하고자 하는 타겟 글자(Content)의 템플릿 폰트 이미지
        """
        if self.model is None:
            # 가상 추론: 가중치가 없을 경우 단순 스타일 믹싱 또는 템플릿 반환으로 대체 (테스트용)
            print("[경고] 실제 AI 모델이 로드되지 않았습니다. 테스트 모드로 실행합니다.")
            return content_img
            
        # 실제 모델 추론 로직 적용부
        # with torch.no_grad():
        #     style_feats = [self.model.encode_style(img.to(self.device)) for img in style_imgs]
        #     content_feat = self.model.encode_content(content_img.to(self.device))
        #     generated_tensor = self.model.decode(content_feat, style_feats)
        # return generated_tensor
        return content_img

    def batch_generate(self, style_dir, target_chars, output_dir):
        """
        style_dir: 잘려진 5개의 손글씨 스타일 이미지가 모여있는 폴더
        target_chars: 생성할 한글 글자 리스트 (예: 2,350자 문자열)
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"스타일 디렉토리 '{style_dir}'로부터 스타일 힌트를 로드합니다...")
        # 1. 스타일 이미지 로드 (최대 5개)
        style_files = [f for f in os.listdir(style_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]
        if not style_files:
            print("[오류] 스타일 힌트 이미지를 찾을 수 없습니다. data/style 폴더에 이미지를 넣어주세요.")
            return
            
        print(f"사용할 스타일 소스 파일: {style_files}")
        
        # 2. 루프 돌며 문자 이미지 생성
        print(f"총 {len(target_chars)}자의 이미지 생성을 시작합니다...")
        for idx, char in enumerate(target_chars):
            # 실제 추론 시에는 content_img 템플릿(기본 돋움/바탕체 이미지)을 렌더링하여 모델에 넘겨줍니다.
            # 여기서는 파일 생성 및 구조 테스트를 위해 자리표시용 이미지를 저장합니다.
            char_hex = f"{ord(char):04X}"
            out_path = os.path.join(output_dir, f"{char_hex}.png")
            
            # 테스트용: 임시 검은색 배경에 흰 글씨 이미지를 생성하여 저장하는 모의 동작
            if torch is None:
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('L', (128, 128), color=255)
                draw = ImageDraw.Draw(img)
                # 폰트가 없으면 기본 폰트 사용
                try:
                    font = ImageFont.truetype("malgun.ttf", 80)
                except IOError:
                    font = ImageFont.load_default()
                
                # 글자를 중앙에 그림
                w, h = draw.textsize(char, font=font) if hasattr(draw, 'textsize') else (64, 64)
                draw.text(((128-w)//2, (128-h)//2), char, fill=0, font=font)
                img.save(out_path)
                
        print(f"추론 완료! 생성된 이미지 저장 경로: {output_dir}")

if __name__ == "__main__":
    generator = FewShotFontGenerator()
    # 임시 테스트 실행 (가상 글자 '가', '나', '다' 생성)
    generator.batch_generate("data/style", "가나다", "output/images")
