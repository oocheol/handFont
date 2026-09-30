import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# FontDiffuser 모듈 경로 주입
FONTDIFFUSER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "fontdiffuser"))
if FONTDIFFUSER_PATH not in sys.path:
    sys.path.append(FONTDIFFUSER_PATH)

# FontDiffuser 모듈 임포트
try:
    from src import (FontDiffuserDPMPipeline,
                     FontDiffuserModelDPM,
                     build_ddpm_scheduler,
                     build_unet,
                     build_content_encoder,
                     build_style_encoder)
    import torchvision.transforms as transforms
except ImportError as e:
    print(f"[경고] FontDiffuser 임포트 실패: {e}")
    FontDiffuserDPMPipeline = None

def _build_ks_x_1001_2350():
    """
    KS X 1001 (완성형 한글 조합) 표준 2,350자 세트를 EUC-KR 코드포인트
    범위(0xB0A1-0xC8FE)를 왕복 디코딩하여 프로그램적으로 생성.
    하드코딩 문자열은 오탈자로 376자 누락 + 124자 오염이 있었음(직접 검증됨).
    """
    chars = []
    for hi in range(0xB0, 0xC9):
        for lo in range(0xA1, 0xFF):
            try:
                ch = bytes([hi, lo]).decode('euc-kr')
            except (UnicodeDecodeError, ValueError):
                continue
            if 0xAC00 <= ord(ch) <= 0xD7A3:
                chars.append(ch)
    return "".join(chars)


COMMON_2350_HANGUL = _build_ks_x_1001_2350()

def load_fontdiffuser_pipeline(ckpt_dir="weights", device="cuda:0"):
    from configs.fontdiffuser import get_parser
    parser = get_parser()
    args = parser.parse_args([])
    
    args.ckpt_dir = ckpt_dir
    args.device = device
    args.num_inference_steps = 20  # GPU 가속 기준 고화질 20스텝 고정
    # 스타일(필체) 유사도 개선: classifier-free guidance_scale를 7.5->3.0으로 낮춤.
    # 높은 CFG는 content 템플릿(맑은고딕 형태)으로 출력을 끌어당겨 필체 개성을
    # 지운다. SCR 스타일 임베딩 코사인 A/B(표본 40자)에서 gs7.5=0.0496 ->
    # gs3.0=0.0620 (+25.0%, src/eval/style_similarity.py)로 실측 확인.
    # gs2.0(0.0627)은 이득이 미미하고 획 소실 위험만 커져 3.0을 승자로 확정.
    args.guidance_scale = 3.0
    
    # 튜플로 이미지 사이즈 변환
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)
    
    # 1. 아웃라인 UNet 구성
    print("[AI 로드] UNet 모델 구조 생성 중...")
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{ckpt_dir}/unet.pth", map_location=device))
    
    # 2. 스타일 인코더 구성
    print("[AI 로드] 스타일 인코더 로드 중...")
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{ckpt_dir}/style_encoder.pth", map_location=device))
    
    # 3. 콘텐츠 인코더 구성
    print("[AI 로드] 콘텐츠 인코더 로드 중...")
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{ckpt_dir}/content_encoder.pth", map_location=device))
    
    # 4. DDPM 스케줄러 구성
    train_scheduler = build_ddpm_scheduler(args=args)
    
    # 5. FontDiffuser 모델 전체 패키징
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder
    )
    model.to(device)
    model.eval()
    
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale
    )
    print("[성공] FontDiffuser SOTA 딥러닝 추론 파이프라인 완벽 로드 완료!")
    return pipe, args


def make_content_image(char, font, size=128):
    """
    FontDiffuser Content Image 생성.
    - 흰 배경(255)에 검은 글씨(0), 정밀 중앙 정렬
    - Morphological dilation 적용: 가는 획(ㅏ 가로획 등)을 굵게 만들어
      모델 Content Encoder가 획 구조를 확실히 인식하도록 강화
    """
    img = Image.new('L', (size, size), color=255)  # 그레이스케일 흰 배경
    draw = ImageDraw.Draw(img)
    try:
        left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
        w, h = right - left, bottom - top
        x = (size - w) // 2 - left
        y = (size - h) // 2 - top
        draw.text((x, y), char, fill=0, font=font)
    except Exception:
        draw.text((size // 4, size // 4), char, fill=0, font=font)
    
    # 흰 배경 검은 글씨 → 반전하여 검은 배경 흰 글씨 (획 부분 = 흰색)
    arr = np.array(img)
    ink_mask = (255 - arr)  # 글씨 부분만 밝게
    
    # Morphological dilation: 가는 획을 팽창시켜 모델이 인식 가능한 두께로 만듦
    # 특히 ㅏ의 짧은 가로획, ㅎ의 점획 등이 잘 인식되도록
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(ink_mask, kernel, iterations=1)
    
    # 다시 흰 배경 검은 글씨로 반전
    result = 255 - dilated
    result_img = Image.fromarray(result.astype(np.uint8))
    
    # RGB로 변환 (FontDiffuser는 3채널 입력)
    return result_img.convert('RGB')


# 한글 자모 인덱스 정의
CHO = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']

# 중성(모음) 그룹별 최우선 스타일 힌트.
# data/style에는 실제로 매/실/효/소/이/새/별/조/영/화 10자만 존재(refine_hints.py 참조).
# 해당 그룹에 직접 대응하는 힌트가 있으면(예: ㅐ그룹→매/새) 그 자체가 최선의 매치이고,
# 없는 그룹(ㅏ/ㅓ/ㅜ 등)만 구조가 가장 가까운 힌트로 대체한다.
JUNG_GROUP_HINTS = [
    (frozenset({0, 2}),              ['화', '별', '영']),  # ㅏ/ㅑ: 화(ㅘ 안의 ㅏ) 및 별/영(ㅕ 우측 획)로 대체
    (frozenset({4, 6}),              ['별', '영']),         # ㅓ/ㅕ: 별/영이 직접 ㅕ 힌트
    (frozenset({1, 3, 5, 7}),        ['매', '새']),         # ㅐ/ㅒ/ㅔ/ㅖ: 매/새가 직접 ㅐ 힌트
    (frozenset({8, 12}),             ['소', '조', '효']),   # ㅗ/ㅛ: 소/조가 직접 ㅗ, 효가 직접 ㅛ
    (frozenset({13, 17}),            ['효', '소', '조']),   # ㅜ/ㅠ: 상하 대칭 구조인 효/소로 대체
    (frozenset({9, 10, 11, 14, 15, 16}), ['화']),           # 복합모음(ㅘ계): 화가 직접 ㅘ 힌트
    (frozenset({18, 19, 20}),        ['이', '실']),         # ㅡ/ㅢ/ㅣ: 이/실이 직접 ㅣ 힌트
]

CHO_HINTS = {
    6: ['매'],                       # ㅁ
    7: ['별'], 8: ['별'],             # ㅂ, ㅃ
    9: ['새', '소', '실'], 10: ['새', '소', '실'],  # ㅅ, ㅆ
    11: ['이', '영'],                # ㅇ
    12: ['조'], 13: ['조'], 14: ['조'],  # ㅈ, ㅉ, ㅊ
    18: ['화', '효'],                # ㅎ
}

FALLBACK_ORDER = ['이', '소', '별', '조', '화', '매', '새', '영', '실', '효']


def get_best_style_char(char, available_chars):
    """
    타겟 글자의 초성/중성 구조에 가장 가까운 스타일 힌트 글자 1개를 선택.

    이전 버전은 최대 3장을 픽셀 평균해 앙상블했으나, FontDiffuser는 스타일
    이미지를 style_encoder뿐 아니라 content_encoder에도 그대로 통과시켜
    "참조 글리프의 구조" 조건으로 함께 사용한다(src/model.py의
    style_content_res_features). 서로 다른 글자를 픽셀 평균하면 스타일·구조
    두 경로 모두에 존재하지 않는 프랑켄글리프가 주입되어 획 왜곡의 원인이
    되므로, 단일 참조 이미지만 사용한다.
    """
    available = set(available_chars)

    if char in available:  # 타겟 글자 자체가 힌트라면 그보다 나은 참조는 없음
        return char

    if not (0xAC00 <= ord(char) <= 0xD7A3):
        for fb in FALLBACK_ORDER:
            if fb in available:
                return fb
        return next(iter(available))

    char_code = ord(char) - 0xAC00
    cho_idx = char_code // 588
    jung_idx = (char_code % 588) // 28

    candidates = []
    for group, hints in JUNG_GROUP_HINTS:
        if jung_idx in group:
            candidates.extend(hints)
            break
    candidates.extend(CHO_HINTS.get(cho_idx, []))
    candidates.extend(FALLBACK_ORDER)

    for c in candidates:
        if c in available:
            return c
    return next(iter(available))


def postprocess_glyph(gray_np, out_size=256):
    """
    디퓨전 raw 그레이스케일 출력(밝을수록 배경, 어두울수록 잉크)을 최종
    흑백 글리프 PNG로 정리.

    이전 버전의 결함들을 수정:
    1. 고정 threshold(175)는 글자마다 다른 잉크 농도를 무시해 획이 옅은 글자를
       통째로 날려버렸다(예: '나'가 완전히 빈 이미지로 저장됨) → 글자별 Otsu
       적응형 threshold로 교체.
    2. 방향별 morphological closing이 배경(255)이 다수인 극성 그대로
       적용되고 있었다. MORPH_CLOSE는 흰 영역(255)을 확장했다가 축소하는
       연산이므로, 배경이 255인 이미지에 그대로 적용하면 얇은 잉크(0) 획을
       깎아 오히려 끊어버린다('가'의 ㄱ이 두 조각으로 갈라지는 원인이었음).
       → 잉크를 255로 반전한 뒤 closing을 적용해야 획을 "잇는" 본래 의도대로
       동작한다.
    3. Connected-Component 노이즈 필터가 고정 400px였는데, 글자마다 총
       잉크량이 다르므로(예: 획이 적은 'ㅣ' 등) 상대 비율 기준으로 교체.
    4. (2026-07 발견) 96px 그레이스케일을 먼저 BICUBIC 업스케일 후 블러+threshold
       하던 순서가 근접한 두 세로획 사이의 좁은 갭을 블러로 뭉개 가짜 연결
       다리를 만들었다 ('에'=ㅇ+ㅔ에서 ㅓ와 ㅣ 사이에 없어야 할 가로선이 생기던
       원인 — ㅐ/ㅒ/ㅔ/ㅖ 계열 371자 전체에 영향). 96px 원본은 두 획이 명확히
       분리되어 있음을 실측 확인 → **원본 해상도에서 먼저 이진화**한 뒤 그
       이진 마스크를 업스케일하도록 순서를 바꿈. LANCZOS4는 이미 확정된 형태를
       리샘플링만 하므로 색 공간에서 갭을 침범하는 블러가 없다.
    """
    # Step 1: 96px 원본 해상도에서 먼저 Otsu 적응형 threshold
    # (업스케일 전에 이진화해야 근접한 두 획 사이 갭이 블러로 뭉개지지 않는다)
    otsu_val, binary96 = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 폴백: 디퓨전 출력이 전체적으로 옅으면 Otsu가 획 대부분을 배경으로
    # 분류해 글자가 통째로 소실되는 케이스가 있음(예: '나' 잉크 0.65%).
    # 잉크 비율이 비정상적으로 낮으면 threshold를 점진 상향해 재시도.
    min_ink_ratio = 0.015
    tval = otsu_val
    while (binary96 == 0).mean() < min_ink_ratio and tval < 245:
        tval = min(245, tval + 15)
        _, binary96 = cv2.threshold(gray_np, tval, 255, cv2.THRESH_BINARY)

    # Step 2: 이미 이진화된 마스크를 4배 업스케일 (96→384). 색 공간 블러가
    # 아니라 형태 리샘플링이므로 갭을 침범하지 않는다.
    ink_up = cv2.resize(cv2.bitwise_not(binary96), (384, 384), interpolation=cv2.INTER_LANCZOS4)
    _, ink = cv2.threshold(ink_up, 127, 255, cv2.THRESH_BINARY)

    # Step 3: 방향별 closing으로 실제 끊긴 획만 보수 (ink=255 기준)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_v = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, vk)
    closed_h = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, hk)
    closed_e = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, ek)
    # union: 어느 방향이든 이어붙인 획은 보존 (ink=255 기준이므로 OR)
    combined = cv2.bitwise_or(closed_v, closed_h)
    combined = cv2.bitwise_or(combined, closed_e)

    # Step 4: 작은 opening으로 잔여 노이즈 점만 제거 (얇은 획 보존 위해 3x3)
    ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, ok)

    # Step 5: CC 크기 필터 - 글자별 총 잉크량에 비례한 상대 기준
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels <= 1:
        filtered = opened
    else:
        total_ink = stats[1:, cv2.CC_STAT_AREA].sum()
        min_area = max(20, total_ink * 0.004)  # 전체 잉크의 0.4% 미만 or 20px 미만은 노이즈
        filtered = np.zeros_like(opened)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == lbl] = 255

    # 표준 컨벤션(흰 배경/검은 잉크)으로 반전 복귀
    result = cv2.bitwise_not(filtered)

    # Step 6: 최종 크기로 리사이즈 + 재이진화
    final_pil = Image.fromarray(result).resize((out_size, out_size), Image.LANCZOS)
    final_np = np.array(final_pil)
    _, final_img = cv2.threshold(final_np, 127, 255, cv2.THRESH_BINARY)

    # Step 7: 윤곽 평활화 - 획 끝 스퍼(spur)·너덜거림 정리 (blur→재이진화 2회,
    # 구조는 보존하고 윤곽만 둥글게). "획 끝이 날림처리됨" 피드백으로 추가.
    for _ in range(2):
        final_img = cv2.GaussianBlur(final_img, (7, 7), 2.0)
        _, final_img = cv2.threshold(final_img, 127, 255, cv2.THRESH_BINARY)
    return final_img


def generate_grayscale_glyph(pipe, **kwargs):
    """Support both upstream PIL output and the local float-tensor extension."""
    try:
        raw = pipe.generate(**kwargs, return_tensor=True)
    except TypeError as error:
        # A clean upstream checkout does not provide the return_tensor option.
        # Propagate unrelated generation errors rather than masking them.
        if "unexpected keyword argument 'return_tensor'" not in str(error):
            raise
        images = pipe.generate(**kwargs)
        return np.asarray(images[0].convert('L'), dtype=np.uint8)

    gray = raw[0, 0] * 0.299 + raw[0, 1] * 0.587 + raw[0, 2] * 0.114
    return (gray.detach().cpu().numpy() * 255).astype(np.uint8)


def run_real_fontdiffuser_inference(pipe, args, style_dir="data/style", output_dir="output/images", device="cuda:0"):
    """
    FontDiffuser weights 4종 + 구조 매칭 기반 단일 참조 스타일로
    2,350자 한글 손글씨 폰트를 생성합니다.

    개선사항:
    - 단일 최적 참조 스타일 선택: get_best_style_char로 픽셀/임베딩 평균 없이
      구조가 가장 가까운 힌트 1장만 사용 (프랑켄글리프 방지)
    - Content Image 품질 개선: FontDiffuser 원본 ttf2im과 동일한 방식 사용
    - 적응형 후처리: postprocess_glyph (Otsu 적응형 threshold + 올바른 극성의
      closing + 상대적 노이즈 필터)
    """
    print("[AI 작동] FontDiffuser 딥러닝 디퓨전 추론 가동 시작...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 스타일 이미지들 로드
    style_files = [f for f in os.listdir(style_dir) if f.lower().endswith('.png')]
    style_hints = {}
    
    for f in style_files:
        char_hex = os.path.splitext(f)[0]
        try:
            char = chr(int(char_hex, 16))
            pil_img = Image.open(os.path.join(style_dir, f)).convert('RGB')
            style_hints[char] = pil_img
        except Exception as e:
            print(f"[경고] 스타일 파일 해석 오류 ({f}): {e}")
    
    print(f"[스타일] 로드된 힌트 글자: {list(style_hints.keys())}")
    
    if not style_hints:
        print("[오류] 스타일 힌트 글자가 하나도 로드되지 않았습니다.")
        return
    
    # 2. 시스템 기본 한글 폰트 로드 (Content Template)
    font_paths = [
        "C:\\Windows\\Fonts\\malgun.ttf", 
        "C:\\Windows\\Fonts\\batang.ttc", 
        "C:\\Windows\\Fonts\\gulim.ttc"
    ]
    target_font = next((p for p in font_paths if os.path.exists(p)), None)
    if not target_font:
        print("[오류] 시스템 맑은고딕 또는 바탕체를 찾을 수 없습니다.")
        return
    
    # FontDiffuser 원본과 동일한 128px 폰트 사이즈
    font = ImageFont.truetype(target_font, 100)
    
    # 전처리 트랜스폼 정의
    content_transforms = transforms.Compose([
        transforms.Resize(args.content_image_size, 
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    style_transforms = transforms.Compose([
        transforms.Resize(args.style_image_size, 
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # 3. 전체 스타일 힌트 텐서 사전 계산 (매번 변환하지 않도록 캐시)
    style_tensor_cache = {}
    for char, pil_img in style_hints.items():
        style_tensor_cache[char] = style_transforms(pil_img)[None, :].to(device)
    
    total = len(COMMON_2350_HANGUL)
    print(f" -> 총 {total}자의 한글에 대해 AI 디퓨전 획 생성을 개시합니다.")
    print(f"    [방식] 구조 매칭 기반 단일 참조 스타일 선택 (픽셀 평균 앙상블 제거)")

    for idx, char in enumerate(COMMON_2350_HANGUL):
        char_hex = f"{ord(char):04X}"
        out_path = os.path.join(output_dir, f"{char_hex}.png")

        # Content Image: FontDiffuser 원본 방식으로 생성 (정밀 중앙 정렬)
        c_img = make_content_image(char, font, size=128)
        content_tensor = content_transforms(c_img)[None, :].to(device)

        # 구조가 가장 가까운 단일 스타일 힌트 선택
        best_char = get_best_style_char(char, style_hints.keys())
        style_tensor = style_tensor_cache[best_char]

        # raw float tensor 직접 수신 [1, 3, 96, 96] 형태
        with torch.no_grad():
            gray_np = generate_grayscale_glyph(pipe,
                content_images=content_tensor,
                style_images=style_tensor,
                batch_size=1,
                order=args.order,
                num_inference_step=args.num_inference_steps,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
                t_start=args.t_start,
                t_end=args.t_end,
                dm_size=args.content_image_size
            )

        # raw tensor: [1, 3, H, W] float32 on GPU, 값 범위 [0, 1]
        # 그레이스케일 변환 (R*0.299 + G*0.587 + B*0.114)
        final_img = postprocess_glyph(gray_np)
        
        cv2.imwrite(out_path, final_img)
        
        if (idx + 1) % 100 == 0:
            print(f"   -> AI 추론 진행도: {idx + 1}/{total} 완료 (현재 자: '{char}')", flush=True)
    
    print("[성공] FontDiffuser AI 추론 완료 및 2,350자 고품질 합성 이미지 저장 성공!")


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[장치 상태] 실행 디바이스: {device}")
    
    try:
        pipe, args = load_fontdiffuser_pipeline("weights", device)
        run_real_fontdiffuser_inference(pipe, args, "data/style", "output/images", device)
    except Exception as e:
        import traceback
        print(f"[오류 발생] 진짜 AI 추론 중 예외 발생: {e}")
        traceback.print_exc()
