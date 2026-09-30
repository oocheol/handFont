"""
A/B 표본 생성 하네스.

주어진 설정(guidance_scale, num_inference_steps, 후처리 완화 여부)으로
표본 글자만 별도 디렉토리에 생성한다. 전체 재생성 없이 스타일 유사도
설정을 빠르게 비교하기 위한 도구.

후처리 변형:
  postprocess='default' -> generator.postprocess_glyph (기존)
  postprocess='soft'    -> closing/opening 완화(획 질감·삐침 보존):
                           blur 축소, closing 커널 축소, opening 제거,
                           CC 필터 완화. 필체 개성을 덜 뭉갠다.
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image, ImageFont
import torchvision.transforms as transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "model"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "eval"))

from generator import (load_fontdiffuser_pipeline, make_content_image,
                       get_best_style_char, postprocess_glyph)

# 대표 표본: 힌트 10자 + 자모/받침 커버 30자
AB_CHARS = list("매별새소실영이조화효"      # 힌트 10자 (자기 스타일 재현 확인)
                "가나다라마바사아자하"      # 기본 음절
                "국권근금강경공관광끝"      # 받침 있는 음절
                "너덜멈넘든능많널넣떻")      # 복잡 음절


def postprocess_soft(gray_np, out_size=256):
    """필체 질감 보존형 후처리 (closing/opening 완화)."""
    gray_pil = Image.fromarray(gray_np)
    upscaled = np.array(gray_pil.resize((384, 384), Image.BICUBIC))
    # 약한 blur (5x5,1.2 -> 3x3,0.8): 획 끝 삐침·질감 보존
    blurred = cv2.GaussianBlur(upscaled, (3, 3), 0.8)
    otsu_val, binary = cv2.threshold(blurred, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    min_ink_ratio = 0.015
    tval = otsu_val
    while (binary == 0).mean() < min_ink_ratio and tval < 245:
        tval = min(245, tval + 15)
        _, binary = cv2.threshold(blurred, tval, 255, cv2.THRESH_BINARY)

    ink = cv2.bitwise_not(binary)
    # 최소한의 연결만: 3x3 ellipse closing 1회 (방향별 5px 커널 제거)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, ek)
    # opening 생략 (질감 보존). CC 필터만 아주 약하게.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    if num_labels <= 1:
        filtered = combined
    else:
        total_ink = stats[1:, cv2.CC_STAT_AREA].sum()
        min_area = max(12, total_ink * 0.002)
        filtered = np.zeros_like(combined)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == lbl] = 255

    result = cv2.bitwise_not(filtered)
    final_pil = Image.fromarray(result).resize((out_size, out_size), Image.LANCZOS)
    final_np = np.array(final_pil)
    _, final_img = cv2.threshold(final_np, 127, 255, cv2.THRESH_BINARY)
    return final_img


def generate_sample(out_dir, guidance_scale=7.5, steps=20, postprocess="default",
                    chars=None, device="cuda:0", seed=0):
    if chars is None:
        chars = AB_CHARS
    os.makedirs(out_dir, exist_ok=True)

    pipe, args = load_fontdiffuser_pipeline(os.path.join(REPO_ROOT, "weights"), device)
    # guidance_scale은 파이프라인 객체에 저장됨
    pipe.guidance_scale = guidance_scale
    args.num_inference_steps = steps

    post_fn = postprocess_soft if postprocess == "soft" else postprocess_glyph

    style_dir = os.path.join(REPO_ROOT, "data", "style")
    style_hints = {}
    for f in os.listdir(style_dir):
        if f.lower().endswith('.png'):
            style_hints[chr(int(os.path.splitext(f)[0], 16))] = \
                Image.open(os.path.join(style_dir, f)).convert('RGB')

    font = ImageFont.truetype("C:\\Windows\\Fonts\\malgun.ttf", 100)
    ct = transforms.Compose([
        transforms.Resize(args.content_image_size,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    st = transforms.Compose([
        transforms.Resize(args.style_image_size,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    cache = {c: st(img)[None, :].to(device) for c, img in style_hints.items()}

    print(f"[AB] out={out_dir} gs={guidance_scale} steps={steps} post={postprocess} "
          f"n={len(chars)}", flush=True)
    for i, char in enumerate(chars):
        content = ct(make_content_image(char, font, size=128))[None, :].to(device)
        style = cache[get_best_style_char(char, style_hints.keys())]
        torch.manual_seed(seed + ord(char))
        with torch.no_grad():
            raw = pipe.generate(
                content_images=content, style_images=style, batch_size=1,
                order=args.order, num_inference_step=steps,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
                t_start=args.t_start, t_end=args.t_end,
                dm_size=args.content_image_size, return_tensor=True)
        gray = ((raw[0, 0] * 0.299 + raw[0, 1] * 0.587 + raw[0, 2] * 0.114)
                .cpu().numpy() * 255).astype(np.uint8)
        img = post_fn(gray)
        cv2.imwrite(os.path.join(out_dir, f"{ord(char):04X}.png"), img)
    print(f"[AB] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gs", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--post", default="default", choices=["default", "soft"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    generate_sample(a.out_dir, a.gs, a.steps, a.post, device=dev, seed=a.seed)
