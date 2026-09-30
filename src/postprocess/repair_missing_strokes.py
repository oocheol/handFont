"""
영역 소실 글리프 표적 수리.

디퓨전 추론이 확률적이라 같은 조건에서도 ㅏ의 가로획 같은 미세 획이
나올 때도, 소실될 때도 있다('다'→'디' 등). component_check로 플래그된
글자만 시드를 바꿔 최대 N회 재생성하고:
  - 영역 검사를 통과하는 시도가 나오면 즉시 교체
  - 끝까지 통과 못 하면 소실 칸 수가 원본보다 적은 최선의 시도로 교체
  - 원본이 가장 나으면 그대로 둠 (과탐된 정상 글리프는 자연히 보존됨)
"""
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "model"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "eval"))

import torch
import numpy as np
import cv2
from PIL import Image, ImageFont
import torchvision.transforms as transforms

from generator import (load_fontdiffuser_pipeline, make_content_image, generate_grayscale_glyph,
                       get_best_style_char, postprocess_glyph)
from component_check import missing_regions, check_directory

MAX_ATTEMPTS = 3


def repair(images_dir=None, device="cuda:0"):
    if images_dir is None:
        images_dir = os.path.join(REPO_ROOT, "output", "images")

    print("[수리] 영역 소실 글리프 스캔 중...")
    bad = check_directory(images_dir)
    print(f"[수리] 대상 {len(bad)}자")
    if not bad:
        return

    pipe, args = load_fontdiffuser_pipeline(os.path.join(REPO_ROOT, "weights"), device)

    style_dir = os.path.join(REPO_ROOT, "data", "style")
    style_hints = {}
    for f in os.listdir(style_dir):
        if f.lower().endswith('.png'):
            style_hints[chr(int(os.path.splitext(f)[0], 16))] = \
                Image.open(os.path.join(style_dir, f)).convert('RGB')

    font = ImageFont.truetype("C:\\Windows\\Fonts\\malgun.ttf", 100)
    ct = transforms.Compose([
        transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    st = transforms.Compose([
        transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    cache = {c: st(img)[None, :].to(device) for c, img in style_hints.items()}

    fixed = kept = improved = 0
    t0 = time.time()
    for i, (char, orig_missing) in enumerate(sorted(bad.items())):
        out_path = os.path.join(images_dir, f"{ord(char):04X}.png")
        content = ct(make_content_image(char, font, size=128))[None, :].to(device)
        style = cache[get_best_style_char(char, style_hints.keys())]

        best_img = None
        best_count = len(orig_missing)

        for attempt in range(MAX_ATTEMPTS):
            torch.manual_seed(1000 + ord(char) * 7 + attempt * 131)
            with torch.no_grad():
                gray = generate_grayscale_glyph(pipe,
                    content_images=content, style_images=style, batch_size=1,
                    order=args.order, num_inference_step=args.num_inference_steps,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    t_start=args.t_start, t_end=args.t_end,
                    dm_size=args.content_image_size)
            candidate = postprocess_glyph(gray)
            m = missing_regions(char, candidate)
            if not m:
                cv2.imwrite(out_path, candidate)
                fixed += 1
                best_img = None
                break
            if len(m) < best_count:
                best_count = len(m)
                best_img = candidate
        else:
            if best_img is not None:
                cv2.imwrite(out_path, best_img)
                improved += 1
            else:
                kept += 1

        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"   -> {i+1}/{len(bad)} (완치 {fixed}, 개선 {improved}, 유지 {kept}) "
                  f"{el/60:.0f}분 경과", flush=True)

    print(f"[수리 완료] 완치 {fixed}, 개선 {improved}, 원본 유지 {kept} / 총 {len(bad)}")


if __name__ == "__main__":
    repair()
