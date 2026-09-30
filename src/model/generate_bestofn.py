"""
Best-of-N 전체 생성 드라이버.

글자당 N개 후보를 생성해:
1. 영역 소실(component_check.missing_regions)이 가장 적은 후보군으로 필터
2. 그중 SCR 스타일 임베딩 유사도가 가장 높은 후보 채택
→ 구조 정확성을 지키면서 필체 유사도를 글자 단위로 최대화.

A/B 실측(표본 39자): 단일 0.0597 → best-of-4 0.0819 (+37%).

이후 파이프라인(순서 고정):
  repair_missing_strokes → remove_strays → normalize_stroke_weight
  → apply_slant(+0.115) → 힌트 10자 원본 재설치 → vectorize → build
이 스크립트는 생성 단계만 담당한다.
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

from generator import (COMMON_2350_HANGUL, load_fontdiffuser_pipeline, generate_grayscale_glyph,
                       make_content_image, get_best_style_char, postprocess_glyph)
from component_check import missing_regions
from style_similarity import (load_scr, embed_pil, load_style_embeddings,
                              similarity_to_style)

N = 4


def run(output_dir=None, device="cuda:0"):
    if output_dir is None:
        output_dir = os.path.join(REPO_ROOT, "output", "images")
    os.makedirs(output_dir, exist_ok=True)

    pipe, args = load_fontdiffuser_pipeline(os.path.join(REPO_ROOT, "weights"), device)
    scr, sdev = load_scr(device=device)
    style_dir = os.path.join(REPO_ROOT, "data", "style")
    style_embs = load_style_embeddings(scr, style_dir, sdev)

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

    total = len(COMMON_2350_HANGUL)
    print(f"[best-of-{N}] {total}자 생성 시작", flush=True)
    t0 = time.time()

    for idx, char in enumerate(COMMON_2350_HANGUL):
        out_path = os.path.join(output_dir, f"{ord(char):04X}.png")
        content = ct(make_content_image(char, font, size=128))[None, :].to(device)
        style = cache[get_best_style_char(char, style_hints.keys())]

        cands = []  # (n_missing, -similarity, img)
        for k in range(N):
            torch.manual_seed(77000 + ord(char) * 19 + k * 1223)
            with torch.no_grad():
                gray = generate_grayscale_glyph(pipe,
                    content_images=content, style_images=style, batch_size=1,
                    order=args.order, num_inference_step=args.num_inference_steps,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    t_start=args.t_start, t_end=args.t_end,
                    dm_size=args.content_image_size)
            img = postprocess_glyph(gray)
            n_miss = len(missing_regions(char, img))
            emb = embed_pil(scr, Image.fromarray(img), sdev)
            sim, _ = similarity_to_style(scr, emb, style_embs, sdev)
            cands.append((n_miss, -sim, img))

        # 구조(소실 최소) 우선, 동률이면 유사도 최고
        cands.sort(key=lambda t: (t[0], t[1]))
        cv2.imwrite(out_path, cands[0][2])

        if (idx + 1) % 100 == 0:
            el = (time.time() - t0) / 60
            eta = el / (idx + 1) * (total - idx - 1)
            print(f"  {idx+1}/{total} ({el:.0f}분 경과, 남은 예상 {eta:.0f}분)", flush=True)

    print(f"[best-of-{N}] 완료: {total}자, {(time.time()-t0)/3600:.1f}시간", flush=True)


if __name__ == "__main__":
    run()
