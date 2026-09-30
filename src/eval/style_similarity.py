"""
스타일(필체) 유사도 정량 지표.

FontDiffuser의 SCR(Style-Contrastive Representation, weights/scr_210000.pth)
모듈을 사용한다. SCR은 InfoNCE로 "같은 필체는 가깝게, 다른 필체는 멀게"
학습된 스타일 임베딩 추출기이므로, 스타일 유사도 측정에 가장 적합하다
(GAN style_encoder의 공간 feature보다 목적에 부합).

절차:
  StyleFeatExtractor -> Projector -> layer별 L2 정규화 임베딩(2048d).
  코사인 유사도 = 정규화 벡터의 내적. nce_layers('0,1,2,3') 평균.

지표:
  - 생성 글리프 임베딩 vs 힌트 10장 각각의 코사인 -> 그 중 mean / max.
  - 여러 글자에 대해 다시 평균 -> 세트 단위 스코어.

Sanity check(--sanity):
  1) 힌트 자기 자신 vs 자기 자신 코사인 ~ 1.0
  2) 힌트 vs 맑은고딕 렌더링(다른 폰트) 코사인은 낮게
  3) 생성 글리프(output/images)는 그 중간~높은 값
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FONTDIFFUSER_PATH = os.path.join(REPO_ROOT, "src", "model", "fontdiffuser")
if FONTDIFFUSER_PATH not in sys.path:
    sys.path.insert(0, FONTDIFFUSER_PATH)

from src import build_scr  # noqa: E402
from configs.fontdiffuser import get_parser  # noqa: E402

NCE_LAYERS = "0,1,2,3"

_TRANSFORM = transforms.Compose([
    transforms.Resize((96, 96),
                      interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def load_scr(ckpt_dir=None, device="cuda:0"):
    if ckpt_dir is None:
        ckpt_dir = os.path.join(REPO_ROOT, "weights")
    parser = get_parser()
    args = parser.parse_args([])
    args.mode = "refinement"  # eval (grad off)
    scr = build_scr(args)
    state = torch.load(os.path.join(ckpt_dir, "scr_210000.pth"), map_location=device)
    scr.load_state_dict(state)
    scr = scr.to(device)
    scr.eval()
    return scr, device


@torch.no_grad()
def embed_pil(scr, pil_img, device):
    """PIL(RGB) 이미지 -> layer별 L2정규화 임베딩 리스트."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    x = _TRANSFORM(pil_img)[None, :].to(device)
    feats = scr.StyleFeatExtractor(x, NCE_LAYERS)
    projs = scr.StyleFeatProjector(feats, NCE_LAYERS)  # 각 [1, 2048], 이미 normalize됨
    return [p[0] for p in projs]


def cosine_layers(emb_a, emb_b):
    """layer별 코사인의 평균(임베딩은 이미 L2정규화 -> 내적=코사인)."""
    sims = [float(torch.dot(a, b)) for a, b in zip(emb_a, emb_b)]
    return sum(sims) / len(sims)


def load_style_embeddings(scr, style_dir, device):
    embs = {}
    for f in sorted(os.listdir(style_dir)):
        if not f.lower().endswith(".png"):
            continue
        ch = chr(int(os.path.splitext(f)[0], 16))
        embs[ch] = embed_pil(scr, Image.open(os.path.join(style_dir, f)), device)
    return embs


def similarity_to_style(scr, gen_emb, style_embs, device):
    """생성 임베딩 vs 모든 힌트 -> (mean_over_hints, max_over_hints)."""
    sims = [cosine_layers(gen_emb, se) for se in style_embs.values()]
    return float(np.mean(sims)), float(np.max(sims))


def score_directory(images_dir, chars, style_dir=None, scr=None, device="cuda:0"):
    """
    images_dir의 지정 글자들 PNG에 대해 힌트 대비 유사도 산출.
    반환: dict(char -> (mean,max)), 및 전체 평균 (avg_mean, avg_max).
    """
    if style_dir is None:
        style_dir = os.path.join(REPO_ROOT, "data", "style")
    close_scr = False
    if scr is None:
        scr, device = load_scr(device=device)
        close_scr = True
    style_embs = load_style_embeddings(scr, style_dir, device)

    per_char = {}
    for c in chars:
        p = os.path.join(images_dir, f"{ord(c):04X}.png")
        if not os.path.exists(p):
            continue
        ge = embed_pil(scr, Image.open(p), device)
        per_char[c] = similarity_to_style(scr, ge, style_embs, device)

    if per_char:
        avg_mean = float(np.mean([v[0] for v in per_char.values()]))
        avg_max = float(np.mean([v[1] for v in per_char.values()]))
    else:
        avg_mean = avg_max = 0.0
    return per_char, avg_mean, avg_max


def _render_font_glyph(char, font_path, size=128):
    font = ImageFont.truetype(font_path, 100)
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    l, t, r, b = draw.textbbox((0, 0), char, font=font)
    w, h = r - l, b - t
    draw.text(((size - w) // 2 - l, (size - h) // 2 - t), char, fill=0, font=font)
    return img.convert("RGB")


def sanity_check(device="cuda:0"):
    scr, device = load_scr(device=device)
    style_dir = os.path.join(REPO_ROOT, "data", "style")
    style_embs = load_style_embeddings(scr, style_dir, device)
    hint_chars = list(style_embs.keys())

    print(f"[SANITY] 힌트 글자: {hint_chars}")

    # 1) 자기 자신
    self_sims = [cosine_layers(style_embs[c], style_embs[c]) for c in hint_chars]
    print(f"[SANITY 1] 힌트 자기자신 코사인: mean={np.mean(self_sims):.4f} "
          f"(min={np.min(self_sims):.4f}, max={np.max(self_sims):.4f})  [~1.0 기대]")

    # 2) 힌트 vs 맑은고딕 렌더링(다른 폰트, 같은 글자) -> 낮아야 함
    malgun = "C:\\Windows\\Fonts\\malgun.ttf"
    cross = []
    for c in hint_chars:
        fe = embed_pil(scr, _render_font_glyph(c, malgun), device)
        cross.append(cosine_layers(fe, style_embs[c]))
    print(f"[SANITY 2] 힌트 vs 맑은고딕(같은 글자,다른 폰트): "
          f"mean={np.mean(cross):.4f} (min={np.min(cross):.4f}, "
          f"max={np.max(cross):.4f})  [낮아야 유효]")

    # 3) 힌트 서로 다른 글자간 (같은 필체 다른 글자) -> 중간~높음
    inter = []
    for i, ci in enumerate(hint_chars):
        for cj in hint_chars[i + 1:]:
            inter.append(cosine_layers(style_embs[ci], style_embs[cj]))
    print(f"[SANITY 3] 힌트끼리(같은 필체,다른 글자): mean={np.mean(inter):.4f}")

    # 4) 생성 글리프(힌트 글자와 동일한 것) vs 힌트
    img_dir = os.path.join(REPO_ROOT, "output", "images")
    gen_vs = []
    for c in hint_chars:
        p = os.path.join(img_dir, f"{ord(c):04X}.png")
        if os.path.exists(p):
            ge = embed_pil(scr, Image.open(p), device)
            m, _ = similarity_to_style(scr, ge, style_embs, device)
            gen_vs.append(m)
    if gen_vs:
        print(f"[SANITY 4] 생성 글리프(힌트글자) vs 힌트 mean-over-hints: "
              f"mean={np.mean(gen_vs):.4f}")

    ok = (np.mean(self_sims) > 0.95 and np.mean(cross) < np.mean(inter))
    print(f"[SANITY 결과] {'유효(PASS)' if ok else '재검토 필요(FAIL)'}: "
          f"자기자신 > 교차폰트, 힌트교차 > 교차폰트")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--images-dir", default=os.path.join(REPO_ROOT, "output", "images"))
    ap.add_argument("--chars", default=None, help="평가할 글자열(기본: proof_sheet SAMPLE_CHARS)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"

    if args.sanity:
        sanity_check(dev)
        return

    if args.chars:
        chars = list(args.chars)
    else:
        sys.path.insert(0, os.path.join(REPO_ROOT, "src", "eval"))
        from proof_sheet import SAMPLE_CHARS
        chars = SAMPLE_CHARS

    per_char, avg_mean, avg_max = score_directory(args.images_dir, chars, device=dev)
    print(f"[유사도] 평가 글자수: {len(per_char)}")
    print(f"[유사도] avg mean-over-hints = {avg_mean:.4f}")
    print(f"[유사도] avg max-over-hints  = {avg_max:.4f}")


if __name__ == "__main__":
    main()
