# HandFont

손글씨 글자 이미지를 스타일 힌트로 사용해 KS X 1001 한글 2,350자를 생성하고 TTF 폰트로 빌드하는 로컬 Python 파이프라인입니다. FontDiffuser로 글자 이미지를 생성하고, OpenCV로 후처리·SVG 벡터화를 수행한 뒤 FontForge로 폰트를 컴파일합니다.

현재 스타일 선택 규칙은 `매·실·효·소·이·새·별·조·영·화` 힌트를 기준으로 작성되어 있습니다. 한 글자씩 잘라 라벨링한 입력을 사용하며, 스캔 페이지를 자동으로 인식하거나 분할하는 제품 UI는 없습니다.

## 구성

| 경로 | 실제 역할 |
| --- | --- |
| `src/preprocess/refine_hints.py` | `data/original/`의 글자별 크롭에서 후보를 선택하고 128px 스타일 힌트 생성 |
| `src/preprocess/label_glyphs.py` | `data/raw/`의 직접 라벨링한 크롭을 전처리. CRAFT/OCR 자동 라벨링은 사용하지 않음 |
| `src/model/generator.py` | EUC-KR 문자셋 구성, 구조에 가까운 단일 힌트 선택, FontDiffuser 추론, 적응형 후처리 |
| `src/model/generate_bestofn.py` | 글자별 후보 4개를 생성해 영역 소실 수와 SCR 스타일 유사도로 선택 |
| `src/postprocess/` | 표적 재생성, 잡티 제거, 획 두께 정규화, 기울기·원본 힌트 적용, 수동 후보 선택 |
| `src/eval/` | 프루프시트, 잉크·연결요소 검사, 영역 검사, 스타일 유사도와 A/B 비교 |
| `src/font_builder/vectorize.py` | 외곽선 단순화와 Catmull-Rom 보간으로 3차 베지어 SVG 생성 |
| `src/font_builder/build_font.py` | FontForge로 SVG를 불러와 비례폭·공백 글리프를 설정하고 TTF 생성 |
| `src/model/fontdiffuser/` | 별도 upstream Git 서브모듈 |

## 환경 준비

주요 실행 경로는 Windows와 CUDA GPU를 전제로 합니다. 콘텐츠 템플릿은 `C:\Windows\Fonts\malgun.ttf`를 사용하며, `generate_bestofn.py`와 표적 수리 스크립트는 기본적으로 `cuda:0`을 사용합니다. `generator.py`만 CUDA가 없으면 CPU를 선택합니다.

```powershell
git clone --recurse-submodules https://github.com/oocheol/handFont.git
cd handFont

# 기존 복제본에서 서브모듈이 없다면:
git submodule update --init --recursive

# 사용하는 GPU/CUDA에 맞는 torch와 torchvision을 먼저 설치합니다.
py -3 -m pip install -r src/model/fontdiffuser/requirements.txt
py -3 -m pip install Pillow numpy fonttools torch torchvision easyocr
```

루트 프로젝트에는 환경 전체를 고정한 lockfile이 없습니다. 위 명령은 upstream 요구사항과 로컬 스크립트 의존성을 설치하는 안내이며, 모든 Python/CUDA 조합에서의 실행을 보장하지 않습니다. 현재 로컬 검증 환경은 Python 3.12.3, PyTorch 2.5.1+cu121, NVIDIA GTX 1650입니다.

FontForge는 별도로 설치해야 합니다. 아래 빌드 명령은 Windows 설치본의 `ffpython.exe`를 사용합니다. 일반 Python에 `fonttools`만 설치해도 `build_font.py`를 실행할 수 있는 것은 아닙니다.

FontDiffuser README의 Model Zoo에서 아래 체크포인트를 확보해 `weights/`에 배치합니다. 파일은 Git에 포함하지 않습니다.

```text
weights/
  unet.pth
  content_encoder.pth
  style_encoder.pth
  scr_210000.pth        # Best-of-N 및 스타일 평가에 필요
```

`src/model/download_weights.py`는 이전 MX-Font용 임시 다운로드 스크립트입니다. 현재 FontDiffuser 체크포인트를 준비하는 도구가 아닙니다.

## 실행

모든 명령은 저장소 루트에서 실행합니다. 스타일 입력은 `data/style/AC00.png`처럼 글자 하나의 유니코드 16진수를 파일명으로 쓰는 PNG입니다. 현재 힌트는 10자이며, 원본·힌트·가중치는 로컬 파일로 준비해야 합니다.

원본 크롭으로 힌트를 다시 만들 때만 다음을 실행합니다. `data/original/`에는 `매1.jpg`, `매2.jpg`, `실.jpg`처럼 글자별 파일을 두며, 완성형 한글 한 글자를 라벨로 사용합니다.

```powershell
py -3 src/preprocess/refine_hints.py
```

생성부터 빌드까지의 기본 순서입니다. 생성 단계는 `generator.py`와 `generate_bestofn.py` 중 하나를 선택합니다. Best-of-N은 글자당 추론을 4회 수행하므로 더 많은 시간이 필요합니다.

```powershell
# 단일 후보 생성
py -3 src/model/generator.py

# 또는, 구조 검사와 스타일 유사도를 이용한 후보 4개 선택
# py -3 src/model/generate_bestofn.py

py -3 src/postprocess/repair_missing_strokes.py
py -3 src/postprocess/remove_strays.py
py -3 src/postprocess/normalize_stroke_weight.py
py -3 src/postprocess/finalize_style.py
py -3 src/font_builder/vectorize.py
& "C:\Program Files\FontForgeBuilds\bin\ffpython.exe" src/font_builder/build_font.py output/svgs output/Font/MyHandWriting.ttf

py -3 src/eval/proof_sheet.py
py -3 src/eval/component_check.py output/images
```

후처리 스크립트는 `output/images/`를 직접 갱신합니다. 특히 `finalize_style.py`는 실행마다 기울기를 추가하므로 동일 이미지에 반복 실행하지 않습니다. 재실행·비교가 필요하면 입력과 산출물 폴더를 먼저 백업합니다.

| 산출물 | 경로 |
| --- | --- |
| 생성 글자 PNG | `output/images/<UNICODE>.png` |
| 벡터 외곽선 SVG | `output/svgs/<UNICODE>.svg` |
| 빌드한 폰트 | `output/Font/MyHandWriting.ttf` |
| 검수 이미지·리포트 | `output/eval/` |

생성기는 한글 2,350자만 생성합니다. 빌더는 SVG 폴더에 있는 글자를 가져오고 공백을 추가하므로, 별도 문장부호 SVG가 있으면 함께 포함됩니다. 영문·숫자 및 한글 전체 11,172자 생성은 구현되어 있지 않습니다.

## 검증과 한계

2026-09-30 로컬 파일을 읽어 확인한 결과:

- `output/images/`와 `output/svgs/`: 한글 2,350자와 문장부호 6자, 목표 한글 누락 0.
- `output/Font/MyHandWriting.ttf`: 한글 2,350자와 문장부호 6자·공백, 빈 한글 글리프 0.
- 일반 결함 검사: 2,356자 중 41자 의심. 영역 검사: 한글 2,350자 중 1,245자 표시.
- 모델 로드와 1자 GPU 추론을 확인했으며, 이 정리 과정에서 전체 문자를 새로 생성하지 않았습니다.

두 검사는 서로 다른 휴리스틱입니다. 표시된 수는 실제 오자 수·정확도·판매 가능한 품질을 증명하지 않으며, 폰트 문장 렌더링과 육안 검수가 필요합니다. SCR 유사도 역시 글자의 의미가 정확하다는 판정이 아닙니다.

`data/`, `weights/`와 대부분의 `output/` 파일은 Git에서 제외됩니다. Git에 기존부터 들어 있던 `output/handFont.ttf`는 별도 구버전 산출물이며 현재 빌드 결과가 아닙니다.

## English

HandFont is a local Python pipeline that generates the 2,350 KS X 1001 Hangul syllables from labeled handwriting crops. It uses FontDiffuser for inference, OpenCV for processing and cubic Bézier SVG outlines, and FontForge for TTF compilation.

Clone with `--recurse-submodules`, prepare the four checkpoints listed above, and run the PowerShell commands from the repository root. Windows font paths and CUDA defaults are used in the main workflow. The ten current reference characters are `매·실·효·소·이·새·별·조·영·화`; scanned-page recognition and automatic OCR labeling are not implemented.

Choose either single-candidate generation or best-of-four generation, then run repair, stray removal, stroke normalization, style finalization, vectorization, and the FontForge `ffpython.exe` build. Style finalization adds slant each time it runs. Back up images before rerunning postprocessing.

The generated font is `output/Font/MyHandWriting.ttf`. Full Unicode Hangul coverage, Latin letters, and digits are outside the current implementation. Automated defect and style scores are diagnostic signals, and character accuracy requires visual review. Input data, checkpoints, and most outputs are intentionally not included in Git; `output/handFont.ttf` is an older tracked artifact.

## License

HandFont's code uses the MIT license in `LICENSE`. FontDiffuser is a separate upstream dependency. Check the upstream code, checkpoint and source-font terms before distributing generated fonts; the root MIT license does not establish rights to those materials.
