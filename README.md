# HandFont

Few-shot 이미지 기반 한글 손글씨 폰트 생성 및 패키징 파이프라인 프로젝트입니다.
A few-shot image-based Korean handwriting font generation and packaging pipeline project.

단 6장의 손글씨 스타일 힌트 이미지(또는 스캔본 이미지)를 입력으로 받아, 전체 한글 상용 2,350자를 자동 생성한 후 컴퓨터에서 사용 가능한 `.ttf` 벡터 폰트 파일로 빌드합니다.
Takes just 6 handwriting style hint images (or scan images) as input, automatically generates all 2,350 common Korean characters, and builds them into a `.ttf` vector font file ready to use on your computer.

---

## 한국어 (Korean)

### 프로젝트 폴더 구조

```text
handFont/
│
├── LICENSE
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/           # 원본 스캔본 배치
│   └── style/         # 원본에서 전처리하여 크로핑한 힌트 글자 (유니코드 16진수 파일명)
│
├── output/
│   ├── images/        # 생성된 2,350자의 개별 PNG 파일
│   ├── svgs/          # 순수 파이썬(OpenCV)을 통해 벡터화된 SVG 파일
│   └── Font/          # 최종 생성된 TTF 폰트 파일 저장 위치
│
└── src/
    ├── preprocess/
    │   ├── crop_glyphs.py   # 기본 이미지 분할
    │   └── label_glyphs.py  # CRAFT 텍스트 탐지를 적용한 정밀 글리프 라벨링 및 전처리
    ├── model/
    │   └── generator.py     # Few-shot Font Generation 모델 추론 및 시뮬레이션
    └── font_builder/
        ├── vectorize.py     # PNG 이미지를 순수 파이썬(OpenCV)으로 SVG 변환
        └── build_font.py    # fontTools 라이브러리로 SVG를 TTF 폰트로 컴파일
```

### 요구사항 및 환경 구축
* **Python 3.8+**
```bash
pip install Pillow torch torchvision numpy opencv-python easyocr fonttools
```

### 사용법 (파이프라인 실행 순서)

1. **손글씨 힌트 이미지 전처리 및 라벨링**
   * 원본 스캔본 이미지들을 `data/raw/` 폴더에 `1(매실효소).jpg`, `2(이새별 이새별).jpg` 와 같이 파일명에 글자를 포함해 저장합니다.
   * 그 다음, 아래 스크립트를 구동하여 AI 기반 CRAFT 텍스트 탐지 모델로 정교하게 한 글자씩 쪼개어 유니코드 파일로 저장합니다.
   ```bash
   python src/preprocess/label_glyphs.py
   ```
2. **한글 2,350자 글씨 이미지 생성**
   * 정제된 힌트 글자의 평균 두께와 번짐 스타일을 계산하여 전체 2,350자의 스타일이 적용된 이미지를 대량 복사/생성합니다.
   ```bash
   python src/model/generator.py
   ```
3. **이미지 벡터화 (SVG 변환)**
   * 생성된 PNG 이미지들을 외곽선 벡터 그래픽(SVG)으로 일괄 변환합니다. (외부 프로그램 `potrace` 없이 OpenCV를 사용해 100% 로컬 가동)
   ```bash
   python src/font_builder/vectorize.py
   ```
4. **TTF 폰트 파일 컴파일**
   * `fonttools` 라이브러리를 사용해 SVG 외곽선들을 하나의 `.ttf` 파일로 컴파일합니다.
   ```bash
   python src/font_builder/build_font.py
   ```
   *최종 결과물은 `output/Font/MyHandWriting.ttf` 경로에 생성됩니다.*

---

## English

### Project Folder Structure

```text
handFont/
│
├── LICENSE
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/           # Place original scan files here
│   └── style/         # Cropped hint characters from raw images (named by Unicode hex)
│
├── output/
│   ├── images/        # Generated 2,350 individual PNG files
│   ├── svgs/          # Vectorized SVG files via pure Python (OpenCV)
│   └── Font/          # Location of the finally compiled TTF font file
│
└── src/
    ├── preprocess/
    │   ├── crop_glyphs.py   # Basic image cropping
    │   └── label_glyphs.py  # Precision glyph labeling using CRAFT text detection
    ├── model/
    │   └── generator.py     # Few-shot Font Generation model inference and simulation
    └── font_builder/
        ├── vectorize.py     # Converts PNG to SVG using pure Python (OpenCV)
        └── build_font.py    # Compiles SVG to TTF font via fontTools library
```

### Prerequisites & Environment Setup
* **Python 3.8+**
```bash
pip install Pillow torch torchvision numpy opencv-python easyocr fonttools
```

### Usage (Pipeline Order)

1. **Preprocessing & Labeling Handwriting Hint Images**
   * Place the raw scan images in `data/raw/` and name them containing target text like `1(매실효소).jpg`, `2(이새별 이새별).jpg`.
   * Run the script below to detect and crop characters precisely into individual Unicode hex files using the AI-based CRAFT detector:
   ```bash
   python src/preprocess/label_glyphs.py
   ```
2. **Generating 2,350 Common Korean Characters**
   * Simulates/generates all 2,350 Korean character images matching the average thickness and curvature of your 10 style hints:
   ```bash
   python src/model/generator.py
   ```
3. **Vectorizing Images (Convert to SVG)**
   * Batch-converts the generated PNG images into vector graphics (SVG) outlines. (Runs 100% locally using OpenCV without needing external tools like `potrace`)
   ```bash
   python src/font_builder/vectorize.py
   ```
4. **Compiling TTF Font File**
   * Compiles the vectorized SVG outlines into a single `.ttf` file using the `fonttools` library:
   ```bash
   python src/font_builder/build_font.py
   ```
   *The final output will be created at `output/Font/MyHandWriting.ttf`.*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
