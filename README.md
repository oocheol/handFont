# HandFont

Few-shot 이미지 기반 한글 손글씨 폰트 생성 및 패키징 파이프라인 프로젝트입니다.

단 5장의 손글씨 스타일 힌트 이미지(또는 스캔본 이미지)를 입력으로 받아, 전체 한글 상용 2,350자를 자동 생성한 후 컴퓨터에서 사용 가능한 `.ttf` 벡터 폰트 파일로 빌드합니다.

## 프로젝트 폴더 구조

```text
handFont/
│
├── LICENSE
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/           # 원본 스캔본 5장 배치
│   └── style/         # 원본에서 전처리하여 크로핑한 힌트 글자 (5개 이미지)
│
├── output/
│   ├── images/        # AI 모델이 생성한 2,350자의 개별 PNG 파일
│   ├── svgs/          # potrace를 통해 벡터화된 SVG 파일
│   └── Font/          # 최종 생성된 TTF 폰트 파일 저장 위치
│
└── src/
    ├── preprocess/
    │   └── crop_glyphs.py  # 스캔본에서 개별 힌트 글자 이미지 분할/정규화
    ├── model/
    │   └── generator.py    # Few-shot Font Generation 모델 추론 인터페이스
    └── font_builder/
        ├── vectorize.py    # PNG 이미지를 Potrace로 SVG 변환
        └── build_font.py   # SVG들을 취합하여 FontForge로 TTF 폰트 컴파일
```

## 요구사항 및 환경 구축

프로젝트 구동을 위해 다음 프로그램 설치가 필요합니다.

### 1. 외부 도구 설치
* **Potrace**: 비트맵 이미지를 벡터 그래픽으로 변환하는 도구
  * [다운로드 및 설치](https://potrace.sourceforge.net/) 후 CLI에서 실행 가능하도록 `PATH` 환경 변수에 추가합니다.
* **FontForge**: 폰트 편집 및 컴파일 엔진
  * [다운로드 및 설치](https://fontforge.org/) 후 FontForge에 내장된 파이썬 실행기(`ffpython`)를 사용해 글꼴 빌드 스크립트를 작동시킵니다.

### 2. Python 라이브러리 설치
```bash
pip install Pillow torch torchvision
```

## 사용법 (파이프라인 실행 순서)

### Step 1: 손글씨 힌트 이미지 전처리
보유한 5개의 원본 스캔본 이미지를 `data/raw/` 폴더에 넣고, 글리프 분할 스크립트를 실행하여 5개의 대표 스타일 힌트 이미지를 추출합니다.
```bash
python src/preprocess/crop_glyphs.py
```
*추출된 이미지는 `data/style/` 폴더에 저장됩니다.*

### Step 2: AI 기반 전체 한글 자소 이미지 생성
준비된 5개의 스타일 힌트 이미지와 사전 학습된 Few-shot Generator 모델을 바탕으로 한글 상용 2,350자 전체의 손글씨 스타일 이미지를 생성합니다.
```bash
python src/model/generator.py
```
*생성된 2,350개의 PNG 파일은 `output/images/` 폴더에 유니코드 번호(예: `AC00.png`)로 저장됩니다.*

### Step 3: 이미지 벡터화 (SVG 변환)
생성된 PNG 이미지들을 Potrace 도구를 활용해 외곽선 벡터 그래픽(SVG)으로 일괄 변환합니다.
```bash
python src/font_builder/vectorize.py
```
*변환된 SVG 파일들은 `output/svgs/` 폴더에 저장됩니다.*

### Step 4: TTF 폰트 파일 컴파일
FontForge 환경을 사용해 SVG 외곽선들을 유니코드 인코딩에 맞추어 하나의 `.ttf` 파일로 빌드합니다.
```bash
ffpython src/font_builder/build_font.py output/svgs/ output/Font/MyHandWriting.ttf
```
*컴파일 결과물인 `MyHandWriting.ttf`를 시스템에 설치하여 자유롭게 사용하실 수 있습니다.*

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 자유롭게 사용 및 배포할 수 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하십시오.
