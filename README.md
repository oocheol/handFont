# HandFont

Few-shot 이미지 기반 한글 손글씨 폰트 생성 및 패키징 파이프라인 프로젝트입니다. 

단 몇 장(예: 5장)의 손글씨 스타일 힌트 이미지를 통해 한글 2,350자를 생성하고, 이를 컴퓨터에서 바로 설치하여 사용할 수 있는 `.ttf` 형식의 벡터 폰트 파일로 빌드합니다.

## 프로젝트 구조

```text
handFont/
│
├── LICENSE
├── README.md
├── .gitignore
│
├── data/              # 원본 스캔본 및 타겟 데이터셋 위치 (git 제외)
├── output/            # 생성된 폰트 이미지 및 최종 TTF 출력 폴더 (git 제외)
├── weights/           # AI 모델 가중치 폴더 (git 제외)
│
└── src/
    ├── preprocess/    # 이미지 전처리 및 개별 글자 자르기
    ├── model/         # Few-shot Font Generation 모델 추론부
    └── font_builder/  # 이진 이미지 벡터화 및 FontForge 기반 TTF 빌드
```

## 설치 및 요구사항

이 프로젝트를 구동하려면 다음 도구들이 필요합니다.

1. **Python 3.8+**
2. **Potrace**: 이미지를 벡터(SVG)로 변환하는 엔진
3. **FontForge**: 폰트 빌드를 위한 라이브러리 및 파이썬 모듈

## 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
