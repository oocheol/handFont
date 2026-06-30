import os
import sys
import xml.etree.ElementTree as ET
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.svgPathPen import SVGPathPen

def parse_svg_path(svg_path):
    """
    SVG 파일에서 <path> 태그의 d 속성(패스 데이터) 목록을 추출합니다.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # SVG 네임스페이스 처리
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        paths = []
        
        # 네임스페이스가 있는 경우와 없는 경우 둘 다 대응
        for path_elem in root.findall('.//svg:path', namespaces):
            d = path_elem.get('d')
            if d:
                paths.append(d)
        if not paths:
            for path_elem in root.findall('.//path'):
                d = path_elem.get('d')
                if d:
                    paths.append(d)
                    
        return paths
    except Exception as e:
        print(f"[경고] SVG 파싱 실패 ({svg_path}): {e}")
        return []

def build_font_from_svgs(svg_dir, output_font_path, font_name="HandFont"):
    """
    fontTools 라이브러리를 사용하여 SVG 파일들로부터
    외부 프로그램(FontForge 등) 의존성 없이 순수 파이썬으로 TTF 폰트 파일을 빌드합니다.
    """
    print(f"새 글꼴 '{font_name}' 빌드를 준비하는 중...")
    
    svg_files = [f for f in os.listdir(svg_dir) if f.lower().endswith('.svg')]
    if not svg_files:
        print("[오류] 빌드할 SVG 파일이 폴더에 존재하지 않습니다.")
        return False
        
    print(f"총 {len(svg_files)}개의 SVG 글리프 파일을 읽어들입니다...")

    # TrueType 글꼴 정보 기본값 설정
    # TrueType의 표준 글리프 크기는 보통 1024 또는 2048 Em-Square를 사용합니다.
    # 우리의 원본 SVG 크기는 128x128 이므로, 이를 1024x1024 스케일로 변환(스케일 팩터: 8)하고
    # y축이 뒤집힌 폰트 규격에 맞춰 상하 반전 및 베이스라인 정렬을 수행합니다.
    # (SVG는 좌측 상단이 (0,0)이지만, TTF 폰트는 좌측 하단이 기준이며 베이스라인 위로 올라갑니다).
    
    scale_factor = 1024 / 128.0
    descender = -150  # 베이스라인 아래로 내릴 오프셋 (한글 받침 등 대비)
    y_offset = 1024 - 150  # 상하 반전 후 배치 오프셋
    
    # 폰트 빌더를 위한 글리프 사전
    glyphs = {}
    # 유니코드 매핑 테이블
    cmap = {}
    
    # 기본 빈 글리프(.notdef) 생성 필수
    # 폰트 규격상 정의되지 않은 글자를 보여줄 .notdef 글리프가 첫 번째 인덱스에 있어야 합니다.
    pen = TTGlyphPen(None)
    pen.moveTo((100, 0))
    pen.lineTo((100, 800))
    pen.lineTo((200, 800))
    pen.lineTo((200, 0))
    pen.closePath()
    glyphs['.notdef'] = pen.glyph()
    
    success_count = 0
    
    for filename in svg_files:
        name_part = os.path.splitext(filename)[0]
        try:
            if len(name_part) == 1:
                codepoint = ord(name_part)
            else:
                hex_str = name_part.upper().replace('U+', '')
                codepoint = int(hex_str, 16)
        except ValueError:
            continue
            
        svg_path = os.path.join(svg_dir, filename)
        paths = parse_svg_path(svg_path)
        
        # 각 글자 고유의 glyph_name 정의 (예: uniAC00)
        glyph_name = f"uni{codepoint:04X}"
        cmap[codepoint] = glyph_name
        
        # 글리프 드로잉 펜 생성
        pen = TTGlyphPen(glyphs)
        
        for path_d in paths:
            # d 값을 분해하여 그리기 명령으로 변환
            # (직선 획 M, L, Z만 사용하므로 단순 수동 변환도 가능)
            # 여기서는 견고성을 위해 수동 파싱 후 그리기 좌표 변환을 거칩니다.
            commands = re.findall(r'([MLZ])\s*([^MLZ]*)', path_d)
            
            for cmd, args_str in commands:
                args = [float(x) for x in args_str.split() if x]
                if cmd == 'M':
                    # 폰트 y축 변환: (128 - y) * scale + descender
                    tx = args[0] * scale_factor
                    ty = (128 - args[1]) * scale_factor + descender
                    pen.moveTo((tx, ty))
                elif cmd == 'L':
                    tx = args[0] * scale_factor
                    ty = (128 - args[1]) * scale_factor + descender
                    pen.lineTo((tx, ty))
                elif cmd == 'Z':
                    pen.closePath()
                    
        # 바이너리 글리프로 컴파일하여 저장
        try:
            glyphs[glyph_name] = pen.glyph()
            success_count += 1
        except Exception as e:
            # 빈 글리프로 대체하여 빌드 깨짐 방지
            pen = TTGlyphPen(glyphs)
            glyphs[glyph_name] = pen.glyph()
            
    print(f" -> 글리프 컴파일 완료: 성공 {success_count}개")

    # FontBuilder를 사용하여 TTF 파일 메타데이터 빌드
    fb = FontBuilder(1024, isTTF=True)
    
    # 글리프 이름 순서 정렬
    glyph_order = ['.notdef'] + sorted(list(cmap.values()))
    fb.setupGlyphOrder(glyph_order)
    
    # 글리프 데이터 등록
    fb.setupGlyf(glyphs)
    
    # 유니코드 cmap 등록
    fb.setupCharacterMap(cmap)
    
    # 수평 메트릭스(폭 정보) 등록
    # 모든 글자 폭(Advance width)을 1024로 고정하여 Monospace 한글 폰트로 지정
    metrics = {gname: (1024, 0) for gname in glyph_order}
    fb.setupHorizontalMetrics(metrics)
    
    # 표준 메타데이터 테이블 등록 (이름, 스타일 등)
    name_strings = {
        'familyName': font_name,
        'styleName': 'Regular',
        'uniqueFontIdentifier': f'{font_name} Regular; 2026-present',
        'fullName': font_name,
        'version': 'Version 1.000',
        'psName': f'{font_name}-Regular',
    }
    fb.setupNameTable(name_strings)
    
    # 표준 OS/2, hhea, head, maxp, post 테이블 구성 (가장 중요)
    fb.setupHead(fontRevision=1.0)
    fb.setupMaxp()
    fb.setupOS2(sTypoAscender=874, sTypoDescender=-150)
    fb.setupHorizontalHeader(ascent=874, descent=-150)
    fb.setupPost(italicAngle=0, underlinePosition=-75, underlineThickness=50)
    
    # 최종 파일 저장
    output_dir = os.path.dirname(output_font_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fb.save(output_font_path)
    print(f"[성공] 폰트 컴파일 완료! 출력 경로: {output_font_path}")
    return True

import re
if __name__ == "__main__":
    if len(sys.argv) < 3:
        # 기본 디렉토리 빌드 테스트
        build_font_from_svgs("output/svgs", "output/Font/MyHandWriting.ttf")
    else:
        build_font_from_svgs(sys.argv[1], sys.argv[2])
