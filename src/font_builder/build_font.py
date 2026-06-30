import os
import sys
import xml.etree.ElementTree as ET
import re
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

def parse_svg_path(svg_path):
    """
    SVG 파일에서 <path> 태그의 d 속성(패스 데이터) 목록을 추출합니다.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        paths = []
        
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
    윈도우 환경에서도 100% 정상 작동하도록 필수 규격을 만족하는 TTF 폰트 파일을 빌드합니다.
    """
    print(f"새 글꼴 '{font_name}' 빌드를 준비하는 중...")
    
    svg_files = [f for f in os.listdir(svg_dir) if f.lower().endswith('.svg')]
    if not svg_files:
        print("[오류] 빌드할 SVG 파일이 폴더에 존재하지 않습니다.")
        return False
        
    print(f"총 {len(svg_files)}개의 SVG 글리프 파일을 읽어들입니다...")

    scale_factor = 1024 / 128.0
    descender = -150
    
    # 1. 폰트 글리프 데이터 사전 및 유니코드 cmap 테이블 준비
    glyphs = {}
    cmap = {}
    
    # 2. 필수 글리프 .notdef, .null, space 생성
    # 윈도우 글꼴 엔진은 폰트 파일 검증 시 아래 세 개의 글리프가 올바르게 정의되어 있어야 승인합니다.
    
    # .notdef (정의되지 않은 글자 표시용 사각형 상자)
    pen = TTGlyphPen(None)
    pen.moveTo((100, 0))
    pen.lineTo((100, 800))
    pen.lineTo((200, 800))
    pen.lineTo((200, 0))
    pen.closePath()
    glyphs['.notdef'] = pen.glyph()
    
    # .null (널 문자)
    pen = TTGlyphPen(None)
    glyphs['.null'] = pen.glyph()
    cmap[0] = '.null'
    
    # space (공백 문자)
    pen = TTGlyphPen(None)
    glyphs['space'] = pen.glyph()
    cmap[32] = 'space'
    
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
            
        # space(32)나 null(0)은 이미 생성했으므로 건너뜀
        if codepoint in (0, 32):
            continue
            
        svg_path = os.path.join(svg_dir, filename)
        paths = parse_svg_path(svg_path)
        
        glyph_name = f"uni{codepoint:04X}"
        cmap[codepoint] = glyph_name
        
        pen = TTGlyphPen(glyphs)
        has_drawn = False
        
        for path_d in paths:
            commands = re.findall(r'([MLZ])\s*([^MLZ]*)', path_d)
            for cmd, args_str in commands:
                args = [float(x) for x in args_str.split() if x]
                if cmd == 'M':
                    tx = args[0] * scale_factor
                    ty = (128 - args[1]) * scale_factor + descender
                    pen.moveTo((tx, ty))
                    has_drawn = True
                elif cmd == 'L':
                    tx = args[0] * scale_factor
                    ty = (128 - args[1]) * scale_factor + descender
                    pen.lineTo((tx, ty))
                    has_drawn = True
                elif cmd == 'Z':
                    try:
                        pen.closePath()
                    except:
                        pass
            
            if has_drawn:
                try:
                    pen.closePath()
                except:
                    pass
                    
        try:
            glyphs[glyph_name] = pen.glyph()
            success_count += 1
        except Exception as e:
            pen = TTGlyphPen(glyphs)
            pen.closePath()
            glyphs[glyph_name] = pen.glyph()
            
    print(f" -> 글리프 컴파일 완료: 성공 {success_count}개")

    # FontBuilder를 사용하여 TTF 파일 빌드
    fb = FontBuilder(1024, isTTF=True)
    
    # 글리프 이름 정렬 (.notdef, .null, space가 처음에 와야 함)
    custom_glyphs = sorted([g for gname, g in cmap.items() if g not in ('.null', 'space')])
    glyph_order = ['.notdef', '.null', 'space'] + sorted(list(set(cmap.values()) - {'.null', 'space'}))
    fb.setupGlyphOrder(glyph_order)
    
    # 글리프 데이터 및 유니코드 캐릭터 맵 등록
    fb.setupGlyf(glyphs)
    fb.setupCharacterMap(cmap)
    
    # 수평 메트릭스(폭 정보 및 좌측 여백 lsb) 등록
    metrics = {gname: (1024, 0) for gname in glyph_order}
    # space의 너비는 조금 줄이고 .null은 0으로 지정
    metrics['space'] = (500, 0)
    metrics['.null'] = (0, 0)
    fb.setupHorizontalMetrics(metrics)
    
    # 메타데이터 이름 테이블 등록
    name_strings = {
        'familyName': font_name,
        'styleName': 'Regular',
        'uniqueFontIdentifier': f'{font_name} Regular; 2026-present',
        'fullName': font_name,
        'version': 'Version 1.000',
        'psName': f'{font_name}-Regular',
    }
    fb.setupNameTable(name_strings)
    
    # 윈도우 OS가 글꼴을 유효하게 렌더링하기 위한 필수 테이블 완성
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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        build_font_from_svgs("output/svgs", "output/Font/MyHandWriting.ttf")
    else:
        build_font_from_svgs(sys.argv[1], sys.argv[2])
