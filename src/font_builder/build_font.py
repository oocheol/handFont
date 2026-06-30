import os
import sys

# FontForge 모듈은 일반 Python에 기본 설치되지 않는 경우가 많으므로 예외 처리
try:
    import fontforge
except ImportError:
    fontforge = None

# 한글 상용 2,350자 목록을 정의 (일부 생략되었지만 필요시 확장 가능하도록 함)
# 실제 한글 2,350자 기본 세트는 대다수의 완성형 폰트 구성에 필수적임
# 여기서는 편의상 생성된 파일들의 이름(예: '가.svg' 혹은 'AC00.svg')을 읽어서 자동으로 글꼴에 주입하는 방식을 취합니다.

def build_font_from_svgs(svg_dir, output_font_path, font_name="HandFont"):
    if fontforge is None:
        print("[오류] 이 스크립트를 실행하려면 FontForge가 설치되어 있어야 하며, FontForge Python 환경(ffpython)에서 작동해야 합니다.")
        print("FontForge 다운로드: https://fontforge.org/")
        return False

    print("새 글꼴 파일을 생성하는 중...")
    font = fontforge.font()
    font.fontname = font_name
    font.fullname = font_name
    font.familyname = font_name
    font.encoding = 'UnicodeFull'

    svg_files = [f for f in os.listdir(svg_dir) if f.lower().endswith('.svg')]
    print(f"총 {len(svg_files)}개의 SVG 글리프 파일을 글꼴에 매핑하는 중...")

    success_count = 0
    for filename in svg_files:
        # 파일 이름은 글자 자체(예: '가.svg') 또는 유니코드 16진수 문자열(예: 'AC00.svg')로 구성되어야 함
        name_part = os.path.splitext(filename)[0]
        
        try:
            # 1. 파일 이름 자체가 단일 문자인 경우 (예: '가')
            if len(name_part) == 1:
                codepoint = ord(name_part)
            # 2. 파일 이름이 유니코드 헥스 값인 경우 (예: 'AC00' 또는 'U+AC00')
            else:
                hex_str = name_part.upper().replace('U+', '')
                codepoint = int(hex_str, 16)
        except ValueError:
            print(f"[경고] 매핑 불가능한 파일 이름 건너뜀: {filename}")
            continue

        # 글리프 생성 및 설정
        glyph = font.createChar(codepoint)
        svg_path = os.path.join(svg_dir, filename)
        
        try:
            glyph.importOutlines(svg_path)
            # 글자 크기 및 오프셋 기본 정렬
            glyph.autoTrace()
            glyph.autoWidth()
            success_count += 1
        except Exception as e:
            print(f"[오류] 글리프 가져오기 실패 ({filename}): {e}")

    print(f"매핑 완료! 성공: {success_count}개 글리프")
    
    # 출력 디렉토리 확인
    output_dir = os.path.dirname(output_font_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 폰트 파일 저장 (.ttf)
    try:
        font.generate(output_font_path)
        print(f"글꼴 파일이 성공적으로 생성되었습니다: {output_font_path}")
        return True
    except Exception as e:
        print(f"[오류] 폰트 파일 저장 실패: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: ffpython build_font.py <svg_directory> <output_font.ttf>")
    else:
        build_font_from_svgs(sys.argv[1], sys.argv[2])
