import os
import sys

# 이 스크립트는 FontForge 파이썬 환경(ffpython)에서 작동해야 합니다.
try:
    import fontforge
except ImportError:
    fontforge = None

def build_font_from_svgs(svg_dir, output_font_path, font_name="HandFont"):
    if fontforge is None:
        print("[오류] 이 스크립트를 실행하려면 FontForge 파이썬 환경(ffpython)에서 작동해야 합니다.")
        print("실행 예: 'C:\\Program Files\\FontForgeBuilds\\ffpython.exe' build_font.py <svg_dir> <output_ttf>")
        return False

    print(f"새 글꼴 '{font_name}' 빌드를 시작합니다 (FontForge 엔진)...")
    font = fontforge.font()
    font.fontname = font_name
    font.fullname = font_name
    font.familyname = font_name
    font.encoding = 'UnicodeFull'

    # 기본 필수 글리프 설정 (.notdef, space 등)
    # FontForge는 자동으로 기본 글꼴 구조를 생성하므로 유니코드 cmap 매핑만 맞춰주면 됩니다.
    svg_files = [f for f in os.listdir(svg_dir) if f.lower().endswith('.svg')]
    print(f"총 {len(svg_files)}개의 SVG 글리프 파일을 읽어들입니다...")

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

        # 글리프 생성 및 SVG 외곽선 주입
        glyph = font.createChar(codepoint)
        svg_path = os.path.join(svg_dir, filename)
        
        try:
            # SVG 불러오기
            glyph.importOutlines(svg_path)
            
            # 외곽선 방향 보정 및 겹침 제거 (ㅁ, ㅇ 등 내부 구멍 뚫림 완벽 보장)
            glyph.correctDirection()
            glyph.removeOverlap()
            
            # 고정 너비 지정 (Monospace)
            glyph.width = 1024
            success_count += 1
        except Exception as e:
            print(f"[경고] 글리프 주입 실패 ({filename}): {e}")

    print(f" -> 컴파일 완료: 성공 {success_count}개")

    # 폰트 출력 디렉토리 보장
    output_dir = os.path.dirname(output_font_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 윈도우 OS에서 100% 정상 인식하는 TTF 글꼴 생성
        font.generate(output_font_path)
        print(f"[성공] 폰트 컴파일 완료! 출력 경로: {output_font_path}")
        return True
    except Exception as e:
        print(f"[오류] 폰트 파일 저장 실패: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        build_font_from_svgs("output/svgs", "output/Font/MyHandWriting.ttf")
    else:
        build_font_from_svgs(sys.argv[1], sys.argv[2])
