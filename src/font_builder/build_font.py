import os
import sys

# 이 스크립트는 FontForge 파이썬 환경(ffpython)에서 작동해야 합니다.
try:
    import fontforge
except ImportError:
    fontforge = None

# em=1000/ascent=800/descent=200은 FontForge 기본값과 동일하게 명시 - 실측 결과
# 256px SVG 콘텐츠가 이 비율로 스케일될 때 획이 ascent/descent 범위 안에 자연스럽게
# 들어옴을 확인함(가장 긴 획 기준 y범위 대략 -85~680).
FONT_EM = 1000
FONT_ASCENT = 800
FONT_DESCENT = 200
SIDE_BEARING = 40           # 좌우 여백 (em 1000 기준 4%)
DEFAULT_EMPTY_WIDTH = 500   # 잉크가 없는 예외 글리프의 기본 advance width


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
    font.em = FONT_EM
    font.ascent = FONT_ASCENT
    font.descent = FONT_DESCENT

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
            # (viewBox 256 기준 콘텐츠가 FontForge에 의해 em=1000 박스로 자동
            # 스케일됨을 실측 확인함 - 별도 좌표 변환 불필요)
            glyph.importOutlines(svg_path)

            # 외곽선 방향 보정 및 겹침 제거 (ㅁ, ㅇ 등 내부 구멍 뚫림 완벽 보장)
            glyph.correctDirection()
            glyph.removeOverlap()

            # 잉크 bbox 기반 비례 너비: 이전 버전은 전 글리프 width=1024 고정
            # (em=1000보다도 넓은) 모노스페이스였음 - 손글씨 커서브 서체에서
            # 글자 폭이 제각각인데 균일 간격을 강제해 부자연스러운 자간이 되던
            # 원인. 좌우 여백만 고정하고 실제 잉크 폭만큼만 advance width를 준다.
            xmin, ymin, xmax, ymax = glyph.boundingBox()
            if xmax > xmin:
                glyph.left_side_bearing = SIDE_BEARING
                glyph.right_side_bearing = SIDE_BEARING
            else:
                glyph.width = DEFAULT_EMPTY_WIDTH  # 잉크 없는 예외 글리프 안전장치
            success_count += 1
        except Exception as e:
            print(f"[경고] 글리프 주입 실패 ({filename}): {e}")

    # 스페이스 글리프 - 이전 버전엔 아예 없어서 일반 텍스트 편집기에서 타이핑 시
    # 공백이 폭 0으로 처리되던 문제
    space = font.createChar(0x20, 'space')
    space.width = int(font.em * 0.4)

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
    if len(sys.argv) == 1:
        success = build_font_from_svgs("output/svgs", "output/Font/MyHandWriting.ttf")
    elif len(sys.argv) == 3:
        success = build_font_from_svgs(sys.argv[1], sys.argv[2])
    else:
        print("사용법: build_font.py [<svg_dir> <output_ttf>]")
        sys.exit(2)
    sys.exit(0 if success else 1)
