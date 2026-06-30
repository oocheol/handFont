import os
import sys
import urllib.request

# MX-Font 한글 pretrained 가중치 파일 예시 다운로드 링크 (또는 LF-Font 등)
# Clova AI FFG Unified Framework 등의 체크포인트 URL 지정 가능
WEIGHTS_URL = "https://huggingface.co/oocheol/handFont/resolve/main/mxfont_korean.pth" # 임시 허깅페이스 호스팅 경로나 대체 공개 가중치 링크
OUTPUT_PATH = "weights/mxfont_korean.pth"

def download_weights():
    os.makedirs("weights", exist_ok=True)
    
    if os.path.exists(OUTPUT_PATH):
        print(f"[정보] 이미 가중치 파일이 존재합니다: {OUTPUT_PATH}")
        return True
        
    print(f"[다운로드 시작] AI 가중치를 다운로드합니다... (저장위치: {OUTPUT_PATH})")
    print("이 작업은 약 200MB~500MB 용량의 가중치를 가져오므로 시간이 다소 걸릴 수 있습니다.")
    
    try:
        # 진행률을 보여주는 다운로드 함수
        def progress_bar(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = read_so_far * 1e2 / total_size
                s = f"\r   -> 다운로드 중: {percent:.2f}% ({read_so_far / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)"
                sys.stdout.write(s)
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r   -> 다운로드 중: {read_so_far / (1024*1024):.1f}MB")
                sys.stdout.flush()

        urllib.request.urlretrieve(WEIGHTS_URL, OUTPUT_PATH, progress_bar)
        print("\n[성공] 가중치 다운로드 완료!")
        return True
    except Exception as e:
        print(f"\n[오류] 다운로드 실패: {e}")
        print("네트워크 연결을 확인하시거나, 가중치 파일을 직접 weights/mxfont_korean.pth 에 다운로드 받아주십시오.")
        return False

if __name__ == "__main__":
    download_weights()
