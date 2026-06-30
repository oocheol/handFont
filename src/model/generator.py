import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# 한글 상용 2,350자 기본 세트 (유니코드 KS X 1001 완성형 한글 중 자주 쓰이는 2,350자)
# 코드를 간결하게 유지하기 위해 완성형 2,350자 셋을 생성하는 헬퍼 목록을 제공합니다.
# 2,350자 목록을 동적으로 빌드하기 위해 대표적인 자모 결합 세트에서 추출하거나 파일에 저장할 수 있습니다.
# 여기서는 널리 쓰이는 한글 상용 2,350자 캐릭터 셋을 정의합니다.

COMMON_2350_HANGUL = (
    "가각간갇갈갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹걘걜걤걉걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괨괩괫괭괴괵괸굄굅굇굉교굔굘굠굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궎궤궝귀귁귄귈귐귑귓규균귤그극근글긁금급긋긍기긱긴길김깁깃깅깊까깍깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄깩꺼꺽껀껄껌껍껏껐껑께껜껨껫껭껴껸껼꼈꼍꼐꼬꼭꼰꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽘꽥꽹꽤꽨꽬꽵꽷꽹꾀꾄꾐꾑꾕뾔꾸꾹꾼꿀꿇꿈꿉꿋꿍꿔꿩꿰뀄뀌뀐뀔뀜뀝뀨끄끈끌끓끔끕끗끝끼끽낀낄낌낍낏낑나낙난날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냥녀녁년녈념념녑녔녕녘녜노녹논놀놂놈놉놋농높놓놔놘놜뇌뇐뇨뇩뇬뇰뇽누눅눈눌눔눕눗눙눠눴뉘뉜뉠뉨뉩뉴뉵뉼늄늉느늑는늘늙늚늠늡늦늪늬늰늴니닉닌닐님닙닛닝닢다닥단닫달닭닮닳담답닷닺닻닿대댁댄댈댐댑댓댔댕댜더덕던덜덞덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨도독돈돋돌돎돔돕돗동돛돝돠돤돨돼됀됄됌됍됐되된될됨됩됫뵤두둑둔둘둠둡둣둥둬뒀뒤뒨뒬뒴뒵뒷듀듄듈듐륭드득든들듦듬듭듯등디딕딘딜딤딥딧딩딪따딱딴딸땀땁땃땄땅때땍땐땔땜땝땟땠땡떠떡떤떨떪떰떱떳떴떵떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똔똴뙈뙨뙬뙵뙷뙹뙤뙨뚜뚝뚠뚤뚫뚱뚸뛔뛰뛴뛸뜀뜁뜅뜨뜯뜬뜰뜸뜹뜻띄띈띌띰띱띠띡띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸량러럭런럴럼럽럿렀렁렇게레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롈롑롓로록론롤롬롭롯롱롸롼뢔뢰뢴룐료룍룐룔룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏류륜률륨륭르륵른를름릅릇릉릐리릭린릴림립릿링마막만맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹먀먁먄먈먕머먹먼멀멂멈멉멋멍멎메멕멘멜멤멥멧멨멩며멱면멸몃몄명몌모목몬몰몲몸몹못몽뫄뫈뫘뫼묀묄묨묍묏묑묘묫무묵문묻물묽묾뭄뭅뭇뭉뭐뭑뭔뭘뭬뮈뮌륄뮌뮤뮨뮬뮴뮹므믁믄믈믐믑믓믕미믹민밀밈밉밋밍및밑바박반받발밝밞밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱌뱍뱐뱜뱡뱨버벅번벌벎범법벗벙베벡벤벨벰벱벳벴벵벼벽변별볍볏볐병볘보복본볼봄봅봇봉봐봔봘뵀뵈뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붜붤붸뷔뷕뷘뷜뷤뷥뷧뷰뷴뷸븀븡브븍븐블븜븝븟븡븨비빅빈빌빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵뺘뺙뺜뺨뺭빼빽뺀뺄뺌뺍뺏뺐뺑뺴뻐뻑뻔뻘뻠뻡뻣뻤뻥뻬뻭뻰뻴뻼뻿뼁뼈뼉뼌뼘뼝뼤뽀뽄뽈뽐뽑뽕뽜뽠뽸뾔뿌뿍뿐뿔뿜뿜뿽뿨쀄쀠쀤쀨쀼쁌쁭쁘쁜쁠쁰쁱쁲쁳쁴삔삘삠삡삣삥사삭산살삷삼삽삿샀상삺샅새색샌샐샘샘샙샛샜생샤샥샨샬샴샵샷샹섀섄서석선설섪섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속손솔솖솜솝솟송솥솨솩솬쇄쇈쇠쇤쇨쇰쇱쇳쇼쇽숀숄쇽숍숏숑수숙순숟술숨숩숫숭숯숴쉈쉐쉔쉘쉠쉡쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슝스슥슨슬슭슴습슷승시식신실싫심십싯싱싶싸싹싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩쌰쌱썅써썩썬썰썲썸썹썻썼썽쎄쎅쎈쎌쎔쎕쎗쎘쎵쎼쏘쏙쏜쏠쏨쏩쏫쏭쏴쏵쐈쐐쐰쐴쐼쐽쐿쑈쑤쑥쑨쑬쑴쑵쑻쑹쒀쒔쒜쒸쒼쓔쓩쓰쓱쓴쓸쓸쓺씀씁쓿씌씐씔씌씨씩씬씰씸씹씻씽아악안앉않알갉앓암압앗았앙앞애액앤앨앰앱앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠어억언얹얻얼얽얾엄업업엇었엉엌엎에엑엔엘엠엡엣엥여역연열엷염엽엿였영옆예옌옐옘옙옛옜오옥온올옭옮옳옴옵옷옹옻와왁완왈왐왑왔왕왜왝왠왬왱외획왼욀욈욉외욋요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읆음읍읏응읔읕읖의의이익인일읽잃임입잇있잉잎자작잔잖잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟝걔쟤져적전절젊점접젓졌정젖제젝젠젤젬젭젯젰젱져젹젼졀졈졉졌졍졔조족존졸졺좀좁졿종좌좍좐좠좽죄죈죌죔죕죗죠죡죤죵주죽준줄줆줌줍줏중줘줬줴쥐쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진질짊짐집짓징짖짙짚짜짝짠짤짧짬짭짭짰짱째짹짼짴쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쪘쩡쩨쩬쩨쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫘쫙쫠쫬쫴쬐쬔쬘쬠쬡쬫쭈쭉쭌쭘쭙쭝쭤쭸쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찌차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챡챤챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳈쳉쳐쳔쳤쳥쳬초촉촌촐촘숍촛총촤촨촬최쵠쵤쵬쵭쵯쵸쵼츄츈츌츔츠측츤츨츰츱츳층치칙친칠칡침칩칫칭치카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컬컴컵컷컸컹케켁켄켈켐켑켓켔켕켜켯켰켱켸코콕콘콜콤콥콧콩콰콱콴괄쾀쾅쾌쾐쾬쾽퀴퀵퀸퀼큄큅귓큐균큘크큭큰클큼클큽큿큵키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텼텡텨텬텼텽톄토톡톤톨톰톱톳통톼퇀돼퇴툔툘툠투툭툰툴툼툽틋퉁퉈퉜퉤튀튁튄튈튐튑튓튜튠튤튬륭트특튼틀틂틈틉틋틔틘틔티틱틴틸팀팁팃팅파팍판팔팖팜팝팟팠팡패팩팬팰팸팹팻팼팽퍄퍅평퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄펵폈평폐폐포폭폰폴폼폽폿퐁퐈퐘푀푄표푠푝푠표푸푹푼풀풂품풉풋풍풔풨퓌퓔퓨퓬퓰퓸륭프픈플픔픗피픽핀필필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헸헹혀혁현혈혐협혓혔형혜혠혤호혹혼홀홂홈홉홋홍화화확환활홤홥왔황홰홱왠홴홽회획회횔횜횝횟효횬횰횸횹후훅훈훌훑훔훕훗훙훠훵훼훽휀휄휨휩휫휘획휜휠휨휩휫휴휵휸휼흄륭흐흑흔흘흙흚흠흡흣흥희흰흰흴흼흽히힉힌힐힘힙힛힝"
)

class MockFewShotFontGenerator:
    def __init__(self, style_dir="data/style"):
        self.style_dir = style_dir
        self.style_images = {}
        
    def load_styles(self):
        """
        data/style/에 생성된 10개의 고유 손글씨 스타일 이미지(.png)를 로드합니다.
        """
        if not os.path.exists(self.style_dir):
            print(f"[경고] 스타일 힌트 디렉토리 {self.style_dir}가 없습니다.")
            return False
            
        style_files = [f for f in os.listdir(self.style_dir) if f.lower().endswith('.png')]
        for filename in style_files:
            char_hex = os.path.splitext(filename)[0]
            try:
                char = chr(int(char_hex, 16))
                img_path = os.path.join(self.style_dir, filename)
                self.style_images[char] = Image.open(img_path).convert('L')
            except Exception as e:
                print(f"[경고] 힌트 로드 오류 ({filename}): {e}")
                
        print(f"[정보] 총 {len(self.style_images)}개의 고유 스타일 힌트 로드 완료.")
        return len(self.style_images) > 0

    def generate_all_fonts(self, output_dir="output/images"):
        """
        10개의 손글씨 스타일 특징을 기반으로, 한글 2,350자의
        손글씨 스타일 이미지를 생성(모의 시뮬레이션)합니다.
        
        기본 바탕/돋움 폰트에서 글씨 획을 생성한 후, 사용자의 손글씨 이미지 특징인
        '획 두께', '윤곽 번짐'을 통계적 필터로 모사하여 실제 폰트 컴파일러가 작동하도록 만듭니다.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.load_styles()
        
        # 시스템 기본 한글 폰트 로드
        font_paths = [
            "C:\\Windows\\Fonts\\malgun.ttf",       # 맑은 고딕
            "C:\\Windows\\Fonts\\batang.ttc",       # 바탕체
            "C:\\Windows\\Fonts\\gulim.ttc"         # 굴림체
        ]
        
        target_font = None
        for path in font_paths:
            if os.path.exists(path):
                target_font = path
                break
                
        if not target_font:
            print("[오류] 시스템에서 기본 한글 폰트를 찾을 수 없습니다. (malgun.ttf, batang.ttc 등)")
            return False
            
        print(f"[정보] 기본 템플릿 글꼴 로드: {target_font}")
        font = ImageFont.truetype(target_font, 80)
        
        # 힌트 글자의 평균 두께(검은색 픽셀 비율) 계산
        black_pixel_ratios = []
        for char, img in self.style_images.items():
            arr = np.array(img)
            # 흰색 배경(255), 글씨(0)
            black_pixels = np.sum(arr < 127)
            ratio = black_pixels / float(arr.size)
            black_pixel_ratios.append(ratio)
            
        avg_black_ratio = np.mean(black_pixel_ratios) if black_pixel_ratios else 0.08
        print(f"[정보] 스타일 힌트 글씨 평균 두께 비율: {avg_black_ratio:.2%}")

        print(f"[정보] 2,350자 손글씨 이미지 모의 생성(Few-shot Simulation) 시작...")
        
        total_count = len(COMMON_2350_HANGUL)
        
        for idx, char in enumerate(COMMON_2350_HANGUL):
            char_hex = f"{ord(char):04X}"
            out_path = os.path.join(output_dir, f"{char_hex}.png")
            
            # 만약 사용자가 적은 10개 힌트 글자에 직접 해당하는 글자라면, 원본 크롭본을 그대로 복사하여 보존
            if char in self.style_images:
                self.style_images[char].save(out_path)
                continue
                
            # 기본 돋움 이미지 생성
            img = Image.new('L', (128, 128), color=255)
            draw = ImageDraw.Draw(img)
            
            # 글씨 중앙 배치를 위한 정렬
            try:
                # Pillow 10+ 호환
                if hasattr(draw, 'textbbox'):
                    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
                    w, h = right - left, bottom - top
                else:
                    w, h = draw.textsize(char, font=font)
            except:
                w, h = 64, 64
                
            draw.text(((128 - w) // 2, (128 - h) // 2 - 10), char, fill=0, font=font)
            
            # OpenCV를 활용해 획의 두께와 부드러움(손글씨 번짐)을 필터링으로 적용
            img_np = np.array(img)
            
            # 이진화 반전 (배경 0, 글씨 255)
            _, thresh = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY_INV)
            
            # 손글씨 특유의 번짐 및 곡선 획 모사 (Dilation / Erosion)
            # 평균 두께 비율이 두꺼울 경우 조금 부풀림
            if avg_black_ratio > 0.09:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                thresh = cv2.dilate(thresh, kernel, iterations=1)
            elif avg_black_ratio < 0.06:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                thresh = cv2.erode(thresh, kernel, iterations=1)
                
            # 가우시안 블러를 얇게 가해 손글씨의 부드러운 획 질감 처리
            blur = cv2.GaussianBlur(thresh, (3, 3), 0)
            _, final_thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
            
            # 다시 색상 반전 (배경 255, 글씨 0)
            final_img = cv2.bitwise_not(final_thresh)
            cv2.imwrite(out_path, final_img)
            
            if (idx + 1) % 500 == 0:
                print(f" -> 생성 진행도: {idx + 1}/{total_count} 완료...")
                
        print(f"[성공] 2,350자 전체 이미지 생성 완료! (저장경로: {output_dir})")
        return True

if __name__ == "__main__":
    generator = MockFewShotFontGenerator()
    generator.generate_all_fonts()
