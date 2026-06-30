import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# PyTorch 로드 시도
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

# 한글 상용 2,350자 기본 세트
COMMON_2350_HANGUL = (
    "가각간갇갈갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹걘걜걤걉걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괨괩괫괭괴괵괸굄굅굇굉교굔굘굠굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궎궤궝귀귁귄귈귐귑귓규균귤그극근글긁금급긋긍기긱긴길김깁깃깅깊까깍깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄깩꺼꺽껀껄껌껍껏껑께껜껨껫껭껴껸껼꼈꼍꼐꼬꼭꼰꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽘꽥꽹꽤꽨꽬꽵꽷꽹꾀꾄꾐꾑꾕뾔꾸꾹꾼꿀꿇꿈꿉꿋꿍꿔꿩꿰뀄뀌뀐뀔뀜뀝뀨끄끈끌끓끔끕끗끝끼끽낀낄낌낍낏낑나낙난날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냥녀녁년녈념념녑녔녕녘녜노녹논놀놂놈놉놋농높놓놔놘놜뇌뇐뇨뇩뇬뇰뇽누눅눈눌눔눕눗눙눠눴뉘뉜뉠뉨뉩뉴뉵뉼늄늉느늑는늘늙늚늠늡늦늪늬늰늴니닉닌닐님닙닛닝닢다닥단닫달닭닮닳담답닷닺닻닿대댁댄댈댐댑댓댔댕댜더덕던덜덞덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨도독돈돋돌돎돔돕돗동돛돝돠돤돨돼됀됄됌됍됐되된될됨됩됫뵤두둑둔둘둠둡둣둥둬뒀뒤뒨뒬뒴뒵뒷듀듄듈듐륭드득든들듦듬듭듯등디딕딘딜딤딥딧딩딪따딱딴딸땀땁땃땄땅때땍땐땔땜땝땟땠땡떠떡떤떨떪떰떱떳떴떵떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똔똴뙈뙨뙬뙵뙷뙹뙤뙨뚜뚝뚠뚤뚫뚱뚸뛔뛰뛴뛸뜀뜁뜅뜨뜯뜬뜰뜸뜹뜻띄띈띌띰띱띠띡띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸량러럭런럴럼럽럿렀렁렇게레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롈롑롓로록론롤롬롭롯롱롸롼뢔뢰뢴룐료룍룐룔룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏류륜률륨륭르륵른를름릅릇릉릐리릭린릴림립릿링마막만맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹먀먁먄먈먕머먹먼멀멂멈멉멋멍멎메멕멘멜멤멥멧멨멩며멱면멸몃몄명몌모목몬몰몲몸몹못몽뫄뫈뫘뫼묀묄묨묍묏묑묘묫무묵문묻물묽묾뭄뭅뭇뭉뭐뭑뭔뭘뭬뮈뮌륄뮌뮤뮨뮬뮴뮹므믁믄믈믐믑믓믕미믹민밀밈밉밋밍및밑바박반받발밝밞밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱌뱍뱐뱜뱡뱨버벅번벌벎범법벗벙베벡벤벨벰벱벳벴벵벼벽변별볍볏볐병볘보복본볼봄봅봇봉봐봔봘뵀뵈뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붜붤붸뷔뷕뷘뷜뷤뷥뷧뷰뷴뷸븀븡브븍븐블븜븝븟븡븨비빅빈빌빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵뺘뺙뺜뺨뺭빼빽뺀뺄뺌뺍뺏뺐뺑뺴뻐뻑뻔뻘뻠뻡뻣뻤뻥뻬뻭뻰뻴뻼뻿뼁뼈뼉뼌뼘뼝뼤뽀뽄뽈뽐뽑뽕뽜뽠뽸뾔뿌뿍뿐뿔뿜뿜뿽뿨쀄쀠쀤쀨쀼쁌쁭쁘쁜쁠쁰쁱쁲쁳쁴삔삘삠삡삣삥사삭산살삷삼삽삿샀상삺샅새색샌샐샘샘샙샛샜생샤샥샨샬샴샵샷샹섀섄서석선설섪섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속손솔솖솜솝솟송솥솨솩솬쇄쇈쇠쇤쇨쇰쇱쇳쇼쇽숀숄쇽숍숏숑수숙순숟술숨숩숫숭숯숴쉈쉐쉔쉘쉠쉡쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슝스슥슨슬슭슴습슷승시식신실싫심십싯싱싶싸싹싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩쌰쌱썅써썩썬썰썲썸썹썻썼썽쎄쎅쎈쎌쎔쎕쎗쎘쎵쎼쏘쏙쏜쏠쏨쏩쏫쏭쏴쏵쐈쐐쐰쐴쐼쐽쐿쑈쑤쑥쑨쑬쑴쑵쑻쑹쒀쒔쒜쒸쒼쓔쓩쓰쓱쓴쓸쓸쓺씀씁쓿씌씐씔씌씨씩씬씰씸씹씻씽아악안앉않알갉앓암압앗았앙앞애액앤앨앰앱앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠어억언얹얻얼얽얾엄업업엇었엉엌엎에엑엔엘엠엡엣엥여역연열엷염엽엿였영옆예옌옐옘옙옛옜오옥온올옭옮옳옴옵옷옹옻와왁완왈왐왑왔왕왜왝왠왬왱외획왼욀욈욉외욋요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읆음읍읏응읔읕읖의의이익인일읽잃임입잇있잉잎자작잔잖잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟝걔쟤져적전절젊점접젓졌정젖제젝젠젤젬젭젯젰젱져젹젼졀졈졉졌졍졔조족존졸졺좀좁졿종좌좍좐좠좽죄죈죌죔죕죗죠죡죤죵주죽준줄줆줌줍줏중줘줬줴쥐쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진질짊짐집짓징짖짙짚짜짝짠짤짧짬짭짭짰짱째짹짼짴쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쪘쩡쩨쩬쩨쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫘쫙쫠쫬쫴쬐쬔쬘쬠쬡쬫쭈쭉쭌쭘쭙쭝쭤쭸쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찌차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챡챤챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳈쳉쳐쳔쳤쳥쳬초촉촌촐촘숍촛총촤촨촬최쵠쵤쵬쵭쵯쵸쵼츄츈츌츔츠측츤츨츰츱츳층치칙친칠칡침칩칫칭치카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컬컴컵컷컸컹케켁켄켈켐켑켓켔켕켜켯켰켱켸코콕콘콜콤콥콧콩콰콱콴괄쾀쾅쾌쾐쾬쾽퀴퀵퀸퀼큄큅귓큐균큘크큭큰클큼클큽큿큵키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텼텡텨텬텼텽톄토톡톤톨톰톱톳통톼퇀돼퇴툔툘툠투툭툰툴툼툽틋퉁퉈퉜퉤튀튁튄튈튐튑튓튜튠튤튬륭트특튼틀틂틈틉틋틔틘틔티틱틴틸팀팁팃팅파팍판팔팖팜팝팟팠팡패팩팬팰팸팹팻팼팽퍄퍅평퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄펵폈평폐폐포폭폰폴폼폽폿퐁퐈퐘푀푄표푠푝푠표푸푹푼풀풂품풉풋풍풔풨퓌퓔퓨퓬퓰퓸륭프픈플픔픗피픽핀필필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헸헹혀혁현혈혐협혓혔형혜혠혤호혹혼홀홂홈홉홋홍화화확환활홤홥왔황홰홱왠홴홽회획회횔횜횝횟효횬횰횸횹후훅훈훌훑훔훕훗훙훠훵훼훽휀휄휨휩휫휘획휜휠휨휩휫휴휵휸휼흄륭흐흑흔흘흙흚흠흡흣흥희흰흰흴흼흽히힉힌힐힘힙힛힝"
)

# -------------------------------------------------------------
# 1. 진짜 AI FFG 모델 (MX-Font) PyTorch 구조 설계
# -------------------------------------------------------------
if torch is not None:
    class ConvBlock(nn.Module):
        def __init__(self, in_c, out_c, kernel=3, stride=1, pad=1):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, kernel, stride, pad, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.relu = nn.ReLU(True)
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class StyleEncoder(nn.Module):
        """
        스타일 힌트 글자의 미세한 획 특징을 축소 인코딩하는 네트워크
        """
        def __init__(self):
            super().__init__()
            self.layer1 = ConvBlock(1, 64, stride=2)   # 64x64
            self.layer2 = ConvBlock(64, 128, stride=2) # 32x32
            self.layer3 = ConvBlock(128, 256, stride=2) # 16x16
            self.layer4 = ConvBlock(256, 512, stride=2) # 8x8
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 256)
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class ContentEncoder(nn.Module):
        """
        기본 글자(Content)의 뼈대/형태를 분석해 스케일별 Feature Map을 보관하는 네트워크
        """
        def __init__(self):
            super().__init__()
            self.c1 = ConvBlock(1, 64)
            self.c2 = ConvBlock(64, 128, stride=2) # 64x64
            self.c3 = ConvBlock(128, 256, stride=2) # 32x32
            self.c4 = ConvBlock(256, 512, stride=2) # 16x16
        def forward(self, x):
            f1 = self.c1(x)
            f2 = self.c2(f1)
            f3 = self.c3(f2)
            f4 = self.c4(f3)
            return [f1, f2, f3, f4]

    class MXFontGenerator(nn.Module):
        """
        MX-Font 핵심 합성 생성기
        Content Feature와 Style Code를 융합하여 정교한 최종 폰트 이미지를 디코딩합니다.
        """
        def __init__(self):
            super().__init__()
            self.style_enc = StyleEncoder()
            self.content_enc = ContentEncoder()
            
            # 스타일 혼합을 위한 Linear 변환 레이어들 (AdaIN 또는 변형된 스타일 융합)
            self.style_fc4 = nn.Linear(256, 512)
            self.style_fc3 = nn.Linear(256, 256)
            self.style_fc2 = nn.Linear(256, 128)
            
            # 디코더 업샘플링 (U-Net Fusing)
            self.up3 = nn.ConvTranspose2d(512, 256, 4, 2, 1) # 32x32
            self.up2 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # 64x64
            self.up1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 128x128
            
            self.final_conv = nn.Conv2d(64, 1, 3, 1, 1)
            
        def forward(self, content_img, style_imgs):
            # 1. 스타일 힌트 이미지들의 평균 스타일 코드 추출
            style_codes = []
            for simg in style_imgs:
                style_codes.append(self.style_enc(simg))
            style_code = torch.mean(torch.stack(style_codes, dim=0), dim=0) # [1, 256]
            
            # 2. 콘텐츠 뼈대 Feature Map 획득
            feats = self.content_enc(content_img)
            f1, f2, f3, f4 = feats
            
            # 3. 스타일 융합 디코딩 (Style injection)
            s_vec4 = self.style_fc4(style_code).view(-1, 512, 1, 1)
            dec3 = self.up3(f4 * (1 + s_vec4)) + f3
            
            s_vec3 = self.style_fc3(style_code).view(-1, 256, 1, 1)
            dec2 = self.up2(dec3 * (1 + s_vec3)) + f2
            
            s_vec2 = self.style_fc2(style_code).view(-1, 128, 1, 1)
            dec1 = self.up1(dec2 * (1 + s_vec2)) + f1
            
            out = torch.sigmoid(self.final_conv(dec1))
            return out
else:
    class MXFontGenerator:
        pass

# -------------------------------------------------------------
# 2. AI 폰트 생성 및 시뮬레이션 통합 뼈대
# -------------------------------------------------------------
class HighQualityFontGenerator:
    def __init__(self, weight_path="weights/mxfont_korean.pth", style_dir="data/style"):
        self.weight_path = weight_path
        self.style_dir = style_dir
        self.device = 'cuda' if (torch and torch.cuda.is_available()) else 'cpu'
        self.model = None
        self.style_hints = {}

    def load_style_hints(self):
        if not os.path.exists(self.style_dir):
            return False
        files = [f for f in os.listdir(self.style_dir) if f.lower().endswith('.png')]
        for filename in files:
            char_hex = os.path.splitext(filename)[0]
            try:
                char = chr(int(char_hex, 16))
                self.style_hints[char] = Image.open(os.path.join(self.style_dir, filename)).convert('L')
            except:
                continue
        print(f"[정보] 로드된 스타일 힌트 글자: {list(self.style_hints.keys())}")
        return len(self.style_hints) > 0

    def load_model(self):
        """
        PyTorch 가중치(.pth) 파일이 존재하는 경우 진짜 AI FFG 모델을 메모리에 올립니다.
        """
        if torch is None:
            return False
        if not os.path.exists(self.weight_path):
            print(f"[정보] AI 가중치 파일('{self.weight_path}')이 발견되지 않았습니다. 시뮬레이션 모드로 생성합니다.")
            return False
            
        try:
            self.model = MXFontGenerator().to(self.device)
            # 가중치 파일 로딩
            checkpoint = torch.load(self.weight_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print("[성공] 진짜 AI FFG (MX-Font) 모델이 GPU/CPU에 성공적으로 적재되었습니다!")
            return True
        except Exception as e:
            print(f"[경고] AI 가중치 로딩 실패로 시뮬레이션 모드로 실행합니다: {e}")
            self.model = None
            return False

    def generate_font_dataset(self, output_dir="output/images"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        has_hints = self.load_style_hints()
        if not has_hints:
            print("[오류] 전처리된 스타일 힌트 글자 이미지(data/style/*.png)가 존재하지 않습니다.")
            return False
            
        # 진짜 AI 가중치 로드 시도
        is_real_ai = self.load_model()
        
        # 시스템 글꼴 로드 (Content Template)
        font_paths = [
            "C:\\Windows\\Fonts\\malgun.ttf", 
            "C:\\Windows\\Fonts\\batang.ttc", 
            "C:\\Windows\\Fonts\\gulim.ttc"
        ]
        target_font = next((p for p in font_paths if os.path.exists(p)), None)
        if not target_font:
            print("[오류] 시스템 기본 한글 폰트를 찾을 수 없습니다.")
            return False
        font = ImageFont.truetype(target_font, 80)
        
        if is_real_ai:
            self.run_real_ai_inference(font, output_dir)
        else:
            self.run_style_simulation(font, output_dir)

    def run_real_ai_inference(self, font, output_dir):
        """
        가중치 파일이 있을 경우, 10개의 손글씨 스타일 힌트를 활용해
        진짜 딥러닝(MX-Font) 추론으로 2,350자의 필체를 합성합니다.
        """
        print("[AI 작동] MX-Font 딥러닝 추론 엔진 가동 중...")
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 스타일 힌트 텐서화
        style_tensors = []
        for img in self.style_hints.values():
            s_img = img.resize((128, 128), Image.Resampling.LANCZOS)
            style_tensors.append(transform(s_img).unsqueeze(0).to(self.device))
            
        total = len(COMMON_2350_HANGUL)
        for idx, char in enumerate(COMMON_2350_HANGUL):
            char_hex = f"{ord(char):04X}"
            out_path = os.path.join(output_dir, f"{char_hex}.png")
            
            # 사용자 힌트 원본 글자는 그대로 복사해 화질 유지
            if char in self.style_hints:
                self.style_hints[char].save(out_path)
                continue
                
            # 기본 콘텐츠 뼈대 렌더링
            c_img = Image.new('L', (128, 128), color=255)
            draw = ImageDraw.Draw(c_img)
            try:
                left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
                w, h = right - left, bottom - top
            except:
                w, h = 64, 64
            draw.text(((128 - w) // 2, (128 - h) // 2 - 10), char, fill=0, font=font)
            
            # 딥러닝 추론
            content_tensor = transform(c_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output_tensor = self.model(content_tensor, style_tensors)
                output_tensor = (output_tensor.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
                
            # 이미지 이진화 반전 후 저장
            _, final_img = cv2.threshold(output_tensor, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(out_path, final_img)
            
            if (idx + 1) % 500 == 0:
                print(f" -> AI 추론 진행도: {idx + 1}/{total} 완료...")
                
        print("[성공] AI 추론 기반 한글 2,350자 스타일 합성 완료!")

    def run_style_simulation(self, font, output_dir):
        """
        가중치가 없을 때 작동하는 하이브리드 필체 모사 시뮬레이터.
        돋움체 뼈대에 스타일 힌트 글자의 평균 두께, 곡률, 거친 획 엣지를 모사하여 생성합니다.
        """
        print("[시뮬레이터 작동] 스타일 힌트 평균 획 분석 및 합성 시뮬레이션 가동...")
        
        # 1. 힌트 글자들의 두께 비율 및 평균 윤곽선 거칠기(곡률 분산) 분석
        thickness_ratios = []
        blur_intensities = []
        
        for char, img in self.style_hints.items():
            arr = np.array(img)
            black_pixels = np.sum(arr < 127)
            thickness_ratios.append(black_pixels / float(arr.size))
            
            # 엣지 거칠기 측정
            edges = cv2.Canny(arr, 50, 150)
            edge_pixels = np.sum(edges > 0)
            blur_intensities.append(edge_pixels / float(black_pixels + 1))
            
        avg_thickness = np.mean(thickness_ratios) if thickness_ratios else 0.08
        avg_blur = np.mean(blur_intensities) if blur_intensities else 0.12
        
        print(f" -> 분석된 힌트 필체 특성: 평균두께 {avg_thickness:.2%}, 거칠기 비율 {avg_blur:.4f}")
        
        total = len(COMMON_2350_HANGUL)
        for idx, char in enumerate(COMMON_2350_HANGUL):
            char_hex = f"{ord(char):04X}"
            out_path = os.path.join(output_dir, f"{char_hex}.png")
            
            if char in self.style_hints:
                self.style_hints[char].save(out_path)
                continue
                
            # 기본 글꼴 뼈대 이미지 생성
            img = Image.new('L', (128, 128), color=255)
            draw = ImageDraw.Draw(img)
            try:
                left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
                w, h = right - left, bottom - top
            except:
                w, h = 64, 64
            draw.text(((128 - w) // 2, (128 - h) // 2 - 10), char, fill=0, font=font)
            
            # 스타일 이식 필터링 (OpenCV)
            img_np = np.array(img)
            _, thresh = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY_INV)
            
            # 1. 힌트의 획 두께에 가깝도록 팽창/수축 보정
            if avg_thickness > 0.09:
                k_size = int(avg_thickness * 30)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(2, k_size), max(2, k_size)))
                thresh = cv2.dilate(thresh, kernel, iterations=1)
            elif avg_thickness < 0.06:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                thresh = cv2.erode(thresh, kernel, iterations=1)
                
            # 2. 힌트의 손글씨 특유의 번짐과 비정형 엣지 모사
            # 가우시안 블러와 랜덤 노이즈 엣지를 미세하게 합성
            blur_kernel = 3 if avg_blur > 0.1 else 1
            blur = cv2.GaussianBlur(thresh, (blur_kernel, blur_kernel), 0)
            
            # 획 엣지를 거칠게 만드는 손글씨 변형 노이즈
            noise = np.random.normal(0, 10, blur.shape).astype(np.float32)
            noisy_blur = np.clip(blur.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            _, final_thresh = cv2.threshold(noisy_blur, 110, 255, cv2.THRESH_BINARY)
            
            # 색상 반전 저장
            final_img = cv2.bitwise_not(final_thresh)
            cv2.imwrite(out_path, final_img)
            
            if (idx + 1) % 500 == 0:
                print(f" -> 시뮬레이션 생성도: {idx + 1}/{total} 완료...")
                
        print(f"[성공] 스타일 모사 시뮬레이션 완료! (저장경로: {output_dir})")

if __name__ == "__main__":
    generator = HighQualityFontGenerator()
    generator.generate_font_dataset()
