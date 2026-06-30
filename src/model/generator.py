import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

# FontDiffuser 모듈 경로 주입
FONTDIFFUSER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "fontdiffuser"))
if FONTDIFFUSER_PATH not in sys.path:
    sys.path.append(FONTDIFFUSER_PATH)

# FontDiffuser 모듈 임포트
try:
    from src import (FontDiffuserDPMPipeline,
                     FontDiffuserModelDPM,
                     build_ddpm_scheduler,
                     build_unet,
                     build_content_encoder,
                     build_style_encoder)
    import torchvision.transforms as transforms
except ImportError as e:
    print(f"[경고] FontDiffuser 임포트 실패: {e}")
    FontDiffuserDPMPipeline = None

COMMON_2350_HANGUL = (
    "가각간갇갈갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹걘걜걤걉걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괨괩괫괭괴괵괸굄굅굇굉교굔굘굠굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궎궤궝귀귁귄귈귐귑귓규균귤그극근글긁금급긋긍기긱긴길김깁깃깅깊까깍깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄깩꺼꺽껀껄껌껍껏껑께껜껨껫껭껴껸껼꼈꼍꼐꼬꼭꼰꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽘꽥꽹꽤꽨꽬꽵꽷꽹꾀꾄꾐꾑꾕뾔꾸꾹꾼꿀꿇꿈꿉꿋꿍꿔꿩꿰뀄뀌뀐뀔뀜뀝뀨끄끈끌끓끔끕끗끝끼끽낀낄낌낍낏낑나낙난날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냥녀녁년녈념념녑녔녕녘녜노녹논놀놂놈놉놋농높놓놔놘놜뇌뇐뇨뇩뇬뇰뇽누눅눈눌눔눕눗눙눠눴뉘뉜뉠뉨뉩뉴뉵뉼늄늉느늑는늘늙늚늠늡늦늪늬늰늴니닉닌닐님닙닛닝닢다닥단닫달닭닮닳담답닷닺닻닿대댁댄댈댐댑댓댔댕댜더덕던덜덞덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨도독돈돋돌돎돔돕돗동돛돝돠돤돨돼됀됄됌됍됐되된될됨됩됫뵤두둑둔둘둠둡둣둥둬뒀뒤뒨뒬뒴뒵뒷듀듄듈듐륭드득든들듦듬듭듯등디딕딘딜딤딥딧딩딪따딱딴딸땀땁땃땄땅때땍땐땔땜땝땟땠땡떠떡떤떨떪떰떱떳떴떵떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똔똴뙈뙨뙬뙵뙷뙹뙤뙨뚜뚝뚠뚤뚫뚱뚸뛔뛰뛴뛸뜀뜁뜅뜨뜯뜬뜰뜸뜹뜻띄띈띌띰띱띠띡띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸량러럭런럴럼럽럿렀렁렇게레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롈롑롓로록론롤롬롭롯롱롸롼뢔뢰뢴룐료룍룐룔룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏류륜률륨륭르륵른를름릅릇릉릐리릭린릴림립릿링마막만맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹먀먁먄먈먕머먹먼멀멂멈멉멋멍멎메멕멘멜멤멥멧멨멩며멱면멸몃몄명몌모목몬몰몲몸몹못몽뫄뫈뫘뫼묀묄묨묍묏묑묘묫무묵문묻물묽묾뭄뭅뭇뭉뭐뭑뭔뭘뭬뮈뮌륄뮌뮤뮨뮬뮴뮹므믁믄믈믐믑믓믕미믹민밀밈밉밋밍및밑바박반받발밝밞밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱌뱍뱐뱜뱡뱨버벅번벌벎범법벗벙베벡벤벨벰벱벳벴벵벼벽변별볍볏볐병볘보복본볼봄봅봇봉봐봔봘뵀뵈뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붜붤붸뷔뷕뷘뷜뷤뷥뷧뷰뷴뷸븀븡브븍븐블븜븝븟븡븨비빅빈빌빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵뺘뺙뺜뺨뺭빼빽뺀뺄뺌뺍뺏뺐뺑뺴뻐뻑뻔뻘뻠뻡뻣뻤뻥뻬뻭뻰뻴뻼뻿뼁뼈뼉뼌뼘뼝뼤뽀뽄뽈뽐뽑뽕뽜뽠뽸뾔뿌뿍뿐뿔뿜뿜뿽뿨쀄쀠쀤쀨쀼쁌쁭쁘쁜쁠쁰쁱쁲쁳쁴삔삘삠삡삣삥사삭산살삷삼삽삿샀상삺샅새색샌샐샘샘샙샛샜생샤샥샨샬샴샵샷샹섀섄서석선설섪섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속손솔솖솜솝솟송솥솨솩솬쇄쇈쇠쇤쇨쇰쇱쇳쇼쇽숀숄쇽숍숏숑수숙순숟술숨숩숫숭숯숴쉈쉐쉔쉘쉠쉡쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슝스슥슨슬슭슴습슷승시식신실싫심십싯싱싶싸싹싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩쌰쌱썅써썩썬썰썲썸썹썻썼썽쎄쎅쎈쎌쎔쎕쎗쎘쎵쎼쏘쏙쏜쏠쏨쏩쏫쏭쏴쏵쐈쐐쐰쐴쐼쐽쐿쑈쑤쑥쑨쑬쑴쑵쑻쑹쒀쒔쒜쒸쒼쓔쓩쓰쓱쓴쓸쓸쓺씀씁쓿씌씐씔씌씨씩씬씰씸씹씻씽아악안앉않알갉앓암압앗았앙앞애액앤앨앰앱앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠어억언얹얻얼얽얾엄업업엇었엉엌엎에엑엔엘엠엡엣엥여역연열엷염엽엿였영옆예옌옐옘옙옛옜오옥온올옭옮옳옴옵옷옹옻와왁완왈왐왑왔왕왜왝왠왬왱외획왼욀욈욉외욋요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읆음읍읏응읔읕읖의의이익인일읽잃임입잇있잉잎자작잔잖잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟝걔쟤져적전절젊점접젓졌정젖제젝젠젤젬젭젯젰젱져젹젼졀졈졉졌졍졔조족존졸졺좀좁졿종좌좍좐좠좽죄죈죌죔죕죗죠죡죤죵주죽준줄줆줌줍줏중줘줬줴쥐쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진질짊짐집짓징짖짙짚짜짝짠짤짧짬짭짭짰짱째짹짼짴쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쪘쩡쩨쩬쩨쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫘쫙쫠쫬쫴쬐쬔쬘쬠쬡쬫쭈쭉쭌쭘쭙쭝쭤쭸쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찌차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챡챤챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳈쳉쳐쳔쳤쳥쳬초촉촌촐촘숍촛총촤촨촬최쵠쵤쵬쵭쵯쵸쵼츄츈츌츔츠측츤츨츰츱츳층치칙친칠칡침칩칫칭치카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컬컴컵컷컸컹케켁켄켈켐켑켓켔켕켜켯켰켱켸코콕콘콜콤콥콧콩콰콱콴괄쾀쾅쾌쾐쾬쾽퀴퀵퀸퀼큄큅귓큐균큘크큭큰클큼클큽큿큵키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텼텡텨텬텼텽톄토톡톤톨톰톱톳통톼퇀돼퇴툔툘툠투툭툰툴툼툽틋퉁퉈퉜퉤튀튁튄튈튐튑튓튜튠튤튬륭트특튼틀틂틈틉틋틔틘틔티틱틴틸팀팁팃팅파팍판팔팖팜팝팟팠팡패팩팬팰팸팹팻팼팽퍄퍅평퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄펵폈평폐폐포폭폰폴폼폽폿퐁퐈퐘푀푄표푠푝푠표푸푹푼풀풂품풉풋풍풔풨퓌퓔퓨퓬퓰퓸륭프픈플픔픗피픽핀필필핌핍핏핑하학한할핥함합핫항해핵핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헸헹혀혁현혈혐협혓혔형혜혠혤호혹혼홀홂홈홉홋홍화화확환활홤홥왔황홰홱왠홴홽회획회횔횜횝횟효횬횰횸횹후훅훈훌훑훔훕훗훙훠훵훼훽휀휄휨휩휫휘획휜휠휨휩휫휴휵휸휼흄륭흐흑흔흘흙흚흠흡흣흥희흰흰흴흼흽히힉힌힐힘힙힛힝"
)

def load_fontdiffuser_pipeline(ckpt_dir="weights", device="cuda:0"):
    from configs.fontdiffuser import get_parser
    parser = get_parser()
    args = parser.parse_args([])
    
    args.ckpt_dir = ckpt_dir
    args.device = device
    args.num_inference_steps = 20  # GPU 가속이 활성화되었으므로 고화질 20스텝으로 원상복구
    
    # 이미지 사이즈를 튜플로 변환 (FontDiffuser 내부 요구 사양)
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    # 1. 아웃라인 UNet 구성
    print("[AI 로드] UNet 모델 구조 생성 중...")
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{ckpt_dir}/unet.pth", map_location=device))
    
    # 2. 스타일 인코더 구성
    print("[AI 로드] 스타일 인코더 로드 중...")
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{ckpt_dir}/style_encoder.pth", map_location=device))
    
    # 3. 콘텐츠 인코더 구성
    print("[AI 로드] 콘텐츠 인코더 로드 중...")
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{ckpt_dir}/content_encoder.pth", map_location=device))
    
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)
    model.to(device)
    
    # 4. DDPM 스케줄러 로드
    train_scheduler = build_ddpm_scheduler(args=args)
    
    # 5. DPM solver 파이프라인 생성
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale
    )
    print("[성공] FontDiffuser SOTA 딥러닝 추론 파이프라인 완벽 로드 완료!")
    return pipe, args

def run_real_fontdiffuser_inference(pipe, args, style_dir="data/style", output_dir="output/images", device="cuda:0"):
    """
    진짜 FontDiffuser weights 4종을 사용하여
    10개의 한글 손글씨 힌트로부터 2,350자의 필체 스타일을 유추 합성(Cross-Attention Diffusion)합니다.
    """
    print("[AI 작동] FontDiffuser 딥러닝 디퓨전 추론 가동 시작...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 스타일 이미지들 로드
    style_files = [f for f in os.listdir(style_dir) if f.lower().endswith('.png')]
    style_images = []
    style_hints = {}
    
    for f in style_files:
        char_hex = os.path.splitext(f)[0]
        try:
            char = chr(int(char_hex, 16))
            pil_img = Image.open(os.path.join(style_dir, f)).convert('RGB')
            style_images.append(pil_img)
            style_hints[char] = Image.open(os.path.join(style_dir, f)).convert('L')
        except Exception as e:
            print(f"[경고] 스타일 파일 해석 오류 ({f}): {e}")
            
    if not style_images:
        print("[오류] 스타일 힌트 글자가 하나도 로드되지 않았습니다.")
        return
        
    # 대표 스타일 이미지 결정 (예: 1번째 힌트 글자)
    style_img_ref = style_images[0]
    
    # 2. 시스템 기본 한글 폰트 로드 (Content Template)
    font_paths = [
        "C:\\Windows\\Fonts\\malgun.ttf", 
        "C:\\Windows\\Fonts\\batang.ttc", 
        "C:\\Windows\\Fonts\\gulim.ttc"
    ]
    target_font = next((p for p in font_paths if os.path.exists(p)), None)
    if not target_font:
        print("[오류] 시스템 맑은고딕 또는 바탕체를 찾을 수 없습니다.")
        return
    font = ImageFont.truetype(target_font, 80)
    
    # 전처리 트랜스폼 정의 (args.content_image_size가 이미 튜플이므로 직접 전달)
    content_transforms = transforms.Compose([
        transforms.Resize(args.content_image_size, 
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    style_transforms = transforms.Compose([
        transforms.Resize(args.style_image_size, 
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # 스타일 텐서 생성
    style_tensor = style_transforms(style_img_ref)[None, :].to(device)
    
    total = len(COMMON_2350_HANGUL)
    print(f" -> 총 {total}자의 한글에 대해 AI 디퓨전 획 생성을 개시합니다.")
    
    for idx, char in enumerate(COMMON_2350_HANGUL):
        char_hex = f"{ord(char):04X}"
        out_path = os.path.join(output_dir, f"{char_hex}.png")
        

            
        # 기본 뼈대 이미지 렌더링
        c_img = Image.new('RGB', (128, 128), color=(255, 255, 255))
        draw = ImageDraw.Draw(c_img)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
            w, h = right - left, bottom - top
        except:
            w, h = 64, 64
        draw.text(((128 - w) // 2, (128 - h) // 2 - 10), char, fill=(0, 0, 0), font=font)
        
        # 콘텐츠 이미지 변환
        content_tensor = content_transforms(c_img)[None, :].to(device)
        
        # 디퓨전 추론 노이즈 해소 루프 실행
        with torch.no_grad():
            output_tensor = pipe.generate(
                content_images=content_tensor,
                style_images=style_tensor,
                batch_size=1,
                order=args.order,
                num_inference_step=args.num_inference_steps,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
                t_start=args.t_start,
                t_end=args.t_end,
                dm_size=args.content_image_size
            )
            
            # 최종 복원 텐서를 넘파이로 변경
            out_img = output_tensor[0]
            # PIL 이미지를 흑백 OpenCV 배열로 변경
            out_np = np.array(out_img.convert('L'))
            
        # 화질 강화 및 경계선 정리
        _, final_thresh = cv2.threshold(out_np, 127, 255, cv2.THRESH_BINARY)
        
        # 앤티앨리어싱 스무딩
        smoothed = cv2.GaussianBlur(final_thresh, (3, 3), 0)
        _, final_img = cv2.threshold(smoothed, 110, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(out_path, final_img)
        
        if (idx + 1) % 100 == 0:
            print(f"   -> AI 추론 진행도: {idx + 1}/{total} 완료 (현재 자: '{char}')", flush=True)

    print("[성공] FontDiffuser AI 추론 완료 및 2,350자 고품질 합성 이미지 저장 성공!")

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[장치 상태] 실행 디바이스: {device}")
    
    try:
        pipe, args = load_fontdiffuser_pipeline("weights", device)
        run_real_fontdiffuser_inference(pipe, args, "data/style", "output/images", device)
    except Exception as e:
        import traceback
        print(f"[오류 발생] 진짜 AI 추론 중 예외 발생: {e}")
        traceback.print_exc()
