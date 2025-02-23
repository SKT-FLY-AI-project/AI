# 프롬프트 명예의 전당 모음.

# 0. 국룰
"""
설명은 반드시 **한글(가-힣)과 영어(a-z)만 사용하여 작성해야 합니다.**
            숫자, 특수문자, 한자는 포함할 수 없습니다.
"""

# 1. 처음으로 대화다운 대화를 성공함.
# VLM
prompt = f"이 이미지를 보고 장면, 색채, 구도, 분위기, 주요 특징을 설명하세요.\n" \
            f"주요 색상: {', '.join(dominant_colors)}\n" \
            f"경계 감지 결과: {'명확함' if edges_detected else '불명확함'}"

# LLM
prompt_template = PromptTemplate(
        input_variables=["title", "vlm_desc", "dominant_colors", "edges_detected"],
        template=f"""
        당신은 그림 설명 전문가입니다.  
        다음 그림에 대해 상세한 설명을 생성해주세요.
        시각장애인에게 설명할 수 있도록 자세하게 작성해 주세요.
        **단, 200자 ~ 500자 사이의 길이로만 생성해야 합니다!**

        - **제목:** "{{title}}"  
        - **VLM 기반 기본 설명:** "{{vlm_desc}}"   

        위 정보를 바탕으로 그림에 대한 상세한 설명을 작성해 주세요.  
        작품의 분위기, 색채, 구도, 표현 기법 등을 분석하고,  
        가능하다면 역사적, 예술적 배경도 함께 제공해 주세요.  
        설명은 반드시 **한글(가-힣)만 사용하여 작성해야 합니다.**  
        영어, 숫자, 특수문자는 포함할 수 없습니다.  
        """
    )
# VTS - answer
prompt = f"""
    사용자가 미술 작품을 감상하고 있습니다.
    이전 대화:
    {context}

    사용자의 입력:
    "{user_input}"

    AI의 역할:
    1. 사용자의 감상에 대해 적절한 반응을 제공합니다.
    2. 새로운 질문을 생성하여 자연스럽게 대화를 이어갑니다.

    AI의 응답 형식:
    1. 반응: (사용자의 감상을 반영한 피드백)
    2. 질문: (VTS 기반의 적절한 추가 질문)
    """
    
# VTS - question    
prompt = f"다음 미술 감상 대화 기록을 참고하여 사용자의 질문에 대답하세요.\n\n"
prompt += "\n".join(history)
prompt += f"\n\n사용자 질문: {user_input}\n\nAI의 답변:"

# 2. 처음으로 VLM과 LLM의 역할을 명확히 나눔.

# VLM
prompt = """
    이 이미지를 보고 핵심 정보만 간결하게 요약하여 제공하세요.

    1. **주요 객체 (3~5개 키워드)**
    - 사람이 있다면 "사람 (명수, 성별별)", 동물이 있다면 "동물 (종류)" 형식으로만 작성하세요.
    - 건물, 자연 요소 등이 있다면 1~3개 주요 요소를 단어로만 나열하세요.

    2️. **색채 및 조명 (3~5개 키워드)**
    - 주요 색상을 3~5개 단어로만 제공하세요 (예: "빨강, 파랑, 노랑").
    - 빛의 방향이나 명암 대비가 있다면 간단한 설명을 덧붙이세요 (예: "빛 방향: 왼쪽 → 오른쪽").

    3️. **구도 및 시각적 흐름 (1~2개 키워드)**
    - 시선이 이동하는 방식이나 주요 패턴이 있다면 1~2개 단어로 제공하세요.
    - 예: "중앙 집중", "대각선 구도", "좌 → 우 시선 이동"

    4️. **분위기 및 감정 (1~2개 키워드)**
    - 이 작품이 주는 감정을 간단한 키워드로 정리하세요.
    - 예: "평온함, 따뜻함", "고요함, 우울함"
    """
    
# LLM
prompt_template = PromptTemplate(
            input_variables=["vlm_desc", "dominant_colors", "edges_detected"],
            template="""
            당신은 그림 설명 전문가입니다.
            다음 그림에 대해 상세한 시각적 설명을 생성해주세요.

            - **VLM 분석 결과:** "{vlm_desc}"
            - **주요 색상:** "{dominant_colors}"
            - **윤곽선 감지 결과:** "{edges_detected}"

            작품의 분위기, 색채, 구도, 표현 기법 등을 상세히 설명해주세요.
            작가나 시대 정보는 알 수 없으므로, 역사적 해설은 포함하지 말고
            순수한 시각적 요소에 집중해주세요.

            설명은 반드시 **한글(가-힣)과 영어(a-z)만 사용하여 작성해야 합니다.**
            숫자, 특수문자, 한자는 포함할 수 없습니다.
            """
        )
prompt_template = PromptTemplate(
        input_variables=list(prompt_variables.keys()),
        template="""
        당신은 그림 설명 전문가입니다.  
        다음 그림에 대해 상세한 설명을 생성해주세요.

        - **제목:** "{title}"
        {correct_artist}
        {correct_period}
        - **VLM 기반 기본 설명:** "{vlm_desc}"
        - **주요 색상:** "{dominant_colors}"
        - **윤곽선 감지 결과:** "{edges_detected}"

        위 정보를 바탕으로 그림에 대한 상세한 설명을 작성해 주세요.  
        작품의 분위기, 색채, 구도, 표현 기법 등을 분석하고,  
        가능하다면 역사적, 예술적 배경도 함께 제공해 주세요.

        {webpage}
        """
    )

# 3. VLM 정제는 잘 되는듯? LLM 정제는 더 필요해보임.
prompt = """
이 그림을 보고 장면을 설명해주세요.  
단순한 키워드가 아니라, 실제로 보고 이야기하듯이 짧은 문장으로 설명해주세요.  
각 요소를 개별적으로 나열하는 것이 아니라, 전체적인 장면을 자연스럽게 묘사해주세요.  

- 어떤 사물이 가장 눈에 띄나요?  
- 사람들은 무엇을 하고 있나요?  
- 색상과 빛의 흐름은 어떤 느낌을 주나요?  
- 전체적인 분위기는 어떻게 표현될 수 있나요?  

너무 길지 않게 2~3문장 정도로 설명해 주세요.
설명은 반드시 **한글(가-힣)과 영어(a-z)만 사용하여 작성해야 합니다.**
⚠️ **한자(漢字)는 절대 포함하지 마세요.** ⚠️  
한자가 포함될 경우, 다시 한글과 영어로만 설명해주세요.
"""




    # 정제 함수 백업
    # 정제 코드
def clean_and_restore_spacing(text):
    """
    VLM(Qwen2.5-VL) 출력에서 불필요한 시스템 메시지 및 프롬프트 반복을 제거하고, 핵심 정보만 유지하는 함수.
    """
    # ✅ 1. "system", "You are a helpful assistant." 같은 AI 시스템 메시지 제거
    text = re.sub(r"(system|You are a helpful assistant\.)", "", text, flags=re.IGNORECASE)

    # ✅ 2. "assistant" 같은 응답 태그 제거 (ex: "assistant 1. 주요 객체")
    text = re.sub(r"assistant\s*\d*\.*", "", text, flags=re.IGNORECASE)

    # ✅ 3. "이 이미지를 보고 ~ 설명하세요" 같은 프롬프트 반복 제거
    text = re.sub(r"이 이미지를 보고.*?설명하세요\.", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"user\s*이\s*그림을\s*보고.*?묘사해주세요\.", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"user\s*이\s*이미지를\s*보고.*?제공하세요\.", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"설명은\s*반드시\s*\*\*.*?\*\*", "", text, flags=re.IGNORECASE)
    
    # ✅ 4. VLM 프롬프트 제거 (프롬프트의 주요 문구 삭제)
    text = re.sub(r"user\s*이\s*그림을\s*보고.*?묘사해주세요\.", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"설명은\s*반드시\s*\*\*.*?\*\*", "", text, flags=re.IGNORECASE)
    
    # ✅ 5. VLM 프롬프트의 예시 문장 제거 (예시가 그대로 포함될 경우)
    text = re.sub(r'예를 들어, ".*?"', "", text, flags=re.DOTALL)
    text = re.sub(r'이런 식으로 이야기해 주세요\.', "", text, flags=re.DOTALL)
    text = re.sub(r"- 어떤 사물이 가장 눈에 띄나요\?.*?- 전체적인 분위기는 어떻게 표현될 수 있나요\?", "", text, flags=re.DOTALL)

    # ✅ 6. 공백 및 줄바꿈 정리
    text = re.sub(r"\s+", " ", text).strip()

    # ✅ 7. 불필요한 기호(-, *, •) 정리 (일관되게 "-" 사용)
    text = re.sub(r"[•*]", "-", text)

    # ✅ 8. 한글과 영어/숫자 사이 띄어쓰기 복원
    text = re.sub(r"([가-힣])([a-zA-Z0-9])", r"\1 \2", text)  # 한글 + 영어/숫자
    text = re.sub(r"([a-zA-Z0-9])([가-힣])", r"\1 \2", text)  # 영어/숫자 + 한글

    return text


# 색보정 백업

def get_color_name(rgb):
    """
    RGB 값을 가장 가까운 색상명으로 변환 (명도 보정 포함)
    """
    # ✅ 명도 조정 적용
    adjusted_rgb = adjust_lightness_and_saturation(rgb)

    min_dist = float('inf')
    closest_color = "알 수 없는 색"

    for name, hex in mcolors.CSS4_COLORS.items():
        r, g, b = mcolors.hex2color(hex)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        dist = np.sqrt((r - adjusted_rgb[0]) ** 2 + (g - adjusted_rgb[1]) ** 2 + (b - adjusted_rgb[2]) ** 2)

        if dist < min_dist:
            min_dist = dist
            closest_color = name
    
    # ✅ 글로벌하게 "회색"이 너무 많으면 자동 보정
    gray_names = ["gray", "grey", "slate", "darkgray", "lightgray", "gainsboro"]
    primary_colors = {
        "blue": "선명한 푸른색",
        "skyblue": "밝은 푸른색",
        "deepskyblue": "진한 푸른색",
        "dodgerblue": "화사한 푸른색",
        "steelblue": "부드러운 푸른색",
        "yellow": "따뜻한 노란색",
        "gold": "황금빛 노란색",
        "red": "강렬한 붉은색",
        "green": "선명한 초록색",
        "limegreen": "맑은 연두색"
    }

    if any(gray in closest_color.lower() for gray in gray_names):
        # 회색 계열이면 원색 계열 중 가장 가까운 색으로 변환
        for key, value in primary_colors.items():
            if key in closest_color.lower():
                closest_color = value
                break

    return closest_color


# 색계열 문제 해결 백업
# 컴퓨터와 인간의 색상차이 조절.
def adjust_lightness_and_saturation(rgb, lightness_factor=1.4, saturation_factor=1.3):
    """
    색상의 명도와 채도를 조정하여 사람이 인식하는 색감과 비슷하게 변환하는 함수.
    - `lightness_factor`: 명도 조정 강도
    - `saturation_factor`: 채도 조정 강도
    """
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # ✅ 명도 범위를 더 넓혀서 밝기 조정 강화
    if l < 0.6:  # 기존 0.5 → 0.6으로 확대 (더 넓은 범위에서 조정)
        l = min(l * lightness_factor, 1.0)  # 1.0을 넘지 않도록 제한
    elif l > 0.8:  # 너무 밝은 색은 과하지 않게 보정
        l = min(l * 1.1, 1.0)  

    # ✅ 채도(Saturation)도 증가하여 원색 계열을 더 살림
    s = min(s * saturation_factor, 1.0)

    # ✅ 특정 계열(파란색, 노란색, 녹색 등)에 대한 추가 보정
    if 180 <= h <= 260:  # 파란색 계열
        l = min(l * 1.4, 1.0)
        s = min(s * 1.3, 1.0)
    elif 40 <= h <= 80:  # 노란색 계열
        l = min(l * 1.5, 1.0)
        s = min(s * 1.4, 1.0)
    elif 80 <= h <= 160:  # 초록색 계열
        l = min(l * 1.4, 1.0)
        s = min(s * 1.3, 1.0)

    # ✅ 새로운 RGB 변환
    new_r, new_g, new_b = colorsys.hls_to_rgb(h, l, s)
    return (int(new_r * 255), int(new_g * 255), int(new_b * 255))


def get_color_name(rgb):
    """
    RGB 값을 가장 가까운 색상명으로 변환 (명도 보정 포함)
    """
    # ✅ 명도 조정 적용
    adjusted_rgb = adjust_lightness_and_saturation(rgb)

    color_categories = {
        "푸른 계열": [
            "blue", "skyblue", "deepskyblue", "dodgerblue", "steelblue", "navy",
            "royalblue", "cornflowerblue", "mediumblue", "midnightblue", "lightskyblue",
            "cadetblue", "powderblue", "slateblue", "darkblue"
        ],
        "초록 계열": [
            "green", "limegreen", "forestgreen", "seagreen", "darkgreen",
            "mediumseagreen", "springgreen", "palegreen", "chartreuse", "lawngreen",
            "mediumaquamarine", "aquamarine", "lightgreen", "darkseagreen"
        ],
        "붉은 계열": [
            "red", "firebrick", "crimson", "darkred", "indianred",
            "lightcoral", "salmon", "darksalmon", "tomato"
        ],
        "노란 계열": [
            "yellow", "gold", "khaki", "goldenrod", "lightgoldenrodyellow",
            "lemonchiffon", "papayawhip", "moccasin", "wheat"
        ],
        "주황 계열": [
            "orange", "darkorange", "coral", "lightsalmon", "sandybrown",
            "chocolate", "burlywood"
        ],
        "보라 계열": [
            "purple", "violet", "orchid", "mediumpurple", "darkorchid",
            "plum", "thistle", "blueviolet", "darkviolet"
        ],
        "갈색 계열": [
            "brown", "saddlebrown", "peru", "tan", "rosybrown",
            "chocolate", "sienna", "darkgoldenrod"
        ],
        "회색 계열": [
            "gray", "darkgray", "lightgray", "slategray", "gainsboro",
            "dimgray", "lightslategray", "darkslategray"
        ],
        "어두운 계열": [
            "black", "dimgray", "darkslategray"
        ],
        "밝은 계열": [
            "white", "snow", "ivory", "whitesmoke", "floralwhite", "linen",
            "beige", "seashell", "oldlace", "cornsilk", "antiquewhite",
            "lavender", "honeydew", "azure", "mintcream", "aliceblue"
        ],
        "청록색 계열": [
            "cyan", "aqua", "turquoise", "darkturquoise", "lightseagreen",
            "mediumturquoise", "paleturquoise"
        ]
    }



    min_dist = float('inf')
    closest_color = ""
    closest_category = ""

    for name, hex in mcolors.CSS4_COLORS.items():
        r, g, b = mcolors.hex2color(hex)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        dist = np.sqrt((r - adjusted_rgb[0]) ** 2 + (g - adjusted_rgb[1]) ** 2 + (b - adjusted_rgb[2]) ** 2)

        if dist < min_dist:
            min_dist = dist
            closest_color = name

    # ✅ 가장 가까운 색상을 대표 색상 계열로 변환
    for category, colors in color_categories.items():
        if any(closest_color.lower() in color for color in colors):
            closest_category = category
            break

    return closest_category

# 주요 색상 추출
def extract_dominant_colors(image, k=5):
    image = image.reshape((-1, 3))
    image = np.float32(image)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    _, counts = np.unique(labels, return_counts=True)
    dominant_colors = palette[np.argsort(-counts)]
    return dominant_colors.astype(int)


============================================================
# 백업
# 컴퓨터와 인간의 색상차이 조절.
def adjust_lightness_and_saturation(rgb, lightness_factor=1.4, saturation_factor=1.3):
    """
    색상의 명도와 채도를 조정하여 사람이 인식하는 색감과 비슷하게 변환하는 함수.
    - `lightness_factor`: 명도 조정 강도
    - `saturation_factor`: 채도 조정 강도
    """
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # ✅ 명도 범위를 더 넓혀서 밝기 조정 강화
    if l < 0.6:  # 기존 0.5 → 0.6으로 확대 (더 넓은 범위에서 조정)
        l = min(l * lightness_factor, 1.0)  # 1.0을 넘지 않도록 제한
    elif l > 0.8:  # 너무 밝은 색은 과하지 않게 보정
        l = min(l * 1.1, 1.0)  

    # ✅ 채도(Saturation)도 증가하여 원색 계열을 더 살림
    s = min(s * saturation_factor, 1.0)

    # ✅ 특정 계열(파란색, 노란색, 녹색 등)에 대한 추가 보정
    if 180 <= h <= 260:  # 파란색 계열
        l = min(l * 1.4, 1.0)
        s = min(s * 1.3, 1.0)
    elif 40 <= h <= 80:  # 노란색 계열
        l = min(l * 1.5, 1.0)
        s = min(s * 1.4, 1.0)
    elif 80 <= h <= 160:  # 초록색 계열
        l = min(l * 1.4, 1.0)
        s = min(s * 1.3, 1.0)

    # ✅ 새로운 RGB 변환
    new_r, new_g, new_b = colorsys.hls_to_rgb(h, l, s)
    return (int(new_r * 255), int(new_g * 255), int(new_b * 255))


def get_color_name(rgb):
    """
    RGB 값을 가장 가까운 색상명으로 변환 (명도 보정 포함)
    """
    # ✅ 명도 조정 적용
    adjusted_rgb = adjust_lightness_and_saturation(rgb)

    color_categories = {
        "푸른 계열": [
            "blue", "skyblue", "deepskyblue", "dodgerblue", "steelblue", "navy",
            "royalblue", "cornflowerblue", "mediumblue", "midnightblue", "lightskyblue",
            "cadetblue", "powderblue", "slateblue", "darkblue"
        ],
        "초록 계열": [
            "green", "limegreen", "forestgreen", "seagreen", "darkgreen",
            "mediumseagreen", "springgreen", "palegreen", "chartreuse", "lawngreen",
            "mediumaquamarine", "aquamarine", "lightgreen", "darkseagreen"
        ],
        "붉은 계열": [
            "red", "firebrick", "crimson", "darkred", "indianred",
            "lightcoral", "salmon", "darksalmon", "tomato"
        ],
        "노란 계열": [
            "yellow", "gold", "khaki", "goldenrod", "lightgoldenrodyellow",
            "lemonchiffon", "papayawhip", "moccasin", "wheat"
        ],
        "주황 계열": [
            "orange", "darkorange", "coral", "lightsalmon", "sandybrown",
            "chocolate", "burlywood"
        ],
        "보라 계열": [
            "purple", "violet", "orchid", "mediumpurple", "darkorchid",
            "plum", "thistle", "blueviolet", "darkviolet"
        ],
        "갈색 계열": [
            "brown", "saddlebrown", "peru", "tan", "rosybrown",
            "chocolate", "sienna", "darkgoldenrod"
        ],
        "회색 계열": [
            "gray", "darkgray", "lightgray", "slategray", "gainsboro",
            "dimgray", "lightslategray", "darkslategray"
        ],
        "어두운 계열": [
            "black", "dimgray", "darkslategray"
        ],
        "밝은 계열": [
            "white", "snow", "ivory", "whitesmoke", "floralwhite", "linen",
            "beige", "seashell", "oldlace", "cornsilk", "antiquewhite",
            "lavender", "honeydew", "azure", "mintcream", "aliceblue"
        ],
        "청록색 계열": [
            "cyan", "aqua", "turquoise", "darkturquoise", "lightseagreen",
            "mediumturquoise", "paleturquoise"
        ]
    }
    
    
    # ✅ 사용자의 입력 유형 분석 (작품 정보 요구 vs 감상 표현)
def classify_user_input(user_input):
    """
    사용자의 입력이 작품 설명을 요구하는지(1-1) vs 자신의 감상을 말하는지(1-2) 분류하는 함수.
    """
    keywords_info = ["이 작품", "설명", "배경", "작가", "의미", "당시 상황"]
    keywords_feeling = ["느낌", "분위기", "인상적", "마음에 들어", "생각", "의견"]

    if any(keyword in user_input for keyword in keywords_info):
        return "info"  # 작품 설명 요청 (1-1)
    elif any(keyword in user_input for keyword in keywords_feeling):
        return "feeling"  # 감상 표현 (1-2)
    return "unknown"