# AI
SKT FLY AI Challenger 6기 패기반 TEAM3 AI 개발
![Image](https://github.com/user-attachments/assets/ff530c91-6f84-476a-be09-519cb90bf350)

  
[서비스 링크(배포 예정)](http://127.0.0.1:8000/)  
  
  
[Git Hub 링크](https://github.com/SKT-FLY-AI-project)  
  

# 0\. Getting Started (시작하기)

```
# << Back-End >>

# +) application.properties에 key 값들 넣어주기
# 0-1. 모든 필요 packages 설치 확인 
# 0-2. 가상환경 생성
    
    ```python
    cd Back-End
    python -m venv venv
    ```
    
# 1. 가상환경 활성화
    
    ```python
    source venv/bin/activate  # Mac/Linux
    venv\Scripts\activate     # Windows
    ```

# 2-0. FastAPI 설치
    
    ```python
    pip install fastapi uvicorn
    ```
    
# 2-1. 서버 실행
    
    ```python
    uvicorn main:app --reload
    ```
    
# 3. 웹 브라우저에서 확인: [127.0.0.1:8000](http://127.0.0.1:8000/)

# << Front-End >>

# +) .env 파일 만들고 key 값들 넣어주기
# 1. 프로젝트 실행
$ 
```

  
  

# 1\. Project Overview (프로젝트 개요)

-   프로젝트 이름: VisionArt – AI 기반 시각장애인을 위한 그림 음성 설명 앱
-   프로젝트 설명: ‘SKT FLY AI Challenger 6기’ 패기반 TEAM3 ([SKT FLY AI Challenger](https://www.skttechacademy.com/))
<!-- -   프로젝트 수상 결과 :  -->
-   프로젝트 목적적 : 시각장애인들이 예술 작품을 감상할 때, AI가 다양한 관점과 견해로 작품의 시각적 정보를 상세히 설명하여 시각적 접근성을 높이는 앱을 개발합니다. AI는 단순한 묘사에 그치지 않고 **사실적 설명**, **예술적 해석**, **역사적 배경, 비평적 견해** 등 여러 시각을 제공하여 사용자(장애인 또는 비장애인) 스스로 작품을 깊이 이해하고 판단할 수 있도록 돕습니다. 기존의 제한된 해설 방식 대신, **다양한 견해의 설명을 통해 예술 감상의 폭을 넓히고 더욱 풍부한 경험을 제공**합니다.
- 프로젝트 설명 :
    - **문화 예술 접근권 확대:** 시각장애인들은 예술 작품을 직접 감상하기 어려워 문화적 소외를 경험하는 경우가 많습니다.
    - **기존 해설 서비스의 한계:** 기존 음성 해설 서비스는 제한된 작품만을 대상으로 하며, 미리 준비된 해설문만 제공됩니다.
    - **맞춤형 설명의 필요성:** AI 기반의 맞춤형 설명으로 각 사용자의 관심사나 설명 난이도를 조절해 더욱 깊이 있는 감상 경험을 제공할 수 있습니다.

  

# 2\. Team Members (팀원 및 팀 소개)

| 김민 | 김민수 | 곽현정 | 안치욱 | 이형록 | 최예준 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| ![김민](https://avatars.githubusercontent.com/u/59240554?v=4) | ![김민수](https://avatars.githubusercontent.com/u/45546892?v=4) | ![곽현정](https://avatars.githubusercontent.com/u/89788679?v=4) | ![안치욱](https://avatars.githubusercontent.com/u/124045490?v=4) | ![이형록](https://avatars.githubusercontent.com/u/195076095?v=4) | ![최예준](https://avatars.githubusercontent.com/u/144091030?v=4) |
| AI | AI | BE | BE | FE | FE |
| [GitHub](https://github.com/doraemon49) | [GitHub](https://github.com/minkim507) | [GitHub](https://github.com/kwakrhkr59) | [GitHub](https://github.com/KooWihC) | [GitHub](https://github.com/hrl090512) | [GitHub](https://github.com/zzunni) |

  
  

# 3\. Key Features (주요 기능)

- 작품 탐지
- AI 설명
- 마이페이지 (이미지와 설명 내역 확인)
- 회원가입
  

# 4\. Tasks & Responsibilities (작업 및 역할 분담)

|   |   |   |   |
| --- | --- | --- | --- |
| 김민 | ![김민](https://avatars.githubusercontent.com/u/59240554?v=4) |  - AI - CNN 그림 탐지지   |
| 김민수 | ![김민수](https://avatars.githubusercontent.com/u/45546892?v=4) |   -   AI - LLM 그림 설명 생성   |
| 곽현정 | ![곽현정](https://avatars.githubusercontent.com/u/89788679?v=4) |   -   백엔드 개발    |
| 안치욱 | ![안치욱](https://avatars.githubusercontent.com/u/124045490?v=4) |   -   백앤드 개발   |
| 이형록 | ![이형록](https://avatars.githubusercontent.com/u/195076095?v=4) |   -   프론트 개발 -   디자인    |
| 최예준 | ![최예준](https://avatars.githubusercontent.com/u/144091030?v=4) |   -   프론트엔드 개발 -   디자인  |
  
  

# 5\. Technology Stack (기술 스택)

## 5.1 Frotend

[![My Skills](https://skillicons.dev/icons?i=figma,flutter,androidstudio,dart&theme=light)](https://skillicons.dev)

## 5.2 Backend

[![My Skills](https://skillicons.dev/icons?i=python,fastapi&theme=light)](https://skillicons.dev)

## 5.3 Cooperation

[![My Skills](https://skillicons.dev/icons?i=git,notion,discord&theme=light)](https://skillicons.dev)

# 6\. Project Structure (프로젝트 구조)

```
<< Back-End >>
📦 


```

```
<< Front-End >>
📦 

```

  
  

# 7\. Development Workflow (개발 워크플로우)

## 브랜치 전략 (Branch Strategy)

우리의 브랜치 전략은 Git Flow를 기반으로 하며, 다음과 같은 브랜치를 사용합니다.

<< Back-End >>

-   Main Branch
    -   배포 가능한 상태의 코드를 유지합니다.
    -   모든 배포는 이 브랜치에서 이루어집니다.
    -   권한이 있는 사용자만 Merge할 수 있습니다. (Branches Rules)-
-   dev Branch
    -   Main Branch에 업로드되기 전, 개발 단계에서 모든 브랜치의 코드를 통합하는 브랜치입니다.
-   feat/login
    -   로그인 및 회원가입 브랜치입니다.

<< Front-End >>

-   Main Branch
    -   배포 가능한 상태의 코드를 유지합니다.
    -   모든 배포는 이 브랜치에서 이루어집니다.
    -   권한이 있는 사용자만 Merge할 수 있습니다. (Branches Rules)
-   dev Branch
    -   Main Branch에 업로드되기 전, 개발 단계에서 모든 브랜치의 코드를 통합하는 브랜치입니다.