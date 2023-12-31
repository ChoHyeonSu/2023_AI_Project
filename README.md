# 2023 AI Project

## 프로젝트 개요

> 본 프로젝트는 인공지능 기술을 활용하여 사용자가 과거에 들었던 노래와 유사한 감정 분포를 가지는 노래 플레이리스트를 생성하는 새로운 접근 방식을 제시합니다. 이 접근 방식은 노래 가사의 감정적 요소를 분석하여, 사용자의 감정 상태에 맞는 음악을 추천하는 방법론에 기반합니다.
> 프로젝트의 목표는 단순한 오락의 수단을 넘어서, 청취자의 감정 상태에 긍정적인 영향을 미치는 음악 플레이리스트 생성 시스템을 구축하는 것입니다. 현대 음악 스트리밍 서비스에서 주로 사용되는 장르 기반 추천 시스템의 한계를 넘어서, 더욱 세밀하고 개인화된 사용자 경험을 제공하기 위해 감정 분석을 통합합니다. 이 프로젝트는 인공지능 모델을 활용하여 노래 가사의 감정 분류를 진행하고, 이를 바탕으로 각 노래의 감정 분포를 분석합니다. 이러한 감정 분포는 플레이리스트 생성의 기준으로 사용되며, 유사한 감정 분포를 가진 노래들을 모아 사용자에게 제공합니다. 이 방식은 기존의 단일 감정 태깅 방식을 넘어서, 노래의 다양한 감정적 요소를 포괄적으로 반영하며, 사용자에게 보다 풍부하고 다채로운 음악 경험을 제공할 것으로 기대됩니다.

## 가사별 감정 분류를 통한 노래 플레이리스트 생성


> Download Pretrained Model
> 사전 훈련된 모델을 사용하여 본 프로젝트를 효율적으로 시작할 수 있습니다. 아래 링크에서 모델을 다운로드하세요:
https://drive.google.com/file/d/1zp8Fprdyr6obOivU8ip0XVNp8yrw9XHX/view?usp=sharing

### 제안 방안

노래를 감정별로 분류하여 유사한 감정의 노래를 만든 플레이리스트를 생성

이를 위해서 2 step으로 나눠서 진행

1step은 노래에 맞는 감정 분류하는 모델 생성

2step은 이 결과를 활용해서 플레이리스트 생성

### 제안 방안 Step 1

**Task** 노래 감정 분류

**가설**

노래 가사를 이용해서 노래의 감정을 분류할 때, 

가사 전체를 하나의 context로 보고, 노래의 감정을 하나로 결정하는 것 보다 

가사별 감정을 분류하여 합한 전체 감정 비율을, 노래의 감정으로 결정하는 것이 노래 domain에 더 적합하다. 

가설의 근거)
노래 한곡에 하나의 감정만 들어 있는 경우가 없기 때문, 

더 자세히 얘기하면 노래는 기승전결이 있는 하나의 스토리인데 이 스토리가 하나의 감정으로 표현될 수 없기 때문

**실험**

input: 노래 한 곡의 가사 전체

output: 감정 분포( 가사별 감정의 summation )

baseline: output이 감정 분포가 아닌 하나의 감정 라벨로 되어있음

baseline과의 비교 방법( 우리 모델이 노래 감정을 더 잘 분류한다를 증명하는 방법 ):

baseline모델도 어차피 확률값으로 클래스를 분류할테니, logit값을 보고 최대인 애를 고르는게 아니라, 그 분포를 모두 출력

이후 우리의 모델과의 분포를 비교

정답값은 가사-감정으로 분류해 놓은 애들의 분포

baseline: 노래 - 감정 라벨로 학습한 모델

baseline output - 감정 하나 (max)

 

output을 감정 6개에 대한 확률값 → ours 결과랑 똑같이 (비율로)

대조군: 가사- 감정 라벨로 학습한 모델에 input으로 노래 한곡의 가사를 넣어 줬을 때 가사별 분석 결과를 합친 감정 분포

golden label: 가사-감정의 pseudo labeling 이 된 것들 중에서 노래 한곡당 감정 분포

### 제안 방안 Step 2

Task: 플레이리스트 생성

→ 새로운 모델을 사용하는 것은 아님. 

→ Step 1의 결과를 활용하여 유사한 감정 분포를 갖는 플레이리스트 생성

Baseline: 랜덤으로 노래를 모은 플레이리스트

Our methods: 감정 분포가 유사한 노래를 모은 플레이리스트

## 설치 방법
이 프로젝트를 로컬 시스템에 설치하기 위해 다음 단계를 따르세요:
1. ```bash
   git clone https://github.com/ChoHyeonSu/2023_AI_Project.git
   ```
2. ```bash
   pip install -r requirements.txt
   ```

## 사용 방법
프로젝트를 사용하는 방법에 대한 간략한 안내입니다:
1. ```bash
   python playlist_generator.py
   ```
2. 사용자의 감정 상태 또는 선호도에 맞는 플레이리스트가 생성됩니다.

## 기여 방법
프로젝트 기여는 다음과 같은 방법으로 할 수 있습니다:
- Issue Tracking: 버그 보고 및 개선 사항 제안
- Pull Request: 코드 개선 및 새로운 기능 추가

기여에 대한 자세한 정보는 `CONTRIBUTING.md` 파일을 참조하세요.

## License
이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

## Contact
프로젝트에 대한 질문이나 제안이 있으시면 다음 연락처로 문의해 주세요:
- Email: jhs18301@gmail.com
- Github: github.com/ChoHyeonSu
