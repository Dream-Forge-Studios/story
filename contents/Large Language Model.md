---
date: '2024-01-30'
title: 'Large Language Model에 관하여'
categories: ['Large Language']
summary: 'LLM을 이해하면서 어떻게 하면 내가 원하는 LLM으로 fine-tuning할 때의 인사이트를 얻어보자'
thumbnail: './test.png'
---

<br>

<br>

나의 목표를 법률 특화 LLM을 만드는 것이다. 이를 위해 LLM을 제대로 이해하면서 fine-tuning할 때의 인사이트를 얻어보려고 합니다.

<div id="Transformer 기반 언어 모델"></div>

## Transformer 기반 언어 모델

### Encoder-only LLM

1. 구조: Encoder 부분만을 포함
2. 작동 방식: 입력 텍스트를 받아 문맥을 이해하고, 이를 바탕으로 텍스트의 의미적 특징을 추출
3. 장점: 빠른 처리 속도, 효율적인 계산, 강력한 문맥 이해 능력.
4. 주요 사용 사례: 텍스트 분류, 개체명 인식, 감정 분석 등 문맥 이해가 중요한 작업.
5. 대표 예시: BERT (Bidirectional Encoder Representations from Transformers).
6. 라벨링 없는 데이터 활용 방법: Masked language model
    - 입력 텍스트의 일부 단어가 마스크(예: [MASK])로 대체되고, 모델은 이 마스크된 단어를 문맥을 기반으로 예측하도록 학습
   
### Decoder-only LLM

1. 구조: Decoder 부분만을 포함
2. 작동 방식: 주어진 입력(시작 토큰 등)으로부터 텍스트를 순차적으로 생성합니다.
3. 장점: 강력한 텍스트 생성 능력, 순차적인 데이터 생성에 적합.
4. 주요 사용 사례: 텍스트 생성, 기계 번역, 챗봇과 같은 대화형 시스템.
5. 대표 예시: GPT (Generative Pre-trained Transformer) 시리즈.
6. 라벨링 없는 데이터 활용 방법: Causal language model
    - 텍스트를 순차적으로 생성하며, 이전에 나온 단어들을 기반으로 다음 단어를 예측

### Encoder-Decoder LLM

1. 구조: Encoder와 Decoder 모두를 포함합니다.
2. 작동 방식: Encoder가 입력 텍스트의 문맥을 이해하고 이를 Decoder에 전달하여, Decoder가 이를 바탕으로 적절한 출력(텍스트, 번역 등)을 생성합니다.
3. 장점: 높은 유연성, 복잡한 작업 수행 능력, 입력과 출력 간의 복잡한 관계 이해.
4. 주요 사용 사례: 기계 번역, 요약, 이미지 캡셔닝, 질문 응답 시스템.
5. 대표 예시: T5 (Text-to-Text Transfer Transformer), Google의 BERT-to-BERT.
6. 라벨링 없는 데이터 활용 방법: Span Corruption
   - 원본 텍스트에서 임의의 연속된 토큰(span)을 선택하여 마스킹하고, 모델이 이 마스킹된 span을 예측하도록 하는 과정

### 비교 분석
- Encoder-Only LLM: 문맥 이해와 분석에 강점을 가지며, 정보 추출 및 분류 작업에 적합
- Decoder-Only LLM: 순차적 텍스트 생성에 강점을 가지며, 창의적인 텍스트 작성이 필요한 작업에 유용
- Encoder-Decoder LLM: 입력과 출력 간의 복잡한 관계를 이해하고 변환하는데 강점을 가지며, 번역이나 요약과 같이 복잡한 변환 작업에 적합

<div id="raw corpora 활용"></div>

## raw corpora 활용

언어 모델을 학습하기 위해서 라벨링된 데이터를 준비하는 과정은 시간이 많이 소요되고 비용이 많이 들 수 있으며, 데이터의 범위나 다양성이 제한될 수 있습니다.

<br>

이러한 문제를 해결하기 위해, 원시 말뭉치(raw corpora)를 활용한 비지도 학습 방식이 제안되었습니다.

<br>

이 방식에서는 데이터 내의 일부 단어를 '마스킹'하고 모델에 이를 예측하도록 합니다. 특히, 언어 생성을 주 목적으로 하는 경우 'Causal Language Modeling'이라는 접근 방식이 사용됩니다.

<img style="width: 80%; margin-top: 40px;" id="output" src="./llm/casual.PNG">

### Emergent Ability

그런데 단순히 뒷말을 예측하도록 학습시켰는데, 의도하지 않은 능력들이 생기는 것을 발견하였습니다. 

<br>

OpenAI의 Alec Radfold가 단순히 뒤에 나올 단어를 예측하는 언어 모델을 RNN으로 만들고 있었는데, 긍정적인 말에 반응하는 뉴런과 부정적인 말에 반응하는 뉴런이 있다는 것을 발견하였으며,

<br>

트렌스포머 기반 언어 모델에서는 문장을 이해하고, 요약하고, 대답하는 등의 능력이 생기는 것을 확인하면서, LLM의 주목하기 시작하였습니다.

<div id="LLM의 발전 과정"></div>

## LLM의 발전 과정

### GPT-1

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="./llm/gpt-1.PNG">

GPT-1은 트렌스포머의 디코더를 이용하여 사전 학습된 모델을 가지고, 다양한 Downstream Task에 사전학습된 모델을 Fine-tuning하여 활용하였습니다.

<br>

Extract 토큰을 통해 특정 Downstream Task을 식별하는데 사용되며, 사전 학습 모델 뒤에 추가되는 Linear 레이어는 출력을 태스크에 적합한 형태로 매핑합니다.

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="./llm/pretrained.PNG">

GPT-1에서 주목할 연구 결과는 명시적인 인간 라벨링이 필요 없으며, 대규모 원시 데이터를 활용하는 비지도학습 활용을 통한 사전 학습이 특정 NLP 태스크에서 효과적이라는 것을 깨달았습니다.

### GPT-2

GPT-2는 사전학습만으로 다양한 Task 처리가 가능함을 주장합니다.

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="./llm/gpt-2.PNG">

데이터 필터링을 통해 고품질 데이터로 모델 크기를 충분히 키워서 학습하니 추가학습 없이 Downstream Task을 Zero-shot으로 해결할 수 있는 가능성을 보입니다.

### GPT-3

#### scaling law

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="./llm/ScalingLaw.PNG">

위의 그래프는 모델의 크기에 따라 데이터의 양과 컴퓨팅 자원 대비 loss 값을 정리한 것입니다.

1. 데이터가 증가함에 따라 모델 크기도 증가해야한다.
2. 모델 크기가 클 경우 적은 데이터로도 loss가 잘 줄어든다.
3. 계산 자원이 제한되어 있을 때, 최적의 loss에 도달하지 못한다.

이러한 연구를 통해 open ai는 가지고 있는 데이터셋의 수에 맞는 적절한 크기에 모델을 출시하게 됩니다. 

<img style="width: 70%; margin-top: 40px; margin-bottom: 40px;" id="output" src="./llm/gpt-3.PNG">

OpenAI는 학습 데이터의 양과 계산 시간을 고려하여, 최적의 loss에 도달하기 위한 적절한 파라미터 크기를 갖춘 GPT-3 175B 모델을 개발했습니다.

<br>

그런데 이렇게 큰 모델을 downstream task를 위해 fine-tuning을 하기에는 적절하지 않습니다.

<br>

GPT-3 175B 모델의 실행에는 175GB의 GPU 메모리가 필요합니다. 16비트 정밀도를 사용할 경우, 메모리 요구량이 약 300GB까지 증가합니다. 

<br>

이 모델을 전체적으로 재학습하기 위해서는 기존 요구량의 5배를 초과하는 GPU 메모리가 필요하므로, 재학습이 어려워 Few-shot Leaning을 통해 downstream task에 적용하는 연구가 이루어졌습니다.

#### Few-shot

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="./llm/shot.PNG">

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="./llm/few-shot.PNG">

Few-shot Leaning을 통해 downstream task 능력이 향상 되는 것을 볼 수 있습니다.

<br>

그런데 여기서 또한 주목해야할 점은 모델 크기가 커지면서(6.7B ~) Emergent Ability가 급격하게 생기는 것을 볼 수 있습니다.

<div id="Model Alignment"></div>

## Model Alignment

GPT 시리즈는 Causal Language Modeling을 기반으로 하여 주로 문장의 뒷부분을 예측하는 데 초점을 맞추고 있으며, 구체적인 문제 해결 방법을 직접적으로 학습하는 것은 아닙니다.

<br>

따라서, 프롬프트 이해 능력을 향상시키고, zero-shot 성능을 개선하는 방향으로 추가 연구가 진행되었습니다.

### Instruction tuning (FLAN)

FLAN은 LLM을 인간의 선호도(feedback)를 바탕으로 미세 조정(fine-tuning)하는 방법을 탐구합니다.

<br>

이는 모델이 다른 태스크들에 대한 Instruction tuning을 통해 학습함으로써, 처음 접하는 태스크에서도 성능이 향상됬다는 점을 주목할만 합니다.

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="llm/dataset.PNG">

위와 같은 다양한 태스크의 데이터셋을 기반으로, 각 데이터셋마다 인간이 직접 10개의 템플릿을 수작업으로 작성합니다. 이 템플릿들은 해당 태스크를 다양한 방식으로 설명합니다.

<br>

더불어, 다양성을 증가시키기 위해 각 데이터셋에 최대 3개의 추가적인 템플릿을 포함시키는데, 이들은 기존 태스크를 '반대로 뒤집는' 형태로 구성됩니다.

<br>

예를 들어, 감정 분석 태스크의 경우, 원래의 태스크는 텍스트(예: 영화 리뷰)의 감정을 분류하는 것이지만, 추가된 템플릿에서는 사용자에게 특정한 감정을 가진 영화 리뷰를 생성하도록 요구할 수 있습니다.

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="llm/template.PNG">

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="llm/zero-perform.PNG">

FLAN을 활용한 모델은 live LLM에 비해 zero-shot(한번도 보지 못한 태스크) 성능에 있어서 큰 향상을 보았으며, 해당 태스크에 특화한 Supervised model에 비교해서도 나은 성능을 보이는 경우도 있었습니다.

<img style="width: 80%; margin-top: 40px; margin-bottom: 40px;" id="output" src="llm/modelsize.PNG">

그런데 주의해야할 점은 모델 크기가 작으면 Instruction tuning이 오히려 해가 된다는 연구가 있습니다.

<br>

위의 도표에서는 8B 이후 부터 도움이 된다고 나와있지만, 이것은 2020년 당시의 연구 결과이며 현재는 6B 이후 부터 도움이 된다고 언급됩니다.

### Instruction tuning의 문제점



<div id="데이터 품질 향상"></div>

## 데이터 품질 향상

1. Quality Filtering

- 목적
  - 잘못된 정보, 오류가 있는 텍스트, 불완전한 문장 등의 저품질 데이터를 제거
  - 편향된 내용이나 부적절한 언어 사용을 포함한 데이터를 필터링
  - 고품질의 데이터만을 사용

- 방법
  - 자동화된 필터링: 특정 키워드나 패턴을 기반으로 자동화된 스크립트를 사용하여 저품질의 데이터를 걸러냅니다. 예를 들어, 욕설이나 혐오 표현을 포함한 텍스트를 자동으로 제거합니다.
  - 수동 검토: 특정 경우에는 데이터셋을 사람이 직접 검토하여 품질을 보장합니다. 이는 특히 민감한 주제나 복잡한 문맥에서 중요합니다.
  - 통계적 방법: 데이터의 분포, 일관성, 완전성 등을 평가하는 통계적 방법을 사용하여 품질을 검증합니다.
  - 소스 기반 필터링: 신뢰할 수 있는 출처에서만 데이터를 수집하거나, 특정 출처의 데이터를 우선적으로 사용합니다.
  - 다양성 및 포괄성 검증: 데이터셋이 다양한 언어, 문화, 주제를 포함하도록 하여, 모델이 광범위한 지식과 관점을 학습하도록 합니다.
  
2. De-duplication

- 목적
  - 학습 데이터셋에서 동일하거나 매우 유사한 콘텐츠를 제거합니다. 이는 데이터셋의 다양성을 보장하고 모델의 일반화 능력을 향상시킵니다.
  - 학습 효율성 향상: 중복 데이터를 제거함으로써, 모델이 불필요하게 동일한 정보를 반복 학습하는 것을 방지합니다.
  - 과적합 방지: 중복 데이터가 많을 경우 모델이 특정 패턴이나 편향에 과도하게 적응할 수 있습니다. De-duplication은 이러한 과적합을 방지합니다.

- 방법
  - 문자열 일치 기반: 가장 간단한 형태는 완전히 일치하는 문자열을 찾아 제거하는 것입니다. 이는 완전한 중복을 제거하는 데 효과적입니다.
  - 유사도 기반: 더 복잡한 방법은 텍스트 간의 유사도를 계산하여, 높은 유사도를 보이는 텍스트를 중복으로 간주하고 제거합니다.
  - 해시 기반: 텍스트 블록의 해시 값을 생성하고, 이를 비교하여 중복을 식별합니다. 이 방법은 대규모 데이터셋에서 효율적으로 중복을 검출할 수 있습니다.
  - N-gram 분석: 텍스트의 N-gram(연속된 N개의 항목)을 분석하여 중복성을 평가합니다. 이는 부분적 중복을 식별하는 데 유용할 수 있습니다.

<div id="Emergence"></div>

## Emergence

OpenAI의 Alec Radfold가 단순히 뒤에 나올 단어를 예측하는 언어 모델을 RNN으로 만들고 있었는데, 긍정적인 말에 반응하는 뉴런과 부정적인 말에 반응하는 뉴런이 있다는 것을 발견

<br>

언어 모델링을 하다 보면 의도하지 않은 능력이 얻게 된다는 것을 발견

<br>

1. DB 인스턴스 클래스: dn.t2.micro
2. 스토리지 자동 조정 활성화 off
3. 백업 보존 기간: 0days
4. 스토리지 사용량 20GiB 넘기지 않기

이렇게 데이터베이스 생성을 마쳤으면, 한국에 맞추어 파라미터 그룹 생성 및 설정을 해주어야한다.

<br>

**파라미터 그룹**

1. RDS > 파라미터 그룹 > 파라미터 그룹 생성
2. 작업 > 편집
3. 값 변경 
   - timezone: Asia/Seoul 
   - character set: utf4mb4
   - collation: utf8mb4_general_ci
4. RDS > 데이터베이스 > 수정
5. DB 파라미터 그룹을 생성한 파리미터 그룹으로 설정
6. 즉시 적용 선택 후 수정 완료되면 재부팅

이제 모든 설정을 맞췄으니 AWS RDS를 DBeaver에 연결해보자

<br>

**DBeaver에 연결**

- DBeaver란?

다양한 데이터베이스 관리 시스템(DBMS)에 대해 통합 데이터베이스 관리 및 개발 도구를 제공하는 오픈 소스 소프트웨어입니다.

<br>

https://dbeaver.io/download/ 해당 링크를 통해 설치할 수 있습니다.

- DBeaver에 연결하기

1. 우선 RDS 데이터베이스의 퍼블릭 엑세스를 예라고 설정한다. 
   - 실제 서비스를 할 때는 아니오로 설정해야한다.
   - 다만 아니오일 때 해당 데이터베이스의 접근하려면 복잡함으로 우선 개발을 마치기 전까지는 예로 설정한다.
2. VPC 보안 그룹 클릭 > 인바운드 규칙 편집 > 규칙 추가 > 내 IP
   - 기본으로 냅두면 연결이 되지 않으니 수정해야한다.
3. DBeaver를 열고 좌측 상단에 콘센트 모양에 +가 붙어있는 아이콘을 선택한다.

<img style="width: 30%; margin-bottom: 40px;" id="output" src="https://velog.velcdn.com/cloudflare/shawnhansh/ff70b719-786a-4145-b33e-22d5ddad7657/4.png">
<img style="width: 80%; margin-bottom: 40px;" id="output" src="https://velog.velcdn.com/cloudflare/shawnhansh/b249e387-f40e-4d8c-9493-374788d5e7ad/image.png">
<img style="width: 80%; margin-bottom: 40px;" id="output" src="https://velog.velcdn.com/cloudflare/shawnhansh/281dca40-f36d-4238-9811-edc2b81244a4/7.png">

- Server Host : 엔드포인트
- Port : 포트
- Database : DB 이름 (초기 데이터베이스 이름 설정 하지 않았으면 비워둠)
- Username : 마스터 사용자 이름
- Password : 비밀번호

<br>

**Lambda에 연결**

<img style="width: 70%; margin-top: 40px;" id="output" src="aws/architecture.PNG">

<br>

이 부분에서 문제가 많았습니다.

1. lambda가 RDS에 연결하기 위해서는 같은 VPC에 위치하여야 합니다.
2. 그런데 vpc에 lambda가 위치하면 외부 api를 호출할 수 없습니다.
3. 그래서 NAT instance를 만들어 이를 통해 외부 api를 호출하여야 합니다.

<br>

여러 시행착오 끝에 그림과 같은 구조로 해결하였습니다.

<br>

- lambda에서 설정
  1. IAM > 역할 > lambda 함수에 해당하는 역할명
  2. 권한 추가: AWSLambdaVPCAccessExecutionRole (lambda에서 VPC를 사용가능하게 함)
  3. Lambda 함수 > 구성 > VPC 편집
  4. NAT instance와 연결된 서브넷 선택
  5. RDS와 동일한 보안 그룹 선택

<div id="NAT Instance"></div>

## NAT Instance

AWS에서 네트워크를 설계할때 NAT의 사용은 필수적입니다. VPC 내에서는 외부 인터넷 접속이 불가하기 때문입니다.

<br>

AWS에서 강력하게 밀어주는 최신식 NAT Gateway 서비스를 이용해서 손쉽게 사설망 외부 통신을 할 수 있지만, 비용 문제가 발생하므로 본 프로젝트에서는 NAT Instance를 사용합니다.

<br>

NAT 인스턴스는 EC2 인스턴스를 NAT용으로 설정해 사용하는, 이른바 불안정하고 한물 간 구식 기술이지만 저렴하며 프리티어를 사용하면 무료로 사용할 수 있습니다.

<br>

**사용 방법**

- EC2 설정

1. EC2 > 인스턴스 > 인스턴스 시작
2. 생성 옵션 설정
   - Application and OS Images (Amazon Machine Image) 
     - EC2 인스턴스를 시작하는 데 사용되며, 운영 체제(OS), 애플리케이션 서버 및 애플리케이션과 같은 소프트웨어가 미리 구성되어 있습니다.
     - NAT Instance로 사용할 것이므로 NAT를 검색하여 프리 티어 사용 가능한 AMI를 선택합니다.
   - 프리티어 사용 가능
     - 일정한 제한된 사용량에 대해서 무료로 제공
   - 키 페어(Key Pair)
     - 보안상의 목적으로 사용되는 한 쌍의 암호화 키
   - 스토리지 구성 
     - SSD 기반은 빠르지만 고비용이고 HDD는 저렴하지만 SSD에 비해 읽기/쓰기 속도가 느림
     - 본인은 저렴한 HDD 기반의 standard(Magnetic) 선택
   - 고급 세부 정보 
     - 많은 경우, 기본 설정만으로도 EC2 인스턴스를 충분히 구동하고 관리할 수 있습니다. 특별한 설정이나 구성이 필요하지 않다면 고급 세부 정보를 변경하지 않아도 됩니다.
     - 고급 세부 정보는 보다 세부적인 설정이 필요한 숙련된 사용자나 특정 요구 사항이 있는 경우에 유용합니다. 예를 들어, 특정 소프트웨어를 자동으로 설치하려면 사용자 데이터를 설정할 수 있습니다.
3. 인스턴스 생성 후 선택 > 작업 > 네트워킹 > 소스/대상 확인변경 중지
   - 일반적으로, AWS에서 실행되는 각 EC2 인스턴스는 '소스/대상 검사(Source/Destination Check)' 기능이 활성화되어 있습니다.
   - 이 기능은 인스턴스가 받거나 보내는 트래픽이 해당 인스턴스를 목적지나 출발지로 하는지를 확인합니다. 즉, 인스턴스는 자신이 최종 목적지나 출발점인 트래픽만 처리합니다.
   - 하지만, NAT 인스턴스의 경우 이 동작이 적합하지 않습니다. NAT 인스턴스는 다른 인스턴스의 트래픽을 인터넷으로 중계하는 역할을 하기 때문에, 자신이 트래픽의 최종 목적지나 출발점이 아닌 경우에도 트래픽을 수신하고 전송할 수 있어야 합니다.
4. 탄력적 IP > 탄력적 IP 주소 할당 > 나머지 냅두고 태그: 키 Name, 값 NAT-coopangpl
5. 탄력적 주소 선택 후 작업 > 탄력적 IP 주소 연결 > NAT Instance 연결
   - 탄력적 IP는 AWS 클라우드 환경에서 고정된 공용 IP 주소를 제공합니다. 이를 통해 인터넷에서 일관된 주소를 사용하여 인스턴스에 접근할 수 있습니다.
6. VPC > 라우팅 테이블 > 라우팅 테이블 생성 > 대상: 0.0.0.0/0, 인스턴스 아까 만든 NAT Instance
7. 서브넷이 총 4개가 있는데 2개는 internet gateway 라우팅 테이블에 연결, 2개는 NAT Instance에 연결된 서브넷에 연결합니다.
8. 그 후 lambda 함수를 NAT Instance에 연결된 서브넷으로 지정하면 외부 api 호출이 가능합니다.

<div id="Amazon EventBridge"></div>

## Amazon EventBridge

Amazon EventBridge는 AWS에서 제공하는 서버리스 이벤트 버스 서비스로, 다양한 소스에서 발생하는 이벤트를 감지하거나 원하는 실행 일정을 정하면, 그에 따라 AWS 서비스, 사용자가 정의한 로직, 또는 서드파티 애플리케이션에 자동으로 반응할 수 있도록 돕습니다.

<br>

이제 모든 준비를 마쳤으니, Amazon EventBridge를 통해 원하는 시간마다 aws Lambda 함수가 실행되게 설정해봅시다. 

<br>

**사용 방법**

<br>

Amazon EventBridge > 일정 > 일정 생성으로 가서 일정 세부 정보 지정을 입력합니다.

<br>

본 프로젝트에서는 매일 오전 10시 15분에 실행되게 하므로 Cron 표현식을 "15 10 * * ? *"로 입력합니다.

<br>

다음으로는 대상 선택을 합니다. aws lambda 함수를 실행시킬 것 이므로 모든 api > aws lambda > invoke > 실행할 함수 선택한 뒤 나머지 입력한 후 일정을 생성합니다.

<div id="Amazon RDS"></div>
