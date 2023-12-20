---
date: '2023-12-05'
title: 'Making Large Language Models Better Data Creators 논문 리뷰'
categories: ['Large Language', 'Data']
summary: 'Making Large Language Models Better Data Creators 완벽 이해하기.'
thumbnail: './test.png'
---

<div id="ABSTRACT"></div>

# ABSTRACT


이 논문에서는 대규모 언어 모델(Large Language Models, LLMs)의 사용이 자연어 처리 분야에서는 큰 발전을 이루었지만, 실제 응용 프로그램에 투입하는 데는 여전히 비용, 응답 시간, 제어, 프라이버시 및 보안 문제 등으로 인한 어려움이 있다고 설명합니다.

<br>

이를 해결하기 위해 인간이 레이블링한 데이터가 필요하지만, 이는 비용과 시간이 많이 듭니다. 

<br>

이에 대한 해결책으로, 논문은 단 하나의 formatting 예시만 필요한 통합 데이터 생성 파이프라인을 제안합니다. 

<br>

이 방법은 비용 효과적이며, 특히  out-of-distribution evaluation(훈련에서 보지 못한 상황)에서 인간 레이블 데이터로 훈련된 모델보다 우수한 성능을 보이는 등, 실제 상황에서의 NLP 시스템 견고성을 향상시키는데 중요한 영향을 미칩니다.

<div id="Introduction"></div>

# Introduction

### LLMs를 레이블러로 사용할 때의 제약

생물의학이나 법률과 같은 전문 분야에서는 데이터가 매우 복잡하고 특수한 정보를 담고 있기 때문에, 이를 정확하게 분석하고 레이블링하기 위해서는 많은 전문 지식과 주의가 필요합니다.

<br>

따라서, 이러한 분야에서 LLMs를 효과적으로 레이블러로 사용하기 위한 데이터를 준비하는 것은 큰 도전이 될 수 있습니다.

### LLMs를 생성기로 사용할 때의 고려사항

LLMs를 효과적으로 데이터 생성기로 사용하기 위해서는, few-shot example을 신중하게 준비해야 합니다(Hartvigsen et al., 2022).

<br>

레이블의 의미를 강조하는 프롬프트를 작성해야 합니다(Wang et al., 2021b; Meng et al., 2022; Ye et al., 2022; Gao et al., 2022). 

예시: "다음 문장이 긍정적인지 부정적인지 분류하시오: [문장]"

<br>

하지만 모든 작업이 의미 있는 레이블을 가지고 있지 않거나, 레이블을 열거할 수 없는 경우가 있습니다.

<br>

예를 들어, '예 vs. 아니오'와 같은 레이블은 문맥 없이는 의미가 없고, 다지선다형 질문의 옵션은 상황마다 달라지는 무한한 레이블 세트를 갖습니다.

### 이 논문에서 제시하는 새로운 데이터 생성 파이프라인

LLM을 통한 데이터 생성을 위한 다양한 접근 방법을 특성화 하기 위한 공식적인 통합 framework를 제시합니다.

<br>

또한 single formatting example만 필요로 하는 새로운 데이터 생성 파이프라인을 제안합니다. 

<br>

이 파이프라인은 여러 종류의 응용 프로그램에 사용될 수 있는 다양하고 다른 종류의 데이터 세트를 만들 수 있으며, 특히 전문 분야에 초점을 맞춘 경우에도 유용합니다.

<br>

이 새로운 파이프라인은 다양한 종류의 데이터셋에 적용 가능하며, 기존 방법들보다 적은 맞춤 설정을 필요로 합니다.

<br>

또한 레이블이 특정 상황이나 문맥에서만 의미를 가지는 데이터에도 적용이 가능합니다.

<div id="2. Formalization of LLM-based data creation"></div>

# 2. Formalization of LLM-based data creation

<div id="Formal Framework and Related Work"></div>

## Formal Framework and Related Work

<img style="width: 50%; margin-bottom: 40px;" id="output" src="./makingImg/labeler.PNG">

$D_U$: 레이블이 없는 데이터

$D_L$: 소수의 레이블이 있는 데이터

$e_l$: 중간 추론 단계 설명

$y$: 생성 레이블

$W_y$: 잘 포맷된 지침 프롬프트

$x$: 학습데이터

### LLM 데이터 생성의 두 가지 전략

**1. labelers로서의 LLM**

- 레이블이 없는 데이터($D_U$)에 대해 레이블을 할당하는데 활용

  <br>

- few-shot setting: 소수의 레이블이 있는 예시($(x_l,y_l) ∈ D_L$)에 조건부로 설정

  <br>
  
- 최근 연구에서 few-shot learning을 할 때, 데이터 샘플의 다양성과 대표성이 중요  (Liu et al., 2022; Rubin et al., 2022; Su et al., 2022).

  <br>

- 법률이나 생물의학과 같은 전문 분야에서는 소수의 예시만 사용할 수 있는 경우가 많아 큰 어려움이 있지만, 본 논문에서는 이 문제를 해결하기 위한 파이프라인 제시(세션 3)

  <br>

- 학습데이터를 중간 추론 단계(Intermediate Reasoning Steps)가 포함되게 구성하는 것($e_l$)이 few shot, zero shot 모두 더 나음
*Intermediate Reasoning Steps 예시

  <br>

문제: "집에 7명의 아이들이 있고, 각각의 아이들에게 3개의 사과를 주려고 합니다. 모두 몇 개의 사과가 필요합니까?"

<br>

일반적인 학습 데이터: 답만 제공됩니다. 예: "21"

<br>

중간 추론 단계가 포함된 학습 데이터:

1. "집에 7명의 아이들이 있습니다."
2. "각 아이에게 3개의 사과를 줍니다."
3. "그러므로, 7명의 아이들 각각에게 3개씩 주어야 하므로, 7 x 3을 계산합니다."
4. "7 x 3은 21입니다."
5. "그래서, 총 21개의 사과가 필요합니다."

**2. generator로서의 LLM**

<br>

labelerx가 입력 x에 기반하여 레이블 y를 예측하는 것과 반대로, generator는 잘 포맷팅된 지시적 프롬프트($W_y$)를 사용하여, 목표 레이블 $y$에 대한 데이터를 생성합니다.

<br>

생성된 결과가 조건부로 설정된 레이블 y에 의존하기 때문에, y는 본질적으로 의미가 있어야 합니다.

예시) y가 영화 리뷰의 감정이라면, Wy는 "영화 리뷰는..."

<br>

*잘 포맷팅된 지시적 프롬프트($W_y$)

1. 명확성(Clearness): 프롬프트는 모델이 수행해야 할 작업을 명확하게 설명합니다. 사용자의 의도와 요구 사항이 분명하게 전달되어야 합니다.

2. 구조화(Structure): 프롬프트는 일관된 형식을 따르며, 필요한 정보를 체계적으로 포함합니다. 이는 모델이 작업을 이해하고 실행하는 데 도움이 됩니다.

3. 지시적(Instructional): 프롬프트는 모델에게 구체적인 지시를 제공합니다. 예를 들어, "다음 문장의 감정을 분석하시오" 또는 "이 문제를 해결하기 위한 단계를 설명하시오"와 같이 구체적인 지시가 포함될 수 있습니다.

4. 목적 지향적(Purpose-Oriented): 프롬프트는 특정 목적이나 작업에 집중되어 있어야 합니다. 이는 모델이 효과적으로 결과를 생성할 수 있도록 합니다.

**3. 그래픽 모델의 사용**

<br>

Graphical Model을 사용하여 시각적으로 표현됩니다. 즉, 어떤 요소가 다른 요소에 어떻게 영향을 미치는지, 어떤 요소가 독립적으로 작동하는지를 나타냅니다.

<div id="3. Example-based Data Creation"></div>

# 3. Example-based Data Creation

<img style="width: 50%; margin-bottom: 40px;" id="output" src="./makingImg/Overview.PNG">

$x_f, y_f$: initial seed formatting example

$W_I$: 지침

<div id="Instruction"></div>

## Instruction

목표는 모델 M이 입력 포매팅 예시($x_f, y_f$)와 같은 형식으로 다양한 예시 세트를 생성하는 것입니다.

<br>

포맷 일관성과 예시 다양성을 보장하기 위해, 시스템 지시사항 $W_I$가 사용됩니다.

<br>

데이터는 {number_of_examples}의 배치 단위로 생성되며, 본 논문에서는 {number_of_examples}를 5로 고정하였습니다.

<br>

레이블 편향을 완화하기 위해, 모델이 생성하는 응답에서 최대한의 변화를 추구하도록 장려합니다.
예를 들어, 답변이 지속적으로 "예"인 데이터에서 반복을 피하는 것과 같습니다.

<div id="Formatting Example"></div>

## Formatting Example

### 포매팅 예시의 역할

데이터 생성 파이프라인에서 필요한 유일한 입력은 단일 포매팅 예시($x_f, y_f$)와 해당 레이블 공간 $Y$입니다.

<br>

이 포매팅 예시는 JSON 구조로 된 프롬프트($W_f$)로 형식화됩니다. 이는 예시가 어떻게 구조화되어야 하는지를 명확히 보여주는 역할을 합니다.

## JSON 구조의 중요성

제공된 단일 JSON 구조의 포맷 프롬프트를 바탕으로, 모델은 JSON 스키마에 부합하는 문법적으로 정확한 출력을 생성할 것으로 기대됩니다.

<br>

JSON과 같은 복잡한 구조화된 출력을 생성하는 것은 도전적일 수 있지만, JSON은 쉽게 파싱할 수 있기 때문에, 생성 시 출력을 검증하는 방법으로 사용됩니다.

<div id="Structure of Formatting Example"></div>

## Structure of Formatting Example

<img style="width: 50%; margin-bottom: 40px;" id="output" src="./makingImg/option.PNG">

### Variable Option

- 논리적인 순서로 구성

  <br>

- 먼저 질문 $x_f$가 제시되고, 그 다음에 답변 후보들 $Y$의 목록이 나오며, 마지막으로 정답 $y_f$가 제시됩니다.

  <br>

- 이 포맷은 질문과 여러 선택지, 그리고 그중 정답을 나타내는 구조를 가지고 있습니다.

### Fixed Option

- Variable Option과 반대로 구성

  <br>

- 이 포맷에서는 먼저 답변 후보들 $Y$가 제시되고, 그 다음에 정답 $y_f$, 마지막으로 질문 $x_f$가 제시됩니다.

  <br>

- 이러한 프롬프트 구성 요소의 역순 배치는 자동 회귀 모델이 미리 정해진 선택지를 가지고 질문을 생성하도록 하기 위한 것입니다. 자유롭게 생성하는 모델은 일관성 없는 출력을 만들어낼 수 있으므로, 고정된 미리 정해진 선택지 세트에 속하지 않는 답변 후보 $Y$를 생성하는 것을 방지하기 위함입니다.

<div id="Self-Reference"></div>

## Self-Reference

하나의 포매팅 예시($x_f$, $y_f$)만을 모든 데이터 생성 반복(iteration)의 참조점으로 사용하는 것은, 생성된 데이터가 광범위하게 커버되고, 다양하며, 균형 잡힌 것을 보장하는 데 한계가 있을 수 있습니다.

  <br>

이러한 한계를 극복하기 위해 Self-Reference 방법이 제안됩니다. 이 방법에서, 모든 후속 생성 단계 $i > 0$에 대한 포매팅 예시 $f_i = (x_{fi}, y_{fi})$는 이전 반복 $i−1$에서 생성된 출력 $(x_{gi−1}, y_{gi−1}) ∈ D_{Gi−1}$에서 샘플링됩니다.

### Random Selection

- 각 반복(iteration) 동안, 다음 단계를 위한 포매팅 예시가 현재 단계의 출력물 중에서 무작위로 선택됩니다.

- 이 방법은 현재 생성된 데이터 세트 내에서 임의의 예시를 선택하여, 다음 데이터 생성 반복에 사용합니다.

- random selection은 데이터의 다양성을 증가시키는 데 도움이 됩니다.

### Contrastive Selection

- 각 반복에서, 이전 포매팅 예시와 가장 큰 의미적 대조를 보이는 예시를 선택합니다.

- 이 접근 방식에서는 사전에 훈련된 Bidirectional Encoder를 사용하여 예시들의 임베딩을 생성하고, $x_f$와 $x_{gi−1}$ 간의 코사인 유사도(Cosine Similarity)를 계산합니다.

- 가장 낮은 유사도를 보이는 인스턴스 $x_{gi−1}$을 선택합니다.

- 대조적 선택은 데이터 세트 내에서 서로 다른 유형의 예시들을 포함시켜 데이터의 균형을 맞추는 데 유용합니다.

### Similar Selection

- 이전 단계의 포매팅 예시와 의미적으로 가장 유사한 새 예시를 선택하여 다음 반복의 데이터 생성에 사용합니다.

### Tree Selection

- 반복적인 샘플링은 모델에 의해 생성된 예상치 못한 내용 변화로 인해, 초기 시드 예시로부터 나중 단계의 생성된 데이터까지 상당한 도메인 이동을 초래할 수 있습니다.

- 이 문제를 피하기 위해, 한 단계에서 생성된 모든 출력물을 후속 반복의 포매팅 예시로 사용합니다.

- 이 접근 방식은 생성된 예시들에 대한 너비 우선 트리 순회로 볼 수 있으며, 다른 세 샘플링 전략이 깊이 우선 탐색 전략을 사용하는 것과 대조됩니다.

- 이 연구의 가설은 탐색 트리의 최소 높이가 주제적으로 더 일관된 샘플을 제공한다는 것입니다.

<div id="4. Experimental Setup"></div>

# 4. Experimental Setup

### Datasets

1. 다지선다형 질문 답변(Multiple-Choice Question Answering, QA):

- variable label space setting에 사용됩니다.
- PIQA, WinoGrande 데이터셋
- 이러한 작업은 문장 내 빈칸 채우기와 같은 다양한 문제를 해결하는 모델의 추론 능력을 요구합니다.

2. 오픈북 예/아니오 질문 답변(Open-book Yes/No QA):

- fixed label space settings에 사용됩니다.
- BoolQ with context, PubMedQA, BioASQ 데이터셋
- 주어진 지문을 이해하고 예측을 하는 능력을 평가합니다.

3. 클로즈드북 예/아니오 질문 답변(Closed-book Yes/No QA):

- fixed label space settings에 사용됩니다.
- BoolQ without context, StrategyQA, CREAK 데이터셋
- 모델의 내재적 지식을 기반으로 답변하는 능력을 평가합니다.

<img style="width: 60%; margin-bottom: 40px;" id="output" src="./makingImg/datasets.PNG">

### Evaluation Details

- 원래 훈련 데이터셋($D_L$)과 LLM 생성 데이터셋($D_G$) 훈련될 때 성능 비교
- 평가에 사용된 기본 모델은 RoBERTa-large (Liu et al., 2019)입니다. 

### Implementation Details

#### 데이터 생성에 사용된 모델

- 2023년 6월 기준 GPT-3.5-turbo 언어 모델이 사용 
- 'temperature'와 'top-p'가 1로 설정
*temperature: 창의성(1보다 클수록 불확실)

*Top-p: 언어 모델이 생성할 수 있는 다음 단어의 후보군을 특정 확률 분포에 따라 제한

<br>

#### 미세 조정 실험

- Adam 최적화 알고리즘(Kingma and Ba, 2014)을 사용
- 최대 시퀀스 길이는 256으로 설정

#### 학습률 및 배치 크기에 대한 그리드 탐색

- 개발 데이터에서 최적의 학습률을 찾기 위해 [3e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 3e-6, 1e-6, 5e-7] 범위에서 그리드 탐색을 수행
- 배치 크기에 대해서도 [4, 8, 16] 범위에서 그리드 탐색을 진행합니다.

<br>

#### 실험 환경

- RTX A5000 그래픽 카드를 사용
- FP32(Floating Point 32-bit) 연산으로 수행

<div id="5. Experimental Results"></div>

# 5. Experimental Results

### Performance Comparison

#### ID Performance(분포 내 데이터에 대한 성능)

  <br>

1. 셋팅

- 원래 데이터셋($D_L$)에서 일부 테스트 세트로 분리
  
  <br>

- 모델들은 원래 데이터셋($D_L$)과 single-shot 파이프라인의 "self-reference" 변형을 사용하여 생성된 데이터셋($D_G$)에서 훈련

  <br>
  
- 두 데이터셋 유형 간의 성능 차이는 백분율 차이($(D_G - D_L)/D_L$)로 표시되어 테이블에 요약

  <br>
  
2. 결과

- 대량의 수작업으로 만들어진 데이터에 대한 대체물이 없음

LLM으로 합성 생성된 데이터를 사용할 때 최대 40.55%까지 성능이 감소

  <br>

- 그러나, LLM은 데이터가 매우 제한적이거나 특수한 영역에서 중요한 역할을 할 수 있음이 입증

WinoGrande, PubMedQA, BioASQ, StrategyQA와 같은 작업에서 $D_G$으로 훈련된 모델의 성능이 비슷하거나 때때로 더 좋은 결과

3. 비교방법

- single-shot 데이터 생성 접근 방식에서 여러 "self-reference" 샘플링 전략의 domain drift 비교

*domain drift: 데이터 생성 과정에서 초기에 설정된 데이터의 특성에서 점차 멀어지는 현상

- 초기에 설정된 하나의 'true formatting example'가 전체 데이터 생성 과정의 기준점

  <br>

- Tree-based exploration strategy은 초기 시드 샘플과 나중에 생성된 인스턴스 간의 의미적 거리를 제한하여 ID data에 대한 성능이 더 높아집니다.

#### OOD Performance(분포 외 데이터에 대한 성능)

  <br>

분포 내(In-Distribution, ID) 데이터는 제어된 환경에서 시스템에 대한 통찰력을 제공하지만, 실제 세계의 응용 프로그램은 종종 훨씬 더 다양하고 혼란스러운 데이터를 처리해야 합니다.

<br>

따라서, 연구자들은 인간이 만든 데이터(즉, 원래 훈련 데이터)와 LLM으로 생성된 데이터를 OOD 설정에서 비교합니다.

1. 셋팅

- ID Performance와 동일

2. 결과

- LM 데이터로 훈련된 모델이 OOD 예측 성능에서 일관되게 더 우수하며, 때로는 상당히 더 나은 성능

  <br>

- 실제 세계 시스템의 견고성과 일반화 가능성에 중요한 의미

3. 비교방법

- OOD 설정에서의 "self-reference" 전략 비교는 Tree-based exploration strategy이 여전히 강력한 전략임을 보여주지만, 다른 샘플링 접근법이 때때로 비슷하거나 더 좋을 수 있음을 나타냅니다.

- 이는 일정 수준의 제어된 잡음이 OOD 테스트 데이터에 일반화하는 데 도움이 될 수 있음을 의미

<br>

<img style="width: 100%; margin-bottom: 40px;" id="output" src="./makingImg/perform.PNG">

### Distribution shift during creation

반복적인 샘플링으로 인한 domain drift가 생성 과정 후반부에 생성된 샘플의 성능에 악영향을 미치는지, 아니면 성능이 정체되는지에 대해 테스트

<img style="width: 100%; margin-bottom: 40px; margin-top: 40px;" id="output" src="./makingImg/distribution.PNG">

1. 셋팅

- 벤치마크 데이터셋 중 하나인 Riddlesense에 대한 누적 데이터 분할의 평가를 수행

  <br>
  
- 훈련 데이터의 증분 비율(10% 블록 단위)을 사용하여 인간 레이블링 데이터와 합성 데이터셋 모두에 대한 모델의 성능을 해당 테스트 세트에서 평가

2. 결과

- 인간 레이블링 데이터를 사용하면 훨씬 빠른 수렴이 이루어짐 (ID 테스트 데이터에서 수행되었기 때문)

  <br>
  
- Random과 Contrast 접근 방식은 주요 평가에서 성능이 떨어졌으며, 누적 분할이 늘어남에 따라 성능이 감소하는 경향

  <br>
  
- Similar와 Tree 접근 방식은 일관되게 더 나은 샘플링 전략으로, 데이터가 추가됨에 따라 성능이 꾸준히 향상

  <br>

- 모든 데이터셋의 최종 상승 추세는 모델이 더 많은 데이터로부터 일반적으로 혜택

### Data creation cost

#### LLM API(2023년 6월 기준 OpenAI의 gpt-3.5-turbo)를 이용한 데이터 생성 비용

<img style="width: 100%; margin-bottom: 40px; margin-top: 40px;" id="output" src="./makingImg/cost.PNG">

- 이 비용은 LLM API, 구체적으로 gpt-3.5-turbo를 사용하여 데이터를 생성하는 데 발생한 비용을 기반

  <br>
  
- 결과는 LLM을 사용한 데이터 생성이 매우 비용 효율적임을 보여줌

  <br>
  
- 생성된 각 데이터셋의 비용이 5달러 미만 (데이터 생성 비용에는 중복되거나 형식에 맞지 않아 거부된 데이터에 대한 비용도 포함)

  <br>

- Tree-based “self-reference” strategy이 가장 높은 성능을 보였을 뿐만 아니라, 경제적으로도 효율적인 전략

  <br>

- Contrastive strategy은 두번째로 나은 비용