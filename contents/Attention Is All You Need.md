---
date: '2023-12-29'
title: 'Attention Is All You Need 논문 리뷰'
categories: ['Large Language']
summary: 'Attention Is All You Need 완벽 이해하기.'
thumbnail: './test.png'
---

<div id="Introduction"></div>

## Introduction

과거 언어 모델링과 기계 번역에서는 순환 신경망(RNN,LSTM,GRU) 모델을 활용하였습니다.

<br>

하지만 순차적 특성은 이전 요소의 처리 결과에 의존하여 시퀀스의 각 부분을 독립적으로 동시에 처리하기 때문에 병렬처리가 어렵고, 특히 긴 시퀀스 길이에서는 메모리 제약으로 인해 배치 처리가 제한됩니다.

**Attention mechanisms**

<br>

Attention mechanisms는 입력 데이터의 모든 부분이 동등한 중요도를 갖지 않는다는 아이디어에서 출발합니다. 모델이 특정 부분에 더 많은 'attention'를 기울여 그 부분의 중요도를 높입니다.

<br>

**Attention mechanisms 작동 원리**

1. Query, Key, Value:

- Query: 현재 타겟 또는 출력 상태를 나타냅니다.
- Key: 입력 데이터의 각 요소를 나타내며, 쿼리와 비교되어 유사도를 계산하는 데 사용됩니다. (Query 부분에서 해당 토큰이 얼마나 중요한지)
- Value: 입력 데이터의 각 요소에 대응하며, 가중치가 적용된 후 최종 출력에 기여합니다. (토큰의 실제 정보)
- 입력 시퀀스의 각 토큰(예: 단어, 문자)마다 고유한 Key와 Value가 할당
- Query는 입력 시퀀스의 각 토큰마다 별도로 생성되지 않고, 대신 현재의 타겟 상태 또는 출력에 대한 Query가 할당 <br> 예) 시퀀스-투-시퀀스 모델(기계번역)에서는 디코더의 각 층마다 하나씩 생성 <br>*Self-Attention에서 Query는 입력 시퀀스의 각 토큰마다 별도로 생성

2. 유사도 계산 및 가중치 할당:

- Query와 각 Key 사이의 유사도를 계산(내적 dot product)하여 attention weight를 결정합니다.
- 이 weight는 입력 데이터의 어느 부분이 현재 출력에 더 중요한지 나타냅니다.
- attention weight는 보통 Softmax 함수를 통해 정규화되어, 모든 weight의 합이 1이 되도록 합니다.
- 이후 Value와 결합하여 context vector를 생성합니다. <br> (현재 출력에 중요성에 따라 입력 데이터의 정보가 반영되도록 하는 계산의 사용)
- 최종적으로 생성된 context vector와 현재의 타겟 상태를 결합하여 최종 출력(예: 번역된 단어, 다음 단어)을 생성하는 데 사용됩니다.

**Transformer의 제안**

<br>

이전에는 attention mechanisms을 순환 신경망과 결합하여 사용하였습니다.

<br>

그런데 Transformer 모델은 순환을 제외하고 전적으로 attention mechanisms에 의존하는 Self-Attention을 활용합니다. 이는 입력과 출력 사이의 전역 의존성을 효과적으로 포착할 수 있게 해주며, 병렬 처리를 크게 향상시킵니다.

<br>

transformer 더 많은 병렬 처리를 가능하게 함으로써, 트레이닝 시간을 단축시키고 번역 품질에서 새로운 최고 기록을 달성할 수 있습니다. 예를 들어, P100 GPU 8개에서 단 12시간만에 트레이닝을 완료할 수 있습니다.

<div id="Background"></div>

## Background

transformer 기본 목표 중 하나는 하나는 순차적 계산을 줄이는 것입니다. 이는 Extended Neural GPU, ByteNet, ConvS2S와 같은 모델들에서도 공통적인 목표로, 이들 모델은 모두 CNN을 기본 구성 요소로 사용합니다.

<br>

하지만 ConvS2S, ByteNet 등에서 두 입력 또는 출력 위치 사이의 관계를 파악하는 데 필요한 연산은 위치 간 거리에 따라 증가합니다. ConvS2S는 선형적으로, ByteNet은 로그 함수적으로 증가합니다.

<br>

이러한 모델들에서 거리가 멀어질수록 서로 관련 있는 신호를 연결하는 것이 더 어려워집니다.

<br>

transformer는 RNN이나 CNN 없이 오로지 Self-Attention에만 의존하는 transduction 모델입니다. 이 점에서 기존 모델들과 크게 차별화됩니다.

<div id="Model Architecture"></div>

## Model Architecture

transformer는 거리에 따라 필요한 연산 수가 증가하는 cnn 기반 모델과 다르게 self-attention을 통해 이 문제를 해결합니다.

<br>

하지만, 이 방식은 모든 입력 토큰들을 하나의 context vector로 평균화함으로써, 개별 토큰들의 고유한 정보나 미묘한 차이가 손실될 수 있습니다. 

<br>

이러한 문제는 Multi-Head Attention을 통해 해결합니다.

<img style="width:60%; margin-top: 40px;" id="output" src="transformer/architecture.PNG">

###  Encoder and Decoder Stacks

**Encoder**

1. 구조

- encoder는 $N=6$개의 동일한 층으로 구성됩니다. 
- 각 층에는 두 개의 sub-layer가 있습니다.

2. 서브층

- 첫 번째 sub-layer는 multi-head self-attention mechanism입니다.
- 두 번째 sub-layer는 positionwise fully connected feed-forward network입니다.

3. residual connection과 layer normalization

- 각 sub-layer 주변에는 Residual Connection이 적용됩니다.
- 이후 각 sub-layer의 출력에는 Layer Normalization가 수행됩니다.
- 즉, 각 sub-layer의 출력은 $LayerNorm(x + Sublayer(x))$ 형태를 갖습니다. 여기서 $Sublayer(x)$는 서브층 자체에 의해 구현된 함수입니다.

4. 출력 차원

- 모델 내의 모든 sub-layer과 embedding layer은 $dmodel = 512$의 출력 차원을 가집니다.

<br>

**Decoder**

1. 구조

- decoder 역시 $N=6$개의 동일한 층으로 구성됩니다.
- 각 층에는 세 개의 sub-layer가 있습니다.

2. 서브층

- encoder와 동일한 두 개의 sub-layer에서 추가로 encoder 출력에 대해 multi-head attention을 수행하는 세 번째 sub-layer 추가

3. residual connection과 layer normalization

- encoder와 동일

4. self-attention sub-layer 수정

- decoder의 self-attention layer는 Masking과 출력 임베딩의 Offset이 추가됩니다.
- Masking
  - 디코더는 현재 시점의 출력을 생성할 때, 현재 시점 이후의 정보를 참조하지 못하도록 해야 합니다.
  - 특정 위치에서는 해당 위치와 그 이후의 위치에 대한 정보를 참조하지 못하도록 마스킹 처리됩니다.
- 출력 임베딩의 Offset
  - 출력 임베딩이 offset by one position된다는 것은, 디코더가 출력을 생성할 때 출력하기 이전 까지만 참조한다는 의미입니다.

###  Scaled Dot-Product Attention

**Scaled Dot-Product Attention의 구조**

1. 입력 차원

- Query와 Key는 차원 $d_k$를 가지며, 값(Value)은 차원 $d_v$를 가집니다.

2. 연산 과정

- Query와 모든 Key의 내적(dot product)을 계산합니다.
- 각 내적 결과를 $\sqrt{d_k}$로 나누어 스케일링합니다.
- Softmax 함수를 적용하여 값을 가중치로 변환합니다.

$Attention(Q,K,V)=softmax( \frac{QK^T}{\sqrt{dk}} )V$

### Multi-Head Attention

**Multi-Head Attention의 개념**

1. linearly project의 사용

- Multi-Head Attention에서는 single attention function를 사용하는 대신, Query, Key, Value를 ℎ번 서로 다른 linearly project을 통해 $d_k, d_k, d_v$차원으로 변환합니다. 

  <br>
  
- 이러한 서로 다른 linearly project를 head라고 부릅니다.

  <br>
  
- 각 head 각기 다른 Query, Key, Value 값을 가지며, 이는 입력 데이터를 서로 다른 방식으로 해석하고, 다양한 정보를 추출할 수 있도록 합니다.

2. 병렬 attention 실행

- 이렇게 투영된 각 Query, Key, Value에 대해 attention function를 병렬로 수행합니다. 
- 이 과정은 $d_v$차원의 출력 값을 생성합니다. 

3. 결합 및 최종 투영

- 어텐션의 결과를 연결(concatenate)한 후, 다시 한 번 project하여 최종 값으로 변환합니다.

$MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O$

$head_i = Attention(QW_i^Q , KW_i^K , W_i^V )$

<br>

$W_i^Q ∈ \mathbb{R}^{d_{model}×dk} , W_i^K ∈ \mathbb{R}^{d_{model}×dk} , W_i^V  ∈ \mathbb{R}^{d_{model}×dv}, W^O ∈ \mathbb{R}^{ hdv×d_{model}} $

<br>

**Multi-Head Attention의 구현**

- 모델은 총 $h=8$개의 $head$를 사용합니다. 
- 각 $head$의 차원 $d_k, d_v$는 $d_{model}/h=64$로 설정됩니다. 
- 각 헤드의 차원이 줄어들기 때문에, 전체 계산 비용은 전체 차원을 사용하는 single head attention과 유사합니다.

<img style="width: 80%; margin-top: 40px;" id="output" src="transformer/attention.PNG">

### Applications of Attention in our Model

1. encoder-decoder attention

- decoder의 multi-head attention 부분
- 이 층에서의 Query는 이전 디코더 층에서 오며, Key와 Value는 인코더의 출력에서 옵니다.
- 이 구조는 decoder 내의 모든 위치가 입력 시퀀스의 모든 위치에 주목할 수 있도록 합니다.
- 이는 전형적인 sequence-to-sequence models에서 볼 수 있는 encoder-decoder attention mechanisms을 모방합니다.

2. encoder 내의 self-attention

- 모든 Query, Key, Value가 같은 곳, 즉 인코더의 이전 층의 출력에서 옵니다.

3. decoder 내의 self-attention

- masking 기법이 사용되어, 아직 생성되지 않은 미래의 단어들에 대한 정보를 차단(−∞로 설정)하고 생성 중인 현재 위치까지만 정보를 참조할 수 있도록 합니다.

### Position-wise Feed-Forward Networks

시스템에는 OpenAI가 개발한 GPT-4 모델이 사용되었습니다.

<br>

GPT-4는 다양한 작업에서 인상적인 성능을 보여주었으며, Uniform Bar Examination(변호사 시험) 통과와 같은 뛰어난 성과를 달성했습니다.

### F1 - 감정적인 메세지 재구성

**Detect a message requiring intervention**

<br>

GPT-4에게 모든 메시지를 전송하고 염증성 여부를 문의하는 방법은 메시지의 양에 따라 비용이 많이 들고 플랫폼에 지연을 초래할 수 있으며, 다른 당사자에게 메시지를 보내기 전에 분석해야 하므로 사용자에게 혼란을 줄 수 있습니다.

<br>

더 정교한 감정적 메시지 감지 방법은 향후 연구에서 탐구가 필요할 것으로 보입니다.

<br>

**Reformulating the message**

<br>

사용된 prompt:

<br>

"당신은 ODR(온라인 분쟁 해결) 플랫폼입니다. 당사자의 채팅 메시지가 주어졌습니다. 내용은 유지하되, 메시지를 덜 대립적이고 원만한 합의에 더 도움이 되도록 재구성하세요. 재구성된 메시지로 직접 응답하고, 설명하지 마세요."

<br>

목표는 메시지를 덜 대립적이고, 원만한 합의에 더 유도하는 방향으로 만드는 것입니다. 또한 사용자의 요구에 따라 좀 더 방어적 혹은 공격적으로 재구성하는 방법에 대한 연구도 필요합니다.

### F2 - 중재자를 위한 메시지 초안 제안

**Generating the message suggestion**

<br>

사용된 prompt:

<br>

"당신은 중재자입니다. 당신의 목표는 두 당사자의 토론을 양 당사자 모두에게 수용 가능한 원만한 해결책으로 유도하는 것입니다. 당사자들 사이의 이 커뮤니케이션에 응답하세요. 중재자의 역할에 충실하되, 당사자들의 대화를 완성하지 마세요. 중립을 유지하고, 어느 한쪽 당사자의 편을 들지 마세요."

<br>

모델에는 대화에서 가장 최근의 10개 메시지가 맥락으로 제공되며, 중재자가 추가 지시를 입력할 수 있습니다.

### F3 - 자동적으로 개입

이는 매우 흥미롭고 강력한 사용 사례가 될 수 있지만 여러 가지 상당한 위험도 내포하고 있습니다. 따라서 그러한 시스템을 구축하기 전에 상당한 연구가 수행되어야 합니다.

<br>

**Triggers**

- 활동이 없는 기간이 일정 시간 지속될 때
- 당사자 간 토론이 격해질 때
- 일정 메시지마다(예: 10개의 메시지마다)
- 당사자 중 한 명이 요청할 때
