---
date: '2024-01-02'
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

<br>

Attention mechanisms는 입력 데이터의 모든 부분이 동등한 중요도를 갖지 않는다는 아이디어에서 출발합니다. 모델이 특정 부분에 더 많은 'attention'를 기울여 그 부분의 중요도를 높입니다.

<br>

**Attention mechanisms 작동 원리**

1. Query, Key, Value:

- Query: 현재 타겟 또는 출력 상태를 나타냅니다.

  <br>

- Key: 입력 데이터의 각 요소를 나타내며, 쿼리와 비교되어 유사도를 계산하는 데 사용됩니다. <br>=> Query 부분에서 해당 토큰이 얼마나 중요한지

  <br>

- Value: 입력 데이터의 각 요소에 대응하며, 가중치가 적용된 후 최종 출력에 기여합니다. <br>=> 토큰의 실제 정보

  <br>

- 입력 시퀀스의 각 토큰(예: 단어, 문자)마다 고유한 Key와 Value가 할당

  <br>

- Query는 입력 시퀀스의 각 토큰마다 별도로 생성되지 않고, 대신 현재의 타겟 상태 또는 출력에 대한 Query가 할당 <br> 예) 시퀀스-투-시퀀스 모델(기계번역)에서는 디코더의 각 층마다 하나씩 생성 <br>*Self-Attention에서 Query는 입력 시퀀스의 각 토큰마다 별도로 생성

2. 유사도 계산 및 가중치 할당:

- Query와 각 Key 사이의 유사도를 계산(내적 dot product)하여 attention weight를 결정합니다.

  <br>

- 이 weight는 입력 데이터의 어느 부분이 현재 출력에 더 중요한지 나타냅니다.

  <br>

- attention weight는 보통 Softmax 함수를 통해 정규화되어, 모든 weight의 합이 1이 되도록 합니다.

  <br>

- 이후 Value와 결합하여 context vector를 생성합니다. <br> (현재 출력에 중요성에 따라 입력 데이터의 정보가 반영되도록 하는 계산의 사용)

  <br>

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

transformer 기본 목표 중 하나는 순차적 계산을 줄이는 것입니다. 이는 Extended Neural GPU, ByteNet, ConvS2S와 같은 모델들에서도 공통적인 목표로, 이들 모델은 모두 CNN을 기본 구성 요소로 사용합니다.

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

$Attention(Q,K,V)=softmax( \frac{QK^T}{\sqrt{d_k}} )V$

### Multi-Head Attention

**Multi-Head Attention의 개념**

1. linearly project의 사용

- Multi-Head Attention에서는 single attention function를 사용하는 대신, Query, Key, Value를 ℎ번 서로 다른 linearly project을 통해 $d_k, d_k, d_v$차원으로 변환합니다. 

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

**구성 요소**

<br>

두 개의 linear transformations과 그 사이의 ReLU 활성화 함수로 구성됩니다.

<br>

$FFN(x)=max(0, xW_1+b_1)W_2+b_2$

<br>

**linear transformations의 일관성과 다양성**

<br>

각 위치에서의 linear transformations은 동일하지만, 층마다 다른 파라미터를 사용합니다. 다시 말해, 인코더와 디코더의 각 층은 독립적인 피드포워드 네트워크 파라미터를 가집니다.

<br>

**kernel size 1의 convolutions으로의 해석**

<br>

이 네트워크는 kernel size가 1인 2개의 convolution으로 해석될 수 있습니다. 이는 각 위치에서의 연산이 주변 위치의 값에 영향을 받지 않는다는 것을 의미합니다.

<br>

입력과 출력의 차원은 $d_{model} = 512$입니다.

<br>

inner-layer(첫번째 linear transformations)의 차원은 $d_{ff}=2048$입니다.

### Embeddings and Softmax

<br>

**Embeddings**

- 입력 토큰과 출력 토큰을 벡터로 변환하기 위해 학습된 임베딩을 사용합니다.

  <br>

- 각 토큰은 $d_{model}$ 차원의 벡터로 변환됩니다.

  <br>

- 임베딩 벡터의 차원 $d_{model}$은 모델의 다른 부분과 일관되게 유지됩니다. (self-attention layer과 feed forword 등 인코더와 디코더의 각 서브층의 입력과 출력은 모두 $d_{model}$차원을 유지)

<br>

**Softmax**

- 디코더의 출력을 다음 토큰의 확률로 변환하기 위해 linear transformations과 softmax funtion를 사용합니다.

  <br>

**가중치 공유**

- 모델은 입력 embeddings 층, 출력 embeddings 층, 그리고 softmax 이전의 linear transformations 간에 동일한 가중치 행렬(임베딩 행렬)을 공유합니다.

*softmax 이전의 linear transformations: $d⋅E^T$ ($E$ 임베딩 행렬, $d$ 디코더 출력)

  <br>

- 이는 모델의 파라미터 수를 줄이고, 효율성을 높이는 데 도움이 됩니다.

<br>

**임베딩 가중치의 스케일링**

<br>

- 임베딩 층에서는 가중치(임베딩 행렬)에 $\sqrt{d_{model}}$을 곱하여 스케일링합니다.

  <br>
  
- 이는 임베딩 벡터의 크기를 조정하고, 학습 과정을 안정화하는 데 도움이 됩니다.

### Positional Encoding

트랜스포머는 순환 구조나 합성곱 구조를 사용하지 않습니다. 따라서 모델이 토큰의 순서를 인식하고, 이를 기반으로 정보를 처리할 수 있도록 위치 정보를 제공해야 합니다.

<br>

**Positional Encoding 구현**

- 입력 임베딩에 위치 인코딩을 더하여, 각 토큰의 위치 정보를 모델에 제공합니다.

  <br>
  
- 이 위치 인코딩은 인코더와 디코더 스택의 하단에서 입력 임베딩과 합쳐집니다.

  <br>

- 위치 인코딩의 차원은 임베딩 벡터의 차원 $d_{model}$과 동일하므로, 두 벡터를 직접 더할 수 있습니다.

  <br>

$PE(pos,2i)=sin(\frac{pos}{10000^{2i/d_{model}}})$

$PE(pos,2i+1)=cos(\frac{pos}{10000^{2i/d_{model}}})$

$pos$는 토큰의 위치(시퀀스 내의 순서), $i$는 임베딩 벡터 내에서의 특정 차원

<br>

**cos, sin 함수 표현의 이점**

<br>

사인과 코사인 함수를 사용한 위치 인코딩은 모델이 상대적 위치 정보를 쉽게 알 수 있도록 합니다. 어떤 고정된 오프셋 $k$에 대해서, $PE_{pos+k}$는 $PE_{pos}$의 선형 함수로 표현될 수 있습니다.

$sin(x+\Delta)=sin(x)cos(\Delta)+cos(x)sin(\Delta)$

$cos(x+\Delta)=cos(x)cos(\Delta)+sin(x)sin(\Delta)$

<br>

학습된 위치 인코딩과 비교했을 때, 사인과 코사인 함수를 사용한 인코딩은 거의 동일한 결과를 보여주었습니다.

<br>

사인과 코사인 함수를 사용하는 이 인코딩 방식은 훈련 중에 보지 못한 더 긴 시퀀스 길이에 대해 모델이 잘 확장될 수 있도록 합니다.

<div id="Why Self-Attention"></div>

## Why Self-Attention

**순환층(recurrent layer)과 비교**

1. 계산 복잡성

- self-attention 층은 모든 위치를 한 번에 연결하므로, 계산 복잡성이 일정합니다. 반면, 순환 층은 시퀀스 길이에 비례하여 계산 복잡성이 증가합니다($O(n)$).

  <br>
  
- 특히, 시퀀스 길이(n)가 표현 차원(d)보다 작은 경우, self-attention 층은 순환 층보다 계산적으로 더 효율적입니다. 이는 최신 기계 번역 모델에서 사용되는 문장 표현(예: 단어 조각 또는 바이트 쌍 표현)에 자주 해당합니다.

2. 병렬 처리 능력

- self-attention 층은 최소한의 순차적 연산만 필요로 하므로, 대부분의 계산을 병렬로 처리할 수 있습니다. 이는 특히 대규모 데이터 처리에 있어 중요한 이점을 제공합니다.

  <br>

- 반면, 순환 층은 각 타임스텝마다 이전 타임스텝의 결과에 의존하기 때문에, 병렬 처리가 어렵습니다.

3. 장거리 의존성 학습

- 네트워크에서 입력과 출력 위치 사이의 경로 길이가 짧을수록 장거리 의존성을 학습하기가 더 쉽습니다. self-attention 층은 모든 입력과 출력 위치 간에 짧은 경로를 제공합니다.

  <br>

- 순환 층은 장거리 의존성 학습에 있어 불리한데, 이는 입력과 출력 사이의 경로 길이가 시퀀스 길이에 따라 증가하기 때문입니다.

4. 제한된 self-attention

- 매우 긴 시퀀스를 처리할 때, 자기 주의는 입력 시퀀스 내의 특정 크기(r)의 이웃만을 고려하도록 제한될 수 있습니다. 이는 최대 경로 길이를 O(n/r)로 증가시키지만, 계산 효율성을 개선할 수 있습니다.
 
  <br>

- 순환 층은 장거리 의존성 학습에 있어 불리한데, 이는 입력과 출력 사이의 경로 길이가 시퀀스 길이에 따라 증가하기 때문입니다.

**CNNs과 비교**

- CNNs의 특징
  1. 하나의 합성곱 층은 커널 너비(k)가 시퀀스 길이(n)보다 작으면, 모든 입력과 출력 위치를 직접 연결하지 않습니다.

      <br>
  
  2. 필요한 층의 수
  
     - 연속적인 커널의 경우: 모든 쌍의 입력-출력 위치를 연결하려면 $O(n/k)$의 합성곱 층이 필요합니다.
     - 확장된(dilated) 커널의 경우: $O(log_k(n))$의 합성곱 층이 필요합니다.
  
  3. 이러한 층을 쌓는 것은 네트워크 내에서 임의의 두 위치 사이의 최장 경로 길이를 증가시킵니다.

      <br>
     
  4. 일반적으로 CNNs 층은 recurrent 층보다 계산 비용이 더 많이 듭니다. 그 비용은 커널의 너비(k)에 비례합니다.

      <br>

  5. 분리 가능한 합성곱(Separable Convolution)은 계산 복잡성을 크게 줄일 수 있으며, 복잡성은 $O(k·n·d + n·d^2)$가 됩니다.

- self-attention의 특징
  1. self-attention의 계산 복잡성은 커널의 크기(k)가 시퀀스 길이(n)와 같은 경우에도 separable convolution과 동일합니다.

      <br>

  2. 시퀀스 내 모든 위치 간의 직접적인 정보 전달을 가능하게 합니다.

      <br>

  3. individual attention heads가 수행하는 작업이나 문장의 구문적 및 의미적 구조와 관련된 행동을 파악하기 용이합니다.

<img style="width: 100%; margin-bottom: 40px;" id="output" src="./transformer/length.PNG">

<div id="Training"></div>

## Training

### Training Data and Batching

- 데이터셋
    - 영어-독일어: WMT 2014 English-German 데이터셋을 사용했습니다. 이 데이터셋은 약 450만 문장 쌍으로 구성되어 있습니다.
    - 영어-프랑스어: 더 큰 WMT 2014 English-French 데이터셋을 사용했습니다. 이 데이터셋은 3600만 문장으로 구성되어 있습니다.
  
- Sentences Encoding
  - 영어-독일어: 바이트 쌍 인코딩(Byte-Pair Encoding, BPE)을 사용하여 약 37,000개 토큰의 공유 소스-타겟 어휘를 생성했습니다.
  - 영어-프랑스어: 32,000개 단어 조각(word-piece) 어휘를 사용했습니다.

- batch 처리
  - 문장 쌍은 대략적인 시퀀스 길이에 따라 배치되었습니다. 
  - 각 훈련 배치는 대략 25,000개의 소스 토큰과 25,000개의 타겟 토큰을 포함했습니다.

### Hardware and Schedule

- 8개의 NVIDIA P100 GPUs를 탑재한 한 대의 머신에서 모델을 훈련했습니다.
    
    <br>
  
- 훈련시간
  - 기본 모델: 논문에서 나온 hyperparameters 설정을 따른 기본 모델은 각 훈련 단계에 대략 0.4초가 소요되었으며, 총 100,000 단계 또는 12시간 동안 훈련되었습니다.
  - 큰 모델: 훈련 단계에 1.0초가 소요되었으며, 총 300,000 단계 또는 3.5일 동안 훈련되었습니다.

### Optimizer

Adam 최적화 알고리즘을 사용했습니다. 설정된 하이퍼파라미터는 $β_1 = 0.9, β_2 = 0.98, ε = 10^{-9}$ 입니다.

<br>

$lrate = d^{−0.5}_{model} · min(step\_num^{−0.5} , step\_num · warmup\_steps^{−1.5} ) $

<br>

이는 첫 warmup_steps 동안 학습률을 선형적으로 증가시킨 다음, 이후에는 단계 번호의 역 제곱근에 비례하여 감소시킵니다.

<br>

warmup_steps는 4000으로 설정되었습니다.

### Regularization

<br>

**Residual Dropout**

- 적용 방법: dropout은 각 sub-layer의 출력에 적용되며, sub-layer 입력에 더해지기 전에 수행됩니다. 이는 또한 인코더와 디코더 스택에서 임베딩과 위치 인코딩의 합에도 적용됩니다.

    <br>

- dropout의 목적: dropout은 모델이 특정 뉴런이나 경로에 과도하게 의존하는 것을 방지하고, 일반화 능력을 향상시키기 위해 사용됩니다. 이는 훈련 과정에서 무작위로 일부 뉴런의 활성화를 drop하여, 네트워크가 더 견고해지도록 합니다.

    <br>
  
- 기본 모델의 dropout 비율: base model의 경우 드롭아웃 비율(Pdrop)은 0.1로 설정됩니다.

**Label Smoothing**

<br>

5개의 클래스 중 1번 클래스가 정답일 경우, 레이블은 [1, 0, 0, 0, 0]처럼 표현하는데 레이블 스무딩이 적용된 새로운 레이블은 [0.9, 0.025, 0.025, 0.025, 0.025]가 됩니다.

- 적용 방법: 훈련 중에 레이블 스무딩을 적용하며, 이는 값 $ϵ_{ls}=0.1$을 사용합니다.

    <br>

- 레이블 스무딩의 목적: 레이블 스무딩은 모델이 너무 확신에 찬 예측을 하는 것을 방지하고, 모델이 불확실성을 더 잘 처리하도록 합니다. 이는 각 훈련 샘플의 레이블을 약간씩 smooth 만들어, 모델이 더욱 smooth한 확률 분포를 학습하게 합니다.

    <br>

- 영향: 레이블 스무딩은 perplexity(모델의 불확실성을 나타내는 지표)에 부정적인 영향을 미칠 수 있지만, 정확도와 BLEU 점수(기계 번역의 성능을 평가하는 지표)를 개선하는 데 도움이 됩니다.

<div id="Results"></div>

## Results

이전에 발표된 모든 모델과 앙상블을 능가했으며, 경쟁 모델들보다 훨씬 적은 훈련 비용(1/4)이 들었습니다.

### Model Variations

<img style="width: 100%; margin-bottom: 40px;" id="output" src="./transformer/results.PNG">

**(A) 어텐션 헤드 수와 차원 변화**

- 실험은 어텐션 헤드의 수와 키(key) 및 값(value) 차원을 변화시키면서 성능 변화를 측정했습니다.
- 단일 헤드 어텐션은 최적 설정보다 0.9 BLEU 점수가 낮았으며, 너무 많은 헤드를 사용할 경우에도 성능이 감소했습니다.

**(B) 어텐션 키 차원 크기 감소의 영향**

- 어텐션 키의 차원 크기를 줄이는 것이 모델 품질에 부정적인 영향을 미쳤습니다.
- 이는 어텐션의 호환성 결정이 쉽지 않으며, 단순한 내적(dot product)보다 더 정교한 호환성 함수가 유용할 수 있음을 시사합니다.

**(C) 및 (D) 큰 모델과 드롭아웃의 중요성**

- 예상대로 더 큰 모델이 더 좋은 성능을 보였으며, 드롭아웃은 과적합을 방지하는 데 매우 유용했습니다.

**(E) 위치 인코딩의 변형**

- 실험에서는 트랜스포머 모델의 기본적인 사인파 위치 인코딩을 학습된 위치 임베딩으로 대체했을 때 거의 동일한 성능을 관찰했습니다.

이러한 실험 결과는 트랜스포머 모델의 다양한 구성 요소가 전체 성능에 중요한 영향을 미치며, 특히 어텐션 메커니즘의 구조와 드롭아웃의 적용이 모델의 품질에 중요한 역할을 한다는 것을 보여줍니다. 또한, 위치 인코딩 방법을 변형해도 성능에 큰 영향이 없음을 확인했습니다. 이러한 결과는 트랜스포머 모델을 최적화하고 개선하는 데 중요한 통찰을 제공합니다.

<div id="Conclusion"></div>

## Conclusion

트랜스포머는 전적으로 attention 메커니즘에 기반한 첫 번째 시퀀스 변환(sequence transduction) 모델로, 인코더-디코더 구조에서 흔히 사용되는 순환 층을 multi head self-attention로 대체했습니다.

<br>

트랜스포머는  recurrent 또는 convolutional 층을 기반으로 하는 아키텍처보다 번역 작업에서 훨씬 빠르게 훈련될 수 있습니다.

<br>

WMT 2014 영어-독일어 및 영어-프랑스어 번역 작업에서 트랜스포머는 새로운 최고 성능(state of the art)을 달성했습니다. 특히 영어-독일어 작업에서는 이전에 보고된 모든 앙상블 모델보다도 뛰어난 성능을 보였습니다.

<br>

연구팀은 텍스트 외의 다른 입력 및 출력 modality를 가진 문제에 트랜스포머를 확장할 계획입니다.

<br>

이미지, 오디오, 비디오와 같이 큰 입력과 출력을 효율적으로 다루기 위한 지역적이고 제한된 attention 메커니즘에 대한 연구를 계획하고 있습니다.