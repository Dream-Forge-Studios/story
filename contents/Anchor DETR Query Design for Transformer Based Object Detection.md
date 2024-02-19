---
date: '2023-10-05'
title: 'Anchor DETR: Query Design for Transformer-Based Object Detection 핵심 요약'
categories: ['CV']
summary: 'Anchor DETR: Query Design for Transformer-Based Object Detection을 이해해보아요'
thumbnail: './test.png'
---


<div id="개선점"></div>

# 개선점

<div id="DETR의 한계점"></div>

## DETR의 한계점

[DETR에 대한 설명](http://localhost:8000/DETR-End-to-End-Object-Detection-with-Transformers/)

<br>

기존의 DETR 모델에서는 "object query"라는 개념을 사용하여 이미지 내의 객체를 탐지합니다. 이 object query는 학습 과정에서 임베딩되며, 특정한 물리적 의미를 가지지 않습니다. 이로 인해 모델이 이미지 내에서 어디에 집중하는지, 즉 어떤 영역을 객체로 인식하는지에 대한 명확한 정보를 얻기 어렵습니다.

<div id="Anchor DETR의 개선점"></div>

## Anchor DETR의 개선점

Anchor DETR 논문에서는 이러한 문제점을 해결하기 위해 **anchor point**라는 개념을 도입하였습니다. 

<br>
Anchor point는 이미지 내의 특정 위치를 기준으로 하여, 그 주변 영역에 대한 정보를 학습하도록 object query를 디자인합니다. 이렇게 되면 object query가 이미지 내의 어떤 영역에 집중하는지 명확하게 알 수 있게 됩니다.
<div></div>
<br>

이러한 개선을 통해 "Anchor DETR"는 학습 속도와 성능 모두에서 큰 향상을 보였습니다. 특히, 기존 DETR이 500 epoch 동안 학습해야 했던 성능을 "Anchor DETR"는 단 50 epoch 만에 달성하였습니다. 이는 학습 시간을 크게 줄여주는 중요한 개선점입니다.

<br>
<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtGonQ%2Fbtr4RzSsCh6%2F6SZc1zIoRAiXqYNHeDjbc0%2Fimg.png">

"Anchor DETR"에서의 개선된 Object Query Design은 이미지 내의 특정 위치를 기준으로 하는 "anchor point"를 도입하여 해당 위치 주변을 학습하도록 설계되었습니다. 

<br>
이를 통해 모델은 이미지 내의 객체를 더욱 효과적으로 탐지할 수 있게 됩니다. 그렇다면, 어떻게 해당 위치 주변을 학습하는 것일까요?

<div id="Anchor Point"></div>

# Anchor Point

Anchor point는 이미지 내의 고정된 위치를 나타냅니다.

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FchotLj%2Fbtr4Ytj8GAr%2FK1KRPptaKLpuSduA89T1fk%2Fimg.png">

논문에서는 두 가지 방법으로 anchor point 를 적용하였습니다. 첫 번재는 grid 방법(아래 그림의 왼쪽)이고 두 번째는 learnable anchor point(그림의 오른쪽)로 Unif(0~1) 에서 anchor point의 갯수만큼 sampling 하는 것 입니다. 결과적으로는 후자가 더 좋은 성능을 가집니다.

1. Spatial Embedding

각 anchor point는 공간적 임베딩을 통해 표현됩니다. 이 임베딩은 anchor point의 위치 정보를 포함하며, 해당 위치 주변의 정보를 학습하는 데 사용됩니다.

2. Attention Mechanism

Transformer의 attention mechanism은 anchor point의 임베딩과 이미지의 피처 맵을 사용하여 해당 위치 주변의 정보를 집중적으로 학습합니다. 

<br>
이를 통해 모델은 anchor point 주변의 객체나 특징을 더욱 정확하게 탐지하게 됩니다.

3. 학습 과정

학습 동안, 모델은 각 anchor point 주변의 객체나 특징을 탐지하는 방법을 학습합니다. 

<br>
예를 들어, 이미지 중앙에 위치한 anchor point는 중앙에 있는 객체를 탐지하는 방법을 학습하게 됩니다. 이러한 방식으로, 모델은 이미지 내의 다양한 위치에서 객체를 탐지하는 방법을 학습하게 됩니다.

<br> 구체적으로 살펴 보자면, 
<img style="width: 100%; margin-bottom: 40px; margin-top: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FwIGfW%2Fbtr43YDamSo%2FnI2zzmtCOuBvXUQRtuUmo1%2Fimg.png">

$Pos_q$는 300개의 anchor point를 포함하고 있습니다. 각 anchor point는 2차원의 위치 정보를 가지고 있습니다. 즉, $Pos_q$는 [300, 2]의 형태를 가집니다.

<br>
이러한 anchor point들은 "Encode()" 함수를 통해 특징(feature)을 확장시킵니다. 결과적으로 각 anchor point는 256차원의 특징 정보를 가지게 되며, 전체적으로 [300, 256]의 형태를 가지게 됩니다.

<br> 추가적으로 pattern embedding을 더해서 사용합니다.
<img style="width: 100%; margin-bottom: 40px; margin-top: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc2AP6m%2Fbtr49T86QFa%2FQE0REXICSsGRsAHTyFFiB0%2Fimg.png">

**Pattern embedding**은 주어진 anchor point에서 여러 개의 예측을 수행하기 위해 도입된 개념입니다. 기본 아이디어는 하나의 anchor point가 여러 가지 패턴 또는 특성을 포착할 수 있도록 하는 것입니다.

<br> 

$$N_p$$는 사용되는 패턴의 개수를 나타냅니다. 예를 들어, $$N_p$$가 3이라면, 각 anchor point는 3가지 다른 패턴 또는 특성을 포착하려고 시도합니다. 이를 통해 모델은 더 다양한 정보를 포착하고, 더욱 복잡한 객체나 특성을 인식할 수 있게 됩니다.

<br>

$C$는 채널의 수를 나타냅니다. $C$는 pattern embedding의 차원 또는 깊이를 나타냅니다.

<img style="width: 100%; margin-bottom: 40px; margin-top: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FDw5jL%2Fbtr4PrHdWOl%2F8K0hOFNbLyb4K1PveMeL9K%2Fimg.png">

최종적인 식은 위와 같이 둘을 더합니다.

<div id="RCDA (Row Column Decouple Attention)"></div>

# RCDA (Row Column Decouple Attention)

attention 메커니즘의 효율성과 성능을 향상시키기 위한 방법입니다. 기본 아이디어는 2D 이미지의 행과 열을 분리하여 처리함으로써 attention 연산의 복잡성을 줄이는 것입니다.

<br>

전통적인 attention 메커니즘에서는 2D 이미지의 모든 픽셀 쌍에 대해 attention 가중치를 계산합니다. 이는 계산적으로 매우 비효율적이며, 특히 이미지의 크기가 클 경우 더욱 그렇습니다.

1. 행과 열의 분리

<br>

$K_f \in R^{H X W X C}$을 $K_{f,x} \in R^{W X C}$, $K_{f,x} \in R^{H X C}$ 두 개의 식으로 만드는 것입니다.

<br>

이미지의 행과 열을 독립적으로 처리합니다. 즉, 각 행과 각 열에 대해 별도의 attention 연산을 수행합니다. 이렇게 하면 2D attention의 복잡성을 1D attention의 복잡성으로 줄일 수 있습니다.

<br> 예를 들어, 이미지가 2D 배열로 표현된다면

```
[[a, b, c],
 [d, e, f],
 [g, h, i]]
```

행 기반의 attention 연산은 다음과 같이 수행됩니다.

<br>
첫 번째 행: [a, b, c] 내부의 요소들 간의 attention 연산

두 번째 행: [d, e, f] 내부의 요소들 간의 attention 연산

세 번째 행: [g, h, i] 내부의 요소들 간의 attention 연산

<br>
열 기반의 attention 연산은 다음과 같이 수행됩니다:

<div></div>
<br>
첫 번째 열: [a, d, g] 내부의 요소들 간의 attention 연산

두 번째 열: [b, e, h] 내부의 요소들 간의 attention 연산

세 번째 열: [c, f, i] 내부의 요소들 간의 attention 연산

2. 효율성

이렇게 행과 열을 분리하여 처리함으로써, 전체 이미지에 대한 2D attention 연산의 복잡성을 크게 줄일 수 있습니다. 각 행 또는 열 내부에서만 attention 연산이 수행되기 때문에, 연산량이 줄어들고 효율성이 향상됩니다.

3. 정보 보존

행과 열을 분리하여 처리하더라도, 중요한 공간적 정보는 여전히 보존됩니다. RCDA는 이미지의 구조와 관계를 유지하면서도 연산의 복잡성을 줄입니다.

<div id="Results"></div>

# Results

<img style="width: 100%; margin-bottom: 40px; margin-top: 40px;" id="output" src="https://user-images.githubusercontent.com/16400591/137062864-d6c6b384-5b6e-44a8-a755-10572dbc49c7.png">
<img style="width: 100%; margin-bottom: 40px; margin-top: 40px;" id="output" src="https://user-images.githubusercontent.com/16400591/137071504-4ce6a52c-8093-438c-b740-99b28419bc2a.png">
<img style="width: 100%; margin-bottom: 40px; margin-top: 40px;" id="output" src="https://user-images.githubusercontent.com/16400591/137071678-bfc13e65-18fc-49b3-bffe-712a159d4907.png">

현재 라벨링이 되어 있는 그림 데이터를 가지고 있어

우선 첫번째로 그림 이미지를 detection 하는 ai 모델을 