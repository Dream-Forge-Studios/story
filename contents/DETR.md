---
date: '2023-10-05'
title: 'DETR: End-to-End Object Detection with Transformers 핵심 요약'
categories: ['CV']
summary: 'DETR: End-to-End Object Detection with Transformers을 이해해보아요'
thumbnail: './test.png'
---


<div id="Abstract"></div>

# Abstract

<div id="기존의 Object Detection 방법들"></div>

## 기존의 Object Detection 방법들

- 기존의 방법들은 non-maximum suppression이나 anchor generation 같은 복잡한 과정과 손수 디자인해야 하는 요소들이 필요했습니다.
- 이러한 과정들은 모델을 복잡하게 만들고, 성능 최적화에 어려움을 주었습니다.

<div id="DETR의 접근 방식"></div>

## DETR의 접근 방식
- DETR은 이러한 복잡한 과정들을 제거하여, detection 파이프라인을 간소화하였습니다.
- DETR의 핵심 구성 요소는 Transformer의 encoder-decoder 구조와 '양자간 매칭(bipartite matching)'을 통한 유니크한 예측입니다.
- 이를 통해, DETR은 object와 이미지 전체의 context 사이의 관계를 추론하고, 최종적인 예측 set을 곧바로 반환할 수 있습니다.

<div id="DETR의 특징"></div>

## DETR의 특징
- 간단한 구조: DETR은 개념적으로 매우 간단하며, 특별한 라이브러리를 필요로 하지 않습니다.
- 고성능: COCO 데이터셋에서 Faster R-CNN과 동등한 정확도와 런타임 속도를 보였습니다.
- 다목적성: DETR은 쉽게 일반화할 수 있어, panoptic segmentation도 생성할 수 있습니다. 이는 경쟁 모델들을 뛰어넘는 성능을 보였습니다.
*Panoptic Segmentation: 이미지에서 모든 픽셀을 레이블링하여 각 픽셀이 어떤 객체에 속하는지, 또는 어떤 세그멘트에 속하는지를 예측하는 작업

<div id="Introduction"></div>

# Introduction

이전 모델들은 위 모델들의 성능은 거의 겹치는 예측들을 후처리하거나, anchor set을 디자인하거나, 휴리스틱하게 target boxes를 anchor에 할당하는 데에 크게 의존합니다.

본 논문은 위와 같은 과정을 간소화 하기 위해서 suurogate task를 패스하고 direct set prediction을 수행하는 방법론을 제안합니다.

*suurogate task: 주요 작업을 직접 수행하기 어려울 때, 그 주요 작업을 대신할 수 있는 보조적인 작업을 의미

<div id="DETR architecture"></div>

# DETR architecture

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://velog.velcdn.com/images%2Fsjinu%2Fpost%2Fb2bbfd51-30f6-4992-b68b-47b1b0200e8b%2Fimage.png">

<div id="Backbone"></div>

## Backbone

input image를 x<sub>img</sub>∈R<sup>3×H<sub>0</sub>×W<sub>0</sub></sup>라 할 때, CNN backbone 모델은 낮은 차원의 
activation map f∈R<sup>C×H×W</sup>를 생성합니다. 본 논문에서는 주로 $$C=2048, H=\frac{H_0}{32}, W=\frac{W_0}{32}$$의 값을 사용합니다.

<div id="Transformer encoder"></div>

## Transformer encoder

1. 1 X 1의 Convolution을 통해 high-level activation map f의 차원을 C에서 d로 변경합니다. 이렇게 함으로써, 새로운 피처 맵 $$z_0$$가 생성됩니다. 이 피처 맵의 차원은 d×H×W입니다.
   <img style="width: 80%; margin-bottom: 40px; margin-top: 40px;" id="output" src="https://kikaben.com/detr-object-detection-with-transformers-2020/images/DETR-CNN.png">

2. encoder는 sequence input을 받습니다. 따라서 피처 맵 $$z_0$$의 높이와 너비를 합쳐서 평탄화(flatten)하여 시퀀스 d×HW 차원의 피처 맵으로 변환합니다.

3. Transformer는 1차원적인 문장을 처리하나 DETR은 2차원적인 이미지를 처리하여 이미지 내의 각 그리드 셀의 세로와 가로 위치($$\frac{d}{2}$$)를 따로 positional encoding한 후 합쳐서 전체 위치 정보를 얻습니다.
   <img style="width: 80%; margin-bottom: 40px; margin-top: 40px;" id="output" src="https://kikaben.com/detr-object-detection-with-transformers-2020/images/DETR-Positional-Encodings.png">
   <img style="width: 60%; margin-bottom: 40px; " id="output" src="https://kikaben.com/detr-object-detection-with-transformers-2020/images/DETR-add-positional-encodings.png">

3. encoder layer는 standard 구조를 가지며, multi-head self-attention module과 feed forward network (FFN) 으로 이루어져 있다.

<div id="Transformer decoder"></div>

## Transformer decoder

1. Transformer는 decoder에 target embedding을 입력하는 반면, DETR은 object queries를 입력합니다.
   - object queries
     - 이미지 내의 객체들을 감지하고 분류하는 작업을 지원하기 위해 특별히 설계된 embedding입니다.
     - Trainable Embedding
     
       object queries는 학습 과정 중에 최적화됩니다. 초기에는 임의의 값으로 설정되지만, 학습 데이터를 통해 점차적으로 업데이트되어, 이미지 내의 객체를 더 정확하게 감지하고 분류하는 데 도움이 되는 값으로 변화합니다.
     - Decoder가 Encoder의 벡터와 자신(Decoder)의 이전 상태를 사용하여 내부적인 연산을 수행하여 다음 결과를 예측합니다.
     - Object Query는 이미지 내의 각 객체를 대표하는 벡터로 작동하며, DETR 모델은 이러한 Object Query를 사용하여 이미지 내의 여러 객체를 감지하고 분류합니다.
2. Transformer는 decoder에서 첫 번째 attention 연산 시 masked multi-head attention을 수행하는 반면, DETR은 multi-head self-attention을 수행합니다.
   -  DETR은 입력된 이미지에 동시에 모든 객체의 위치를 예측하기 때문에 별도의 masking 과정을 필요로 하지 않습니다. 

3.  Transformer는 Decoder 이후 하나의 head를 가지는 반면, DETR는 두 개의 head를 가집니다.  Transformer는 다음 token에 대한 class probability를 예측하기 때문에 하나의 linear layer를 가지는 반면, DETR은 이미지 내 객체의 bounding box와 class probability를 예측하기 때문에 각각을 예측하는 두 개의 linear layer를 가집니다.

<div id="Prediction feed-forward networks(FFNs)"></div>

## Prediction feed-forward networks(FFNs)

1. Decoder에서 출력한 output embedding을 3개의 linear layer와 ReLU activation function으로 구성된 FFN에 입력하여 최종 예측을 수행합니다. 
2. FFN은 이미지에 대한 class label과 bounding box에 좌표(normalized center coordinate, width, height)를 예측합니다. 
3. 이 때 예측하는 class label 중 ∅ 은 객체가 포착되지 않은 경우로, "background" class를 의미합니다.

<div id="Auxiliary decoding losses"></div>

## Auxiliary decoding losses

1. Auxiliary decoding losses는 중간 레이어에서의 예측에 대한 손실을 의미합니다
1. 학습 시, 각 decoder layer마다 FFN을 추가하여 auxiliary loss를 구합니다. 
2. 이러한 보조적인 loss를 사용할 경우 모델이 각 class별로 올바른 수의 객체를 예측하도록 학습시키는데 도움을 준다고 합니다. 
3. 추가한 FFN은 서로 파라미터를 공유하며, FFN 입력 전에 사용하는 layer normalization layer도 공유합니다.

<img style="width: 60%; margin-top: 0px; " id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUdR3Y%2FbtrWkvCHHae%2FrkQ3eZRXle9uAd0kKoskXk%2Fimg.png">
spatial positional encoding은 주로 key (k)와 query (q)에 적용된 이유는 Value (v)는 attention weights와 결합되어 최종 출력을 생성합니다. 
<div></div>
<br>
위치 정보는 이미 attention weights 계산에 사용되었기 때문에, value에 다시 위치 정보를 추가하는 것은 중복이 될 수 있습니다.

<div id="Object detection set prediction loss"></div>

# Object detection set prediction loss

<div id="Hungarian algorithm"></div>

## Hungarian algorithm

Hungarian algorithm은 두 집합 사이의 일대일 대응 시 가장 비용이 적게 드는 bipartite matching(이분 매칭)을 찾는 알고리즘입니다.

1. 두 집합 $$I$$와 $$J$$가 있습니다. 여기서 $$I$$ 집합의 각 원소를 $$J$$ 집합의 원소와 짝지을 때 발생하는 비용을 $$c(i,j)$$라고 합니다. 예를 들어, 일정한 작업을 수행하는 데 필요한 시간이나 비용 등을 생각하면 됩니다.

2. 헝가리안 알고리즘의 목표는 이 비용을 최소화하는 최적의 짝을 찾는 것입니다. 즉, $$I$$ 집합의 각 원소가 $$J$$ 집합의 어떤 원소와 짝지어져야 가장 전체 비용이 적게 드는지를 찾아내는 것입니다.

3. 이 때, "permutation"이라는 용어는 짝을 지을 때의 최적의 순서나 방법을 의미합니다. 예를 들어, 어떤 작업자가 어떤 작업을 수행하는 것이 가장 효율적인지를 결정하는 것과 같은 개념입니다.

4. 헝가리안 알고리즘은 이러한 비용을 표현한 행렬을 기반으로 동작하며, 이 행렬에서 최적의 짝을 찾아냅니다.

간단히 말하면, Hungarian algorithm은 두 집합 사이에서 각 원소를 최적으로 짝지어 전체 비용을 최소화하는 방법을 찾아내는 알고리즘입니다.

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FciPXoJ%2FbtrUyA7cNVl%2FzTVkQCpxtOLaK6eKgpD6s1%2Fimg.png">

위의 표 각 칸에는 해당 예측 경계 상자와 실제 객체 위치가 짝지어졌을 때의 비용이 적혀 있습니다. 예를 들어, 첫 번째 예측된 경계 상자가 두 번째 실제 객체 위치와 짝지어졌을 때의 비용이 11이라는 것을 알 수 있습니다.

<br>

이제 이 표를 바탕으로 가장 비용이 적게 드는 방법으로 예측된 경계 상자와 실제 객체 위치를 짝지어야 합니다. 예를 들어, 첫 번째 예측 경계 상자는 첫 번째 실제 객체 위치와, 두 번째 예측 경계 상자는 두 번째 실제 객체 위치와 짝지어지는 경우, 전체 비용은 32가 됩니다.

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FMlrj0%2FbtrUuId7Bsa%2FHQNjKRXOECQVXm6k8QLEvk%2Fimg.png">

반면 이 경우는 permutation을 [3, 4, 1, 5, 2]인 경우, cost가 12로 상대적으로 매우 낮으며 가장 바람직하게 matching된 것을 확인할 수 있습니다.

Hungarian algorithm은 이처럼 cost에 대한 행렬을 입력 받아, matching cost가 최소인 permutation을 출력합니다.

<div id="Bounding box Loss"></div>

## Bounding box Loss

DETR에서는 bounding box를 예측할 때 Hungarian algorithm과 bounding box loss를 사용합니다.

대부분의 기존 방법들은 'anchor'라는 기준점을 사용하여 bounding box를 예측합니다. 이 방법은 예측된 bounding box가 크게 벗어나지 않도록 도와줍니다.

<br>
그러나 DETR은 이런 기준점 없이 bounding box를 바로 예측합니다. 그 결과, 예측된 bounding box의 크기나 위치가 다양하게 나타날 수 있습니다. 만약 단순히 l1 loss라는 방법만 사용하여 예측의 정확성을 측정한다면, 큰 bounding box와 작은 bounding box에서의 오차가 다르게 측정될 수 있습니다. 

<div></div>
<br>
예를 들어, 큰 bounding box에서는 오차가 크게 나타나고, 작은 bounding box에서는 오차가 작게 나타날 수 있습니다.
이런 문제를 해결하기 위해, DETR에서는 l1 loss와 함께 generalized IoU(GIoU) loss라는 다른 방법도 함께 사용합니다. 이 두 가지 방법을 조합함으로써, bounding box의 크기에 상관없이 예측의 정확성을 더 잘 측정할 수 있게 됩니다.

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkQHRk%2FbtrULPjWa9X%2Fjm3BH61IDJMKuRePahUqP0%2Fimg.png">
*GIoU loss 이미지

$$
GIoU=IoU(b_{σ(i)},\hat{b_i})−\frac{|B(b_{σ(i)},\hat{b_i})|∖b_{σ(i)}∪\hat{b_i}}{|B(b_{σ(i)},\hat{b_i})|}
$$

$$
L_{iou}(b_{σ(i)},\hat{b_i})=1−GIoU
$$

$$
L_{box}(b_{σ(i)},\hat{b_i})=λ_{iou}L_{iou}(b_{σ(i)},\hat{b_i})+λ_{L1}||b_{σ(i)}−\hat{b_i}||_{1}
$$

<div id="Find optimal matching"></div>

## Find optimal matching 

1. 두 개의 set에 대하여 bipartite matching을 수행하기 위해, 아래의 cost를 minimize할 수 있는 N의 permutation을 탐색합니다.

$$\hat{σ}=argmin_{σ∈S_N} \sum_{N}^{i}L_{match}(y_i,\hat{y}_σ(i))$$

<br>

$$L_{match}(y_i,\hat{y}_σ(i))$$는 ground truth $y_i$ 와 index가 $i$인 prediction 사이의 pair-wise matching cost 입니다.

<br>

$L_{match}=−1$<sub>{ci≠∅}</sub>$\hat{p}_{σ(i)}(ci)+1$<sub>{ci≠∅}</sub>$L_{box}(b_i,\hat{b}_{σ(i)})$

<br>

$c_i$는 target class label을 의미하며, $b_i ∈ [ 0 , 1 ]^4$ 는 ground truth box의 좌표와 이미지 크기에 상대적인 width, height를 의미합니다.
class probability를 $\hat{p}_{σ ( i )} ( c i )$ 로, predicted box를 $\hat{b}_{σ ( i )}$ 로 정의합니다.

<br>

$L_{box}$는 bounding box loss를 의미합니다.

<div id="Compute Hungarian loss"></div>

## Compute Hungarian loss 

$L_{Hungarian}(y,\hat{y})=∑^{N}_{i=1}[−log\hat{p}_{\hat{σ}(i)}(ci)+1$
<sub>{ci≠∅}</sub>$L_{box}(b_i,\hat{b}_{\hat{σ}}(i))]$

<br>

실제 학습 시 예측하는 객체가 없는 경우인 $c_i = ∅$에 대하여 log-probability를 1/10로 down-weight한다고 합니다. 이는 실제로 객체를 예측하지 않는 negative sample의 수가 매우 많아 class imbalance를 위해 해당 sample에 대한 영향을 감소시키기 위함입니다.