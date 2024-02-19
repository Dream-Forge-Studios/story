---
date: '2023-10-05'
title: 'DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection 논문 리뷰'
categories: ['CV']
summary: 'DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection을 이해해보아요'
thumbnail: './test.png'
---

<div id="기존 DETR 모델의 문제점"></div>

# 기존 DETR 모델의 문제점

1. 이전 DETR 모델은 동일한 대상의 중복 출력 문제
   - DETR은 NMS 없이 end-to-end로 학습되며, 이는 중복된 예측이 발생할 수 있게 만듭니다.
   - 예측된 바운딩 박스와 ground truth 바운딩 박스와 매칭하여 손실을 계산합니다. 그러나 이 방식은 중복된 예측에 대해 명확한 패널티를 부과하지 않습니다.

2. object queries의 초기화 방식이 비효율적
   - DETR에서 쿼리들은 일반적으로 무작위로 초기화 되기 때문에 특별한 구조나 패턴을 갖지 않아 학습에 더 많은 시간이 필요할 수 있습니다.
   - DETR의 초기화 방식은 사전 지식(priors)을 포함하지 않습니다.

DINO는 기존 DETR 모델의 문제점을 개선하며 더 나은 정확성과 모델 크기와 데이터셋 크기를 줄이는 뛰어난 확장성을 보여줍니다.

<br> 그렇다면 어떻게 기존 문제점을 개선했는지 알아봅시다.

<div id="contrastive denoising training"></div>

# contrastive denoising training

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://miro.medium.com/v2/resize:fit:1100/0*zxfKfsnBqJF55gmn">

Contrastive Denoising Training은 ground truth 데이터를 기반으로 positive과 negative의 예시를 모두 사용하여 모델을 훈련합니다.

<div id="positive sample과 negative sample을 만드는 방법"></div>

## positive sample과 negative sample을 만드는 방법

<br>
두 개의 hyper-parameters 1과 2가 있으며, 여기서 1<2이며 positive query와 negative query를 생성합니다. positive query는 1보다 작은 노이즈 스케일을 가지며 해당하는 Ground Truth 상자를 재구성하는 반면, 내부 사각형과 외부 사각형 사이의 negative query는 1보다 크고 2보다 작은 노이즈 스케일을 갖습니다.

<div></div>
<br>
쉽게 생각해서 Positive Query라는 것은 객체의 실제 Ground Truth Bbox가 포함된 이미지 영역을 의미하며, Negative Query는 그 외의 배경 영역을 의미합니다.

<div></div>
<br>
이미지에서 n개의 Ground Truth 상자가 있는 경우 CDN 그룹은 각 GT 상자가 긍정적인 쿼리와 부정적인 쿼리를 생성하는 2 × n 쿼리를 갖게 됩니다.

<div></div>
<br>
Positive Query에는 GIOU Loss가 적용되고, Negative Query에서는 Focal Loss가 적용됩니다.

<div></div>
<br>

CDN 디코더는 Noise 섞인 샘플 쿼리를 처리하여 객체의 존재 여부를 판별하고, 바운딩 박스의 중심 좌표를 다시 예측함으로써 Denoising을 수행합니다. 

<div></div>
<br>
이러한 방법은 훈련 중 동일한 대상의 중복 출력을 방지하여 모델의 일대일 매칭을 개선하였습니다.

<div id="Mixed Query Selection"></div>

# Mixed Query Selection

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://miro.medium.com/v2/resize:fit:1100/0*do93orxA_r0gXKmb">

그림을 보면

<br>
(a)는 이미지의 인코더 특성을 사용하지 않고 static embedding을 사용합니다. 이는 embedding 무작위로 초기화로 비효율적입니다.

<div></div>
<br>

(b)는 이미지의 인코더의 위치와 특성을 참고하여 embedding을 초기화 합니다. 하지만 추가적인 정제 없이 초기 내용 특성이기 때문에 디코더에 대해 모호하고 오해를 불러일으킬 수 있습니다.

<br>

Mixed Query Selection인 (c)는 위치 정보만을 사용하여 앵커 박스를 초기화하고, 내용 쿼리는 이전처럼 정적으로 유지합니다.

<br>이는 위치 쿼리만 향상시키고 내용 쿼리를 학습 가능하게 유지하여, 모델이 인코더로부터 더 포괄적인 내용 특성을 추출할 수 있도록 돕습니다.

<div id=" Look Forward Twice"></div>

#  Look Forward Twice

<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpgL7H%2FbtrDEjg1fD7%2FxoiXSgKqoVdgb7qCzY3r2K%2Fimg.png">

look forward once 방법은 특정 레이어의 매개변수를 업데이트할 때 해당 레이어의 박스 손실만을 고려합니다. 

<br>그러나 이후 레이어의 개선된 박스 정보가 이전 레이어의 박스 예측을 수정하는데 더 도움이 될 것이라고 주장합니다.

<br>논문에서는 look forward twice라는 새로운 방법을 제안하여, layer-i의 매개변수 업데이트를 layer-i와 layer-(i + 1)의 손실 모두에 의존하게 합니다. 

<br>그리하여 각 예측 오프셋은 두 번의 박스 업데이트에 사용됩니다.

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbpk4nu%2FbtrDDsSN3UA%2FEmr4Aik2MqZ6dobVpVacFK%2Fimg.png">

수식으로 자세하게 살펴보자면,

<br>$∆b_i$는 이전에 예측된 박스의 위치나 크기($b_{i-1}$)에서 $Layer_i$에서 박스의 위치나 크기로의 변화량을 의미하며, $b′_i$와 $b_i^{(pred)}$를 업데이트 하는데 사용된다.

<br>위의 수식에서 Detach는 $b′_i$(undetach 상태의 $b_i$)를 받아 그래디언트 전파를 차단한 상태의 $b_i$를 생성합니다. 

<br>이렇게 함으로써, 모델은 $b′_i$에서 $b_i$로 전환할 때 그래디언트 전파를 차단하고, 이후 레이어에서의 예측을 위해 $b′_i$와 $Δb_i$를 사용하여 $b_i^{(pred)}$를 계산할 수 있습니다.

<div id="Model Architecture"></div>

# Model Architecture

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FnFnmR%2FbtrDBLemenl%2FkzgSgKZj08dVYGCdVIjhq1%2Fimg.png">

1. ResNet, Swin 등의 백본 모델에서 다중 스케일 이미지 특성 추출

2. 이미지 특성에 대한 Positional Embedding으로 인코더에 입력

3. 디코더에서 Positional Query를 활용하여 앵커 박스 초기화(Mixed Query Selection)

4. Deformable Attention 연산(Look forward Twice)

5. 최종 출력값은 예측된 앵커 박스 및 분류 결과

6. Constrative Denoising을 통한 앵커 박스 및 분류 결과 개선
[출처] 빅웨이브에이아이 기술블로그

<div id="Experiment"></div>

# Experiment

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FDvTAb%2FbtrDE86WdQu%2FI0YMEgzQRjgKY8SeliDBi1%2Fimg.png">

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FO4B57%2FbtrDE8lxTw8%2F3JALGXqwBBbyQAJGfpyOk0%2Fimg.png">

<img style="width: 100%; margin-top: 40px; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmJ46i%2FbtrDBtjY4zA%2FdvTtih0RBjdjUJjL1NupB0%2Fimg.png">
