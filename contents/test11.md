---
date: '2020-07-29'
title: 'BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation'
categories: ['Web', 'SEO', 'Optimization']
summary: 'BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation 논문 리뷰'
thumbnail: './test.png'
---

semantic segmentation 기존 논문에서는 inference 속도를 향상시키기 위해 low-level detail을 포기했다.

* Low-level Information (저수준 정보)

  1. 기본적인 이미지 특성: 픽셀 값, 색상, 밝기, 그라디언트, 질감, 가장자리 및 방향과 같은 기본적인 이미지 특성을 나타냅니다.
  2. 세부 사항: Low-level 정보는 이미지의 구체적인 세부 사항을 포착합니다.
  3. 필터 응답: 초기 CNN 계층에서 추출되는 특성은 주로 low-level 정보에 중점을 둡니다. 예를 들어, Sobel, Scharr 및 Gabor와 같은 필터는 가장자리, 방향, 질감과 같은 low-level 정보를 포착하는 데 사용됩니다.

* High-level Information (고수준 정보)

  1. 추상화: 이미지의 고수준 의미나 맥락을 나타냅니다. 객체의 유형, 관계, 액션 및 씬의 전반적인 의미를 포함합니다.
  2. 객체 인식 및 분류: High-level 정보는 특정 객체나 개체의 카테고리를 인식하는 데 중점을 둡니다.
  3. CNN의 깊은 계층: CNN 아키텍처의 깊은 계층들은 high-level 정보를 포착합니다. 초기 계층은 가장자리나 질감과 같은 low-level 특성을 학습하는 반면, 더 깊은 계층은 얼굴, 자동차, 개 등의 복잡한 객체를 포착하는 특성을 학습합니다.

본 논문에서는 spatial detail(Low-level)과 categorical semantics(High-level)를 모두 충족시키는 네트워크를 제안하며 real-time으로 semantic segmentation을 진행합니다.

<div id="Bilateral Segmentation Network"></div>

# Bilateral Segmentation Network
<img style="width: 100%; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbmyWHR%2FbtrcVkFsuAf%2FihgtgxdaDnKruAxgjkWzS1%2Fimg.png">
이 네트워크는 빠른 추론 속도와 높은 정확도를 모두 달성하기 위해 설계되었습니다.

  <div id="1. Detail Branch"></div>

## 1. Detail Branch
>이미지의 저수준 특징, 즉 세부적인 정보와 구조를 포착하는 데 중점을 둔 부분입니다. 
> 
> 이러한 세부 정보는 세맨틱 분할의 정확성에 매우 중요합니다.

- 저수준(low-level)의 정보를 효과적으로 캡처하기 위해 풍부한 채널 수(channel capacity)를 갖는다.

  - 특징의 다양성: 저수준 정보는 주로 이미지의 세밀한 질감, 모서리, 색상 변화와 같은 부분에 포함됩니다. 이러한 정보는 다양한 특징들로 구성되어 있습니다. 따라서 이런 다양한 정보를 효과적으로 표현하려면 네트워크에 충분한 채널 용량이 필요합니다.
  - Spatial Details: 저수준의 정보는 고수준의 의미 정보보다 공간적 세부 정보(spatial details)를 갖는 경향이 있습니다. 이러한 세부 정보를 잘 캡처하려면 더 많은 채널이 필요할 수 있습니다.

- 넓은 공간적 크기(spatial size)를 가지고 있기 때문에 residual connection을 사용하지 않는다.

  - 계산 및 메모리 효율성
    
    - 넓은 spatial size를 가진 feature map에 residual connection을 적용하면 연산과 메모리 사용량이 늘어납니다. 
    - BiSeNet V2는 효율성을 중요하게 생각하기 때문에, 추가적인 연산량을 줄이기 위해 Detail Branch에서는 residual connection을 사용하지 않을 수 있습니다.

  - 정보의 통합
    - Residual connections는 주로 네트워크가 깊어질 때, 그래디언트 소실 문제를 완화하는 데 도움을 줍니다. 
    - 하지만 Detail Branch에서는 이미 저수준의 특징을 충분히 추출할 수 있으므로, 추가적인 residual connection 없이도 학습에 필요한 정보를 충분히 전달할 수 있을 것입니다.

- 구조
  - 총 3개의 layer로 이루어져 있으며 각각 convolution과 batch normalization, ReLu 활성화 함수가 포함
  - 최종적으로 input의 1/8 크기의 feature map이 출력
  
  <div id="2. Semantic Branch"></div>
## 2. Semantic Branch
>이미지의 고수준 의미 정보를 포착하는 데 중점을 둔 부분입니다.

 <img style="width: 100%; margin-bottom: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtRwXY%2FbtrcNkzPqPK%2FwHjQUjRkHRYPF2K0VL0Vf0%2Fimg.png">

1. Stem Block

- 입력 이미지의 초기 특징을 추출하는 역할을 합니다. 
- 여러 합성곱 계층, 배치 정규화, 그리고 활성화 함수를 통해 이미지의 기본적인 특징들을 캡처합니다.

2. Fast-down sampling

- 이미지의 해상도를 빠르게 줄이는 동시에 중요한 특징 정보를 보존합니다. 
- 여러 합성곱 계층과 최대 풀링(Max Pooling) 또는 스트라이드가 2 이상인 합성곱을 통해 이미지의 해상도를 줄입니다.
- 네트워크의 연산량을 줄이고, receptive field를 확장시키는 데 사용됩니다.
*Receptive field란, 출력에 영향을 주는 입력 이미지의 영역을 의미하는데, 이 영역을 넓히면 네트워크 출력의 한 부분이 입력 이미지의 더 큰 부분을 고려하게 해줍니다.

3. Gather and Expansion Layer
   <img style="width: 100%; margin-top: 40px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FLFwgq%2FbtrcOd1F6AH%2FIPsV9KsdRk6I3Ocof2dBv0%2Fimg.png">
- MobileNetv2(a)의 역병목과 달리, GE Layer(c)에는 하나의 추가적인 3x3 합성곱이 있습니다. 그러나 이 레이어는 계산 비용과 메모리 접근 비용에도 친화적입니다. 또한, 이 레이어 덕분에 GE Layer는 역병목보다 더 높은 특징 표현 능력을 가지게 됩니다.
- 이 계층은 특징 맵의 공간 해상도를 조절하고 특징을 확장하는 역할을 합니다. 
- 특징 맵의 해상도를 줄이는 (Gather) 합성곱 연산과 해상도를 높이는 (Expansion) 전치 합성곱 (Transposed Convolution) 연산을 결합하여 사용합니다.

4. Context Embedding Block

-  Global Average Pooling (GAP)을 사용하여 특징 맵의 전역적인 맥락 정보를 파악합니다. 
- GAP은 특징 맵의 각 채널에 대해 평균 값을 계산하여 고수준의 의미론적 정보를 캡처합니다. 
- 이후, 이 정보는 다시 원래의 공간 해상도로 확장되어 원래의 특징 맵과 결합됩니다.


<br>

- Stem Block 이 후 Context Embedding Block
  - global contextual 정보를 효율적으로 얻기 위해서 average pooling과 resisual connection을 사용

  <div id="3. Gather and Expansion Layer"></div>
## 3. Gather and Expansion Layer
> 고수준 의미 정보와 저수준 세부 정보를 효과적으로 통합하는 데 도움을 줍니다.