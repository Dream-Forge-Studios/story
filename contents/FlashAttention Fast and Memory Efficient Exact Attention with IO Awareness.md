---
date: '2023-12-06'
title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness 논문 리뷰'
categories: ['LLM']
summary: 'FlashAttention 완벽 이해하기.'
thumbnail: './test.png'
---

<div id="ABSTRACT"></div>

# ABSTRACT

전통적인 트랜스포머 모델들은 긴 시퀀스를 처리할 때 많은 시간과 메모리를 필요로 합니다. 

<br>

이는 self-attention 메커니즘의 시간과 메모리 복잡도가 시퀀스 길이에 따라 제곱으로 증가하기 때문입니다. 

<br>

FlashAttention은 GPU 메모리 계층 사이의 읽기/쓰기를 고려하는 IO-aware attention 알고리즘으로, <u>트랜스포머 모델의 처리 속도를 향상시키기 위해 개발</u>되었습니다. 

<br>

이 방법은 <u>tiling을 사용하여 GPU의 고대역폭 메모리(HBM)와 온칩 SRAM 사이의 메모리 읽기/쓰기 횟수를 줄입니다.</u> 

<br>

*tiling: 큰 데이터 세트나 계산 작업을 더 작고 관리하기 쉬운 부분(타일)으로 나누는 기술

*HBM: GPU가 대량의 데이터를 저전력으로 빠르게 처리할 수 있게 함

*SRAM: 캐시 메모리로 사용, DRAM보다 빠른 접근 속도

<br>

FlashAttention은 표준 주의력 알고리즘보다 적은 HBM 접근이 필요하며, 다양한 SRAM 크기에 대해 최적화되어 있습니다.

<br>

또한, FlashAttention은 block-sparse attention으로 확장되어, 기존의 어떤 approximate attention 방법보다 빠르게 작동합니다.

*block-sparse attention: 입력 시퀀스의 특정 부분에 대한 attention을 계산

<br>

FlashAttention은 BERT-large, GPT-2, long-range arena 모델들의 훈련 속도를 대폭 향상시켰습니다. 

<br>

예를 들어, BERT-large 모델에서는 기존의 훈련 속도 기록보다 15% 빠르고, GPT-2 모델에서는 3배, long-range arena에서는 2.4배의 속도 향상을 보였습니다.

<br>

이러한 향상된 속도 덕분에 FlashAttention은 더 긴 문맥을 가진 트랜스포머 모델을 가능하게 하여, 모델의 품질을 높입니다. (예: GPT-2에서의 낮은 혼란도(perplexity)와 긴 문서 분류에서의 성능 향상)

<br>

FlashAttention을 사용한 트랜스포머는 Path-X 챌린지(시퀀스 길이 16K)와 Path-256(시퀀스 길이 64K)에서 최초로 평균 이상의 성능을 달성했습니다.


<div id="Introduction"></div>

# Introduction

트랜스포머의 핵심인  self-attention module이 시퀀스 길이에 따라 시간과 메모리 복잡도가 제곱으로 증가하기 때문에 긴 시퀀스에 대한 처리는 여전히 어려운 문제입니다.

<br>

많은 approximation attention 방법들(sparse approximation, low-rank approximation 등)이 주의력의 계산과 메모리 요구 사항을 줄이기 위해 개발되어 시퀀스 길이에 대해 거의 선형적인 계산 요구 사항을 가지게 하지만, 실제 처리 시간(wall-clock speedup)에서 개선을 보이지 않아 널리 채택되지 않았습니다. 

<br>

이러한 접근법들의 주된 문제 중 하나는 FLOP(부동 소수점 연산의 수) 감소에 초점을 맞추는 것인데 메모리 접근 시간, 데이터 전송 시간, 다양한 하드웨어 자원의 효율적 사용 등은 모두 처리 시간에 영향을 미치지만 FLOP와는 직접적인 관련이 없습니다.

<br>

또한, 메모리 접근(IO)에서 발생하는 오버헤드를 무시하는 경향이 있습니다.

<img style="width: 100%; margin-bottom: 40px; margin-top: 40px;" id="output" src="https://raw.githubusercontent.com/hazyresearch/flash-attention/master/assets/flashattn_banner.jpg">

본 논문에서는 attention 알고리즘을 개선하기 위한 새로운 접근법인 **IO-aware**을 제시합니다.

<br>

**IO-aware**란 <u>메모리 접근(입출력, IO)을 세심하게 고려하는 것</u>을 말합니다. 구체적으로는, 다양한 종류의 메모리(빠른 메모리 SRAM와 비교적 느린 메모리 HBM) 사이에서 데이터를 읽고 쓰는 과정을 주의 깊게 관리하는 것입니다. 

<br>

현대의 GPU에서는 계산 속도가 메모리 속도를 앞서고 있으며, 트랜스포머 모델에서 대부분의 연산은 메모리 접근에 의해 제한됩니다. 

<br>

즉, <u>계산 자체보다는 데이터를 메모리에서 읽고 쓰는데 더 많은 시간이 소요</u>되는 경우가 많습니다.

<br>

**IO-aware** 알고리즘은 데이터베이스 조인, 이미지 처리, 수치 선형 대수학 등 메모리 접근이 주요한 시간 소모 요소인 <u>메모리 제한 연산에서 중요한 역할</u>을 합니다.

<br>

그러나 현재 널리 사용되는 딥러닝 프레임워크인 파이썬의 PyTorch와 TensorFlow와 같은 인터페이스들은 메모리 접근을 세밀하게 제어할 수 있는 기능을 제공하지 않습니다.

<br>

그래서 고안된 FlashAttention의 주요 목표는 attention matrix를 HBM에 읽고 쓰는 것을 피하는 것입니다. 

<br>

이를 위한 전략으로는

<div id="전략"></div>

## 전략

1. tiling: 입력을 블록으로 나누어 블록 내에서 여러번 softmax reduction를 수행한 후 결과를 통합합니다.
*softmax reduction: 모델이 어떤 토큰에 더 많은 '주의'를 기울여야 하는지 계산하는 것

2. forward pass 과정에서 소프트맥스 정규화 인자를 SRAM에 저장하여, backward pass 과정에서 HBM에서 읽지 않고 SRAM에서 attention을 빠르게 재계산합니다.

    <br>
   
3. CUDA에서 구현되어 메모리 접근을 세밀하게 제어할 수 있습니다.
   - 모든 attention 연산을 하나의 GPU 커널로 통합하여, 연산 효율성을 높입니다.
    *서로 다른 GPU 커널을 사용하면, 각 커널 간의 컨텍스트 전환(context switching)이 필요합니다.

이를 통한 구체적인 성능 개선으로는

<div id="성능 개선"></div>

## 성능 개선

1. 재계산으로 인한 FLOP 증가에도 불구하고, 표준 attention 대비 빠른 실행 속도(예: GPT-2에서 최대 7.6배 빠름)와 더 적은 메모리 사용(시퀀스 길이에 선형적)을 달성합니다.

    <br>
   
2. IO complexity
    - IO complexity는 컴퓨터 알고리즘에서 데이터의 입출력(Input/Output) 작업이 얼마나 복잡한지를 나타내는 지표입니다.
   
      <br>
      
    - FlashAttention은 표준 attention 대비 훨씬 적은 HBM 접근이 필요하여 IO complexity가 낮습니다.
   
      <br>
      
    - FlashAttention의 HBM 접근 복잡도: $O(N^2d^2M^{-1})$<br>* $O()$: 알고리즘의 성능 상한선
    - 표준 attention의 HBM 접근 복잡도: $Ω(Nd+N^2)$<br>* $Ω()$: 알고리즘의 성능 하한선<br>* 시퀀스의 길이: $N$, SRAM 크기: $M$, attention head의 차원: $d$

FlashAttention의 확장 버전인 Block-Sparse FlashAttention이 있습니다.

<div id="다양한 확장 가능성"></div>

## 다양한 확장 가능성

본 논문에서는 다중 GPU에서의 주의력, 커널 회귀, 블록 희소 행렬 곱셈과 같은 다양한 연산에 FlashAttention을 확장할 수 있음을 논의합니다.

<br>

**Block-Sparse FlashAttention**

<br>

FlashAttention의 확장으로, Block-Sparse attention 계산을 수행합니다. 이는 특히 매우 긴 시퀀스 데이터를 처리할 때 유용합니다.

<br>

Block-Sparse attention는 모든 토큰 간이 아닌 특정 블록 내의 토큰들 끼리만 attention 계산을 하는 것을 의미합니다.

<div id="성능 평가 및 벤치마킹"></div>

## 성능 평가 및 벤치마킹

1. 모델 훈련 속도 향상: FlashAttention을 사용하면 트랜스포머 모델의 훈련이 기존 방법보다 더 빠르게 진행됩니다. 예를 들어, BERT-large, GPT-2, Long-range arena 모델들이 기존 대비 각각 15%, 3배, 2.4배 빠른 속도로 훈련됩니다.

2. 모델 품질 향상: FlashAttention은 더 긴 시퀀스를 처리할 수 있게 함으로써 모델의 품질을 향상시킵니다. GPT-2에서는 복잡도(perplexity)가 0.7 개선되었고, 긴 문서 분류 작업에서는 성능이 6.4 포인트 향상되었습니다.

3. Path-X 및 Path-256 도전과제 수행: FlashAttention을 사용한 트랜스포머 모델은 Path-X 도전과제에서 처음으로 기회 수준 이상의 성능을 보였으며, Block-sparse FlashAttention을 사용한 모델은 더 긴 64K 시퀀스에서 Path-256 도전과제를 수행할 수 있었습니다.

4. attention 벤치마킹: FlashAttention은 표준 attention 구현보다 최대 3배 빠르며, 128부터 2K까지의 일반적인 시퀀스 길이에서 더 나은 성능을 보입니다. 시퀀스 길이가 1K 이상인 경우, 일부 approximate attention methods (e.g., Linformer)가 더 빠르기 시작하지만, Block-sparse FlashAttention은 현재 알려진 모든 approximate attention methods보다 빠릅니다.
