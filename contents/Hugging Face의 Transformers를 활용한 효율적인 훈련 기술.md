---
date: '2024-04-30'
title: 'Hugging Face의 Transformers를 활용한 효율적인 훈련 기술'
categories: ['LLM']
summary: 'llm을 활용하면서 자꾸 out of memory 오류가 나타나, 이를 해결할 수 있는 효율적인 훈련 기술에 대해 알아보자.'
thumbnail: './test.png'
---

<div id="efficient training on a single GPU"></div>

# efficient training on a single GPU

single GPU 활용할 때, 메모리 활용을 최적화하고 훈련 속도를 높이는 실용적인 기술을 설명합니다.

<br>

또한, multi gpu를 활용할 때에도 이러한 접근 방식을 적용할 수 있으며, 추후에 multi gpu에 대한 효율적 훈련 방법도 제시되어있습니다.

<br>

LLM을 훈련할 때는 데이터 처리량/훈련 시간, 모델 성능을 동시에 고려해야합니다.

<br>

처리량(샘플/초)을 최대화하면 훈련 비용이 낮아집니다. 이는 일반적으로 GPU를 최대한 활용하여 GPU 메모리를 한계까지 채움으로써 달성됩니다.

<br>

원하는 배치 크기가 GPU 메모리의 한계를 초과하는 경우, gradient accumulation과 같은 메모리 최적화 기술이 도움이 될 수 있습니다.

<br>

그러나 선호하는 배치 크기가 메모리에 맞는다면, 훈련을 느리게 할 수 있는 메모리 최적화 기술을 적용할 필요는 없습니다.

<br>

큰 배치 크기를 사용할 수 있다고 해서 반드시 그래야 하는 것은 아닙니다. 하이퍼파라미터 튜닝의 일부로, 최상의 결과를 제공하는 배치 크기를 결정한 다음, 자원을 그에 맞게 최적화해야 합니다.

<div id="Batch size choice"></div>

## Batch size choice

최적의 성능을 달성하기 위해 적절한 배치 크기를 식별하는 것으로 시작하세요.

<br>

$2^N$크기의 batch sizes와 input/output neuron를 사용한 것이 권장됩니다. 이는 

<br>

이는 많은 컴퓨터 시스템과 하드웨어는 2의 거듭제곱 단위로 데이터를 처리하는데 최적화되어 있습니다. 이러한 설정은 GPU나 CPU에서 데이터를 더 효율적으로 처리할 수 있게 해줍니다.

<br>

종종 8의 배수를 사용하지만, 사용하는 하드웨어와 모델의 데이터 타입에 따라 더 높을 수 있습니다.

<br>

참고로,  fully connected layers(일반 행렬 곱셈(GEMMs)에 관여하는)에 대해 NVIDIA가 제공하는 [입력/출력 뉴런 수 및 배치 크기]('https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#input-features')에 대한 권장사항을 확인하세요.

<br>

[텐서 코어 요구사항]('https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc')은 데이터 타입과 하드웨어에 기반하여 배수를 정의합니다. 예를 들어, fp16 데이터 타입의 경우 8의 배수가 권장되지만, A100 GPU를 사용하는 경우에는 64의 배수를 사용하세요.

<br>

작은 파라미터에 대해서는 [Dimension Quantization Effects]('https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization')도 고려하세요. 여기서 타일링이 발생하고 올바른 배수는 상당한 속도 향상을 가져올 수 있습니다.

<div id="Gradient Accumulation"></div>

## Gradient Accumulation

Gradient Accumulation 방법은 전체 배치에 대해 그래디언트를 한 번에 계산하는 대신 작은 increments(증분)으로 그래디언트를 계산하려는 것을 목표로 합니다.

<br>

batch를 여러 작은 batch로 나누어 각 그래디언트를 계산한 후, 이 그래디언트들을 accumulation하여 실제로는 큰 배치 하나를 처리하는 것과 동등한 업데이트를 수행할 수 있습니다.

<br>

메모리 제한으로 인해 큰 배치를 사용할 수 없는 경우 유용한 기술이지만, 추가적인 계산 부하를 초래하여 훈련 시간이 증가될 수 있으며, train loss가 발생합니다.

<br>

[RTX-3090]('https://github.com/huggingface/transformers/issues/14608#issuecomment-1004392537') 및 [A100]('https://github.com/huggingface/transformers/issues/15026#issuecomment-1005033957')에 대한 batch size와 gradient accumulation 벤치마크를 참조하세요.

```
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
```

<div id="Gradient Checkpointing"></div>

## Gradient Checkpointing

일부 대형 모델의 경우 배치 크기를 1로 설정하고 gradient accumulation을 사용하더라도 여전히 메모리 문제가 발생할 수 있습니다.

<br>

이는 그래디언트를 계산하기 위해 forward pass에서 모든 activations를 저장해야 하기 때문에 상당한 메모리 부담을 초래하기 때문입니다.

<br>

activations을 버리고 backward pass 시 필요할 때 다시 계산하는 대안은 상당한 계산 부담을 도입하고 훈련 과정을 느리게 할 수 있습니다.

<br>

Gradient Checkpointing은 이 두 접근 방식 사이의 타협을 제공하며, 계산 그래프 전반에 걸쳐 전략적으로 선택된 activations만 저장함으로써 그래디언트를 위해 다시 계산해야 하는 activations의 일부만 필요로 합니다.

['자세한 설명']('https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)

<br>

```
training_args = TrainingArguments(
    per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args
)
```
<div id="Mixed precision training"></div>

## Mixed precision training

Mixed precision training은 특정 변수에 대해 낮은 정밀도의 수치 포맷을 사용함으로써 모델 훈련의 계산 효율성을 최적화하는 기술입니다.

<br>

전통적으로 대부분의 모델은 32비트 부동 소수점 정밀도(fp32 또는 float32)를 사용하여 변수를 표현하고 처리합니다. 그러나 모든 변수가 정확한 결과를 달성하기 위해 이렇게 높은 정밀도를 필요로 하는 것은 아닙니다.

<br>

일부 변수의 정밀도를 16비트 부동 소수점(fp16 또는 float16)과 같은 낮은 수치 포맷으로 줄임으로써 계산 속도를 높일 수 있습니다.

<br>

이 접근법에서 일부 계산은 반정밀도로 수행되고 일부는 여전히 전체 정밀도로 수행되기 때문에, 이 방법을 혼합 정밀도 훈련이라고 합니다.

<br>

혼합 정밀도 훈련은 가장 일반적으로 fp16 (float16) 데이터 타입을 사용하여 달성됩니다.

<br>

그러나 일부 GPU 아키텍처(예: Ampere architecture)는 bf16 및 tf32(CUDA internal data type) 데이터 타입을 제공합니다. 이러한 데이터 타입 간의 차이에 대해 자세히 알아보려면 [NVIDIA 블로그]('https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/')를 참조하십시오.

### fp16

fp16은 16비트로 표현되는 부동 소수점 데이터 타입입니다. 이 포맷은 1비트의 부호 비트(sign bit), 5비트의 지수 비트(exponent bits), 그리고 10비트의 유효 숫자 비트(significand bits, 또는 mantissa)로 구성됩니다. fp16의 동적 범위(dynamic range)는 제한적이지만, 빠른 계산과 메모리 효율성 때문에 널리 사용됩니다.

<br>

mixed precision training의 주요 이점은 activations를 반정밀도(fp16)로 저장함으로써 메모리 사용량을 줄이는 데 있습니다.

<br>

그러나 이 과정에서 gradients도 fp16로 계산되지만, optimization step에서는 다시 fp32을 사용하기 때문에 이 단계에서는 메모리 절약이 이루어지지 않습니다.

<br>

mixed precision training은 계산 속도를 빠르게 할 수 있는 장점이 있지만, 모델이 GPU 상에서 16비트와 32비트 정밀도 두 가지 형태로 동시에 존재하게 되므로, 특히 작은 배치 크기에서는 더 많은 GPU 메모리를 사용하게 됩니다.

<br>

이로 인해 GPU 상에 모델이 차지하는 공간이 원래의 1.5배가 됩니다. 이는 빠른 계산 속도의 이점을 제공하면서도, 동시에 메모리 사용량이 늘어날 수 있음을 의미합니다.

```
training_args = TrainingArguments(per_device_train_batch_size=4, fp16=True, **default_args)
```

### BF16

bf16은 Google에서 개발한 16비트 부동 소수점 포맷으로, 특히 딥 러닝에서 사용되며, fp32와 같은 지수 범위를 가지고 있습니다. bf16은 1비트의 부호 비트, 8비트의 지수 비트, 그리고 7비트의 유효 숫자 비트로 구성됩니다. 이 구성 덕분에 bf16은 fp16보다 훨씬 넓은 범위의 숫자를 표현할 수 있지만, 정밀도는 다소 낮습니다.

<br>

Ampere 이상의 하드웨어에 접근할 수 있다면 혼합 정밀도 훈련 및 평가에 bf16을 사용할 수 있습니다. bf16은 fp16보다 정밀도는 낮지만 훨씬 더 큰 동적 범위를 가집니다.

<br>

fp16에서는 가질 수 있는 가장 큰 숫자가 65535이며, 그 이상의 숫자는 오버플로우를 일으킵니다. bf16 숫자는 최대 3.39e+38까지 가능하며, 이는 fp32와 거의 같습니다.

```
training_args = TrainingArguments(bf16=True, **default_args)
```

### TF32

Ampere 하드웨어는 'tf32'라는 마법 같은 데이터 타입을 사용합니다. 이 데이터 타입은 fp32와 동일한 숫자 범위(8비트 지수)를 가지고 있지만, 23비트의 정밀도 대신에 fp16과 같은 10비트의 정밀도만을 가지며, 총 19비트만을 사용합니다.

<br>

"마법 같다"라는 표현은 기존의 fp32 훈련 및/또는 추론 코드를 사용하면서 tf32 지원을 활성화하면 최대 3배의 처리량 향상을 얻을 수 있다는 의미에서 사용됩니다.

<div id="Optimizer choice"></div>

## Optimizer choice

트랜스포머 모델을 훈련할 때 가장 일반적으로 사용되는 옵티마이저는 Adam 또는 AdamW 입니다.

<br>

Adam은 이전 그래디언트의 롤링 평균을 저장함으로써 좋은 수렴성을 달성하지만, 모델 파라미터 수만큼의 추가 메모리를 요구합니다. 이를 해결하기 위해 다른 옵티마이저를 사용할 수 있습니다.

<br>

예를 들어, NVIDIA GPU용으로 NVIDIA/apex가 설치되어 있거나, AMD GPU용으로 ROCmSoftwarePlatform/apex가 설치되어 있다면, adamw_apex_fused는 지원되는 모든 AdamW 옵티마이저 중에서 가장 빠른 훈련 경험을 제공합니다.

<br>

Trainer는 이외에 사용할 수 있는 다양한 옵티마이저를 통합합니다: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision, adafactor, adamw_bnb_8bit 등이 있습니다. 더 많은 옵티마이저는 제3자 구현을 통해 플러그인할 수 있습니다.

<br>

AdamW 옵티마이저의 두 가지 대안에 대해 자세히 살펴보겠습니다:

adafactor: Trainer에서 사용 가능

adamw_bnb_8bit: Trainer에서도 사용 가능하지만, 시연을 위해 아래에서 제3자 통합을 제공합니다.

<br>

예를 들어, "google-t5/t5-3b"와 같은 30억 파라미터 모델의 경우 비교하면:

표준 AdamW 옵티마이저는 각 파라미터마다 8바이트를 사용하므로 24GB의 GPU 메모리가 필요합니다(8*3 => 24GB).

Adafactor 옵티마이저는 각 파라미터마다 약 4바이트 이상을 사용하므로 12GB 이상이 필요합니다(4*3에 약간 더).

8bit BNB 양자화 옵티마이저는 옵티마이저 상태가 모두 양자화된 경우 단 6GB만 사용합니다(2*3).

<div id="How 🤗 Transformers solve tasks"></div>

# How 🤗 Transformers solve tasks

<div id="Speech and audio"></div>

## Speech and audio

Wav2Vec2는 라벨이 없는 음성 데이터로 사전 훈련된 자기 지도(self-supervised) 모델로, 라벨이 있는 데이터에 대해 미세 조정하여 오디오 분류와 자동 음성 인식 작업에 사용됩니다. 이 모델은 네 가지 주요 구성 요소로 이루어져 있습니다:

<img style="width: 100%; margin-bottom: 0px;" id="output" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/wav2vec2_architecture.png">

1. 특성 인코더(Feature encoder): 
   - 원시 오디오 파형을 받아 평균이 0이고 unit variance(단위 분산)으로 정규화한 후, 각각 20ms 길이의 특성 벡터 시퀀스로 변환합니다.  <br>* unit variance: 데이터셋의 분산을 1로 조정
   - 오디오 파형은 본질적으로 연속적이기 때문에, 텍스트 시퀀스가 단어로 분리되는 것처럼 개별 단위로 나눌 수 없습니다. 
   이러한 모델들은 지식을 직접 수정하고 확장할 수 있으며, 접근한 지식을 검사하고 해석할 수 있다는 장점이 있습니다.

2. 양자화 모듈(Quantization module): 
   - 특성 벡터는 이산적인 음성 단위를 학습하려는 목표를 가진 양자화 모듈로 전달됩니다. 
   - 음성 단위는 코드북(어휘로 생각할 수 있음)이라고 하는 코드워드 모음에서 선택됩니다. 코드북에서 연속 오디오 입력을 가장 잘 나타내는 벡터 또는 음성 단위가 선택되어 모델을 통해 전달됩니다.
   
3. 컨텍스트 네트워크(Context network): 
   - 특성 벡터의 절반 가량이 무작위로 마스킹되며, 마스킹된 특성 벡터는 상대적 위치 임베딩을 추가하여 Transformer 인코더인 컨텍스트 네트워크로 공급됩니다.
   
4. 사전 훈련 목표(Pretraining objective): 
   - 컨텍스트 네트워크의 사전 훈련 목표는 대조적인 작업(contrastive task)입니다. 
   - 모델은 마스킹된 예측의 참 양자화된 음성 표현을 거짓 집합으로부터 예측해야 하며, 이는 모델이 가장 유사한 컨텍스트 벡터와 양자화된 음성 단위(대상 레이블)를 찾도록 장려합니다.


이제 Wav2Vec2가 사전 훈련되었으므로, 오디오 분류나 자동 음성 인식을 위해 자신의 데이터에 미세 조정할 수 있습니다!

### Audio classification

오디오 분류를 위해서는 Wav2Vec2 모델 기반 위에 sequence classification head를 추가합니다. 

<br>

이 classification head는 선형 레이어로서, 인코더의 은닉 상태를 받아들입니다. 

<br>

hidden states는 각 오디오 프레임에서 학습된 특성을 나타내며, 길이가 다를 수 있습니다. 고정 길이의 하나의 벡터를 생성하기 위해, hidden states들은 먼저 풀링되고, 그 다음 클래스 레이블에 대한 logits으로 변환됩니다. 

<br>

logits과 타겟 사이의 cross-entropy loss이 계산되어 가장 가능성 있는 클래스를 찾습니다.

### Automatic speech recognition

자동 음성 인식을 위해서는 Wav2Vec2 모델 기반 위에 언어 모델링 헤드를 연결성 시간 분류(Connectionist Temporal Classification, CTC)를 위해 추가합니다. 

- CTC
  - 주로 자동 음성 인식(Automatic Speech Recognition, ASR)과 같이 입력(오디오 신호)과 출력(텍스트) 사이의 시간적 매핑이 명확하지 않은 작업에 사용됩니다. 
  - CTC는 입력 시퀀스 내의 각 시점에 대해 가능한 모든 출력 시퀀스의 확률을 계산하여, 가장 가능성 높은 시퀀스를 예측합니다.

언어 모델링 헤드는 linear layer로서, 인코더의 은닉 상태를 받아들이고 이를 logits으로 변환합니다. 

<br>

각 logits은 토큰 클래스를 나타내며(토큰의 수는 작업 어휘에서 온다), logits과 타겟 사이의 CTC 손실이 계산되어 가장 가능성 있는 토큰 시퀀스를 찾습니다. 이 시퀀스는 그 후에 텍스트로 디코딩됩니다.

<div id="Computer vision"></div>

## Computer vision

컴퓨터 비전 작업에 접근하는 데에는 주로 두 가지 방법이 있습니다:

1. 이미지를 일련의 패치로 분할하고 Transformer를 사용하여 병렬 처리하는 방법
   - ViT(Vision Transformer)가 이 방법의 대표적인 예입니다. ViT는 이미지를 작은 패치로 나누고, 각 패치를 Transformer 모델에 입력으로 사용하여 이미지를 처리합니다. 
   - 이 접근 방식은 Transformer가 텍스트 처리에서 보여준 강력한 성능을 이미지 분류와 같은 비전 작업에 활용하려는 시도로, 이미지 내 패치 간의 관계를 학습하여 효과적으로 이미지를 분류할 수 있습니다.

2. 모던 CNN(Convolutional Neural Network)을 사용하는 방법
   - ConvNeXT와 같은 모던 CNN은 전통적인 컨볼루셔널 레이어에 의존하면서도 현대적인 네트워크 디자인을 채택합니다. 
   - 이러한 모델은 이미지의 공간적 계층적 특성을 효과적으로 학습하여 다양한 비전 작업에 적용됩니다. 
   - ConvNeXT는 기존의 CNN 아키텍처를 개선하여 더 나은 성능과 효율성을 제공합니다.

추가적으로 Transformer와 컨볼루션을 혼합하는 방법이 있습니다.

<br>

예를 들어, Convolutional Vision Transformer(LeViT)와 같은 모델은 Transformer와 컨볼루셔널 레이어의 장점을 결합합니다. 이러한 접근 방식은 두 가지 방법의 강점을 결합하여 더욱 강력한 모델을 만들려고 합니다.

### Image classification

#### Transformer

<br>
<br>

<img style="width: 100%; margin-top: 0px;" id="output" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg">

<br>

1. 이미지를 패치로 분할
   - 이미지는 서로 겹치지 않는 정사각형 패치로 분할되며, 각 패치는 벡터 또는 패치 임베딩으로 변환됩니다. 

     <br>
  
   - 패치 임베딩은 2D 컨볼루셔널 레이어로부터 생성되며, 이 레이어는 베이스 Transformer에 적합한 입력 차원(각 패치 임베딩당 768개의 값)을 생성합니다.  <br> 예를 들어, 224x224 픽셀 이미지를 196개의 16x16 이미지 패치로 분할할 수 있습니다.  
  
     <br>
    
   - 텍스트가 단어로 토크나이징되는 것처럼, 이미지는 패치 시퀀스로 "토크나이징"됩니다.

2. 학습 가능한 임베딩 추가
   - 특별한 [CLS] 토큰(이미지를 대표하는 정보를 포함)에 대한 학습 가능한 임베딩이 패치 임베딩의 시작 부분에 BERT와 같이 추가됩니다. 

     <br>

   - [CLS] 토큰의 최종 hidden state는 붙어 있는 분류 헤드로의 입력으로 사용되며, 다른 출력은 무시됩니다. 

     <br>

   - 이 토큰은 모델이 이미지의 표현을 인코딩하는 방법을 학습하는 데 도움을 줍니다.

3. 위치 임베딩 추가
   -  모델이 이미지 패치의 순서를 알지 못하기 때문에, 패치와 학습 가능한 임베딩에 위치 임베딩이 추가됩니다. 
   - 위치 임베딩 역시 학습 가능하며 패치 임베딩과 동일한 크기를 가집니다. 마지막으로, 모든 임베딩은 Transformer 인코더로 전달됩니다.

4. 출력 처리
   - ViT의 사전 훈련 목표는 단순히 분류입니다. 
   - 다른 분류 헤드와 마찬가지로, MLP 헤드는 출력을 클래스 레이블에 대한 로짓으로 변환하고, 가장 가능성 높은 클래스를 찾기 위해 크로스 엔트로피 손실을 계산합니다.

#### CNN

ConvNeXT는 새롭고 현대적인 네트워크 디자인을 채택하여 성능을 향상시킨 CNN(컨볼루셔널 신경망) 아키텍처입니다.

1. 각 단계의 블록 수 변경 및 “패치화”
   - 이미지를 더 큰 스트라이드와 해당 커널 크기로 "패치화"하여 비중첩 슬라이딩 윈도우를 만듭니다. 이 패치화 전략은 ViT가 이미지를 패치로 분할하는 방식과 유사합니다.

2. 병목(bottleneck) 레이어
   - 채널 수를 줄인 후 다시 복원하여 1x1 컨볼루션을 더 빠르게 수행하고 depth를 늘립니다. 
   - 반대로 채널 수를 확장하고 축소하는 역병목(inverted bottleneck)은 더 메모리 효율적(입력 채널 수가 많아 덜 깊게 layer를 구성)입니다.

3. 3x3 컨볼루셔널 레이어 대체
   - 병목 레이어에서 일반적인 3x3 컨볼루셔널 레이어를 깊이별 컨볼루션(depthwise convolution)으로 대체합니다. 
   - 일반적인 3x3 컨볼루셔널 레이어를 깊이별 컨볼루션(depthwise convolution)으로 대체함으로써 얻을 수 있는 주요 이점
     - 계산 효율성의 증가
       - 파라미터 수 감소: 예를 들어, 입력이 64개의 채널을 가지고 있고, 3x3 컨볼루션을 사용하는 경우, 각 컨볼루션 필터는 64개의 채널 모두에 적용되므로, 필터 당 9(3x3) * 64개의 파라미터가 필요합니다. 반면, 깊이별 컨볼루션에서는 각 채널에 대해 단 9개의 파라미터만 필요하게 됩니다.
     - 모델 성능의 개선
       - 특성 추출의 효율성: 각 채널마다 독립적인 필터를 적용함으로써, 깊이별 컨볼루션은 채널 별 특성을 더 세밀하게 추출할 수 있습니다. 이는 모델이 입력 데이터의 다양한 특성을 더 잘 학습하게 하고, 결과적으로 전반적인 모델의 성능을 향상시킬 수 있습니다.
       - 과적합 감소: 파라미터 수가 줄어들면 모델이 훈련 데이터에 과적합되는 경향도 감소합니다. 이는 모델이 더 일반화된 특성을 학습하게 하여 새로운 데이터에 대해 더 잘 일반화할 수 있게 합니다.
   - 깊이별 컨볼루션은 각 입력 채널에 대해 별도로 컨볼루션을 적용한 후 마지막에 다시 쌓습니다.

4. ViT의 global receptive 모방
   - ConvNeXT는 커널 크기를 7x7로 증가시켜 ViT의 주의 메커니즘 덕분에 한 번에 더 많은 이미지를 볼 수 있는 global receptive 효과를 모방하려고 시도합니다.

5. 레이어 디자인 변경
   - 활성화 및 정규화 레이어 수를 줄이고, 활성화 함수를 ReLU에서 GELU로 변경하며, BatchNorm 대신 LayerNorm을 사용하는 등 Transformer 모델을 모방하는 여러 레이어 디자인 변경을 채택합니다.

<img style="width: 70%; margin-top: 0px;" id="output" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/convnext_architecture.png">

- 96-d: feature map이 96개의 차원
- 1x1: 1x1 컨볼루션
- d7x7: depthwise convolution

### Object detection

DETR(DEtection TRansformer)는 CNN과 Transformer 인코더-디코더를 결합한 최초의 끝까지(end-to-end) 객체 탐지 모델입니다. 

<br>

이 모델은 전통적인 객체 탐지 접근 방식과 다른 새로운 메커니즘을 도입하여 객체 탐지 문제를 해결합니다.

<br>

<img style="width: 100%; margin-top: 0px;" id="output" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/detr_architecture.png">

1. CNN 백본
  - 사전 훈련된 CNN 백본은 입력 이미지를 받아 저해상도의 특성 맵을 생성합니다. 이 특성 맵은 이미지의 고수준(high-level) 표현을 포함하며, 1x1 컨볼루션을 통해 차원이 축소됩니다.

2. 특성 벡터와 위치 임베딩
  - Transformer가 순차적인 모델이기 때문에, 특성 맵은 일련의 특성 벡터로 평탄화되고, 위치 임베딩과 결합됩니다. 이를 통해 각 벡터가 이미지 내의 어느 위치에 해당하는지 모델이 인식할 수 있게 합니다.

3. 인코더와 디코더
- 특성 벡터들은 인코더를 통과하여 이미지 표현을 학습합니다. 인코더의 hidden state는 object queries와 결합되어 디코더로 전달됩니다. 
- object queries는 모델이 이미지 내의 다양한 객체를 인식하고 위치를 결정하기 위해 사용하는 학습 가능한 임베딩이며, 디코더 입력 전 일반적으로 무작위로 초기화되며, 디코더의 attention 레이어를 거치며 업데이트됩니다.

4. object detection head
- 디코더의 hidden state는 object detection head로 전달되어, 각 객체 쿼리에 대한 bounding box 좌표와 클래스 레이블을 예측합니다. 이 때, 객체가 없는 경우 'no object' 클래스가 예측됩니다.

5. 병렬 디코딩과 bipartite matching loss
- DETR은 모든 object queries를 병렬로 디코딩하여 N개의 최종 예측을 생성합니다. 여기서 N은 쿼리의 수입니다. 
- 처음에 객체 수를 알 수 없기 때문에 object queries는 일반적으로 충분히 크게 설정됩니다.
- 훈련 중에는 bipartite matching loss을 사용하여 고정된 수의 예측과 고정된 집합의 실제 레이블을 비교합니다. 이 손실 함수는 예측과 실제 레이블 간의 일대일 할당을 찾도록 합니다.

### Image segmentation

Mask2Former는 모든 유형의 이미지 분할 작업을 해결하기 위한 범용 아키텍처입니다. 

<br>

기존의 분할 모델들이 instance(같은 종류의 객체라도 개별적으로 식별), semantic(카테고리만), panoptic(instance + semantic) segmentation과 같은 특정 하위 작업에 맞춰져 개발된 반면, Mask2Former는 이러한 모든 작업을 마스크 분류 문제로 간주합니다. 

<br>

여기서 마스크 분류는 주어진 이미지에 대해 N개의 세그먼트로 픽셀을 그룹화하고, 해당 세그먼트의 N개 마스크와 그에 해당하는 클래스 레이블을 예측합니다.

