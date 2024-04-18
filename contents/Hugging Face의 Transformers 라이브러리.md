---
date: '2024-04-04'
title: 'Hugging Face의 Transformers 라이브러리'
categories: ['LLM']
summary: 'llm을 효율적으로 학습 하는 방법을 알아보자'
thumbnail: './test.png'
---

Hugging Face의 Transformers 라이브러리는 자연어 처리(NLP), 컴퓨터 비전, 오디오 처리 및 다중 모달 입력을 포함한 다양한 기계 학습 작업을 위한 대규모 Transformer 모델을 쉽게 사용할 수 있도록 설계된 오픈 소스 프로젝트입니다.

<br>

Hugging Face의 공식 문서를 기반으로 필요한 내용을 정리하였습니다.

<div id="Philosophy"></div>

# Philosophy

- 사용자가 가능한 쉽고 빠르게 이용할 수 있도록 하는 것

이를 위해 라이브러리는 사용자가 배워야 할 abstractions의 수를 크게 제한했습니다. 사실상 거의 모든 abstractions를 제거하고, 모델 사용에 필요한 세 가지 표준 클래스만을 요구합니다:

1. Configuration(구성): 모델의 구성과 관련된 정보(예: 모델의 하이퍼파라미터)를 담당합니다.
2. Models(모델): 실제 Transformer 모델을 나타냅니다.
3. Preprocessing Class(전처리 클래스): 입력 데이터를 모델이 이해할 수 있는 형태로 변환하는 역할을 합니다. 이는 자연어 처리(NLP)에서는 토크나이저, 컴퓨터 비전에서는 이미지 프로세서, 오디오 처리에서는 특징 추출기, 그리고 다중 모달 입력에서는 해당 프로세서를 포함합니다.

이 클래스들은 사전 훈련된 인스턴스를 사용하여 from_pretrained() 메서드를 통해 간단하고 통일된 방식으로 초기화할 수 있습니다.

<br>

이 메서드는 필요한 경우 사전 훈련된 체크포인트에서 관련 클래스 인스턴스와 연관된 데이터(configurations’ hyperparameters, tokenizers’ vocabulary, models’ weights)를 다운로드, 캐시하고 로드합니다.

<br>

이 체크포인트는 Hugging Face Hub 또는 사용자의 저장된 체크포인트에서 제공될 수 있습니다.

<br>

또한, 라이브러리는 두 가지 주요 API를 제공합니다:

<br>

1. Pipeline(): 주어진 작업에 대해 모델을 빠르게 추론하기 위해 사용됩니다.
2. Trainer: PyTorch 모델을 빠르게 훈련하거나 미세 조정하기 위한 도구입니다. TensorFlow 모델도 Keras의 fit 메서드와 호환됩니다.

따라서, Transformers 라이브러리는 신경망을 위한 modular toolbox가 아닙니다. 

<br>

라이브러리를 확장하거나 기반으로 새로운 기능을 구축하려면, 표준 Python, PyTorch, TensorFlow, Keras 모듈을 사용하고 라이브러리의 기본 클래스를 상속하여 모델 로딩 및 저장과 같은 기능을 재사용할 수 있습니다.

- 최신 기술(state-of-the-art) 모델을 제공하며, 이 모델들이 원래 모델의 성능과 가능한 한 가깝게 동작하도록 하는 것

이를 위해 라이브러리는 각 아키텍처별로 적어도 한 가지 예제를 제공하며, 이 예제는 해당 아키텍처의 공식 저자들이 제공한 결과를 재현합니다.

<br>

이 목표를 달성하기 위해, Transformers 라이브러리의 코드는 가능한 한 원본 코드 베이스와 가깝게 유지됩니다. 이러한 접근 방식은 두 가지 주요 이점을 가집니다:

1. 성능의 일관성: 원본 모델의 코드와 매우 유사하게 구현함으로써, 원본 저자들이 보고한 성능과 유사한 결과를 얻을 수 있습니다. 이는 연구자나 개발자가 모델을 평가하고 사용할 때, 기대할 수 있는 성능에 대한 신뢰성을 높입니다.
2. 교차 프레임워크 호환성: 일부 코드는 TensorFlow에서 시작하여 PyTorch로 변환되었거나 그 반대의 경우도 있을 수 있습니다. 이 과정에서, 특정 프레임워크의 관례나 스타일을 완벽하게 따르지 않을 수도 있습니다. 그러나, 이러한 접근은 다른 프레임워크 사용자들에게도 모델을 접근 가능하게 하며, 다양한 기술 스택에서도 최신 모델을 활용할 수 있게 합니다.

- hidden-states와 attention weights 접근

Transformers는 단일 API를 통해 모든 히든 스테이트(hidden-states)와 어텐션 가중치(attention weights)에 접근할 수 있는 기능을 제공합니다. 

<br>

이를 통해 사용자는 모델이 결정을 내리는 데 있어 중요한 역할을 하는 내부 메커니즘을 분석할 수 있습니다.

- 표준화된 전처리 클래스와 기본 모델 API

이러한 표준화는 다양한 모델 간에 쉽게 전환할 수 있게 해줍니다. 사용자는 다른 모델을 실험하고 비교할 때 일관된 방식으로 작업할 수 있습니다.

- 단어장과 임베딩에 새로운 토큰 추가

미세 조정(fine-tuning)을 위해 새로운 토큰을 단어장(vocabulary)과 임베딩(embeddings)에 쉽고 일관된 방식으로 추가할 수 있는 기능을 제공합니다. 

<br>

이는 특정 작업이나 도메인에 모델을 맞춤화하는 데 중요합니다.

- mask and prune Transformer heads

모델의 특정 부분을 비활성화하거나 제거하여 모델의 크기를 줄이거나 성능을 개선할 수 있는 간단한 방법을 제공합니다. 

<br>

이는 모델의 효율성을 높이고, 특정 작업에 대한 중요한 특징을 강조하는 데 사용될 수 있습니다.

- PyTorch, TensorFlow 2.0, Flax 간의 쉬운 전환

이 기능은 한 프레임워크에서 훈련을 진행하고 다른 프레임워크에서 추론을 수행할 수 있게 해줍니다. 이는 다양한 기술 스택을 가진 팀이나 프로젝트에서 유연성을 제공합니다.

<div id="Transformers가 할 수 있는 일"></div>

# Transformers가 할 수 있는 일

<div id="Audio"></div>

## Audio

<br>

오디오 및 음성 처리 작업은 다른 유형의 데이터 처리 작업과 몇 가지 주요한 차이점이 있습니다. 주된 차이점은 오디오 입력이 연속적인 신호라는 점입니다. 텍스트와 달리, 원시 오디오 파형은 문장이 단어로 나뉘는 것처럼 깔끔하게 이산적인 청크로 나눌 수 없습니다. 

<br>

이 문제를 해결하기 위해, 원시 오디오 신호는 일반적으로 정기적인 간격으로 샘플링됩니다. 간격 내에서 더 많은 샘플을 취하면, 샘플링 레이트가 높아지고 오디오는 원본 오디오 소스와 더 유사하게 됩니다.

<br>

과거의 접근 방식에서는 유용한 특성을 추출하기 위해 오디오를 사전 처리했습니다. 현재는 오디오 및 음성 처리 작업을 시작할 때 원시 오디오 파형을 직접 특성 인코더에 공급하여 오디오 표현을 추출하는 것이 더 일반적입니다. 

<br>

이 방식은 사전 처리 단계를 단순화하고 모델이 가장 필수적인 특성을 학습할 수 있도록 합니다.

<br>

즉, 현대의 오디오 및 음성 처리 모델은 복잡한 사전 처리 절차 없이도 직접 원시 오디오 데이터를 처리할 수 있는 능력을 갖추고 있습니다. 이는 모델이 오디오 데이터의 중요한 특성을 직접 학습하도록 하여, 효율성을 높이고 더 나은 성능을 달성할 수 있도록 합니다.

- Audio classification
  - 음향 장면 분류(Acoustic scene classification): 오디오를 장면 레이블(예: “사무실”, “해변”, “경기장”)로 분류합니다. 이 작업은 오디오 녹음이 어디에서 이루어졌는지를 식별하는 데 사용될 수 있습니다.

  - 음향 이벤트 탐지(Acoustic event detection): 오디오를 소리 이벤트 레이블(예: “자동차 경적”, “고래 울음”, “유리 깨짐”)로 분류합니다. 이는 특정 소리 이벤트를 식별하고 분류하는 데 유용합니다.

  - 태깅(Tagging): 여러 소리가 포함된 오디오(예: 새소리, 회의에서의 발화자 식별)에 레이블을 붙입니다. 이 작업은 오디오에 포함된 다양한 소리 유형을 식별하여 각각에 대한 태그를 할당하는 데 초점을 맞춥니다.

  - 음악 분류(Music classification): 음악을 장르 레이블(예: “메탈”, “힙합”, “컨트리”)로 분류합니다. 이 작업은 음악 트랙을 장르별로 분류하고 조직하는 데 사용됩니다.
```
from transformers import pipeline

classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
preds

[{'score': 0.4532, 'label': 'hap'},
 {'score': 0.3622, 'label': 'sad'},
 {'score': 0.0943, 'label': 'neu'},
 {'score': 0.0903, 'label': 'ang'}]
```

- Automatic speech recognition

자동 음성 인식(Automatic Speech Recognition, ASR)은 음성을 텍스트로 변환하는 기술입니다.

<br>

Transformer 아키텍처가 도움을 준 주요 도전 중 하나는 저자원 언어에서의 ASR입니다. 

<br>

대량의 음성 데이터에서 사전 훈련을 수행하고, 저자원 언어의 레이블이 붙은 단 한 시간 분량의 음성 데이터에만 모델을 미세 조정함으로써, 이전에 100배 더 많은 레이블이 붙은 데이터로 훈련된 ASR 시스템과 비교하여 여전히 고품질의 결과를 생성할 수 있습니다.

<br>

이는 Transformer 모델이 대규모 데이터에서 학습한 일반적인 언어적, 음성적 특성을 이용하여, 상대적으로 적은 양의 데이터로도 효과적인 학습이 가능함을 의미합니다. 

br>

이런 접근 방식은 특히 데이터가 부족한 언어에 대한 ASR 시스템 개발에서 큰 진전을 의미하며, 다양한 언어와 방언에 대한 접근성과 포용성을 크게 향상시킵니다.

```
from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

<div id="Computer vision"></div>

## Computer vision

- Image classification
  - 헬스케어: 의료 이미지를 레이블링하여 질병을 감지하거나 환자의 건강을 모니터링합니다. 예를 들어, X-ray나 MRI 스캔을 분석하여 특정 질병의 존재를 확인하거나, 질병의 진행 상태를 추적할 수 있습니다.

  - 환경: 위성 이미지를 레이블링하여 삼림 벌채를 모니터링하거나, 야생 지역 관리에 정보를 제공하거나, 산불을 탐지합니다. 이를 통해 환경 변화를 시각화하고, 자연 재해에 대응하며, 지속 가능한 자원 관리를 지원할 수 있습니다.

  - 농업: 작물의 이미지를 레이블링하여 식물의 건강을 모니터링하거나, 땅 사용 모니터링을 위한 위성 이미지를 레이블링합니다. 이를 통해 농작물의 성장 상태를 추적하고, 수확량을 예측하며, 질병이나 해충의 발생을 조기에 감지할 수 있습니다.

  - 생태학: 동물이나 식물 종의 이미지를 레이블링하여 야생 동물 인구를 모니터링하거나 멸종 위기 종을 추적합니다. 이는 생물 다양성 연구와 보존 노력에 중요한 정보를 제공할 수 있습니다.

```
from transformers import pipeline

classifier = pipeline(task="image-classification")
preds = classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
print(*preds, sep="\n")

{'score': 0.4335, 'label': 'lynx, catamount'}
{'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
{'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
{'score': 0.0239, 'label': 'Egyptian cat'}
{'score': 0.0229, 'label': 'tiger cat'}
```

- Object detection

  - 자율 주행 차량: 다른 차량, 보행자, 교통 신호등과 같은 일상적인 교통 객체를 탐지합니다. 이 정보는 자율 주행 시스템이 주변 환경을 이해하고 안전한 주행 경로를 결정하는 데 필수적입니다.

  - 원격 감지: 재난 모니터링, 도시 계획, 날씨 예측 등에 사용됩니다. 위성 이미지나 항공 이미지에서 중요한 객체를 식별하고 분석함으로써, 자연 재해의 영향을 평가하거나 도시의 확장을 계획하는 데 도움을 줍니다.

  - 결함 탐지: 건물의 균열이나 구조적 손상, 제조 결함을 탐지합니다. 이는 품질 관리와 유지 보수 작업에서 중요한 의미를 가지며, 위험을 예방하고 비용을 절감하는 데 기여합니다.

```
from transformers import pipeline

detector = pipeline(task="object-detection")
preds = detector(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]
preds

[{'score': 0.9865,
  'label': 'cat',
  'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]
```

- Image segmentation
  - 인스턴스 분할(Instance segmentation): 객체의 클래스를 레이블링할 뿐만 아니라, 각 객체의 구별 가능한 인스턴스도 레이블링합니다(예: "dog-1", "dog-2"). 이는 동일한 유형의 여러 객체를 식별하고 각각을 개별적으로 처리할 수 있게 해줍니다.

  - 파노프틱 분할(Panoptic segmentation): 의미론적 분할(Semantic segmentation)과 인스턴스 분할의 결합입니다. 이는 각 픽셀을 의미론적 클래스로 레이블링하고 객체의 각 구별 가능한 인스턴스도 레이블링합니다. 이를 통해 이미지의 구조를 보다 상세하게 이해할 수 있으며, 개별 객체와 그 배경을 모두 정확하게 분류할 수 있습니다.

```
from transformers import pipeline

segmenter = pipeline(task="image-segmentation")
preds = segmenter(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
print(*preds, sep="\n")

{'score': 0.9879, 'label': 'LABEL_184'}
{'score': 0.9973, 'label': 'snow'}
{'score': 0.9972, 'label': 'cat'}
```

- Depth estimation
  - 스테레오(Stereo): 두 이미지를 비교하여 깊이를 추정합니다. 이 두 이미지는 동일한 장면에 대해 약간 다른 각도에서 촬영되며, 이러한 차이를 분석함으로써 각 픽셀의 깊이 정보를 추출할 수 있습니다. 스테레오 방식은 두 눈으로 물체의 거리를 판단하는 인간의 시각 시스템과 유사한 원리를 사용합니다.
  - 단안(Monocular): 단일 이미지에서 깊이를 추정합니다. 이 방식은 한 장의 이미지만을 사용하여 깊이 정보를 추출하며, 이는 훨씬 더 도전적인 작업입니다. 단안 깊이 추정은 이미지의 특정 단서(예: 텍스처 그라데이션, 객체 크기, 관점)를 분석하여 깊이 정보를 유추합니다.

```
from transformers import pipeline

depth_estimator = pipeline(task="depth-estimation")
preds = depth_estimator(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
```

<div id="Natural language processing"></div>

## Natural language processing

- Text classification
  - 감성 분석: 텍스트를 긍정적 또는 부정적과 같은 어떤 극성에 따라 레이블링합니다. 이는 정치, 금융, 마케팅과 같은 분야에서 의사 결정을 정보 제공하고 지원하는 데 사용될 수 있습니다. 감성 분석을 통해, 기업은 소비자의 의견을 분석하고, 제품이나 서비스에 대한 반응을 이해할 수 있습니다.
  - 콘텐츠 분류: 텍스트를 어떤 주제에 따라 레이블링하여 뉴스 및 소셜 미디어 피드에서 정보를 조직하고 필터링하는 데 도움을 줍니다(날씨, 스포츠, 금융 등). 콘텐츠 분류를 통해, 사용자는 관심 있는 주제의 정보를 쉽게 찾을 수 있으며, 정보의 홍수 속에서 중요한 내용을 식별할 수 있습니다.

```
from transformers import pipeline

classifier = pipeline(task="sentiment-analysis")
preds = classifier("Hugging Face is the best thing since sliced bread!")
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
preds

[{'score': 0.9991, 'label': 'POSITIVE'}]
```

- Token classification
  - 명명된 엔티티 인식(Named Entity Recognition, NER): 토큰을 조직, 사람, 위치 또는 날짜와 같은 엔티티 카테고리에 따라 레이블링합니다. NER은 특히 생물의학 분야에서 인기가 높으며, 이때 유전자, 단백질 및 약물 이름과 같은 엔티티를 레이블링할 수 있습니다. 예를 들어, 환자 기록에서 특정 질병 이름이나 약물을 자동으로 식별하고 분류하는 데 사용됩니다.

  - 품사 태깅(Part-of-Speech Tagging, POS): 토큰을 명사, 동사, 형용사와 같은 품사에 따라 레이블링합니다. POS는 두 개의 동일한 단어가 문법적으로 어떻게 다른지(예: "은행"이 명사인 경우와 동사인 경우)를 번역 시스템이 이해하는 데 도움을 줍니다. 이는 문장 구조를 분석하고 문장의 의미를 더 정확하게 파악하는 데 필수적입니다.

```
from transformers import pipeline

classifier = pipeline(task="ner")
preds = classifier("Hugging Face is a French company based in New York City.")
preds = [
    {
        "entity": pred["entity"],
        "score": round(pred["score"], 4),
        "index": pred["index"],
        "word": pred["word"],
        "start": pred["start"],
        "end": pred["end"],
    }
    for pred in preds
]
print(*preds, sep="\n")

{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}
```

- Question answering
  - 추출적(Extractive): 질문과 일부 문맥이 주어졌을 때, 답변은 모델이 문맥에서 추출해야 하는 텍스트의 범위입니다. 예를 들어, 사용자가 "바람과 함께 사라지다는 어느 나라에서 제작되었나요?"라고 물었을 때, 모델은 제공된 문맥(예: 관련된 문서나 글)에서 직접 "미국"이라는 답변을 찾아내야 합니다.

  - 추상적(Abstractive): 질문과 일부 문맥이 주어졌을 때, 답변은 문맥에서 생성됩니다. 이 접근 방식은 Text2TextGenerationPipeline에 의해 처리되며, QuestionAnsweringPipeline과는 다릅니다. 추상적 질문 응답 모델은 문맥을 바탕으로 새로운 답변을 생성하여, 답변이 직접적으로 문맥에서 발췌되지 않은 경우에도 정보를 제공할 수 있습니다.

```
from transformers import pipeline

question_answerer = pipeline(task="question-answering")
preds = question_answerer(
    question="What is the name of the repository?",
    context="The name of the repository is huggingface/transformers",
)
print(
    f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
)

score: 0.9327, start: 30, end: 54, answer: huggingface/transformers
```

- Summarization
  - 추출적(Extractive): 원본 텍스트에서 가장 중요한 문장을 식별하고 추출합니다. 이 방식은 원본 텍스트의 특정 부분을 그대로 가져와 요약을 구성하기 때문에, 요약된 텍스트는 원본 문서의 실제 단어와 문장으로만 구성됩니다.

  - 추상적(Abstractive): 원본 텍스트에서 타겟 요약(입력 문서에 없는 새로운 단어를 포함할 수 있음)을 생성합니다. SummarizationPipeline은 추상적 접근 방식을 사용합니다. 이 방식은 원본 텍스트를 기반으로 새로운 단어나 문장을 생성하여 요약을 만들기 때문에, 보다 자연스럽고 읽기 쉬운 요약을 제공할 수 있습니다.

```
from transformers import pipeline

summarizer = pipeline(task="summarization")
summarizer(
    "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
)

[{'summary_text': ' The Transformer is the first sequence transduction model based entirely on attention . It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention . For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers .'}]
```

- Translation
  - 초기에는 번역 모델이 대부분 단일 언어(monolingual)에 초점을 맞췄지만, 최근에는 여러 쌍의 언어 간에 번역할 수 있는 다언어(multilingual) 모델에 대한 관심이 증가하고 있습니다. 이러한 다언어 모델은 하나의 모델을 사용하여 여러 언어 간의 번역을 가능하게 하며, 이는 특히 자원이 제한적인 언어에 대한 번역 품질을 향상시키는 데 도움이 됩니다.

```
from transformers import pipeline

text = "translate English to French: Hugging Face is a community-based open-source platform for machine learning."
translator = pipeline(task="translation", model="google-t5/t5-small")
translator(text)

[{'translation_text': "Hugging Face est une tribune communautaire de l'apprentissage des machines."}]
```

- Language modeling
  - 인과적(Causal): 모델의 목표는 시퀀스에서 다음 토큰을 예측하는 것이며, 미래의 토큰은 마스킹됩니다. 이 유형의 모델링은 특정 단어 다음에 어떤 단어가 나올지를 예측하는 것에 초점을 맞추며, 텍스트 생성과 같은 작업에서 주로 사용됩니다.
  ```
  from transformers import pipeline

  prompt = "Hugging Face is a community-based open-source platform for machine learning."
  generator = pipeline(task="text-generation")
  generator(prompt)  # doctest: +SKIP
  ```
  - 마스킹(Masked): 모델의 목표는 시퀀스 내에서 마스킹된 토큰을 예측하는 것으로, 시퀀스 내의 모든 토큰에 대한 전체 접근을 허용합니다. 이 접근 방식은 모델이 주어진 문맥에서 특정 단어가 어떤 단어인지를 추측하도록 하며, BERT와 같은 모델에서 사용되는 전략입니다.
  ```
  text = "Hugging Face is a community-based open-source <mask> for machine learning."
  fill_mask = pipeline(task="fill-mask")
  preds = fill_mask(text, top_k=1)
  preds = [
  {
  "score": round(pred["score"], 4),
  "token": pred["token"],
  "token_str": pred["token_str"],
  "sequence": pred["sequence"],
  }
  for pred in preds
  ]
  preds
  
  [{'score': 0.2236,
  'token': 1761,
  'token_str': ' platform',
  'sequence': 'Hugging Face is a community-based open-source platform for machine learning.'}]
  ```
  
- Multimodal
  - 모델이 특정 문제를 해결하기 위해 여러 데이터 형태(텍스트, 이미지, 오디오, 비디오)를 처리해야 하는 작업입니다. 이미지 캡셔닝(Image captioning)은 멀티모달 작업의 예시로, 모델이 이미지를 입력으로 받아 이미지를 설명하거나 이미지의 특정 속성을 설명하는 텍스트 시퀀스를 출력합니다.
  - 이미지 캡셔닝과 같은 작업에서 모델은 이미지 임베딩과 텍스트 임베딩 사이의 관계를 학습합니다.
  - Document question answering
    - 문서로부터 자연어 질문에 대한 답변을 제공하는 작업입니다. 텍스트를 입력으로 받는 토큰 수준의 질문 응답 작업과 달리, 문서 질문 응답은 문서에 대한 질문과 함께 문서의 이미지를 입력으로 받아 답변을 반환합니다.
    - 문서 질문 응답은 구조화된 문서를 파싱하고 그것으로부터 핵심 정보를 추출하는 데 사용될 수 있습니다. 예를 들어, 영수증에서 총액과 거스름돈을 추출하는 것과 같은 작업이 가능합니다.

```
from transformers import pipeline
from PIL import Image
import requests

url = "https://datasets-server.huggingface.co/assets/hf-internal-testing/example-documents/--/hf-internal-testing--example-documents/test/2/image/image.jpg"
image = Image.open(requests.get(url, stream=True).raw)

doc_question_answerer = pipeline("document-question-answering", model="magorshunov/layoutlm-invoices")
preds = doc_question_answerer(
    question="What is the total amount?",
    image=image,
)
preds

[{'score': 0.8531, 'answer': '17,000', 'start': 4, 'end': 4}]
```

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
<div id="Methods"></div>

# Methods

1. 검색기 (Retriever)

    - 입력 $x$에 대해 텍스트 패시지를 반환하는 검색기 $p_η(z∣x)$를 포함합니다. 여기서 $η$는 검색기의 파라미터를 나타냅니다. 이 검색기는 주어진 쿼리에 대해 상위 K개의 텍스트 패시지를 반환합니다.

2. 생성기 (Generator)

    - 검색된  text passages $z$, 이전의 토큰 $y_{1:i−1}$, 그리고 원본 입력 $x$를 바탕으로 현재 토큰 $y_i$를 생성하는 생성기 $p_θ(y_i∣x,z,y_{1:i−1})$를 포함합니다. 여기서 $θ$는 생성기의 파라미터입니다.

<div id="Models"></div>

## Models

<img style="width: 100%; margin-right: 0px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="rag/ragModel.PNG">

- 쿼리 인코더(Query Encoder): 사용자의 쿼리나 질문 $x$를 밀집된 벡터로 변환합니다.
- 문서 인덱스(Document Index): 가능한 모든 문서 $z$의 밀집된 벡터 표현을 저장합니다. 이 인덱스는 문서를 빠르게 검색할 수 있게 해주는 역할을 합니다.
- 사전 훈련된 seq2seq 모델 (Generator): 이 모델은 입력 시퀀스를 받아 출력 시퀀스를 생성합니다. 이 경우, 입력 시퀀스는 사용자 쿼리 $x$와 검색된 문서 $z_i$의 조합이며, 출력 시퀀스는 최종 예측 $y$입니다.
- 쿼리에 대한 Top-K 문서의 검색: 사용자 쿼리 $x$에 대해, 최대 내적 검색(Maximum Inner Product Search, MIPS) 알고리즘을 사용하여 가장 관련성이 높은 상위 K개의 문서 $z_i$를 찾습니다. MIPS는 쿼리 벡터와 문서 벡터 간의 내적을 최대화하는 문서를 빠르게 찾는 알고리즘입니다.

RAG 모델은 검색된 문서를 잠재 변수로 취급하고, 이를 통해 생성기와 검색기를 end-to-end 학습할 수 있습니다. 이 모델은 잠재 문서를 다르게 marginalize하여 생성된 텍스트에 대한 분포를 생성하는 두 가지 접근 방식을 제안합니다:

- RAG-Sequence
  - 이 방식은 모든 타겟 토큰을 예측할 때 같은 문서를 사용
  - $P_{RAG-Sequence}(y|x)≈\sum_{z∈top-k(p(·|x))}p_η(z∣x)\prod_{i=1}^{N}p_θ(y_i∣x,z,y_{1:i−1})$
- RAG-Token
  - 이 방식은 각 타겟 토큰을 예측할 때 다른 문서를 사용
  - $P_{RAG-Sequence}(y|x)≈\prod_{i=1}^{N}\sum_{z∈top-k(p(·|x))}p_η(z∣x)p_θ(y_i∣x,z,y_{1:i−1})$

<div id="Retriever: DPR"></div>

## Retriever: DPR

Bi-encoder 구조를 사용합니다. 

<br>

Bi-encoder는 DPR은 문서($d(z)$)와 쿼리($q(x)$)를 각각 인코딩하는 두 개의 BERT 모델을 사용합니다. 이 두 인코더는 각각 문서 인코더($BERT_{d}$)와 쿼리 인코더($BERT_{q}$)로, 문서와 쿼리를 밀집 벡터로 변환합니다.

<br>

$p_η(z∣x) ∝ exp(d(z)^Tq(x))$

$d(z)=BERT_{d}(z), q(x)=BERT_{q}(x)$

<br>

RAG 모델의 검색기 DPR는 사전 훈련된 bi-encoder를 사용하여 초기화됩니다. 이 검색기는 TriviaQA와 Natural Questions 데이터셋에 대한 문서를 검색하도록 훈련되었습니다. 이러한 방식으로, RAG 모델은 높은 품질의 문서 검색 능력을 바로 사용할 수 있게 됩니다.

<div id="Generator: BAR"></div>

## Generator: BAR

본 논문에서는 생성기($p_θ(y_i∣x,z,y_{1:i−1})$)는 어떤 인코더-디코더 모델을 사용하여도 모델링될 수 있지만, 여기서는 400M 파라미터를 가진 사전 훈련된 seq2seq 변환기인 BART-large를 사용합니다.

<br>

BART 생성기의 파라미터 $θ$는 "parametric memory"로 언급됩니다. 이는 모델의 학습 가능한 파라미터를 의미하며, 모델이 어떻게 입력을 처리하고 적절한 출력을 생성할지 결정하는 데 중요한 역할을 합니다.

<div id="Training"></div>

## Training

이 부분은 RAG 모델의 검색기(retriever)와 생성기(generator) 컴포넌트를 함께 학습시키는 과정에 대해 설명하고 있습니다. 

<br>

이 과정은 어떤 문서가 검색되어야 하는지에 대한 직접적인 지도가 없이 진행됩니다. 학습의 목표는 입력/출력 쌍 ($x_j, y_j$)의 훈련 데이터셋을 사용하여, 각 타겟에 대한 negative marginal log-likelihood를 최소화하는 것입니다.

<br>

이는 모델이 주어진 입력 $x_j$에 대해 정확한 출력 $y_j$를 생성할 확률을 최대화하는 것을 목표로 합니다.

<br>

문서 인코더를 학습하는 것은 계산 비용이 많이 듭니다. 왜냐하면 문서 인덱스를 주기적으로 업데이트해야 하기 때문입니다.

<br>

그러나 이 연구에서는 문서 인코더와 인덱스를 고정시키고, 강력한 성능을 유지하면서 쿼리 인코더 $BERT_q$와 BART generator만 fine-tuning하는 것으로 충분하다고 발견했습니다.

<div id="Decoding"></div>

## Decoding

RAG-Sequence와 RAG-Token 모델은 각 모델은 주어진 입력 x에 대해 최적의 출력 y를 추정하는 과정에서 서로 다른 접근 방식을 사용합니다.

- RAG-Token
  - standard beam decoder이 사용됩니다.
    - 빔 크기(beam size): 이는 알고리즘이 각 단계에서 고려할 최상위 후보 시퀀스의 수를 결정합니다. 빔 크기가 1이면, 빔 검색은 그리디 검색(greedy search)과 동일해집니다. 반면, 빔 크기가 크면 클수록 더 많은 후보를 탐색하지만, 계산 비용도 증가합니다.
    - 단계별 탐색: 디코딩 과정에서, 알고리즘은 각 시간 단계에서 가능한 모든 다음 토큰을 고려하고, 이전 단계의 각 후보 시퀀스에 대해 가능한 모든 확장을 생성합니다. 그런 다음 이 확장된 시퀀스들의 점수를 평가하여 최상위 빔 크기만큼의 후보를 선택합니다.
    - 최종 선택: 시퀀스의 끝을 나타내는 특별 토큰(예: <eos> 또는 </s>)이 생성되거나, 특정 길이에 도달하면 시퀀스 생성을 멈춥니다. 이후, 최종 후보들 중에서 가장 높은 총 점수를 가진 시퀀스를 최적의 출력으로 선택합니다.
  - standard beam decoder이 사용 가능한 이유
    - RAG-Token 모델은 각 타겟 토큰 $y_i$을 생성할 때마다 다른 문서 $z_i$를 참조할 수 있습니다. 
    - 이는 문서 선택 과정이 동적으로 이루어지며, 각 토큰이 생성될 때 최적의 문서가 선택되어 그 문서의 정보를 기반으로 토큰의 확률을 계산합니다. 
    - 이러한 방식은 각 단계에서 생성될 가능성이 있는 다음 토큰들에 대해 표준 빔 디코더를 사용하여 효율적으로 최적의 시퀀스를 찾을 수 있게 합니다.
    
- RAG-Sequence
  - RAG-Sequence 모델에서는 전체 출력 시퀀스 $y$에 대해 동일한 문서 집합 $z$를 사용합니다. 이는 각 타겟 토큰이 생성될 때 같은 문서 집합에 기반한다는 것을 의미하며, 전체 시퀀스에 대한 확률을 개별 토큰별 확률로 단순 분해하기 어렵게 만듭니다. 
  - 각 토큰의 생성이 동일한 문서 집합의 정보에 의존하므로, RAG-Sequence는 standard beam decoder를 직접적으로 사용하기 어려운 구조를 가집니다.
  
    <br>
  
  - Thorough Decoding
    - 입력 $x$에 대해 관련된 각 문서 $z$마다 빔 검색을 실행합니다. 이것은 각 문서를 기반으로 가능한 가설(시퀀스)들의 집합 $Y$을 생성합니다.
    - 생성된 각 가설 $y$에 대해, 모든 관련 문서에서의 확률을 계산합니다. 특정 가설이 어떤 문서의 빔에 나타나지 않는 경우, 그 문서에 대해 추가적인 순방향 패스를 실행하여 가설의 확률을 계산합니다.
    - 각 문서에 대해 계산된 가설 $y$의 확률을 $p_θ(y∣x,z)$와 문서의 우선순위 확률 $p_η(z∣x)$에 기반하여 가중치를 부여하고, 이 확률들을 모든 문서에 대해 합산합니다. 이를 통해 최종 가설들의 확률을 결정합니다.
    
    <br>
  
  - Fast Decoding
    - 긴 출력 시퀀스에 대해, 가설 집합 $Y$가 커지고 많은 순방향 패스가 필요할 수 있습니다. 이를 효율적으로 만들기 위해, 가설 $y$가 특정 문서 $z_i$에서 생성되지 않았다면 그 확률을 대략 0으로 가정하는 근사치를 사용합니다.
    - 이 근사를 통해, 가설이 처음부터 생성되지 않은 문서에 대한 추가적인 순방향 패스를 실행할 필요가 없어집니다. 이는 계산 비용을 크게 줄이면서도 효과적인 디코딩을 가능하게 합니다.

<div id="Experiments"></div>

# Experiments

본 논문에서는 Wikipedia 데이터 덤프를 non-parametric 지식 소스로 사용하며, 구체적으로 2018년 12월의 덤프를 사용합니다.

<br>

문서 인코더를 사용하여 각 문서에 대한 임베딩을 계산하기 위해 FAISS(Facebook AI Similarity Search) 라이브러리를 사용하여 단일 MIPS(Maximum Inner Product Search) 인덱스를 구축합니다. 여기에는 빠른 검색을 위한 Hierarchical Navigable Small World(HNSW) 근사 방식이 사용됩니다.

<br>

훈련 동안에는 각 쿼리에 대해 상위 k개 문서를 검색합니다. 여기서 k는 {5, 10} 중 하나로 설정됩니다.

### Open-domain Question Answering

- 실세계 응용과 지식 집약적 태스크에서 중요한 역할을 합니다.
- 질문과 답변을 입력-출력 텍스트 쌍 $(x,y)$으로 처리하며, RAG 모델은 답변의 negative log-likelihood를 최소화하여 훈련됩니다.
- RAG 모델과 다른 접근 방식 비교
  - extractive QA paradigm: 답변이 검색된 문서에서 추출된 텍스트 범위로 제한되는 방식
  - Closed-Book QA: RAG와 마찬가지로 답변을 생성하지만, 검색을 활용하지 않고 오로지 parametric knowledge에만 의존하는 접근 방식
- 사용된 데이터셋
  - Natural Questions (NQ), TriviaQA (TQA), WebQuestions (WQ), CuratedTrec (CT)
  - CT와 WQ는 상대적으로 작은 데이터셋이므로, NQ에서 훈련된 RAG 모델로 초기화하여 사용합니다.
  - 연구에서는 이전 작업에서 사용된 동일한 훈련/개발/테스트 분할을 따르며, 정확도 평가는 Exact Match (EM) 점수를 통해 이루어집니다. <br>
    이전 연구: https://aclanthology.org/P19-1612/, https://arxiv.org/abs/2004.04906
  - T5 모델과의 비교를 위해, TriviaQA의 Wiki 테스트 세트에서도 성능을 평가합니다.

### Abstractive Question Answering

RAG 모델이 단순히 텍스트에서 답변을 추출하는 것을 넘어, 자유 형식의 추상적 텍스트 생성을 통해 질문에 답변할 수 있습니다.

<br>

RAG 모델의 자연어 생성 능력을 지식 집약적 환경에서 평가하기 위해 MSMARCO NLG 태스크 v2.1을 사용합니다.

- MSMARCO NLG 태스크
  - 각 질문에 대해 검색 엔진에서 검색된 10개의 gold passages(질문과 관련이 높은 문서나 텍스트 조각), 그리고 검색된 passages들로부터 주석이 달린 전체 문장 답변이 포함됩니다.
    
    <br>
    
  - 제공된 passages를 사용하지 않고, 질문들과 답변들만을 사용하여 MSMARCO를  open-domain abstractive QA task로 취급합니다.

    <br>

  - 한계
    - 일부 MSMARCO 질문들은 gold passages에 접근하지 않고는 참조 답변과 일치하는 방식으로 답변할 수 없는 경우가 있습니다.
    - 일부 MSMARCO 질문들은 Wikipedia만을 사용해서는 답변할 수 없습니다.

###  Jeopardy Question Generation

Jeopardy는 한 사실에 대해 해당 정답를 추측하는 독특한 형식을 가집니다.

<br>

예를 들어, "The World Cup"은 "In 1986 Mexico scored as the first country to host this international sports competition twice."라는 질문에 대한 답변이 됩니다.

<br>

비교를 위한 BART 모델을 100K 훈련, 14K 개발, 27K 테스트 예제를 포함하는 SearchQA 데이터셋 분할를 활용하여 학습합니다.

<br>

- 평가 방법
  - SQuAD에 튜닝된 Q-BLEU-1 메트릭을 사용하여 평가합니다. Q-BLEU는 정답 일치에 더 높은 가중치를 두는 BLEU의 변형이며, 질문 생성에 대한 표준 메트릭보다 인간 판단과 더 높은 상관관계를 가집니다.
  - 인간 평가
    - 생성된 질문의 사실성과 특정성을 평가하기 위해 두 가지 인간 평가를 수행합니다. 사실성은 선언이 신뢰할 수 있는 외부 소스로 확인될 수 있는지 여부를 의미하며, 특정성은 입력과 출력 사이의 높은 상호 의존성을 의미합니다.
    - 평가 방식: 평가자들에게 답변과 두 개의 생성된 질문(하나는 BART에서, 다른 하나는 RAG에서 생성)을 보여주고, 네 가지 옵션 중 하나를 선택하도록 합니다: 질문 A가 더 나음, 질문 B가 더 나음, 둘 다 좋음, 둘 다 나쁨.

###  Fact Verification

FEVER 태스크는 Wikipedia를 사용하여 자연어 주장이 지지(supported), 반박(refuted)되는지, 아니면 결정하기에 충분한 정보가 없는지(not enough information)를 분류하는 것을 요구합니다. 

<br>

이 작업은 주장과 관련된 증거를 Wikipedia에서 검색하고, 이 증거를 기반으로 주장이 참인지 거짓인지, 또는 Wikipedia만으로 검증할 수 없는지를 추론하는 것을 포함합니다. 따라서 FEVER는 검색 문제와 함께 어려운 추론 작업을 결합한 것입니다.

<div id="Results"></div>

# Results

### Open-domain Question Answering

RAG는 Open-domain QA 태스크에서 최신 모델들과 비교하여 새로운 최고 기록을 세웠습니다.

<br>

RAG는 'closed-book'(오직 파라메트릭 지식만을 사용하는) 접근 방식의 생성 융통성과 'open-book'(검색 기반) 접근 방식의 성능을 결합합니다.

<br>

RAG는 REALM이나 T5+SSM과 같이 "salient span masking(중요한 구문을 마스킹)"과 같은 비용이 많이 드는, 특수한 사전 훈련 없이도 강력한 결과를 보여줍니다.

<br>

RAG의 검색기는 Natural Questions과 TriviaQA에서의 검색 감독을 사용하여 초기화된 DPR의 검색기를 사용합니다.

<br>

RAG는 DPR QA 시스템의 문서 재순위 부여(re-ranking)나 추출적 독자(extractive reader) 없이 검색과 생성을 결합한 접근 방식으로 질문에 대한 답변을 생성하며 뛰어난 성능을 보입니다.

<br>

또한 이는 실제 답변이 문서에 그대로 존재하지 않더라도 답변에 대한 단서가 포함된 문서를 사용하여 정확한 답변을 생성할 수 있습니다. 이는 표준 추출적 접근 방식에서는 불가능한 일입니다.

<br>

<img style="width: 50%; margin-right: 0px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="rag/openbook.PNG">

### Abstractive Question Answering

Open MS-MARCO NLG 태스크에서 RAG-Sequence 모델이 BART 모델을 초과하는 성능을 보였습니다. 

<br>

구체적으로, RAG-Sequence는 BART에 비해 2.6 BLEU 점수와 2.6 Rouge-L 점수로 우수한 성능을 보였습니다. 이는 상당히 인상적인 결과입니다.

<br>

비교되는 상태 최고의 모델들은 참조 답변을 생성하는 데 필요한 특정 정보를 포함하는 gold passages에 접근할 수 있지만, RAG-Sequence는 그러한 접근 없이도 높은 성능을 달성했습니다.

<br>

모든 질문이 Wikipedia만으로 답변할 수 있는 것은 아니지만, RAG-Sequence는 이러한 한계를 극복하고 인상적인 결과를 보였습니다.

<br>

생성된 답변의 질에 대한 평가에서 RAG 모델은 BART에 비해 덜 허구를 만들어내고(fewer hallucinations), 더 자주 사실적으로 정확한 텍스트를 생성하는 것으로 나타났습니다. 

<br>

이는 RAG 모델이 단순히 정보를 재생산하는 것을 넘어서, 실제적이고 정확한 정보를 바탕으로 답변을 생성할 수 있음을 시사합니다.

<br>

나중에 RAG 생성물이 BART 생성물보다 더 다양하다는 것을 보여주는 결과도 제시됩니다. 

<br>

이는 RAG 모델이 생성 과정에서 더 넓은 범위의 정보를 활용하고, 더 창의적이고 다양한 답변을 만들어낼 수 있음을 의미합니다.

### Jeopardy Question Generation

Jeopardy 질문 생성 태스크에서 RAG-Token과 RAG-Sequence 모델, 그리고 BART 모델의 성능을 비교하는 내용을 다룹니다. 

<br>

RAG-Token 모델이 이 태스크에서 RAG-Sequence 모델보다 더 우수한 성능을 보였으며, 두 RAG 모델 모두 BART 모델을 Q-BLEU-1 메트릭에서 능가했다고 합니다. 

<br>

인간 평가 결과에서는 RAG 모델이 사실성(factuality)과 구체성(specificity) 측면에서 더 뛰어난 것으로 나타났습니다.

<br>

평가자들은 RAG가 BART보다 더 사실적인 답변을 생성한다고 평가했습니다. BART가 더 사실적인 경우는 7.1%에 불과했고, RAG가 더 사실적인 경우는 42.7%였으며, 두 모델 모두 사실적인 경우는 17%였습니다.

<br>

Jeopardy 질문은 종종 두 개의 별개 정보를 포함합니다. RAG-Token은 여러 문서의 내용을 결합하여 응답을 생성할 수 있는 능력 때문에, 이 태스크에 더 적합할 수 있습니다.

<br>

예시를 통해, RAG가 "The Sun Also Rises"와 "A Farewell to Arms" 같은 제목을 생성할 때 특정 문서에 의존하지 않고도 제목을 완성할 수 있음을 보여줍니다. 이는 모델의 파라메트릭 지식이 충분하다는 것을 시사합니다.

<br>

BART 모델만 사용해도 "The Sun Also Rises"와 "A Farewell to Arms" 같은 제목을 완성할 수 있었습니다. 이는 파라메트릭 및 비파라메트릭 메모리가 함께 작동하여, 비파라메트릭 구성요소가 생성을 안내하고 파라메트릭 메모리에 저장된 구체적 지식을 끌어내는 방식을 보여줍니다.

### Fact Verification

- RAG 모델의 FEVER 태스크 성능
  - 3방향 분류(지지, 반박, 판단 불가): RAG 모델의 성능은 복잡한 파이프라인 시스템과 도메인 특화 아키텍처를 가진 최신 모델들과 4.3% 이내의 차이로 비슷합니다. 이 최신 모델들은 intermediate retrieval supervision을 사용하여 훈련되지만, RAG는 그러한 감독 없이도 비슷한 성능을 달성합니다.
  - 2방향 분류(지지 또는 반박): RAG는 gold passages가 주어진 상태에서 주장을 참 또는 거짓으로 분류하는 RoBERTa 모델과 비교됩니다. RAG는 주장만이 제공되고 스스로 증거를 검색함에도 불구하고, 이 모델과 2.7% 이내의 정확도 차이를 보입니다.

- RAG에 의한 문서 검색과 gold passages
  - RAG가 검색한 문서와 FEVER에서 gold passages로 주석이 달린 문서 사이의 일치도를 분석했습니다. 이는 RAG가 증거를 적절히 검색할 수 있는지 평가하기 위함입니다.
  - 결과적으로, 상위 검색된 문서 중 71%가 gold passages였으며, 상위 10개 검색된 문서 중 90%에 gold passages가 포함되었습니다.

<img style="width: 50%; margin-right: 0px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="rag/Table2.PNG">

### Fact Verification

- Generation Diversity
  - 생성된 내용의 다양성을 평가하기 위해, 다른 모델들이 생성한 n-gram의 비율(유니크한 n-gram 수를 전체 n-gram 수로 나눈 값)을 계산합니다.
  - RAG-Sequence 모델이 생성한 답변이 RAG-Token 모델보다 더 다양하며, 두 모델 모두 BART 모델보다 훨씬 더 다양한 답변을 생성합니다. 이는 어떠한 다양성 촉진 디코딩 기법을 사용하지 않고도 달성된 결과입니다.

- Retrieval Ablations
  - RAG의 핵심 기능 중 하나는 태스크에 관련된 정보를 검색하는 것을 학습하는 것입니다.
  - 검색기를 학습 과정 동안 고정시키는(ablation) 실험을 통해 검색 메커니즘의 효과를 평가합니다. 학습된 검색이 모든 태스크에 대한 결과를 개선합니다.
  - RAG의 Dense Retriever와 기존의 단어 중첩 기반 검색기인 BM25를 비교합니다. 특정 태스크에서는 BM25가 더 나은 성능을 보일 수 있으며(예: FEVER), 이는 FEVER 주장이 엔티티 중심적이고 단어 중첩 기반 검색에 적합하기 때문입니다.

<img style="width: 100%; margin-right: 0px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="rag/Table6.PNG">

### Index hot-swapping

Parametric-only models인 T5나 BART와 같은 모델들은 세상의 변화에 따라 모델의 행동을 업데이트하기 위해 추가 훈련이 필요합니다.

<br>

하지만, RAG는 지식을 쉽게 업데이트할 수 있습니다.

- 실험 설정: 2016년 12월의 DrQA Wikipedia 덤프를 사용하여 인덱스를 구축하고, 이 인덱스를 사용하는 RAG의 출력을 메인 결과(2018년 12월의 인덱스 사용)와 비교합니다.
- 실험 방법: 2016년부터 2018년 사이에 변경된 82명의 세계 지도자 목록을 준비하고, "Who is {position}?" (예: "Who is the President of Peru?")와 같은 템플릿을 사용하여 각 인덱스로 NQ RAG 모델을 쿼리합니다.
- 결과
  - 2016년 인덱스를 사용할 때 RAG는 2016년의 세계 지도자에 대해 70% 정확도로 올바른 답변을 합니다.
  - 2018년 인덱스를 사용할 때는 2018년의 세계 지도자에 대해 68% 정확도로 올바른 답변을 합니다.
  - 인덱스가 맞지 않을 경우(예: 2018년 인덱스로 2016년 지도자를 물을 때, 또는 그 반대) 정확도는 현저히 낮아집니다(각각 12%와 4%).

### Effect of Retrieving more documents

이 내용은 RAG 모델이 훈련 중에 검색하는 문서의 수(5개 또는 10개 검색된 잠재 문서)에 따른 성능 변화를 탐구한 결과를 요약하고 있습니다. 

<br>

훈련 단계에서 검색된 문서의 수를 조정했을 때 성능 차이가 크게 나타나지 않았으며, 테스트 시점에서 검색하는 문서의 수를 조정함으로써 성능과 실행 시간에 영향을 줄 수 있는 유연성이 있다는 점을 보여줍니다.

<br>

테스트 시 더 많은 문서를 검색할수록 개방형 QA 태스크의 결과가 단조롭게 개선되는 것을 보여줍니다. 하지만, RAG-Token 모델의 경우에는 검색 문서 수가 10개일 때 성능이 최고점에 도달합니다.

<br>

RAG-Token에서는 더 많은 문서를 검색할 때 Rouge-L 점수는 향상되지만 Bleu-1 점수는 소폭 하락하는 경향을 보입니다. 반면, RAG-Sequence의 경우 이러한 효과가 덜 두드러집니다.

<br>

<img style="width: 100%; margin-right: 0px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="rag/Figure3.PNG">
