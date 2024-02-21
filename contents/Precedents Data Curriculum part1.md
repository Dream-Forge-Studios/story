---
date: '2023-12-16'
title: '법률 LLM 개발에 있어서 판례 Data Curriculum - Part 1: 사건과 관련된 법률 찾기 Task'
categories: ['LLM', 'Legal']
summary: '고성능의 법률 도메인 특화 LLM을 만들기 위한 판례 Data Curriculum의 관한 연구(사건과 관련된 법률 찾기 Task)'
thumbnail: './test.png'
---

<div id="Introduction"></div>

## Introduction

본 내용은 법률 분야에 특화된 언어 모델(Legal Language Model, LLM)을 개발을 위한 Data Curriculum에 관한 연구입니다. 

<br>

Data Curriculum란,  LLM이 학습 데이터를 어떤 방식으로 결합하고, 제공되는 순서에 따라 모델의 성능이 변화할 수 있다는 개념에 기반한 연구입니다.

<br>

기존 연구에서 법률 Task와 관련한 Data Curriculum 연구가 많이 부족한 실정입니다. 이번 연구를 통해 법률 LLM 발전에 기여하고 싶습니다.

<br>

본 연구에서 판례데이터를 활용하여 연구를 진행합니다. 판례 데이터는 법률 분야에서 신뢰도가 높으며, 접근성이 우수한 자료입니다.

<br>

그러므로 법률 LLM 개발에 있어 판례 데이터의 효과적 활용이 핵심적인 요소로 작용합니다.

<br>

본 연구는 소규모 데이터 세트를 활용하여 진행됩니다. 방대한 학습 데이터를 사용할 경우 모델의 정확도는 향상될 수 있지만, 그로 인해 학습 시간이 지나치게 길어지는 문제에 직면합니다.

<br>

따라서, 본 연구에서는 DoReMi 연구를 참고하여 초기에는 소량의 데이터를 사용한 연구를 진행한 뒤, 이후에 대규모 데이터로 모델을 학습시키는 접근 방식을 채택합니다.

[관련 논문](https://arxiv.org/pdf/2305.10429.pdf)

<br>

본 연구는 판례 Data Curriculum의 첫번째 연구로써 사건과 관련된 법률 찾기 Task를 수행합니다.

<div id="Training"></div>

## Training

### environment

Mistral-7B-Instruct-v0.1을 기반으로 한국어에 특화한 Synatra-7B-v0.3-dpo를 base model로 4비트 양자화와 LORA를 활용하여 A600 GPU 4대를 활용하여 학습을 진행하였습니다.

### method

gpt의 발전과정에서 영감을 받아 Causal language modeling을 통한 사전학습 방식으로 학습을 진행합니다.

### Dataset

[허깅페이스 판례 데이터](https://huggingface.co/datasets/joonhok-exo-ai/korean_law_open_data_precedents)에서 DoReMi 연구를 참고하여, 소규모 데이터 세트를 활용하여 연구를 진행합니다. 구체적으로는 민사 판례 중 임금 사건에 관련된 데이터만을 대상으로 학습을 진행합니다.

<div id="Research"></div>

## Research

### 1. 판례 전체(법률 추출) vs 전문만

판례 데이터는 판시사항/판결요지/참조조문/참조판례/전문으로 이루어져 있습니다.

<br>

이 중 판시사항/판결요지/참조조문/참조판례는 전문의 내용을 요약하고 키워드를 추출한 부분으로 볼 수 있습니다.

<br>

따라서, 본 연구에서는 판례의 전체 구성요소와 법적 판단의 핵심을 담고 있는 전문만을 대상으로 학습을 진행함으로써, Task 데이터에 대한 학습 효과를 검증하였습니다.

### 2. 법률 빈도수의 영향

본 연구는 판례 데이터의 특징을 중심으로 진행되었습니다. 현실 세계에서는 일상적으로 자주 언급되는 법률도 있고, 그렇지 않은 법률도 있습니다.

<br>

이러한 현상은 피할 수 없는 부분입니다. 한 가지 판례에서 여러 법률이 언급될 수 있기 때문에, 데이터 내에서 법률의 언급 비율을 균등하게 맞추는 것은 실현 불가능하며, 이는 사용 가능한 데이터의 양을 감소시키는 단점으로 이어집니다.

<br>

이에 따라, 본 연구에서는 사용 가능한 데이터의 양과 법률 언급 빈도수 사이의 적절한 비율을 찾기 위한 연구를 수행하였습니다.

<div id="Evaluation"></div>

## Evaluation

- 평가 질문 생성

평가 질문 생성은 [Zhen Wan ET AL. (2023)](https://arxiv.org/pdf/2310.03328.pdf)의 연구, 즉 GPT-4가 법률 관련 정보 제공 시 정확한 판단을 내릴 수 있다는 연구를 바탕으로 GPT-4를 활용하여 진행됩니다.

<br>

구체적인 프롬프트:

```
{법률명}

{법률 설명}

법률 LLM을 테스트하기 위한 데이터를 만들고 있는데, 위의 법과 관련된 사건 3가지만 만들어줘
```

<br>

위의 프롬프트로 생성한 평가 질문을 통한 아래의 질문 형식에 각 모델이 생성한 답변의 논리성을 비교하여 평가합니다.

```
{사건}와 관련된 법은?
```

본 연구의 주목적이 사건에 맞는 법률을 찾는 Task이나, Causal language modeling을 통한 사전학습 방식을 통해서는 정확한 법률을 찾아주는 것이 불가능하여 답변의 논리성을 평가합니다.

<br>

본 연구에서의 논리성의 개념은 질문의 의도를 잘 파악하여, 질문의 유용한 답변을 하는가 입니다.

<div id="Result"></div>

## Result

**1. 판례 전체(법률 추출) vs 전문만**

- 판례 전체
  - 구제적인 법으로 답변하는 경향이 높음
  - 하지만 답변으로 제공한 법이 질문과 관계없는 경우가 많음
  - Casual Langugae Modeling의 특성과 법률명의 형식으로 인해 법률을 답변하고 그 이후로 의미없이 숫자만 들어난 법으로 답변
  ```
  정답: 근로기준법 제18조(단시간근로자의 근로조건은 그 사업장의 같은 종류의 업무에 종사하는 통상 근로자의 근로시간을 기준으로 산정한 비율에 따라 결정되어야 한다.)
  
  질문: '소프트웨어 개발자로 일하는 단시간 근로자 진수는 계약상 주 10시간 근무합니다. 진수는 연차 유급휴가를 요구했지만, 회사는 그의 근로시간이 15시간 미만이라는 이유로 거부했습니다.'와 관련된 법은?
  답변: 해당 사건은 근로기준법 제41조 제1항 제2호, 제42조 제1항, 제43조 제1항, 제44조 제1항, 제45조 제1항, 제46조 제1항, 제47조 제1항, 제48조 제1항, 제49조 제1항 ... (중략)
  
  *근로기준법 제41조:  사용자는 각 사업장별로 근로자 명부를 작성하고 근로자의 성명, 생년월일, 이력, 그 밖에 대통령령으로 정하는 사항을 적어야 한다.
  ```

- 전문만
  - 구체적인 법이 포함되어 있지 않는 경우가 많음
  - 답변으로 구체적인 법이 제공된 경우, 판례 전체로 학습할 때와 마찬가지로 관계없는 법률이 작성되어 있음
  - 논리적으로는 질문과 관계되어있는 유용한 답변 제공
  ```
  정답: 근로기준법 제18조(단시간근로자의 근로조건은 그 사업장의 같은 종류의 업무에 종사하는 통상 근로자의 근로시간을 기준으로 산정한 비율에 따라 결정되어야 한다.)

  질문: '마케팅 회사에서 일하는 단시간 근로자 리나는 주 14시간 근무합니다. 리나는 같은 부서의 통상 근로자들과 비교해 현저히 낮은 시간당 임금을 받고 있습니다.'와 관련된 법은?
  답변: 이 사건은 근로자가 연차 유급휴가를 요구하고 그 요구를 거부하는 회사의 이유에 관한 것입니다. 근로자가 연차 유급휴가를 요구하기 위해서는 근로자가 근로하는 근로자의 근로시간이 1년간 1년의 1/4 이상이어야 하고,. ... (중략)
  ```

- 결론
  - 판례 전체와 이유만 학습했을 때 모두 질문과 관련된 정확한 법률을 답변하지 못함 <br>=> 판례데이터를 활용한 사전학습만으로는 사건과 관련된 법률명 추출이 불가능함
  - 답변의 논리성을 따져보았을 때는 판례 전체를 제공하는 것보다는 이유만 학습했을 때가 좀 더 유용한 답변을 제공

<br>

**2. 법률 빈도수의 영향**

<img style="width: 100%; margin-top: 40px;" id="output" src="legalData/Figure_1.PNG">
<img style="width: 100%; margin-top: 40px;" id="output" src="legalData/Figure_2.PNG">

<br>

그림과 같은 법률 빈도수를 가진 A와 B 데이터로 학습을 진행한 후 답변을 비교해보았습니다.

- A의 빈도수가 많이 차이남에도 불가하고, 많이 언급된 법률을 주로 답변하는 경향은 없었습니다.
- A가 B에 비해 대부분 논리적인 답변을 제공하였습니다.

이러한 결과로 미루어보았을 때, 법률에 빈도수 보다는 다양한 경우의 판례 데이터를 제공하는 것이 효과적이라는 결론을 얻게 되었습니다.

<div id="Conclusion"></div>

## Conclusion

Casual Language Modeling을 통한 사전학습을 진행했을 때, 사건과 관련된 정확한 법률적 답변을 도출하는 것은 어려웠습니다. 이는 [Zhen Wan ET AL. (2023)](https://arxiv.org/pdf/2310.03328.pdf)의 연구와 일치합니다.

<br>

그러나 판례 데이터를 활용함으로써, LLM이 법적 판단을 내릴 수 있는 'emergent ability'가 발생함을 확인했습니다.

<br>

이는 판례 데이터가 LLM의 법적 판단 능력을 강화하는 데 효과적임을 나타냅니다.

<br>

사건과 관련된 정확한 법률을 식별하는 작업에서 판례 데이터를 통해 LLM의 법적 판단 능력이 개선되면, LLM의 답변을  [Zhen Wan ET AL. (2023)](https://arxiv.org/pdf/2310.03328.pdf)의 연구와 같이 RAG를 통해 해결할 수 있습니다.

<br>

이러한 연구는 법률 도메인에서 LLM에서 RAG 연구의 중요성을 시사합니다.

### 추가적인 연구

#### 평가 답변 생성 prompt 변경 

<br>

연구를 시작할 때 사건과 관련된 법률을 찾는 Task를 목표로 하였지만, Casual Language Modeling을 통한 사전학습에서는 불가능하며, 판례데이터는 법적 판단 능력을 강화하는데 효과적임을 확인하였습니다.

<br>

그래서 판례데이터의 효과에 맞게 아래와 같이 질문을 변경하여 답변을 비교하였습니다.

```
{사건}에 대한 법적 판단은?
```

그 결과 target Task가 담겨있는 학습데이터로 Casual Language Modeling을 통한 사전학습한 LLM에게 target Task로 질문을 하면 일반화 성능이 떨어졌습니다.

<br>

이전 연구에서 법률 추출 Task 데이터(판례 전체)로 Casual Language Modeling을 통한 사전학습한 LLM에게 법률 추출 질문을 할 때 성능이 떨어졌던 것과 같이

<br>

법적 판단 Task 데이터(판례의 전문만)로 Casual Language Modeling을 통한 사전학습한 LLM에게 법적 판단 질문을 하면 좋지 않은 성능을 보였습니다.

(질문과 상관없이 학습데이터에서 본 특수한 경우에 관한 답변)

```
질문: '마케팅 회사에서 일하는 단시간 근로자 리나는 주 14시간 근무합니다. 리나는 같은 부서의 통상 근로자들과 비교해 현저히 낮은 시간당 임금을 받고 있습니다.'이에 대한 법적 판단은?

전문만 학습한 모델의 답변: 

이 사건에서 주장하는 사실은 다음과 같습니다.
1. 리나는 2008. 1. 1.부터 2010. 12. 31.까지 마케팅 회사에서 일하는 단시간 근로자로 근무하였고, 그 기간 동안 주 14시간 근무하였다. ... (중략)

판례 전체를 학습한 모델의 답변: 

리나가 단시간 근로자로서 주 14시간 근무하고 있다면, 단시간 근로자에 대한 시간외 근로에 관한 법률에 따라 시간외 근로에 대한 가산 임금을 지급받을 수 있습니다. ... (중략)
```

이러한 결과를 통해 이전 연구에서 판례 전체를 제공하는 것을 문제로 보았는데, 

<br>

그것이 아니라 target Task가 담겨있는 학습데이터로 Casual Language Modeling을 통한 사전학습을 하면 일반화 성능이 떨어지는 문제로 기인한 것입니다.

### 판례 데이터 활용 방법

1. 판례 데이터는 법적 판단 능력을 향상시켜주는 데이터이다.
2. 법률 빈도수의 차이는 큰 영향이 없으며, 다양한 상황의 데이터를 제공하는 것이 중요하다.
3. 하지만 법적 판단과 관련된 내용인 전문만으로 학습시키면 LLM의 일반화 능력이 떨어진다. 

<div id="추가로 연구되야할 사항"></div>

## 추가로 연구되야할 사항

1. 같은 문장을 반복해서 생성하는 현상

```
1. 제46조에 관한 법은 공기업 경영규정 제31조 제3항 및 제4항, 제46조에 관한 법은 공기업 경영규정 제31조 제3항 및 제4항, 제46조에 관한 법은 공기업 경영규정 제31조 제3항 및 제4항,
2. 개인 E가 친구 F의 차량을 사용하여 택시 서비스를 제공하는 것이 아니라 개인 E가 친구 F의 차량을 사용하여 택시 서비스를 제공하는 것이 아니라 개인 E가 친구 F의 차량을 사용하여 택시 서비스를 제공하는 것이 아니라
```

<br>

본 연구는 오로지 판례 데이터를 활용하여 학습을 수행했습니다. 이 과정에서, 동일한 문장이 반복적으로 생성되는 문제가 나타났습니다.

<br>

이 문제의 발생 원인으로는 데이터의 다양성 부족(판례 데이터만 활용)과 파라미터 설정 문제 등이 있거나, 인간의 의도를 모델이 학습하게 하는 Model Alignment 기법을 적용한 추가적인 연구가 필요함을 확인하였습니다.

2. 파라미터의 영향

```
training_arguments_f = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 4,
    optim = "adamw_torch",
    save_steps= 30,
    logging_steps= 30,
    learning_rate= 4e-4,
    weight_decay= 0.01,
    fp16= False,
    bf16= False,
    max_grad_norm= 1.0,
    max_steps= -1,
    warmup_ratio= 0.06,
    group_by_length= False,
    lr_scheduler_type= "cosine",
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False,
    save_total_limit=5,
    report_to="wandb"
)
```

처음 학습을 진행하 였을 때 위와 같은 파라미터로 진행하였는데,

```
"질문": '외국인 근로자 E는 그의 출신 국가 때문에 회사 F의 사회적 활동에서 배제되고 있습니다. E는 이를 차별적인 처우로 여기고 이의를 제기합니다.'와 관련된 법은?
"답변": 
이 사건 법원은 다음과 같은 이유로 원고의 상고를 기각하고 있습니다.
1. 원고의 상고이유에 대하여
가. 원고는 이 사건 법원에서 이 사건 법원의 판결에 불복하여 상고하였으나, 상고이유에 대하여 이 사건 법원의 판단은 정당하고 수긍할 수 있으므로 상고를 기각하기로 하였습니다. ... (중략)
```

<br>

위와 같이 질문에 맞지 않고, 단순히 학습데이터에서 본 말을 나열하고 있습니다.

<br>

```
training_arguments_a = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 1,
    gradient_accumulation_steps= 4,
    optim = "paged_adamw_8bit",
    save_steps= 30,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant",
    report_to="wandb"
)
```

위의 파라미터로 학습을 진행했을 때는

```
"질문": '외국인 근로자 E는 그의 출신 국가 때문에 회사 F의 사회적 활동에서 배제되고 있습니다. E는 이를 차별적인 처우로 여기고 이의를 제기합니다.'와 관련된 법은?
"답변": 이 문제는 노동조합 및 노동관계조정법 제30조에 의하여 규정된 임금 차등금지 원칙에 관한 것입니다. 이 원칙은 같은 업무를 수행하는 근로자에 대하여 동일한 임금을 지급하여야 한다는 것을 규정하고 있습니다. ... (중략)
```

<br>

상당히 논리적인 답변을 하고 있습니다.

<br>

학습에 있어서 파라미터 설정은 굉장히 중요한 작업이므로, 파라미터에 대한 연구가 중요합니다.

<br>

하지만  이 연구의 궁극적인 목표는 Casual Langugae Modeling으로 사전학습 후 Model Alignment를 했을 때의 성능을 향상시키는 것에 있습니다.

<br>

현재는 Casual Language Modeling을 이용한 사전학습에 초점을 맞춰 연구가 진행 중입니다. 이 과정에서 선택된 두 번째 파라미터는 사전학습 단계에는 적합할 수 있으나, Model Alignment 단계에서는 반드시 최적이라고 할 수 없습니다.

<br>

따라서 파라미터에 대한 추가적인 연구는 사전학습 후 Model Alignment 단계의 성능 향상을 목표로 하는 본 연구의 다음 단계에서 진행될 예정입니다.

3. 다른 사건의 데이터

본 연구에서 법률 빈도수의 차이는 큰 영향이 없으며, 다양한 상황의 데이터를 제공하는 것이 중요하다는 결과를 확인하였습니다.

<br>

이에 따라, 임금과 관련된 판례 데이터만을 학습하는 것보다 다양한 사건의 판례 데이터를 활용하는 것이 임금 관련 법적 판단을 할 때에도 효과적일 것이라는 가능성이 보입니다.

<br>

그러므로 임금이 아닌 다른 사건들의 판례와 함께 학습하였을 때의 연구를 진행할 예정입니다.