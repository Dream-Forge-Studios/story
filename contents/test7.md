---
date: '2023-02-14'
title: '인공지능 필수 기초 1탄'
categories: ['AI']
summary: '인공지능에 대한 필수 기초에 대해 총정리 하였습니다.'
thumbnail: './test.png'
---

<div id="1. weight와 bias"></div>

# 1. weight와 bias

<div style="display: flex; margin-top: 0px">
<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*upfpVueoUuKPkyX3PR3KBg.png">
</div>

> <u>weight</u>는 input의 정보를 얼마나 보낼건지(중요도), <u>bias</u>는 각 노드의 민감도를 정한다.

---

<div id="2. 선형회귀"></div>

# 2. 선형회귀

<div style="display: flex; margin-top: 0px">
<div style="width: 50%;"></div>
<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://upload.wikimedia.org/wikipedia/commons/b/be/Normdist_regression.png">
<div style="width: 50%;"></div>
</div>

> 입력과 출력의 관계(함수)를 선형으로 추정하는 것

즉, ax+b 함수에서 입력과 출력의 관계를 잘 설명하는 a,b를 구하는 것

<br>
그렇다면 어떻게 a,b를 찾아내는 것인가?

<div id="loss(cost) funtion"></div>

## loss(cost) funtion

> 예측값과 실제값의 차이

loss를 최소화하는 a,b가 바로 최적에 a,b이다.

선형회귀에서는 mse(mean squared error - 실제값과 예측값 차이의 제곱의 평균)를 사용하여 loss를 구하는 것이 일반적이다.

<div style="display: flex; margin-top: 0px">
<div style="margin-top: 10px">(</div>
<img style="width: 300px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?L = e_1^2 + e_2^2 + e_3^2 + e_4^2 + \cdots + e_n^2 ">
<div style="margin-top: 10px">)</div>
</div>

<br>
<br>
그렇다면 loss를 최소화하는 a,b를 어떻게 찾아가야할까?

a,b를 일일이 바꿔가면서 하는 것일까? 이는 너무 비효율적이다.

이제 loss를 최소화하는 다양한 방법들을 알아보자.

---

<div id="3. Gradient descent(경사하강법)"></div>

# 3. Gradient descent(경사하강법)

> 처음 랜덤으로 a,b를 정한 다음 모든 값에 대한 loss를 구한 후 <u>gradient(기울기)의 반대 방향</u>으로 이동

<div id="기울기의 반대 방향을 향해 이동하는 이유는?"></div>

## 기울기의 반대 방향을 향해 이동하는 이유는?

> 기울기가 0에 가까운 지점이 최적에 값이며 기울기는 가장 가파른 방향을 향하기 때문이다.

하지만 계속 가파른 방향의 반대방향으로 향하다 보면 최적의 값을 지나버릴 수 있다.

그래서 Learning rate가 필요하다.

<div id="Learning rate"></div>

## Learning rate

> 얼마나 이동할 것인지를 조절하는 것, 기호로는 <img style="width: 16px; margin-right: 8px; margin-left: 4px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\alpha">

<div style="display: flex; margin-top: 0px">
<img style="width: 300px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}
a_{k+1} \\ b_{k+1}
\end{bmatrix} = \begin{bmatrix}
a_{k} \\ b_{k}
\end{bmatrix} - \alpha \begin{bmatrix}
\frac{\partial L}{\partial a} \\ \frac{\partial L}{\partial b}
\end{bmatrix}">
</div>

<div id="Gradient descent의 단점"></div>

## Gradient descent의 단점

1. 모든 값에 대한 loss를 고려해서 너무 오래 걸린다.

2. local minimum
<div style="display: flex; margin-top: -40px">
<div style="width: 50%;"></div>
<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="   https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Extrema_example_original.svg/440px-Extrema_example_original.svg.png">
<div style="width: 50%;"></div>
</div>
<br>

이러한 단점을 해결하기 위한 방법은 어떤 것이 있을까?

---

<div id="4. SGD"></div>

# 4. SGD

> 랜덤하게 비복원추출로 데이터 하나씩 뽑아서 loss를 만듬

<div style="display: flex; margin-top: 0px">
<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99848D4C5B613C0C15">
</div>

<div id="왜 gradient가 +를 향하지 않는가?"></div>

## 왜 gradient가 +를 향하지 않는가?

하나만 보고 방향을 결정하기 때문이다.

하나만 보기 때문에 빠르다는 장점이 있다. 

또한 local minimum 으로부터 탈출의 기회가 되기도 한다.

---

<div id="5. mini-batch SGD"></div>

# 5. mini-batch SGD

> 데이터를 하나씩 뽑는 것이 아니라 batch size만큼 뽑아 비복원추출하여 loss를 만듬

<div style="display: flex; margin-top: 0px">
<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F9987E53A5D0E1B7C2E">
</div>

<br>

현재는 gpu가 병렬 연산을 가능하게 하므로 빠르게 계산할 수 있다.

<div id="hyperparameter"></div>

## hyperparameter
> 정해줘야 하는 값

1. Epoch : 전체 데이터를 얼마나 반복할 것 인가

2. Batch size : 몇 개씩 볼 것 인가

3. Learning rate : 얼마큼 갈 것인가

4. Initial weight

5. model architecture (layer 수, node 수, activation 함수 등)

6. loss 함수

* 적절한 값

1. batch size 8k 까지만

<div style="display: flex; margin-top: 0px">
<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://mblogthumb-phinf.pstatic.net/MjAxODEyMjRfMjMg/MDAxNTQ1NjE5MzY1NDMx.N6wnv9T-HolFs8i310qG3mXeR0SKaAWoLEjWOokVAXog.S5lQjzsk5VUhlUAfvlMl_hsZkWeYaIzKSwYEdlW1fSEg.PNG.wideeyed/13.png?type=w800">
</div>

2. batch size 두배하면 learning rate도 두배 (Linear Scaling Rule)

3. warmup

<div style="display: flex; margin-top: 0px">
<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbTWPA7%2Fbtq1Ntn3oSX%2FWj1DImFnZHkLznhTeDPGu0%2Fimg.png">
</div>

4. Weight initialization

   <br>
   U() 균일 분포, N() 정규 분포, 
    
   N in 이전 레이어의 노드 수, N out 다음 레이어의 노드 수
   
- LeCun

<div style="display: flex; margin-top: 6px; margin-left: 30px">
<img style="width: 200px; margin-right: 20px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?w\sim U(-\sqrt{\frac{3}{N_{in}}}, \sqrt{\frac{3}{N_{in}}})">
<div style="margin-top: 18px">or</div>
<img style="width: 140px; margin-right: 8px; margin-left: 18px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?w\sim N(0, \sqrt{\frac{1}{N_{in}}})">
</div>
<br>

   - Xavier (sigmoid/tanh 사용하는 신경망)
   
   <div style="display: flex; margin-top: 10px; margin-left: 30px">
   <img style="width: 300px; margin-right: 20px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?w\sim U(-\sqrt{\frac{6}{N_{in}+N_{out}}}, \sqrt{\frac{6}{N_{in}+N_{out}}})">
   <div style="margin-top: 16px">or</div>
   <img style="width: 200px; margin-right: 8px; margin-left: 18px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?w\sim N(0, \sqrt{\frac{2}{N_{in}+N_{out}}})">
   </div>
   <br>

- He (ReLU 사용하는 신경망)

   <div style="display: flex; margin-top: 30px; margin-left: 18px">
   <img style="width: 200px; margin-right: 20px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?w\sim U(-\sqrt{\frac{6}{N_{in}}}, \sqrt{\frac{6}{N_{in}}})">
   <div style="margin-top: 18px">or</div>
   <img style="width: 140px; margin-right: 8px; margin-left: 18px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?w\sim N(0, \sqrt{\frac{2}{N_{in}}})">
   </div>

---

<div id="6. Adam"></div>

# 6. Adam
> momentum과 RMSprop을 합쳐놓은 최적화 알고리즘

<div id="momentum과 RMSprop"></div>

## momentum과 RMSprop

<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 50px; margin-bottom: 0px;" id="output" src="test7Img/mom.PNG">
<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 80px; margin-bottom: 0px;" id="output" src="test7Img/rms.PNG">
<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 80px; margin-bottom: 0px;" id="output" src="test7Img/adam.PNG">

---

<div id="7. K-Fold Cross Validation"></div>

# 7. K-Fold Cross Validation

<div id="Validation"></div>

## Validation

1. training 데이터 중 학습 도중 test할 때 쓰는 데이터

2. 하이퍼파라미터 선택을 위한 데이터

<img style="width: 100%; margin-right: 8px; margin-left: 0px; margin-top: 50px; margin-bottom: 0px;" id="output" src="test7Img/kf.PNG">
