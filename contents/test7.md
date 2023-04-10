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

<div id="적절한 batch size를 정하는 방법"></div>

## 적절한 batch size를 정하는 방법

딥러닝에서는 시행을 무한번 반복하고 산술 평균을 구하는 **기댓값(Expectation)**
을 의미

- 질량 함수의 기대값 정의

- 밀도(연속) 함수의 기대값 정의

<div style="display: flex; margin-top: 0px">
<img style="width: 250px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?E[X]=\int_{-\infty }^{\infty }xp(x)dx ">
</div>

<div style="display: flex; margin-top: 36px">
<div style="margin-top: 10px">보통</div>
<img style="width: 120px; margin-right: 8px; margin-left: 10px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?E[X]=\mu ">
<div style="margin-top: 10px">로 표기</div>
</div>

<div id="분산"></div>

## 분산

- 퍼진 정도
<div style="display: flex; margin-top: -30px"></div>

- 편차(평균과의 차이)를 제곱

  *절댓값을 안 쓰는 이유는 (-7, -1, 1, 7)과 (-4,-4, 4, 4)을 절댓값으로 분산을 구하면 같게 나온다.
<div style="display: flex; margin-top: -30px"></div>

- 질량 함수의 분산 정의

<div style="display: flex; margin-top: 0px">
<img style="width: 280px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?V[X]=\sum_i(x_i-\mu)^2p_i ">
</div>

- 밀도(연속) 함수의 분산 정의

<div style="display: flex; margin-top: 0px">
<img style="width: 340px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?V[X]=\int_{-\infty }^{\infty }(x_i-\mu)^2p(x)dx ">
</div>

<div style="display: flex; margin-top: 20px"></div>

- 표준편차 (<img style="width: 14px; margin-right: 2px; margin-left: 2px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\sigma ">)

<div style="display: flex; margin-top: 0px">
<img style="width: 90px; margin-right: 8px; margin-left: 0px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\sqrt{V[X]} ">
</div>

<div style="display: flex; margin-top: 24px"></div>
*루트를 하기 전에는 값이 너무 커지므로 단위를 맞춰주기 위해 사용

---

<div id="14. 균등 분포와 정규 분포"></div>

# 14. 균등 분포와 정규 분포

<div id="균등 분포"></div>

## 균등 분포

<div style="margin-top: 10px;">
<img style="width: 440px;" id="output" src="test4Img/ud_graph.PNG">
</div>

- Uniform distribution

<div style="display: flex; margin-top: -30px"></div>

- 평평하게 생겼다. (주사위, 동전)

<div style="display: flex; margin-top: -30px"></div>

- 식

<div style="display: flex; margin-top: 0px">
<img style="width: 168px; margin-right: 8px; margin-left: 20px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(x)=\left\{\begin{matrix}\frac{1}{b-a}  \\ 0 \end{matrix}\right.">
<img style="width: 90px; margin-right: 8px; margin-left: 20px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\begin{matrix} a\leq x\leq b\\otherwise \end{matrix}">
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 140px; margin-right: 8px; margin-left: 20px; margin-top: 40px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?X \sim U(a,b)">
</div>

- 평균

<div style="display: flex; margin-top: 0px">
<img style="width: 100px; margin-right: 8px; margin-left: 20px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{1}{2}(a+b)">
</div>

- 분산

<div style="display: flex; margin-top: 0px">
<img style="width: 130px; margin-right: 8px; margin-left: 20px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{1}{12}(b-a)^2">
</div>

<div id="정규 분포"></div>

## 정규 분포

<div style="margin-top: 10px;">
<img style="width: 440px;" id="output" src="http://infoso.kr/wp/wp-content/uploads/2020/10/%EC%A0%95%EA%B7%9C%EB%B6%84%ED%8F%AC1.png">
</div>

- Normal distribution or Gaussian distribution

<div style="display: flex; margin-top: -30px"></div>

- 종모양 (키)

<div style="display: flex; margin-top: -30px"></div>

- 식

<div style="display: flex; margin-top: 0px">
<img style="width: 280px; margin-right: 8px; margin-left: 20px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(x)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}">
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 160px; margin-right: 8px; margin-left: 20px; margin-top: 40px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?X \sim N(\mu,\sigma^2)">
</div>

<div style="display: flex; margin-top: 10px"></div>

- 평균

<div style="display: flex; margin-top: 0px">
<img style="width: 22px; margin-right: 8px; margin-left: 20px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\mu">
</div>

<div style="display: flex; margin-top: 10px"></div>

- 분산

<div style="display: flex; margin-top: 0px">
<img style="width: 36px; margin-right: 8px; margin-left: 20px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\sigma^2">
</div>

---

<div id="15. 최대 우도 추정 (MLE)"></div>

# 15. 최대 우도 추정 (MLE)

<div id="조건부확률과 likehood"></div>

## 조건부확률과 likehood

<div style="margin-top: 10px;">
<img style="width: 440px;" id="output" src="test4Img/abBox.PNG">
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 140px; margin-right: 8px; margin-left: 26px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(cb|A)=\frac{1}{2}">
<img style="width: 140px; margin-right: 8px; margin-left: 94px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(cb|B)=\frac{2}{3}">
<img style="width: 100px; margin-right: 8px; margin-left: 40px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?likehood">
</div>

<div style="display: flex; margin-top: 10px">
<img style="width: 140px; margin-right: 8px; margin-left: 26px; margin-top: 8px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(b|A)=\frac{1}{2}">
<img style="width: 140px; margin-right: 8px; margin-left: 94px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(b|B)=\frac{1}{3}">
<img style="width: 100px; margin-right: 8px; margin-left: 40px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?likehood">
</div>

<div style="display: flex; margin-top: 20px">
<div style="font-size: 24px; margin-left: 40px;">조건부확률</div>
<div style="font-size: 24px; margin-left: 124px;">조건부확률</div>
</div>

- likehood
  - 위의 그림에서는 동시에 A상자에서 어떤 공이 나올지와 B상자에서 어떤 공이 나올지에 대한 확률을 의미
  - 조건부 확률 값, but 확률 분포는 아니다. (합이 1이 아님)

<div id="MLE란"></div>

## MLE란

<div style="display: flex; margin-top: 36px">
<div style="margin-top: 10px">정규 분포를 예시를 들어보면</div>
<img style="width: 400px; margin-right: 8px; margin-left: 10px; margin-top: 6px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?z_1=x+n_1, z_2=x+n_2, n \sim N(0,\sigma ^2)">
<div style="margin-top: 10px">일 때</div>
</div>

<div style="display: flex; margin-top: 36px">
<img style="width: 110px; margin-right: 8px; margin-left: 0px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(z_1,z_2|x)">
<div style="margin-left: 10px"> <u>"x가 뭐였길 때 z가 저렇게 나와을까?"</u>에 대한 답을 찾고자 하는 것</div>
</div>

<div style="display: flex; margin-top: 36px">
<div style="margin-top: 10px">그렇다면</div>
<img style="width: 70px; margin-right: 0px; margin-left: 10px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?z_1, z_2">
<div style="margin-top: 10px; margin-left: 10px;">이 나올 확률이 높은 경우 즉, <u>likehood가 최대가 되도록 하는 x</u>를 구하면 된다. </div>
</div>

<div style="display: flex; margin-top: 36px">
<div style="margin-top: 10px">*예시에서는 정규분포 이므로 순간변화량(기울기)이 0이 되는 값이 최대</div>
</div>

<div style="display: flex; margin-top: 26px">
<div style="margin-top: 26px">구해보면</div>
<img style="width: 280px; margin-right: 0px; margin-left: 10px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(z_1,z_2|x)=p(z_1|x)p(z_2|x)">
<div style="margin-top: 24px; margin-left: 10px;">(독립시행 가정),</div>
<img style="width: 280px; margin-right: 0px; margin-left: 10px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(z_1|x)=\frac{1}{\sqrt{2\pi \sigma^2}}e^{-\frac{(z_1-x)^2}{2\sigma^2}}">
</div>

<div style="display: flex; margin-top: 36px">
<img style="width: 130px; margin-right: 0px; margin-left: 0px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\hat{x}=arg max_x">
<img style="width: 600px; margin-right: 0px; margin-left: 8px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(z_1|x)p(z_2|x)=log p(z_1|x)p(z_2|x)=-\frac{(z_1-x)^2}{2\sigma^2}-\frac{(z_2-x)^2}{2\sigma^2}">
</div>

<div style="display: flex; margin-top: 26px">
<div style="margin-top: 24px">미분하면</div>
<img style="width: 340px; margin-right: 0px; margin-left: 18px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{z_1-\hat x}{\sigma^2}+\frac{z_1-\hat x}{\sigma^2}=\frac{z_1+z_2}{2}=0">
</div>

<div style="display: flex; margin-top: 26px">
<div style="margin-top: 24px">즉 여기서는 들어온 measurement의 평균을 구하는 것이 MLE이다.</div>
</div>

---

<div id="16. 최대 사후 확률 (MAP)"></div>

# 16. 최대 사후 확률 (MAP)

>likelihood 뿐만 아니라 prior distribution(사전 분포)까지 고려한 posterior(사후 확률)를 maximize 하는 것

<div id="Bayesian rule"></div>

## Bayesian rule

<div style="display: flex; margin-top: 10px">
<img style="width: 400px; margin-right: 0px; margin-left: 8px; margin-top: 0px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(x|z)=\frac{p(x,z)}{p(z)}=\frac{p(z|x)p(x)}{p(z)}">
</div>

<br>

어떻게 저런 식이 나왔을까?

<div style="display: flex; margin-top: 26px">
<img style="width: 240px; margin-right: 0px; margin-left: 0px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(A|B)=\frac{p(A \cap B)}{p(B)}">
</div>

<div style="display: flex; margin-top: 26px">
<img style="width: 240px; margin-right: 0px; margin-left: 0px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(B|A)=\frac{p(A \cap B)}{p(A)}">
<div style="margin-top: 32px; margin-left: 10px;">이면</div>
<img style="width: 280px; margin-right: 0px; margin-left: 24px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(A \cap B)=p(B|A)p(A)">
</div>

<div id="MAP 식"></div>

## MAP 식

<div style="display: flex; margin-top: 26px">
<img style="width: 600px; margin-right: 0px; margin-left: 0px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?\hat{x}=argmax_x\frac{p(z|x)p(x)}{p(z)}=argmax_x p(z|x)p(x)">
</div>

<div style="display: flex; margin-top: 26px">
<div style="margin-top: 24px">MLE와 비교했을 때</div>
<img style="width: 50px; margin-right: 0px; margin-left: 10px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p(x)">
<div style="margin-top: 24px; margin-left: 10px;">가 추가된 것으로 x의 분포를 사전에 알고 있다는 의미이다. (prior distribution)</div></div>

단, 잘못된 사전 정보는 오히려 추정 성능에 악영향을 미친다.

---

<div id="17. 정보 이론 기초"></div>

# 17. 정보 이론 기초

<div id="Entropy"></div>

## Entropy

- 불확실성

- 균등분포일 때 1

<div id="Cross-entropy"></div>

## Cross-entropy

- 실제로는 <img style="width: 20px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p_i">를 따르지만 <img style="width: 18px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q_i">로 구하는 경우

  (실제 값을 모르는 경우, 결과로 정수값이 필요한데 정수값으로 나오지 않는 경우 등)
- 딥러닝에서는 <img style="width: 20px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p_i">가 출력, 최대한 <img style="width: 18px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q_i">와 비슷하게 만드려고 노력

<div id="KL-divergence"></div>

## KL-divergence

- <img style="width: 15px; margin-right: 4px; margin-left: 0px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?p">와 <img style="width: 13px; margin-right: 4px; margin-left: 4px; margin-top: 20px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q">의 분포 차이 (<img style="width: 60px; margin-right: 4px; margin-left: 4px; margin-top: 10px; margin-bottom: 0px;" id="output" src="https://latex.codecogs.com/svg.image?q - p">)

<div id="Mutual information"></div>

## Mutual information

- x, y가 관련이는 정도(독립이면 0)
