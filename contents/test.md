---
date: '2023-02-14'
title: '딥러닝 필수 기초 수학 1탄'
categories: ['AI']
summary: '딥러닝을 위해 반드시 알아야할 필수 기초 수학을 총정리 하였습니다.'
thumbnail: './test.png'
---

<div id="1. 함수"></div>

# 1. 함수

![funMainImg.PNG](testImg/funMainImg.PNG)

<div id="함수의 표현"></div>

## 함수의 표현

#### 1. 한 개 입력(x), 한 개 출력인 경우

<br>
<img style="width: 120px" alt='TeX' src="https://latex.codecogs.com/svg.image?f(x)=x^{2}">
<br>

#### 2. 두 개 입력(x,y), 한 개 출력인 경우
<br>
<img style="width: 140px" id="output" src="https://latex.codecogs.com/svg.image?f(x,y)=yx^{2}">
<br>

#### 3. 한 개 입력(x), 두 개 출력인 경우 (벡터 한개 출력)

<br>
<img style="width: 130px" id="output" src="https://latex.codecogs.com/svg.image?f(x)=\begin{bmatrix}x \\ x^{2}\end{bmatrix}">
<br>

#### 4. 두 개 입력(x,y), 두 개 출력인 경우 (벡터 한개 출력)
<br>
<img style="width: 180px" id="output" src="https://latex.codecogs.com/svg.image?f(x,y)=\begin{bmatrix}x+y \\ xy^{2}\end{bmatrix}">
<br>

[//]: # (<p></p>)
<br>
*여러개 출력인 경우에 그래프는 그냥 각각 그려주면 된다.

- 3번인 경우
<div>
<img style="width: 180px" id="output" src="testImg/oneFgraph.PNG">  ㅤㅤㅤ<img style="width: 190px" id="output" src="testImg/twoFgraph.PNG">
</div>
<br>

---

<div id="2. 로그 함수"></div>

# 2. 로그 함수

#### <img style="width: 60px" id="output" src="https://latex.codecogs.com/svg.image?log_{a}x">는 a(밑)를 몇 승 해야 x(진수)냐라는 의미

로그 함수는 딥러닝에서 가장 많이 사용하는 함수이다. 

그 이유는 로그를 통해 큰 수를 작게 만들고 복잡한 계산을 간편하게 하기 때문이다.

<div id="로그 그래프"></div>

## 로그 그래프
<p style=" FLOAT: none; CLEAR: none"><span class="imageblock" style="display: inline-block; width: 267px;  height: auto; max-width: 100%;" href="https://t1.daumcdn.net/cfile/tistory/2245BF3B534EAAEF32?original" data-lightbox="lightbox" data-alt="null"><img src="https://t1.daumcdn.net/cfile/tistory/2245BF3B534EAAEF32" style="max-width: 100%; height: auto;" srcset="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&amp;fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F2245BF3B534EAAEF32" width="267" height="225" alt="로그함수 그래프" filename="logarithmic_function_graph_movement_1.png" filemime="image/png"></span>&nbsp; <span class="imageblock" style="display: inline-block; width: 265px;  height: auto; max-width: 100%;" href="https://t1.daumcdn.net/cfile/tistory/223EFE3B534EAAEF35?original" data-lightbox="lightbox" data-alt="null"><img src="https://t1.daumcdn.net/cfile/tistory/223EFE3B534EAAEF35" style="max-width: 100%; height: auto;" srcset="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&amp;fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F223EFE3B534EAAEF35" width="265" height="226" alt="로그함수 그래프의 평행이동 - x축 방향" filename="logarithmic_function_graph_movement_2.png" filemime="image/png"></span><br><span class="imageblock" style="display: inline-block; width: 266px;  height: auto; max-width: 100%;" href="https://t1.daumcdn.net/cfile/tistory/222A493B534EAAEF0F?original" data-lightbox="lightbox" data-alt="null"><img src="https://t1.daumcdn.net/cfile/tistory/222A493B534EAAEF0F" style="max-width: 100%; height: auto;" srcset="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&amp;fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F222A493B534EAAEF0F" width="266" height="225" alt="로그함수 그래프의 평행이동 - y축 방향" filename="logarithmic_function_graph_movement_3.png" filemime="image/png"></span>&nbsp;<span class="imageblock" style="display: inline-block; width: 266px;  height: auto; max-width: 100%;" href="https://t1.daumcdn.net/cfile/tistory/254F743B534EAAF006?original" data-lightbox="lightbox" data-alt="null"><img src="https://t1.daumcdn.net/cfile/tistory/254F743B534EAAF006" style="max-width: 100%; height: auto;" srcset="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&amp;fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F254F743B534EAAF006" width="266" height="226" alt="로그함수 그래프의 평행이동 - x, y축 방향" filename="logarithmic_function_graph_movement_4.png" filemime="image/png"></span></p>

<br>

<div id="중요한 성질"></div>

## 중요한 성질

<div style="margin-top: -30px"></div>

1. <img style="width: 220px" id="output" src="https://latex.codecogs.com/svg.image?log_{a}xy=log_{a}x+log_{a}y">
<div style="margin-bottom: -20px"></div>

2. <img style="width: 220px" id="output" src="https://latex.codecogs.com/svg.image?log_{a}xy=log_{a}x+log_{a}y">
<div style="margin-bottom: -35px"></div>

3. <img style="width: 176px" id="output" src="https://latex.codecogs.com/svg.image?log{_{}a}^{m}x=\frac{1}{m}log^{a}x">
<div style="margin-bottom: -40px"></div>

4. <img style="width: 130px" id="output" src="https://latex.codecogs.com/svg.image?log_{a}b=\frac{log_{c}b}{log_{c}a}">
<div style="margin-bottom: -30px"></div>

5. <img style="width: 130px" id="output" src="https://latex.codecogs.com/svg.image?log_{a}b=\frac{1}{log_{b}a}">
<div style="margin-bottom: -30px"></div>

6. <img style="width: 116px" id="output" src="https://latex.codecogs.com/svg.image?a^{log_{a}x}=x">
<div style="margin-bottom: -30px"></div>

7. <img style="width: 150px" id="output" src="https://latex.codecogs.com/svg.image?a^{log_{b}c}=c^{log_{b}a}">
<div style="margin-bottom: -30px"></div>

<div style="margin-top: 30px"></div>

<div id="자연 상수 e"></div>

## 자연 상수 e 

밑 없이 <img style="width: 26px; margin-right: 4px; margin-left: 4px;" id="output" src="https://latex.codecogs.com/svg.image?log">만 쓰여있으면 밑이 e라고 생각하면 된다.

자연상수 e는 연속 성장을 표현하기 위해 고안된 상수이다. 연속적인 세상을 표현하기 위해 만들어졌다고 이해하면 된다.

구체적인 의미는 <u>**100%의 성장률**</u>을 가지고 <u>**1회 연속 성장할 때 얻게 되는 성장량**</u>이다.

<br>

만약 50% 성장률을 가지고 1회 연속 성장한다면 <img style="width: 20px" id="output" src="https://latex.codecogs.com/svg.image?e^\frac{1}{2}">

100% 성장률로 2회 연속 성장한다면 그 성장량은 <img style="width: 18px" id="output" src="https://latex.codecogs.com/svg.image?e^2">

즉, <img style="width: 20px" id="output" src="https://latex.codecogs.com/svg.image?e^x">라는 식에서 지수 x가 갖는 의미는 <u>성장횟수 x 성장률</u>이다.

---

<div id="3. 벡터와 행렬 (선형대수학)"></div>

# 3. 벡터와 행렬 (선형대수학)
<br>
<div style="display: flex"><div style="padding: 32px 0px;">행벡터 : </div>
<img style="width: 40px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}1\\2\\3\end{bmatrix}">
<div style="padding: 32px 20px;">열 벡터 : </div>
<img style="width: 90px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}1&2&3\\\end{bmatrix}">
</div>

<div style="display: flex; margin-top: 10px">
<div style="padding: 32px 0px;">행렬 : </div><img style="width: 90px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}1&2&3\\4&5&6\\\end{bmatrix}">
</div>
위의 행렬에서 2행 3열(2x3)의 값은 6
<br></br>

- 딥러닝에서 중요한 이유 : 연립방정식을 쉽게 표현하기 위해

연립 1차 방정식
(n차 방정식은 최고차항을 기준)

<br>
<div style="display: flex">
<img style="width: 150px" id="output" src="https://latex.codecogs.com/svg.image?\left\{\begin{matrix}2x+5y=10\\9x+6y=30\end{matrix}\right.">
<div style="padding: 12px 12px;">을 행렬로 표현하면</div>
<img style="width: 170px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}2&5\\9&6\\\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}%20=\begin{bmatrix}10\\30\end{bmatrix}">
</div>

<br>
상수가 같은 세 개의 연립 1차 방정식
<br>

<br>
<div style="display: flex">
<img style="width: 170px" id="output" src="https://latex.codecogs.com/svg.image?\left\{\begin{matrix}2x_{1}+5y_{1}=10\\9x_{1}+6y_{1}=30\end{matrix}\right.">
<div style="margin-left: 20px"></div>
<img style="width: 170px" id="output" src="https://latex.codecogs.com/svg.image?\left\{\begin{matrix}2x_{2}+5y_{2}=12\\9x_{2}+6y_{2}=20\end{matrix}\right.">
<div style="margin-left: 20px"></div>
<img style="width: 170px" id="output" src="https://latex.codecogs.com/svg.image?\left\{\begin{matrix}2x_{3}+5y_{3}=15\\9x_{3}+6y_{3}=19\end{matrix}\right.">
<div style="padding: 12px 12px;">을 행렬로 표현하면</div>
</div>
<br>
<img style="width: 360px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}2&5\\9&6\\\end{bmatrix}\begin{bmatrix}x_{1}&x_{2}&x_{3}\\y_{1}&y_{2}&y_{3}\\\end{bmatrix}%20=%20\begin{bmatrix}10&12&15\\30&20&19\\\end{bmatrix}">
<br>

<div id="주요 성질"></div>

## 주요 성질
1. **짝이 맞아야 곱할 수 있다.**

2x<u>2</u> 행렬과 <u>2</u>x3 행렬은 곱할 수 있고 2x<u>2</u> 행렬과 <u>3</u>x2 행렬은 곱할 수 없다.

2. **결과 사이즈는 맨 앞 x 맨 뒤 이다.**

<u>2</u>x4 행렬 곱하기 4x2 행렬 곱하기 2x<u>3</u> 행렬은 2x3이다.

3. **교환 법칙은 성립되지 않는다.**

<div id="벡터의 방향과 크기"></div>

## 벡터의 방향과 크기

벡터 [2,1]를 좌표평면에 나타내면 

(화살표는 방향, 직선의 길이는 크기)

<br>
<div style="width: 400px; z-index: -1;" >
<img src="testImg/vertorP.png">
</div>
<br>

**크기와 방향이 같으면 같은 벡터이다.** 시점이 달라도 같은 벡터일 수 있다.

위의 벡터와 아래 벡터는 시점은 다르지만 같은 벡터이다.

<br>
<div style="width: 400px; z-index: -1;" >
<img src="testImg/vertorP2.png">
</div>
<br>

### 벡터의 크기(화살표의 길이) 구하는 방법

1. L2-norm : <img style="width: 80px" id="output" src="https://latex.codecogs.com/svg.image?\sqrt{x^{2}+y^{2}}"> (피타고라스 정리)
<pr></pr>
2. L1-norm : <img style="width: 70px" id="output" src="https://latex.codecogs.com/svg.image?\left|x\right|+\left|y\right|"> (절댓값)

<br></br>

### 벡터 표기 법 : <img style="width: 20px; margin: -5px 10px" id="output" src="https://latex.codecogs.com/svg.image?\vec{x}">

---

<div id="4. 전치와 내적"></div>

# 4. 전치와 내적

<div id="전치(Transpose)"></div>

## 전치(Transpose)

<img style="width: 110px; margin-bottom: 10px" id="output" src="https://latex.codecogs.com/svg.image?A_{ij}%20=%20A^{T}_{ji}">
전치하면 i행 j열이 j행 i열된다.

<div style="display: flex; margin-top: 60px">
<img style="width: 140px" id="output" src="	https://latex.codecogs.com/svg.image?A=\begin{bmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\\\end{bmatrix}">
<div style="padding: 12px 12px;">일 때 전치하면</div>
<img style="width: 140px" id="output" src="	https://latex.codecogs.com/svg.image?A^{T}=\begin{bmatrix}a_{11}&a_{21}\\a_{12}&a_{22}\\\end{bmatrix}">
</div>

<img style="width: 160px; margin-top: 64px" id="output" src="https://latex.codecogs.com/svg.image?(a\vec{x})^{T}=\vec{x}^{T}a^{T}">
<div style="display: flex; margin-top: 30px">
<img style="width: 220px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\\\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\\\end{bmatrix}%20=%20\begin{bmatrix}b_{1}\\b_{2}\\\end{bmatrix}">
<div style="padding: 12px 12px;">일 때 전치하면</div>
<img style="width: 280px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}x_{1}&x_{2}\\\end{bmatrix}%20\begin{bmatrix}a_{11}&a_{21}\\a_{12}&a_{22}\\\end{bmatrix}=%20\begin{bmatrix}b_{1}&b_{2}\\\end{bmatrix}">
</div>
<br>
<br>

<div id="내적 (dot product)"></div>

## 내적 (dot product)

<img style="width: 330px; margin-bottom: 20px" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}A_{1}\\A_{2}\end{bmatrix}\cdot%20%20\begin{bmatrix}B_{1}\\B_{2}\end{bmatrix}=A_{1}B_{1}+A_{2}B_{2}=\vec{A}^{T}\vec{B}">

위에서 점 표시( <img style="width: 4px;" id="output" src="https://latex.codecogs.com/svg.image?\cdot"> ) 내적을 의미

두 벡터를 내적한 결과는 스칼라이다. (스칼라는 방향없이 크기만 가지는 값)

<br>

#### 내적은 닮은 정도를 나타낸다.

<img style="width: 200px; margin-top: 20px" id="output" src="https://latex.codecogs.com/svg.image?\vec{A}^{T}\vec{B}=\left\|\vec{A}\right\|\left\|\vec{B}\right\|cos\theta">

1. <img style="width: 32px; margin-right: 5px;" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{A}\right\|"> 밑에 아무 것도 쓰여있지 않으면 L2-norm을 나타낸다. <img style="width: 40px;" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{A}\right\|_{1}">는 L1-norm

2. <img style="width: 11px; margin-right: 5px;" id="output" src="https://latex.codecogs.com/svg.image?\theta">는 두 벡터의 사잇각

<div style="margin-top: -10px; margin-bottom: 10px">
<u>내적이 닮은 정도를 나타내는 이유</u>는 빗변의 길이에 <img style="width: 46px; margin-left: 3px;" id="output" src="https://latex.codecogs.com/svg.image?cos\theta">
을 곱하면 밑변의 길이가 나오기 때문이다.
</div>

[cos(코사인)에 대해 이해가 되지 않는다면](https://hub1234.tistory.com/m/42)

<br>

두 벡터를 그래프로 그려보았을 때

<div style="width:500px; margin-top: 10px;">
<img id="output" src="testImg/dotProduct.png">
</div>

<div style="display: flex; margin-top: 20px;">
<div style="margin-top: -10px">위의 그림과 같이 
<img style="width: 100px; margin-right: 4px" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{A}\right\|\left\|\vec{B}\right\|cos\theta">에서
<img style="width: 70px; margin-left: 8px" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{B}\right\|cos\theta">
를 통해 방향이 다른 
<img style="width: 34px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{B}\right\|">
을 
<img style="width: 34px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?\left\|\vec{A}\right\|">
와 같은 방향에서 비교할 수 있게 된다.</div>
</div>

---

<div id="5. 극한과 입실론 - 델타 논법"></div>

# 5. 극한과 입실론-델타 논법

극한은 제대로 정의하는데는 100년이 걸렸을 만큼 쉬운 개념이 아니다.

우리가 고등학교 교육과정에서는


> 함수 <img style="width: 34px; margin-right: 4px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">에서 x가 a에 <u>한 없이 가까워질 때</u>, <img style="width: 34px; margin-right: 4px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">의 값이 일정한 값 L에 <u>한 없이 가까워지면</u> 함수 <img style="width: 34px; margin-right: 4px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">는 L에 수렴한다고 하고, L을 <img style="width: 34px; margin-right: 4px; margin-top: 10px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">의 극한값 또는 극한

이라 한다고 배웠다.
<br><br>

하지만 이는 극한을 직관적으로 이해시키기 위한 개념일 뿐 <u>오개념(계속 변화하는 상태가 아님)</u>을 만든다. 극한의 정의를 완전하게 이해하는 것은 너무 어려워 고등학교에서는 어쩔 수 없이 위와 같이 정의했다.
<br><br>

극한은 <u>입실론-델타 논법</u>을 통해 제대로 정의 된다.

> 입실론( <img style="width: 8px" id="output" src="https://latex.codecogs.com/svg.image?\epsilon "> )는 <img style="width: 36px; margin-left: 4px; margin-right: 6px;" id="output" src="https://latex.codecogs.com/svg.image?f(x)">의 오차를 표현하고, 델타( <img style="width: 9px" id="output" src="https://latex.codecogs.com/svg.image?\delta "> )는 <img style="width: 12px; margin-left: 5px;" id="output" src="https://latex.codecogs.com/svg.image?x "> 의 오차를 의미하는데,
> <br> <div style="display: flex; margin-top: 10px;"><div>임의의 </div><img  style="width: 50px; margin-left: 10px; margin-right: 6px; margin-top: -4px;" id="output" src="https://latex.codecogs.com/svg.image?\epsilon>0"> 에 대하여 <img style="width: 140px; margin-left: 10px; margin-right: 6px; margin-top: 4px;" id="output" src="https://latex.codecogs.com/svg.image?0<\left|x-a\right|<\delta ">의 범위에서 <img style="width: 140px; margin-left: 10px; margin-right: 6px; margin-top: 4px;" id="output" src="	https://latex.codecogs.com/svg.image?\left|f(x)-L\right|%3C\epsilon "> 이게 하는 <img style="width: 50px; margin-left: 10px; margin-right: 6px; margin-top: 0px;" id="output" src="	https://latex.codecogs.com/svg.image?\delta%3E0">가 존재하면 다음과 같이 정의한다.</div>
> <img style="width: 160px; margin-top: 20px;" id="output" src="https://latex.codecogs.com/svg.image?\displaystyle \lim_{x\to a}f(x)=L">

위의 글만 보면 이해하기가 정말 어렵다.

쉽게 이해하기 위해 극한이 존재하는 경우를 살펴보면

**임의의 <img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: 16px" id="output" src="https://latex.codecogs.com/svg.image?f(x)"> 오차 범위 안에 해당하는
<img style="width: 12px; margin-left: 5px;" id="output" src="https://latex.codecogs.com/svg.image?x "> 의 오차들에 해당하는
<img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: 16px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">가 모두 처음 <img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: 16px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">의 오차 범위 안에 있어야 한다.**

<br>

**1. 극한이 존재하는 경우**

<div style="display: flex">
<img style="width: 410px; margin-top: 20px; margin-right: 10px" id="output" src="testImg/g1.PNG">
<img style="width: 410px; margin-top: 20px" id="output" src="testImg/g2.PNG">
</div>
<div style="display: flex">
<img style="width: 410px; margin-top: 10px; margin-right: 10px" id="output" src="testImg/g3.PNG">
<img style="width: 410px; margin-top: 10px" id="output" src="testImg/g4.PNG">
</div>

<br>

**2. 극한이 존재하지 않는 경우**

<img style="width: 410px; margin-top: 20px; margin-right: 10px" id="output" src="testImg/gx.PNG">

<div style="display: flex; margin-top: 36px">
<div>
<img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: -4px" id="output" src="https://latex.codecogs.com/svg.image?f(x)"> 오차 범위 <img style="width: 10px; margin-left: 10px; margin-right: 6px; margin-top: -8px" id="output" src="https://latex.codecogs.com/svg.image?\frac{1}{2}"> 안에 해당하는
<img style="width: 12px; margin-left: 5px; margin-top: -5px; margin-right: 4px;" id="output" src="https://latex.codecogs.com/svg.image?x "> 의 오차들에 해당하는
<img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: 5px" id="output" src="https://latex.codecogs.com/svg.image?f(x)"> 중 처음 <img style="width: 36px; margin-left: 4px; margin-right: 6px; margin-top: 5px" id="output" src="https://latex.codecogs.com/svg.image?f(x)">의 오차 범위 안의 벗어나는 값이 있다.
</div>
</div>

<br>

[참고 자료](https://www.youtube.com/watch?v=JEe1rDCQ13E)

---

<div id="6. 미분과 도함수"></div>

# 6. 미분과 도함수

미분이란 **순간 변화율**이다. 이해하기 쉽게 그래프에서  **순간 기울기**라고 생각하면 된다.

<br>
그렇다면 기울기란?

<div id="기울기"></div>

## 기울기

>  기울기는 어떠한 직선이 수평으로 증가한 크기만큼 수직으로 얼마나 증가하였는지 나타내는 값

<div style="display: flex; margin-top: 0px"> 
기울기 구하는 방법 : <img style="width: 34px; margin-right: 8px; margin-left: 8px; margin-top: -12px" id="output" src="https://latex.codecogs.com/svg.image?\frac{\Delta%20y}{\Delta%20x}"> ( <img style="width: 34px; margin-right: 6px; margin-left: 8px; margin-top: -15px" id="output" src="https://latex.codecogs.com/svg.image?\Delta%20x">는 <img style="width: 13px; margin-right: 6px; margin-left: 8px; margin-top: -10px" id="output" src="https://latex.codecogs.com/svg.image?x"> 증가량 )
</div>

<div id="순간 변화율"></div>

## 미분 (순간 변화율)

<div style="display: flex; margin-top: 0px">
<div>
예를 들어<img style="width: 100px; margin-right: 6px; margin-left: 6px;" id="output" src="	https://latex.codecogs.com/svg.image?f(x)=x^{2}">함수에서의 <img style="width: 54px; margin-right: 6px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?x=1"> 일 때 미분값(순간변화율)을 생각해보자.
</div>
</div>

<br>
<div style="display: flex; margin-top: 0px">
<div>
우선 <img style="width: 54px; margin-right: 6px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?x=1">일 때에서 <img style="width: 54px; margin-right: 6px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?x=2">일 때 까지의 변화율은 (1,1)와 (2,4) 연결한 직선의 기울기인 3이다.
</div>
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 320px; margin-top: 20px; margin-right: 10px" id="output" src="testImg/sg1.PNG"> 
<div style="display: flex; margin-top: 0px">
<div style="margin-top: 150px; margin-right: 10px;  margin-left: 20px;">*구하는 식 : </div> <img style="width: 100px; margin-top: 20px; margin-right: 10px" id="output" src="https://latex.codecogs.com/svg.image?\frac{4-1}{2-1}=3">
</div>
</div>

<br>

<div style="display: flex; margin-top: 0px">
<div>
그렇다면 순간 변화율은 <img style="width: 16px; margin-right: 6px; margin-left: 6px;" id="output" src="	https://latex.codecogs.com/svg.image?x">가 몇일 때 까지의 변화율일까? 바로 <img style="width: 16px; margin-right: 6px; margin-left: 6px;" id="output" src="	https://latex.codecogs.com/svg.image?x">의 증가량이 0에 가장 가까운 값이다.
</div>
</div>

<img style="width: 320px; margin-top: 20px; margin-right: 10px" id="output" src="testImg/sg2.PNG"> 

<br>
<div style="display: flex; margin-top: 0px">
즉 <img style="width: 54px; margin-right: 6px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?x=1">일 때 미분값(순간변화율)을 식으로 표현하면
</div>

<img style="width: 300px; margin-top: 30px;" id="output" src="https://latex.codecogs.com/svg.image?\displaystyle \lim_{\Delta x\to 0}\frac{f(1+\Delta x)-f(1)}{\Delta x}">

<br>
이다.

<div id="도함수"></div>

## 도함수

그렇다면 여기서 <img style="width: 16px; margin-right: 6px; margin-left: 6px;" id="output" src="	https://latex.codecogs.com/svg.image?x">의 다양한 값의 미분값(순간변화율)을 표현한다면

<img style="width: 300px; margin-top: 30px;" id="output" src="https://latex.codecogs.com/svg.image?\displaystyle \lim_{\Delta x\to 0}\frac{f(x+\Delta x)-f(x)}{\Delta x}">

<br>
<div>위와 같은 식이 나온다. 이것이 바로 <u>도함수</u>이다.</div>

<div style="display: flex; margin-top: 10px">
<div>
그리고 위의 식을 간단하게 <img style="width: 28px; margin-right: 6px; margin-left: 10px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{dy}{dx}">
(분수는 아님)로 표현한다.
</div>
</div>

<div style="display: flex; margin-top: 20px">
<div>
만약 <img style="width: 54px; margin-right: 4px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?x=1">일 때 미분값은 <img style="width: 92px; margin-right: 6px; margin-left: 10px;" id="output" src="https://latex.codecogs.com/svg.image?\left.\begin{matrix}\frac{dy}{dx}\end{matrix}\right|_{x=1}">라고 표현한다.
</div>
</div>

<div id="딥러닝에서 많이 쓰이는 도함수"></div>

## 딥러닝에서 많이 쓰이는 도함수

<div style="display: flex; margin-top: 0px"><div style="margin-top: 10px">1. </div><img style="width: 34px; margin-right: 4px; margin-left: 14px;" id="output" src="https://latex.codecogs.com/svg.image?x^{n}"><div style="font-size: 20px">→</div><img style="width: 76px; margin-right: 4px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?nx^{n-1}"></div>
<br>
<div style="display: flex; margin-top: 0px"><div style="margin-top: 10px">2. </div><img style="width: 34px; margin-right: 4px; margin-left: 14px;" id="output" src="https://latex.codecogs.com/svg.image?e^{x}"><div style="font-size: 20px">→</div><img style="width: 34px; margin-right: 4px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?e^{x}"></div>
<br>
<div style="display: flex; margin-top: 0px"><div style="margin-top: 18px">3. </div><img style="width: 60px; margin-right: 10px; margin-left: 14px;" id="output" src="https://latex.codecogs.com/svg.image?logx"><div style="font-size: 20px; margin-top: 12px">→</div><img style="width: 18px; margin-right: 4px; margin-left: 10px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{1}{x}"></div>
<br>
<div style="display: flex; margin-top: 0px"><div style="margin-top: 18px">4. </div><img style="width: 74px; margin-right: 10px; margin-left: 14px;" id="output" src="https://latex.codecogs.com/svg.image?log_2x"><div style="font-size: 20px; margin-top: 12px">→</div><img style="width: 70px; margin-right: 4px; margin-left: 10px;" id="output" src="	https://latex.codecogs.com/svg.image?\frac{1}{log2}\frac{1}{x}"></div>
<br>
<div style="display: flex; margin-top: 8px"><div style="margin-top: 7px">5. </div><img style="width: 160px; margin-right: 10px; margin-left: 14px;" id="output" src="https://latex.codecogs.com/svg.image?f(x)+g(x)"><div style="font-size: 20px; margin-top: 3px">→</div><img style="width: 180px; margin-right: 4px; margin-left: 10px;" id="output" src="	https://latex.codecogs.com/svg.image?f^{'}(x)+g^{'}(x)"></div>
<br>
<div style="display: flex; margin-top: 12px"><div style="margin-top: 6px">6. </div><img style="width: 80px; margin-right: 10px; margin-left: 14px;" id="output" src="https://latex.codecogs.com/svg.image?af(x)"><div style="font-size: 20px; margin-top: 3px">→</div><img style="width: 90px; margin-right: 4px; margin-left: 10px;" id="output" src="	https://latex.codecogs.com/svg.image?af^{'}(x)"></div>
<br>
<div style="display: flex; margin-top: 12px"><div style="margin-top: 6px">7. </div><img style="width: 120px; margin-right: 10px; margin-left: 14px;" id="output" src="https://latex.codecogs.com/svg.image?f(x)g(x)"><div style="font-size: 20px; margin-top: 3px">→</div><img style="width: 280px; margin-right: 4px; margin-left: 10px;" id="output" src="	https://latex.codecogs.com/svg.image?f^{'}(x)g(x)+f(x)g^{'}(x)"></div>

---

<div id="7. 연쇄법칙"></div>

# 7. 연쇄법칙

<img style="width: 80px; margin-right: 4px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?(x^{2}+1)^{2}">을 미분하는데 연쇄법칙으로 생각해보자!

이해를 돕기 위해

<br>
<div style="display: flex;">
<img style="width: 20px; margin-right: 4px; margin-left: 6px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?x"><div style="margin-top: 15px">→</div>
<img style="width: 34px; margin-right: 0px; margin-left: 8px; margin-top: -6px;" id="output" src="https://latex.codecogs.com/svg.image?x^{2}"><div style="margin-top: 15px">→</div>
<div><img style="width: 100px; margin-right: 8px; margin-left: 10px;  margin-top: 4px;" id="output" src="https://latex.codecogs.com/svg.image?x^{2}+1"></div><div style="margin-top: 15px">→</div>
<img style="width: 130px; margin-right: 4px; margin-left: 6px;" id="output" src="https://latex.codecogs.com/svg.image?(x^{2}+1)^{2}"></div>

<br>
위의 그림 처럼 변화 과정을 그린다. 

그런 다음 뒤에서 부터 전에 값으로 앞에 값을 미분한다고 생각하면 연쇄법칙이 된다.

<img style="width: 600px; margin-right: 4px; margin-left: 6px; margin-top: 50px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{d(x^{2}+1)^{2}}{dx}=\frac{d(x^{2}+1)^{2}}{d(x^{2}+1)}\frac{d(x^{2}+1)}{dx^{2}}\frac{dx^{2}}{dx}">
<img style="width: 500px; margin-right: 4px; margin-left: 6px; margin-top: 40px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{d(x^{2}+1)^{2}}{dx}=2(x^{2}+1)\cdot1\cdot2x">

---

<div id="8. 편미분과 그라디언트"></div>

# 8. 편미분과 그라디언트

<div id="편미분"></div>

## 편미분

편미분은 **여러개 변수로 이루어진 함수를 미분할 때 각각에 대해 미분** 하는 것

<div style="display: flex; margin-top: 20px">
<img style="width: 140px; margin-right: 4px; margin-left: 0px; margin-top: -2px;" id="output" src="https://latex.codecogs.com/svg.image?f(x,y)=yx^{2}">로 살펴보면 
</div>
<div style="display: flex; margin-top: 14px">
<img style="width: 16px; margin-right: 4px; margin-left: 0px; margin-top: -2px;" id="output" src="https://latex.codecogs.com/svg.image?x">에 대한 편미분
(<img style="width: 16px; margin-right: 4px; margin-left: 4px; margin-top: -2px;" id="output" src="https://latex.codecogs.com/svg.image?x">에 대한 변화율), 
<img style="width: 12px; margin-right: 4px; margin-left: 10px; margin-top: -2px;" id="output" src="https://latex.codecogs.com/svg.image?y">에 대한 편미분
(<img style="width: 12px; margin-right: 2px; margin-left: 4px; margin-top: -2px;" id="output" src="https://latex.codecogs.com/svg.image?y">에 대한 변화율)이다.
</div>

<div style="display: flex; margin-top: 0px">
<div style="margin-top: 26px">기호로 표현하면 </div>
<img style="width: 30px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{\partial%20f}{\partial%20x}">
<div style="margin-top: 26px">,</div>
<img style="width: 30px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{\partial%20f}{\partial%20y}">
<div style="margin-top: 26px">이고 다른 변수들은 전부 상수로 취급하고 미분하면 된다.</div>
</div>

<br>

<div style="display: flex; margin-top: 0px">
<img style="width: 160px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?f(x,y)=yx^{2}">
<div style="margin-top: 26px">를 예로 들면</div>
<img style="width: 100px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{\partial f}{\partial x}=2yx">
<div style="margin-top: 26px">,</div>
<img style="width: 90px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?\frac{\partial f}{\partial y}=x^{2}">
<div style="margin-top: 26px">이다.</div>
</div>

<div id="그라디언트"></div>

## 그라디언트

편미분한 것을 벡터로 묶은 것

<div style="display: flex; margin-top: 0px">
<img style="width: 160px; margin-right: 4px; margin-left: 0px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?f(x,y)=yx^{2}">
<div style="margin-top: 26px">의 그라디언트는</div>
<img style="width: 60px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}%202yx\\x^{2}\end{bmatrix}">
</div>

<img style="width: 260px; margin-right: 4px; margin-left: 0px; margin-top: 30px;" id="output" src="https://latex.codecogs.com/svg.image?\begin{bmatrix}\frac{%20\partial%20f}{%20\partial%20x}\\\frac{%20\partial%20f}{%20\partial%20y}\end{bmatrix}\left.\begin{matrix}%20\\\end{matrix}\right|_{x=1,y=1}%20=%20\begin{bmatrix}2%20\\1\end{bmatrix}">

### 그라디언트의 의미

>여러개 변수를 미분 값(순간 변화율)을 합쳐서 해당 함수가 어떻게 나아가고 있는지 알 수 있다. 

---

<div id="9. 테일러 급수"></div>

# 9. 테일러 급수

>  어떤 임의의 함수를 다항함수로 나타내는 것

<div style="display: flex; margin-top: 0px">
<img style="width: 480px; margin-right: 8px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?f(x)=c_{0}+c_{1}x^{1}+c_{2}x^{2}+c_{3}x^{3}+c_{4}x^{4}+...">
<div style="margin-top: 19px">에서</div>
<img style="width: 26px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?c_{n}">
<div style="margin-top: 19px">을 구하면 된다.</div>
</div>

<br>
구하는 방법은 알고 싶은 부분(x)에 값을 넣어주고 계속 미분하면 된다.
<br>
<br>

<div style="display: flex; margin-top: 0px">
<img style="width: 64px; margin-right: 8px; margin-left: 0px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?cos(x)">
<div style="margin-top: 19px">을 예로 들면</div>
<img style="width: 60px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?x=0">
<div style="margin-top: 19px">일 때를 알고 싶다면</div>
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 22px; margin-right: 8px; margin-left: 0px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?c_{0}">
<div style="margin-top: 19px">는 x에 0을 대입해서</div>
<img style="width: 50px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?cos0">
<div style="margin-top: 19px">을 구하면 된다.</div>
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 22px; margin-right: 8px; margin-left: 0px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?c_{1}">
<div style="margin-top: 19px">은 한 번 미분해서</div>
<img style="width: 460px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?-sinx=0+c_{1}+2c_{2}x+3c_{3}x^2+5c_{4}x^3+...">
</div>

<div style="display: flex; margin-top: 0px">
<div style="margin-top: 19px">*</div>
<img style="width: 46px; margin-right: 8px; margin-left: 0px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?cosx">
<div style="margin-top: 14px">를 미분하면</div>
<img style="width: 60px; margin-right: 4px; margin-left: 10px; margin-top: 6px;" id="output" src="https://latex.codecogs.com/svg.image?-sinx">
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 22px; margin-right: 8px; margin-left: 0px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?c_{2}">
<div style="margin-top: 19px">은 두 번 미분해서</div>
<img style="width: 400px; margin-right: 4px; margin-left: 10px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?-cosx=2c_{2}+3\cdot2c_{3}x+4\cdot3c_{4}x^2+...">
</div>

<div style="display: flex; margin-top: 0px">
<img style="width: 22px; margin-right: 8px; margin-left: 0px; margin-top: 10px;" id="output" src="https://latex.codecogs.com/svg.image?c_{3}">
<div style="margin-top: 19px">은 세 번 미분해서</div>
<img style="width: 340px; margin-right: 4px; margin-left: 10px; margin-top: 18px;" id="output" src="https://latex.codecogs.com/svg.image?sinx=3\cdot2c_{3}+4\cdot3\cdot2c_{4}x+...">
</div>

<br>
이런식으로 구하면 된다.

<div id="테일러 급수 식"></div>

## 테일러 급수 식
위에서 보면 규칙을 찾을 수 있다. 이 규칙을 활용하여 식으로 표현한다면
<div style="display: flex; margin-top: 0px">
<img style="width: 66px; margin-right: 8px; margin-left: 0px; margin-top: 28px;" id="output" src="https://latex.codecogs.com/svg.image?x=a">
<div style="margin-top: 36px">일 때 값을 알고 싶다면</div>
<img style="width: 120px; margin-right: 4px; margin-left: 10px; margin-top: 18px;" id="output" src="https://latex.codecogs.com/svg.image?c_{n}=\frac{f^{n}(a)}{n!}">
</div>

<br>
<br>

하지만 테일러 급수는 **모든 것을 표현할 수 있는 것은 아니다.** 

ln(x) 함수는 x > 2인 경우에는 수렴 하지 못한다.