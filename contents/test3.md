---
date: '2020-07-29'
title: 'position이 fixed일 때 margin: 0 auto가 안 먹히면'
categories: ['Web']
summary: 'position이 fixed일 때 margin: 0 auto가 안 먹힐 때 해결 방법'
thumbnail: './test.png'
---

margin: 0 auto는 가운데 정렬에 기능을 한다.

앞에 0은 위아래 공백이고 auto는 좌우공백을 의미하는데, auto로 하면 좌우공백을 일정하게 하여 화면의 크기가 바뀌어도 가운데 정렬을 유지할 수 있다.

하지만 position이 fixed일 때는 margin: 0 auto만 적으면 가운데 정렬이 되지 않고 추가로 아래와 같이 입력해주어야한다. 

<div id="해결방법"></div>

# 해결방법

```
.class-name {
   position: fixed;
   margin: 0 auto;
   left: 0;
   right: 0;
}
