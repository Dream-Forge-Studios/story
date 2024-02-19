---
date: '2023-02-16'
title: 'react 기반 기술 블로그에 목차에 하이라이트 추가하기'
categories: ['WEB/APP']
summary: 'Interaction Observer API를 활용하여 다양한 기능을 가지고 있는 동적 목차를 추가해보자.'
thumbnail: './test.png'
---

스크롤에 의해 헤더안으로 제목이 가려지면 그에 해당하는 목차에 하이라이트를 하는 기능을 구현해 볼 것이다.

이 기능을 구현하기 위해선 해당 위치를 파악하여 스크롤에 따라 이벤트를 일으켜야 한다.

이를 쉽게 구현하기 위한 방법인 Interaction Observer API를 소개할 것이다.

<div id="Interaction Observer API"></div>

# Interaction Observer API

<div id="왜 사용하는가?"></div>

## 왜 사용하는가?

기존 scroll 이벤트는 스크롤을 할 때마다 지속적으로 실행되어 큰 부하를 줄 수 있다.

하지만 Interaction Observer API는 비동기적으로 원하는 타겟의 가시성이 변경될 때만 이벤트가 실행된다.

또한 쉽게 구현할 수 있도록 기능이 제공된다.

<div id="사용 방법"></div>

## 사용 방법

```
const observer = new IntersectionObserver(callback, options); // 관찰자 생성
observer.observe(element); // 관찰 대상 등록
```

우선 관찰자를 생성한다. options의 값이 충족될 때마다 callback이 호출된다.

### options의 값

- `root`

  <br>
  대상의 가시성을 확인하기 위한 뷰포트로 사용되는 요소입니다. 대상의 조상이어야 합니다. 지정되지 않거나 null인 경우 브라우저 뷰포트가 기본값입니다.
  
  <br>
  <br>
  여기서 뷰포트란 눈에 보이는 화면을 말하는 것이다.
  <br>
  <br>
  
- `rootMargin`
  
  <br>
  루트 주위에 여백을 둡니다. margin예를 들어 " (상단, 오른쪽, 하단, 왼쪽) 와 같은 CSS 속성과 유사한 값을 가질 수 있습니다 10px 20px 30px 40px". 값은 백분율이 될 수 있습니다. 이 값 세트는 교차점을 계산하기 전에 루트 요소 경계 상자의 각 측면을 늘리거나 줄이는 역할을 합니다. 기본값은 모두 0입니다.

  <br>
  <br>
  
- `threshold`

  <br>
  관찰자의 콜백이 실행되어야 하는 대상의 가시성 비율을 나타내는 단일 숫자 또는 숫자 배열입니다. 가시성이 50% 표시를 통과할 때만 감지하려는 경우 값 0.5를 사용할 수 있습니다. 가시성이 또 다른 25%를 지날 때마다 콜백을 실행하려면 배열 [0, 0.25, 0.5, 0.75, 1]을 지정합니다. 기본값은 0입니다(단 하나의 픽셀이라도 표시되는 즉시 콜백이 실행됨을 의미). 1.0 값은 모든 픽셀이 표시될 때까지 임계값이 통과된 것으로 간주되지 않음을 의미합니다.
  
위의 옵션값들이 충족되면 callback 함수가 실행된다.

element에는 관찰하고자 하는 값을 넣어준다. 여기서는 목차에 하이라이트를 해줄 것으로 목차에 들어갈 문장을 넣어준다.

### 목차를 위한 구현 방법

#### 1. 관찰자 생성

````
import {Dispatch, SetStateAction} from 'react';

const observerOption = {
    threshold: 0.4,
    rootMargin: '-60px 0px 0px 0px',
};

export const getIntersectionObserver = (setState: Dispatch<SetStateAction<string>>) => {
    let direction = '';
    let prevYposition = 0;

    // scroll 방향 check function
    const checkScrollDirection = (prevY: number) => {
        if (window.scrollY === 0 && prevY === 0) return;
        else if (window.scrollY > prevY) direction = 'down';
        else direction = 'up';

        prevYposition = window.scrollY;
    };

    // observer
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            checkScrollDirection(prevYposition);
            if ((direction === 'down' && !entry.isIntersecting) ||
                (direction === 'up' && entry.isIntersecting)) {
                setState(entry.target.innerHTML);
            }
        });
    }, observerOption);

    return observer;
}
````

나는 헤더에 제목이 가려질 때 목차에 하이라이트를 해줄 것이므로 헤더의 높이인 60px를 rootMargin에 넣어주었다.

threshold는 이런 저런 값을 넣어보니 0.4가 해당 기능에 적절하였다.

특정 대상이 없기 때문에 root는 지정하지 않았다.

callback 함수의 관해서는 scroll 방향을 체크하였고, 

- entries 값
  - `boundingClientRect` : 관찰 대상의 사각형 정보(DOMRectReadOnly)
  - `intersectionRect` : 관찰 대상의 교차한 영역 정보(DOMRectReadOnly)
  - `intersectionRatio` : 관찰 대상의 교차한 영역 백분율(intersectionRect 영역에서 boundingClientRect 영역까지 비율, Number)
  - `isIntersecting` : 관찰 대상의 교차 상태(Boolean)
  - `rootBounds` : 지정한 루트 요소의 사각형 정보(DOMRectReadOnly)
  - `target` : 관찰 대상 요소(Element)
  - `time` : 변경이 발생한 시간 정보(DOMHighResTimeStamp)

에서 isIntersecting를 활용하여 관찰 대상이 들어올 때(`true`)는 스크롤을 올릴 때, 관찰 대상이 나갈 때(`false`)는 스크롤을 내릴 때의 조건으로 하였다.
<br>
<br>

#### 2. 관찰 대상 설정

```
const PostContent: FunctionComponent<PostContentProps> = function ({ html }) {

    const [currentInnerHTML, setCurrentInnerHTML] = useState<string>(''); //현재 목차
    const [headingEls, setHeadingEls] = useState<Element[]>([]); //관찰 대상

    useEffect(() => {
        const observer = getIntersectionObserver(setCurrentInnerHTML);
        const headingElements = Array.from(document.querySelectorAll('h1, h2'));
        setHeadingEls(headingElements);
        headingElements.map((header) => {
            observer.observe(header);
        });
    }, []);
    console.log(currentInnerHTML)

    return (
        <ContentWrapper>
            <MarkdownRenderer dangerouslySetInnerHTML={{ __html: html }} />
            <Toc>
                <Toc2>
            {headingEls.map((a) => {
                let HighLight: boolean = false;
                if (currentInnerHTML == a.innerHTML){
                    HighLight = true;
                }

                if (a.tagName == 'H1'){
                    return <H1 HighLight={HighLight}>{a.innerHTML}</H1>
                }
                else if (a.tagName == 'H2') {
                    return <H2 HighLight={HighLight}>{a.innerHTML}</H2>
                }
                }
            )}
                </Toc2>
            </Toc>
        </ContentWrapper>
    )
}

export default PostContent
```

h1과 h2 태그를 관찰 대상으로 정하였고, 현재 목차와 관찰 대상이 같으면 하이라이트를 주었다.
<br>
<br>

[Interaction Observer API를 좀 더 자세히 알고싶다면](https://heropy.blog/2019/10/27/intersection-observer/)




