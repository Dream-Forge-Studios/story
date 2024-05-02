import React, {FunctionComponent, useState, useEffect} from 'react'
import styled from '@emotion/styled'
import {getIntersectionObserver} from '../../lib/observer'
import {css} from "@emotion/react";

interface PostContentProps {
    html: string
}

const MarkdownRenderer = styled.div`
  // Renderer Style
  display: flex;
  flex-direction: column;
  width: 800px;
  margin: 0 auto;
  padding: 100px 0;
  word-break: break-all;

  // Markdown Style
  line-height: 1.8;
  font-size: 16px;
  font-weight: 400;

  // Apply Padding Attribute to All Elements
  p {
    padding: 3px 0;
  }

  // Adjust Heading Element Style
  h1,
  h2,
  h3 {
    font-weight: 800;
    margin-bottom: 30px;
  }

  * + h1,
  * + h2,
  * + h3 {
    margin-top: 80px;
  }

  hr + h1,
  hr + h2,
  hr + h3 {
    margin-top: 0;
  }

  h1 {
    font-size: 30px;
  }

  h2 {
    font-size: 25px;
  }

  h3 {
    font-size: 20px;
  }

  // Adjust Quotation Element Style
  blockquote {
    margin: 30px 0;
    padding: 5px 15px;
    border-left: 2px solid #000000;
    font-weight: 800;
  }

  // Adjust List Element Style
  ol,
  ul {
    margin-left: 20px;
    padding: 30px 0;
  }

  // Adjust Horizontal Rule style
  hr {
    border: 1px solid #000000;
    margin: 100px 0;
  }

  // Adjust Link Element Style
  a {
    color: #4263eb;
    text-decoration: underline;
  }

  // Adjust Code Style
  pre[class*='language-'] {
    margin: 30px 0;
    padding: 15px;
    font-size: 15px;

    ::-webkit-scrollbar-thumb {
      background: rgba(255, 255, 255, 0.5);
      border-radius: 3px;
    }
  }

  code[class*='language-'],
  pre[class*='language-'] {
    tab-size: 2;
  }

  // Markdown Responsive Design
  @media (max-width: 800px) {
    width: 100%;
    padding: 80px 20px;
    line-height: 1.6;
    font-size: 14px;

    h1 {
      font-size: 23px;
    }

    h2 {
      font-size: 20px;
    }

    h3 {
      font-size: 17px;
    }

    img {
      width: 100%;
    }

    hr {
      margin: 50px 0;
    }
  }
`

const H1 = styled.div<{HighLight: boolean}>`
  padding: 6px 0px;
  opacity: ${props => props.HighLight ? "1" : "0.5"};
  font-size: 17px;
  color: ${props => props.HighLight ? "#8ddb8c" : "#000"};

  &:hover{
    color: -webkit-link;
    cursor: pointer;
    text-decoration: underline;
  }
`

const H2 = styled.div<{HighLight: boolean}>`
  padding: 6px 0px;
  border-left: 1px solid #000;
  color: ${props => props.HighLight ? "#8ddb8c" : "#000"};
  font-size: 17px;
  opacity: ${props => props.HighLight ? "1" : "0.5"};
  margin-left: 5px;
  padding-left: 10px;

  &:hover{
    color: -webkit-link;
    cursor: pointer;
    text-decoration: underline;
  }
`

const ContentWrapper = styled.div`
  display: flex;
  //display: block;
  //margin: 0 auto;
`

interface TocProps {
    top: number;
}

const Toc = styled.div<TocProps>`
  display: block;
  position: fixed;
  width: 500px;
  //margin: -200px auto;
  margin: ${({ top }) => top}px auto;
  left: 0;
  right: 0;
`

const Toc2 = styled.div`
  margin-left: 700px;
  width: 300px;
`

const MenuDivStyle = css`
  width: 150px;
`
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
    const initialTop = 100;  // 초기 위치를 100px로 설정합니다.
    const [tocTop, setTocTop] = useState(initialTop);

    useEffect(() => {
        const handleScroll = () => {
            const scrolledAmount = window.scrollY;
            const freezePoint = 300; // 스크롤 고정 지점을 300px로 설정합니다.
            let newTop = Math.max(0, initialTop - scrolledAmount); // 스크롤에 따라 top 값을 감소시킵니다.

            if (scrolledAmount >= freezePoint) {
                newTop = -180; // 스크롤이 freezePoint 이상이면 top 값을 0으로 고정합니다.
            }

            setTocTop(newTop);
        };

        window.addEventListener('scroll', handleScroll);

        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, []);

    return (
        <ContentWrapper>
            <MarkdownRenderer dangerouslySetInnerHTML={{ __html: html }} />
            <Toc top={tocTop}>
                <Toc2>
            {headingEls.map((a) => {
                let HighLight: boolean = false;
                if (currentInnerHTML == a.innerHTML){
                    HighLight = true;
                }
                if (a.tagName == 'H1'){
                    return <H1 HighLight={HighLight}><a href={"#" + a.innerHTML}><div css={MenuDivStyle}>{a.innerHTML}</div></a></H1>
                }
                else if (a.tagName == 'H2') {
                    return <H2 HighLight={HighLight}><a href={"#" + a.innerHTML}><div css={MenuDivStyle}>{a.innerHTML}</div></a></H2>
                }
                }
            )}
                </Toc2>
            </Toc>
        </ContentWrapper>
    )
}

export default PostContent
