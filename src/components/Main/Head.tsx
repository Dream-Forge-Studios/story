import React, { FunctionComponent } from 'react'
import styled from '@emotion/styled'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
    faHomeLgAlt,
    faBars,
} from '@fortawesome/free-solid-svg-icons'
import { useBreakpoint } from 'gatsby-plugin-breakpoints';
import {Link} from "gatsby";

const Wrapper = styled.div`
  display: flex;
  justify-content: space-between;
  width: 1020px;;
  height: 80px;
  margin: 0 auto;
  color: #2b76b9;
  @media (max-width: 1024px) {
    width: 100%;
    height: 55px;
    padding: 0 20px;
  }
`

const MenuWrapper = styled.div`
  display: flex;
  font-size: 16px;
  font-weight: 600;
  margin-top: 30px;
  margin-bottom: 20px;
  color: #2cacd1;
  @media (max-width: 1020px) {
    margin-top: 20px;
    font-size: 14px;
  }
`

const Logo = styled.div`
  font-size: 30px;
  position: relative;
  margin-top: 20px;
  margin-bottom: 10px;
  @media (max-width: 1020px) {
    font-size: 20px;
    margin-top: 15px;
  }
  @media (max-width: 720px) {
    margin-top: 15px;
    margin-bottom: 15px;
  }
`

const MenuTitle1 = styled.div`
`

const MenuTitle2 = styled.div`
  margin-left: 20px;
`

const MiddleTitle = styled.div`
  font-size: 20px;
  font-weight: 1000;
  margin: 10px;
  position: absolute;
  left: 45%;
  @media (max-width: 1536px) {
    left: 40%;
  }
  @media (max-width: 1020px) {
    font-size: 15px;
    left: 38%;
  }
  @media (max-width: 768px) {
    font-size: 15px;
    left: 36%;
  }
  @media (max-width: 720px) {
    font-size: 15px;
    left: 28%;
  }
  @media (max-width: 540px) {
    font-size: 15px;
    left: 36%;
  }
  @media (max-width: 414px) {
    font-size: 15px;
    left: 30%;
  }
  @media (max-width: 390px) {
    font-size: 15px;
    left: 29%;
  }
  @media (max-width: 375px) {
    font-size: 15px;
    left: 28%;
  }
  @media (max-width: 280px) {
    font-size: 13px;
    left: 25%;
  }
`

const MiddleFirst = styled.div`
  display: flex;
  justify-content : center;
  @media (max-width: 1020px) {
    font-size: 15px;
    left: 35%;
  }
  @media (max-width: 280px) {
    font-size: 13px;
    left: 28%;
  }
`

const MiddleTwo = styled.div`
  display: flex;
  justify-content : center;
`

const MenuBar = styled.div`
  font-size: 20px;
  margin-top: 15px;
  margin-bottom: 15px;
`
const Head: FunctionComponent = function ({

                                                                     }) {
    const breakpoints = useBreakpoint();
    return (
            <Wrapper>
                <Logo>
                    <Link to={'/'}>
                <FontAwesomeIcon icon={faHomeLgAlt}  size="1x" />
                    </Link>
                </Logo>
                <MiddleTitle>
                    <MiddleFirst>Coding makes</MiddleFirst>
                    <MiddleTwo>my dream come true</MiddleTwo>
                </MiddleTitle>
                {       !breakpoints.sm ?         <MenuWrapper>
                    <MenuTitle1>프로젝트</MenuTitle1>
                    <MenuTitle2>블로그</MenuTitle2>
                    <MenuTitle2>소개</MenuTitle2>
                </MenuWrapper> :
                    <MenuBar>
                    <FontAwesomeIcon icon={faBars} color={'#2b76b9'} size="1x" />
                    </MenuBar>}

            </Wrapper>
    )
}

export default Head
