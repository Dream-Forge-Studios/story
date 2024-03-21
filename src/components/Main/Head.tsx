import React, { FunctionComponent } from 'react'
import styled from '@emotion/styled'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
    faHomeLgAlt,
    faBars,
} from '@fortawesome/free-solid-svg-icons'
import { useBreakpoint } from 'gatsby-plugin-breakpoints';
import {Link} from "gatsby";

const Header = styled.div`
  position: fixed;
  width: 100%;
  height: 60px;
  margin: 0 auto;
  left: 0;
  right: 0;
  //background-color: #21212150;
  opacity: 0.4;
  //@media (max-width: 1024px) {
  // width: 100%;
  // height: 55px;
  // padding: 0 20px;
  //}
`

const Wrapper = styled.div`
  z-index: 1;
  position: fixed;
  margin: 0 auto;
  left: 0;
  right: 0;
  height: 60px;
  width: 1020px;
  display: flex;
  justify-content: space-between;
  margin: 0 auto;
  color: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(7px);
  @media (max-width: 1020px) {
    width: 100%;
    padding: 0px 20px;
  }
`

const MenuWrapper = styled.div`
  display: flex;
  font-size: 16px;
  font-weight: 600;
  margin-top: 20px;
  margin-bottom: 20px;
  color: rgba(0, 0, 0, 0.8);
  @media (max-width: 1020px) {
    margin-top: 20px;
    font-size: 14px;
  }
`

const Logo = styled.div`
  font-size: 24px;
  position: relative;
  margin-top: 10px;
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

const MenuTitle2 = styled(Link)`
  margin-left: 20px;
`

// const MenuBar = styled.div`
//   font-size: 20px;
//   margin-top: 15px;
//   margin-bottom: 15px;
// `
const Head: FunctionComponent = function ({

                                                                     }) {
    // const breakpoints = useBreakpoint();
    return (
        <div>
            <Wrapper>
                <Logo>
                    <Link to={'/'}>
                <FontAwesomeIcon icon={faHomeLgAlt}  size="1x" />
                </Link>
                </Logo>
                {/*{       !breakpoints.sm ?         <MenuWrapper>*/}
                {/*    <MenuTitle1>Project</MenuTitle1>*/}
                {/*    <MenuTitle2>Blog</MenuTitle2>*/}
                {/*    <MenuTitle2>Introduce</MenuTitle2>*/}
                {/*</MenuWrapper> :*/}
                {/*    <MenuBar>*/}
                {/*    <FontAwesomeIcon icon={faBars} color={'#2b76b9'} size="1x" />*/}
                {/*    </MenuBar>}*/}
                <MenuWrapper>
                    <MenuTitle1>Project</MenuTitle1>
                    <MenuTitle2  to={`/blog`}>Blog</MenuTitle2>
                    <MenuTitle2>Introduce</MenuTitle2>
                  </MenuWrapper>
            </Wrapper>
            <Header></Header>
        </div>
    )
}

export default Head
