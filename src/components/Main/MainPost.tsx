import React, {FunctionComponent, useState, Component  } from "react";
import styled from "@emotion/styled";
import {PostListType} from "../../types/PostItem.types";
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {
    faChevronRight,
} from '@fortawesome/free-solid-svg-icons'
import {Link} from "gatsby";

const RepresentativeProjectWrapper = styled.div`
  width: 1020px;
  margin: 0 auto;

  @media (max-width: 1020px) {
    width: 100%;
    margin-top: 50px;
    padding: 0 20px;
  }
  @media (max-width: 720px) {
    width: 100%;
    margin: 30px auto 0;
    padding: 0 20px;
  }
`

const TitleWrapper = styled.div`
  display: flex;
  justify-content: space-between;
  width: 1020px;
  @media (max-width: 1020px) {
    width: 100%;
    margin-left: 0px;
  }
`

const Title = styled.div`
  margin-top: 5px;
  font-size: 30px;
  font-weight: 700;

  @media (max-width: 1020px) {
    font-size: 25px;
  }
  @media (max-width: 720px) {
    font-size: 22px;
  }
`


const MoreWrapper3 = styled.div`
  font-size: 14px;
  color: grey;
  display: flex;
  margin-top: 28px;
  @media (max-width: 1024px) {
    font-size: 14px;
  }
  @media (max-width: 720px) {
    font-size: 14px;
    margin-top: 18px;
  }
`

const MoreText = styled.div`
  margin-right: 7px;
  margin-top: -3px;
  font-size: 15px;

  @media (max-width: 1024px) {
    font-size: 14px;
  }
`

const MainPost: FunctionComponent<PostListType> = function ({

                                                                         }) {
    return (
        <RepresentativeProjectWrapper>
            <TitleWrapper>
                <Title>Representative Project</Title>
                {/*<MoreWrapper3> <MoreText>더보기</MoreText> <FontAwesomeIcon icon={faChevronRight} size="1x"/>*/}
                {/*</MoreWrapper3>*/}
            </TitleWrapper>
        </RepresentativeProjectWrapper>
    )
}

export default MainPost