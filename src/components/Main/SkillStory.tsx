import React, {FunctionComponent  } from "react";
import styled from "@emotion/styled";
import {PostListType} from "../../types/PostItem.types";
import {Link} from "gatsby";

import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';

const SkillStoryWrapper = styled.div`
  //width: 1020px;;
  margin: 80px auto 0;

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

const CategoryWrapper = styled.div`
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  margin: 60px 0px;
`

const Category = styled(Link)`
  align-items: center;
  padding: 10px 20px;
  margin-right: 10px;
  margin-bottom: 10px;
  background-color: #ffffff; // White background
  border-radius: 20px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); // subtle shadow
  cursor: pointer;
  transition: all 0.3s ease;
  width: 230px;
  
  &:hover {
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); // slightly more prominent shadow on hover
  }

  @media (max-width: 720px) {
    padding: 8px 15px;
  }
`

const CategoryName = styled.div`
  font-size: 24px;
  font-weight: 700; // bold text
  text-align: left;

  @media (max-width: 720px) {
    font-size: 16px;
  }
`

const CategoryText = styled.div`
  font-size: 14px;
  text-align: left;
  color: rgb(122, 122, 122);
  margin-top: 10px;

  @media (max-width: 720px) {
    font-size: 16px;
  }
`
const SkillStory: FunctionComponent<PostListType> = function ({

                                                                          }) {
    return (
        <SkillStoryWrapper>
            <TitleWrapper>
                <Title>Skill Story</Title>
            </TitleWrapper>
            <CategoryWrapper>
                <Category to={`/blog/?category=LLM`}>
                        <CategoryName>
                            LLM
                        </CategoryName>
                    <CategoryText>법률 domain LLM을 개발하고 있습니다.</CategoryText>
                    </Category>
                <Category to={`/blog/?category=WEB/APP`}>
                    <CategoryName>
                        WEB/APP
                    </CategoryName>
                    <CategoryText>상상하던 서비스들을 실현하고 있습니다.</CategoryText>
                </Category>
                <Category to={`/blog/?category=CV`}>
                    <CategoryName>
                        CV
                    </CategoryName>
                    <CategoryText>Object Detection에 관한 논문을 정리하였습니다.</CategoryText>
                </Category>
                <Category to={`/blog/?category=AI BASIC`}>
                    <CategoryName>
                        AI BASIC
                    </CategoryName>
                    <CategoryText>AI 기초 내용을 정리하였습니다.</CategoryText>
                </Category>
            </CategoryWrapper>
     </SkillStoryWrapper>
    )
}

export default SkillStory