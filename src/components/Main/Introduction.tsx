import React, { FunctionComponent } from 'react'
import styled from '@emotion/styled'
import { IGatsbyImageData } from 'gatsby-plugin-image'
import ProfileImage from 'components/Main/ProfileImage'

type IntroductionProps = {
    profileImage: IGatsbyImageData
}

const Background = styled.div`
  width: 100%;
  background-image: linear-gradient(60deg, #3d3393 0%, #2b76b9 37%, #2cacd1 65%, #35eb93 100%);
  color: #ffffff;
`



const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: flex-start;
  width: 1020px;
  height: 320px;
  margin: 0 auto;

  @media (max-width: 1020px) {
    width: 100%;
    height: 240px;
    padding: 0 20px;
  }
  @media (max-width: 720px) {
    width: 100%;
    height: 220px;
    padding: 0 20px;
  }
`

const SubTitle = styled.div`
  font-size: 20px;
  font-weight: 500;

  @media (max-width: 1020px) {
    font-size: 15px;
  }
  @media (max-width: 720px) {
    font-size: 13px;
  }
`

const Title = styled.div`
  margin-top: 5px;
  font-size: 35px;
  font-weight: 700;

  @media (max-width: 1020px) {
    font-size: 25px;
  }
  @media (max-width: 720px) {
    font-size: 20px;
  }
`
const Introduction: FunctionComponent<IntroductionProps> = function ({
                                                                         profileImage,
                                                                     }) {
    return (
        <Background>
            <Wrapper>
                <ProfileImage profileImage={profileImage} />
                <div>
                    <SubTitle>Nice to Meet You,</SubTitle>
                    <Title>I'm a developer who realizes imagination.</Title>
                </div>
            </Wrapper>
        </Background>
    )
}

export default Introduction

