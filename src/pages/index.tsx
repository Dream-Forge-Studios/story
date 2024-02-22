import React, { FunctionComponent, useState, useEffect } from 'react';
import Introduction from 'components/Main/Introduction'
import { graphql } from 'gatsby'
import { IGatsbyImageData } from 'gatsby-plugin-image'
import Template from 'components/Common/Template'
import SkillStory from "components/Main/SkillStory";
import MainPost from "components/Main/MainPost";

type IndexPageProps = {
    location: {
        search: string
    }
    data: {
        site: {
            siteMetadata: {
                title: string
                description: string
                siteUrl: string
            }
        }
        file: {
            childImageSharp: {
                gatsbyImageData: IGatsbyImageData
            }
            publicURL: string
        }
    }
}



const IndexPage: FunctionComponent<IndexPageProps> = function ({
                                                                   data: {
                                                                       site: {
                                                                           siteMetadata: { title, description, siteUrl },
                                                                       },
                                                                       file: {
                                                                           childImageSharp: { gatsbyImageData },
                                                                           publicURL,
                                                                       },
                                                                   },
                                                               }) {

    // 화면 너비 상태를 관리하는 state
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        // 화면 너비를 검사하는 함수
        const handleResize = () => {
            setIsMobile(window.innerWidth < 720);
        };

        // 컴포넌트가 마운트될 때 이벤트 리스너를 추가
        window.addEventListener('resize', handleResize);

        // 초기 화면 너비 검사 실행
        handleResize();

        // 컴포넌트가 언마운트될 때 이벤트 리스너를 제거
        return () => window.removeEventListener('resize', handleResize);
    }, []);


    return (
        <Template
            title={title}
            description={description}
            url={siteUrl}
            image={publicURL}
        >
            <Introduction profileImage={gatsbyImageData} />
            {!isMobile && <SkillStory/>}
            <MainPost/>
        </Template>
    )
}

export default IndexPage

export const getPostList = graphql`
  query getPostList {
    site {
      siteMetadata {
        title
        description
        siteUrl
      }
    }
    file(name: { eq: "profile-image" }) {
      childImageSharp {
        gatsbyImageData(width: 120, height: 120)
      }
      publicURL
    }
  }
`;
