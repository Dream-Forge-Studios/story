import React, { FunctionComponent } from 'react'
import styled from '@emotion/styled'
import { Link } from 'gatsby'
import { GatsbyImage } from 'gatsby-plugin-image'
import { PostFrontmatterType } from 'types/PostItem.types'

type PostItemProps = PostFrontmatterType & { link: string }

const PostItemWrapper = styled(Link)`
  display: flex;
  flex-direction: column;
  transition: 0.3s box-shadow;
  cursor: pointer;
  border-radius: 10px 10px 10px 10px;
`

const ThumbnailImage = styled(GatsbyImage)`
  width: 100%;
  height: 260px;
  border-radius: 10px 10px 10px 10px;
  @media (max-width: 720px) {
    width: 100%;
    height: 220px;
  }
  @media (max-width: 360px) {
    width: 100%;
    height: 180px;
  }
`

const PostItemContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 15px;
`
const Title = styled.div`
  display: -webkit-box;
  overflow: hidden;
  margin-bottom: 3px;
  text-overflow: ellipsis;
  white-space: normal;
  overflow-wrap: break-word;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  font-size: 20px;
  font-weight: 800;
`

const Summary = styled.div`
  display: -webkit-box;
  overflow: hidden;
  margin-top: auto;
  text-overflow: ellipsis;
  white-space: normal;
  overflow-wrap: break-word;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  font-size: 14px;
  font-weight: 600;
  margin-top: 12px;
  color: gray;
  @media (max-width: 720px) {
    font-size: 12px;
  }
`

const RepresentItem: FunctionComponent<PostItemProps> = function ({
                                                                 title,
                                                                      summary,
                                                                 thumbnail: {
                                                                     childImageSharp: { gatsbyImageData },
                                                                 },
                                                                 link,
                                                             }) {
    return (
        <PostItemWrapper to={link}>
            <ThumbnailImage image={gatsbyImageData} alt="Post Item Image" />
            <PostItemContent>
                <Title>{title}</Title>
                <Summary>{summary}</Summary>
            </PostItemContent>
        </PostItemWrapper>
    )
}

export default RepresentItem