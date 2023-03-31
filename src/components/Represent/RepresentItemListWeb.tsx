import React, {FunctionComponent, useState} from 'react'
import styled from '@emotion/styled'
import { Link } from 'gatsby'
import { GatsbyImage } from 'gatsby-plugin-image'
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";
import {faChevronLeft, faChevronRight} from "@fortawesome/free-solid-svg-icons";
import {PostListItemType, PostListType} from "../../types/PostItem.types";
import RepresentItem from "components/Represent/RepresentItem";

const RepresentWrapper = styled.div`
  display: flex;
`

const ProjectListWrapper = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-gap: 20px;
  width: 1020px;
  margin: 0 auto;
  padding: 32px 0 100px;
`

const RightArrow = styled.div`
  width: 60px;
  margin-top: 16%;
`

const LeftArrow = styled.div`
  width: 24px;
  margin-top: 16%;
  margin-left: 36px;
`

const represenDatas0: Array<string> = ["426df633-3c30-5fa1-b46c-873914beb57b", "b2c31e45-11e0-5a1e-bf0e-3ba5b007ed52"]
const represenDatas1: Array<string> = ["c996f946-3d0c-5aae-a7db-57b50d900e74", "bcf3467f-3644-5321-be98-6df0eb415193"]
const RepresentItemListWeb: FunctionComponent<PostListType> = function ({
                                                                               posts
                                                                           }) {
    const [number, setNumber] = useState(0);
    const onIncrease = () => {
        setNumber(number + 1);
    }
    const onDecrease = () => {
        setNumber(number - 1);
    }
    const represenDatas = number == 1 ? represenDatas0 : represenDatas1
    // @ts-ignore
    const newArray = posts.filter(function (element: PostListItemType) {
        return represenDatas.indexOf(element.node.id) > -1;
    });
    return (
        <RepresentWrapper>
            { number == 1 ?    <RightArrow onClick={onDecrease}><FontAwesomeIcon icon={faChevronLeft} size="2x"/></RightArrow> : <RightArrow></RightArrow>}

            <ProjectListWrapper>
                {posts.map(
                    ({
                         node: {
                             id,
                             fields: {slug},
                             frontmatter,
                         },
                     }: PostListItemType) => (
                        <RepresentItem {...frontmatter} link={slug} key={id}/>
                    ),
                )}
            </ProjectListWrapper>
            {number == 0 ?    <LeftArrow onClick={onIncrease}><FontAwesomeIcon icon={faChevronRight} size="2x"/></LeftArrow> : <LeftArrow></LeftArrow>}
        </RepresentWrapper>
    )
}

export default RepresentItemListWeb
