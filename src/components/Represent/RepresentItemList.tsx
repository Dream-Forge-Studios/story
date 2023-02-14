import React, {FunctionComponent, useState} from 'react'
import styled from '@emotion/styled'
import {Link} from 'gatsby'
import {PostListItemType, PostListType} from "../../types/PostItem.types";
import RepresentItem from "components/Represent/RepresentItem";

import Slider from 'react-slick';
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';
import {faChevronLeft, faChevronRight} from "@fortawesome/free-solid-svg-icons";
import {FontAwesomeIcon} from "@fortawesome/react-fontawesome";
import {useBreakpoint} from 'gatsby-plugin-breakpoints';

const RepresentWrapper = styled.div`
  margin: 0 auto;
  padding: 32px 0 100px;
  width: 1024px;
  @media (max-width: 1020px) {
    width: 100%;
  }
  @media (max-width: 720px) {
    width: 100%;
    padding: 14px 0 100px;
  }
`


const RightArrow = styled.div`
  position: absolute;
  top: 150px;
  right: -40px;
`

const LeftArrow = styled.div`
  position: absolute;
  top: 150px;
  left: -40px;
`

function NextArrow(props: any) {

    const {onClick, onIncrease, number} = props;
    if (number == 0) {
        return (
            <RightArrow onClick={onIncrease}>
                <FontAwesomeIcon icon={faChevronRight} size="1x" onClick={onClick}/>
            </RightArrow>);
    } else {
        return <RightArrow></RightArrow>
    }
}

function PrevArrow(props: any) {
    const {onClick, onDecrease, number} = props;

    if (number == 1) {
        return (<LeftArrow onClick={onDecrease}>
            <FontAwesomeIcon icon={faChevronLeft} size="1x" onClick={onClick}/>
        </LeftArrow>);
    } else {
        return <LeftArrow></LeftArrow>
    }
}

const represenDatas: Array<string> = ["426df633-3c30-5fa1-b46c-873914beb57b", "b2c31e45-11e0-5a1e-bf0e-3ba5b007ed52", "c996f946-3d0c-5aae-a7db-57b50d900e74", "bcf3467f-3644-5321-be98-6df0eb415193"]
const RepresentItemList: FunctionComponent<PostListType> = function ({
                                                                               posts
                                                                           }) {

    const [number, setNumber] = useState(0);
    const onIncrease = () => {
        setNumber(number + 1);
    }
    const onDecrease = () => {
        setNumber(number - 1);
    }

    const breakpoints = useBreakpoint();

    const settings = {
        dots: breakpoints.md ? true : false,
        infinite: false,
        arrow: breakpoints.md ? false : true,
        speed: 500,
        slidesToShow: breakpoints.sm ? 1 : 2,
        slidesToScroll: breakpoints.sm ? 1 : 2,
        nextArrow: breakpoints.md ? '' : <NextArrow onIncrease={onIncrease} number={number}/>,
        prevArrow: breakpoints.md ? '' : <PrevArrow onDecrease={onDecrease} number={number}/>,

    }

    const newArray = posts.filter(function (element: PostListItemType) {
        return represenDatas.indexOf(element.node.id) > -1;
    });

    return (
        <RepresentWrapper>
            <StyledSlider {...settings}>
                {newArray.map(
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
            </StyledSlider>
        </RepresentWrapper>
    )
}

export default RepresentItemList

const StyledSlider = styled(Slider)`
  .slick-slide > div {
    // 자식 안에 div
    margin: 10px;
    box-sizing: border-box;
  }
`;