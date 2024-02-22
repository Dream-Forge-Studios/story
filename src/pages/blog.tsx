import React, {FunctionComponent, useMemo} from 'react'
import styled from '@emotion/styled'
import GlobalStyle from 'components/Common/GlobalStyle'
import Footer from 'components/Common/Footer'
import CategoryList from 'components/Main/CategoryList'
import PostList, { PostType } from 'components/Main/PostList'
import Head from "components/Main/Head";
import {graphql} from "gatsby";
import {PostListItemType} from "../types/PostItem.types";
import {IGatsbyImageData} from "gatsby-plugin-image";
import queryString, {ParsedQuery} from "query-string";

type BlogPageProps = {
    location: {
        search: string
    }
    data: {
        allMarkdownRemark: {
            edges: PostListItemType[]
        }
        file: {
            childImageSharp: {
                gatsbyImageData: IGatsbyImageData
            }
        }
    }
}

export type CategoryListProps = {
    selectedCategory: string
    categoryList: {
        [key: string]: number
    }
}


const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
`

const BlogPage: FunctionComponent<BlogPageProps> = function ({
                                                                 location: { search },
                                                                 data: {
                                                                     allMarkdownRemark: { edges },
                                                                     file: {
                                                                         childImageSharp: { gatsbyImageData },
                                                                     },
                                                                 },
                                                             }) {

    const parsed: ParsedQuery<string> = queryString.parse(search)
    const selectedCategory: string =
        typeof parsed.category !== 'string' || !parsed.category
            ? 'All'
            : parsed.category

    const categoryList = useMemo(
        () =>
            edges.reduce(
                (
                    list: CategoryListProps['categoryList'],
                    {
                        node: {
                            frontmatter: { categories },
                        },
                    }: PostType,
                ) => {
                    categories.forEach(category => {
                        if (list[category] === undefined) list[category] = 1;
                        else list[category]++;
                    });

                    list['All']++;

                    return list;
                },
                { All: 0 },
            ),
        [],
    )
    return (
        <Container>
            <GlobalStyle />
            <Head />
            <CategoryList
                selectedCategory={selectedCategory}
                categoryList={categoryList}
            />
            <PostList selectedCategory={selectedCategory} posts={edges} />
            <Footer />
        </Container>
    )
}

export default BlogPage

export const getPostList = graphql`
  query getPostList {
    site {
      siteMetadata {
        title
        description
        siteUrl
      }
    }
    allMarkdownRemark(
      sort: { order: DESC, fields: [frontmatter___date, frontmatter___title] }
    ) {
      edges {
        node {
          id
          fields {
            slug
          }
          frontmatter {
            title
            summary
            date(formatString: "YYYY.MM.DD.")
            categories
            thumbnail {
              childImageSharp {
                gatsbyImageData(width: 300, height: 300)
              }
            }
          }
        }
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
