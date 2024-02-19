import React, { FunctionComponent } from 'react'
import styled from '@emotion/styled'
import GlobalStyle from 'components/Common/GlobalStyle'
import Footer from 'components/Common/Footer'
import CategoryList from 'components/Main/CategoryList'
import Introduction from 'components/Main/Introduction'
import PostList from 'components/Main/PostList'
import Head from "components/Main/Head";
const CATEGORY_LIST = {
    LLM: 5,
    CV: 3,
}

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
`

const BlogPage: FunctionComponent = function () {
    return (
        <Container>
            <GlobalStyle />
            <Head />
            <CategoryList selectedCategory="Web" categoryList={CATEGORY_LIST} />
            {/*<PostList />*/}
            <Footer />
        </Container>
    )
}

export default BlogPage