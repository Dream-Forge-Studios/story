import { IGatsbyImageData } from 'gatsby-plugin-image'

export type PostFrontmatterType = {
    title: string
    date: string
    categories: string[]
    summary: string
    thumbnail: {
        childImageSharp: {
            gatsbyImageData: IGatsbyImageData
        }
        publicURL: string
    }
}

export type PostListItemType = {
    node: {
        id: string
        fields: {
            slug: string
        }
        frontmatter: PostFrontmatterType
    }
}

export type PostListType = {
    posts: PostListItemType[],
}

export interface PostPageItemType {
    node: {
        html: string
        frontmatter: {
            title: string
            summary: string
            date: string
            categories: string[]
            thumbnail: {
                childImageSharp: {
                    gatsbyImageData: any // Gatsby 이미지 데이터 타입에 맞게 조정
                }
                publicURL: string
            }
        }
    }
}
