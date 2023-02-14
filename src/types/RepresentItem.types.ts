import { IGatsbyImageData } from 'gatsby-plugin-image'

export type RepresentFrontmatterType = {
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

export type RepresentItemType = {
    node: {
        id: string
        fields: {
            slug: string
        }
        frontmatter: RepresentFrontmatterType
    }
}