const myCustomQueries = {
  xs: '(max-width: 320px)',
  sm: '(max-width: 720px)',
  md: '(max-width: 1024px)',
  l: '(max-width: 1536px)',
  represent2: '(max-width: 1030px)',
  represent1: '(max-width: 660px)',
};

module.exports = {
  siteMetadata: {
    title: `주니어 개발자의 개발 블로그`,
    description: `주니어 개발자로서의 저를 표현한 블로그입니다.`,
    author: `Hyun`,
    siteUrl: 'https://my-website-link.com', // 배포 후 변경 예정
  },
  plugins: [
    {
      resolve: "gatsby-plugin-breakpoints",
      options: {
        queries: myCustomQueries,
      },
    },
    `gatsby-plugin-fontawesome-css`,
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [
          `gatsby-remark-emoji`,  // <-- this line adds emoji
        ]
      }
    },
    {
      resolve: 'gatsby-plugin-typescript',
      options: {
        isTSX: true,
        allExtensions: true,
      },
    },
    `gatsby-plugin-emotion`,
    `gatsby-plugin-react-helmet`,
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `contents`,
        path: `${__dirname}/contents`,
      },
    },
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `images`,
        path: `${__dirname}/static`,
      },
    },
    `gatsby-transformer-sharp`,
    `gatsby-plugin-image`,
    {
      resolve: `gatsby-plugin-sharp`,
      options: {
        defaults: {
          formats: ['auto', 'webp'],
          quality: 100,
          placeholder: 'blurred',
        }
      }
    },
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [
          {
            resolve: 'gatsby-remark-smartypants',
            options: {
              dashes: 'oldschool',
            },
          },
          {
            resolve: 'gatsby-remark-prismjs',
            options: {
              classPrefix: 'language-',
            },
          },
          {
            resolve: 'gatsby-remark-images',
            options: {
              maxWidth: 768,
              quality: 100,
              withWebp: true,
            },
          },
          {
            resolve: 'gatsby-remark-copy-linked-files',
            options: {},
          },
          {
            resolve: 'gatsby-remark-external-links',
            options: {
              target: '_blank',
              rel: 'nofollow',
            },
          },
          {
            resolve: 'gatsby-plugin-canonical-urls',
            options: {
              siteUrl: 'https://my-website.com/',
              stripQueryString: true,
            },
          },
          'gatsby-plugin-sitemap',
          {
            resolve: 'gatsby-plugin-robots-txt',
            options: {
              policy: [{ userAgent: '*', allow: '/' }],
            },
          },
        ],
      },
    },
  ],
};