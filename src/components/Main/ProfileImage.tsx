import React, { FunctionComponent } from 'react'
import styled from '@emotion/styled'
import { GatsbyImage, IGatsbyImageData } from 'gatsby-plugin-image'

type ProfileImageProps = {
    profileImage: IGatsbyImageData
}

const ProfileImageWrapper = styled(GatsbyImage)`
  width: 120px;
  height: 120px;
  margin-bottom: 30px;
  border-radius: 50%;

  @media (max-width: 1020px) {
    width: 80px;
    height: 80px;
  }
  @media (max-width: 720px) {
    width: 80px;
    height: 80px;
    margin-bottom: 20px;
  }
`

const ProfileImage: FunctionComponent<ProfileImageProps> = function ({
                                                                         profileImage,
                                                                     }) {
    return <ProfileImageWrapper image={profileImage} alt="Profile Image" />
}

export default ProfileImage