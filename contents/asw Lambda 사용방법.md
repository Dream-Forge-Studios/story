---
date: '2023-02-16'
title: 'aws Lambda 사용방법'
categories: ['Etc']
summary: 'aws Lambda에서 원하는 파이썬 코드를 일정 시간이 되면 자동으로 실행되게 해보자.'
thumbnail: './test.png'
---

<br>

<br>

저는 쿠팡 api로 가격데이터를 매일 가져와서 가격의 흐름을 확인할 수 있는 웹 서비스를 기획하고 있습니다.

<br>

그전에 로컬에서 해당 코드를 매일 실행시키는 것은 번거로운 과정이니, aws Lambda를 통해 매일 일정한 시간에 api를 호출하여 s3에 엑셀로 정리하는 기능을 구현해보려고 합니다.

<div id="aws Lambda란?"></div>

## aws Lambda란?

Amazon Web Services(AWS)에서 제공하는 서버리스 컴퓨팅 서비스입니다. 이 서비스는 개발자가 서버 관리에 신경 쓰지 않고 코드 실행에 집중할 수 있게 해주는 플랫폼입니다.

<br>

코드를 업로드하고 실행 조건(트리거)을 설정하기만 하면 AWS가 나머지를 관리합니다.

<br>

사용한 컴퓨팅 자원에 대해서만 비용을 지불합니다. 요청 수와 실행 시간에 따라 요금이 책정되며, 코드가 실행되지 않을 때는 비용이 발생하지 않습니다.

<div id="사용 방법"></div>

## 사용 방법

aws Lambda 대시보드에서 함수를 생성하고 코드 입력창에 단순히 입력하고 Depoly하여 코드를 반영하고 Test를 하면 간단히 실행해볼 수 있습니다.

<br>

**라이브러리 추가**

<br>

코드를 추가하는 것은 어렵지 않으나, 라이브러리를 추가하는 경우 처음 접하는 사람에게는 어려울 수 있습니다.

```
#python 디렉토리 생성
mkdir python

# python 패키지 설치 (amazon linux 에서는 기본 pip 명령어가 python 2.X 이므로 pip3를 입력해야 함)
$ pip3 install -t ./python pytz
```

위와 같이 AWS Lambda의 기본 환경에 이미 포함되어 있는 라이브러리 외 필요한 라이브러리를 python 디렉토리에 설치합니다.

<br>

한가지 주의해야할 점은 AWS Lambda 환경은 Linux 기반입니다. 그래서 pandas 같이 window, mac, linux 버전이 다른 라이브러리는 linux 버전 라이브러리를 설치해야합니다.

<br>

방법은 https://pypi.org/project/pandas/#files 해당 링크에 들어가 자신의 버전에 맞는 파일을 다운로드 후 압축을 풀어 python 디렉토리에 넣어주면 됩니다.

<br>

또한 requests 라이브러리를 사용하는 과정에서 오류가 났는데 requests는

```
$ pip3 install -t ./python requests==2.29.0 urllib3==1.26.16
```

위와 같은 버전으로 설치해야 AWS Lambda의 환경에서 오류가 나지 않습니다.

<br>

이렇게 라이브러리 설치가 모두 끝났다면, python 디렉토리가 포함되도록 .zip으로 압축합니다.

<br>

압축한 파일은 Lambda > 계층 경로로 가 계층을 생성 후, 다시 함수 화면으로 가 Add a layer 해주면 됩니다.

<div id="S3 연결하기"></div>

## S3 연결하기

AWS Lambda 함수에서 S3에 읽고 쓰기 권한을 가져오는 방법입니다.

<br>

Amazon S3에 버킷 생성 후 권한 탭의 버킷 정책에

```
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::123456789012:service-role/lambda-role"
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    }
  ]
}
```

위와 같이 입력하면 됩니다.

<br>

`123456789012` 부분은 AWS 계정 ID이며, `lambda-role`은 함수 > 구성탭 > 권한의 역할 이름, `your-bucket-name`은 대상 S3 버킷의 이름입니다.




