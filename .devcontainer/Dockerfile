FROM python:3.11.8

# パッケージの追加とタイムゾーンの設定
RUN apt-get update && apt-get install -y \
    tzdata \
    vim \
    git \
    git-secrets \
    mecab libmecab-dev mecab-ipadic-utf8 \
&&  ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
&&  apt-get clean \
&&  rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo

# pip, jupyterlab
RUN python3 -m pip install --upgrade pip \
&&  pip install --no-cache-dir \
    jupyterlab

# Python,データ分析でつかうパッケージ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
