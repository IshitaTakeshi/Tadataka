# Tadataka

[![Build Status](https://travis-ci.org/IshitaTakeshi/Tadataka.svg?branch=develop)](https://travis-ci.org/IshitaTakeshi/Tadataka)
[![codecov](https://codecov.io/gh/IshitaTakeshi/Tadataka/branch/develop/graph/badge.svg)](https://codecov.io/gh/IshitaTakeshi/Tadataka)

[Dataset](https://drive.google.com/drive/folders/1gDXYusi9ilMIQO5aUUHbV3jqBOCdrG3W)

## 使い方

### インストールせずに使う場合

1. リポジトリをcloneする

```
git clone git@github.com:IshitaTakeshi/Tadataka.git
cd Tadataka
git checkout nikkei
```

2. データセットを `datasets/` ディレクトリに展開する

```
unzip <path to saba.zip> -d datasets
```

3. PYTHONPATHの設定

`Tadataka` ディレクトリにいることを確認し，`PYTHONPATH` を設定する

```
PYTHONPATH=$(pwd)
```

4. 実行

```
python3 nikkei/explanation.py
python3 nikkei/vo.py
```
