# 🧰 common-py-tools

研究用途の Python 実験・解析コードに共通して利用できるユーティリティ関数群をまとめたライブラリです。
FFT 処理、可視化、フィッティング、データ読み込みなど、科学技術計算に頻出する処理を簡単に行えるように設計されています。

---

## 📦 各モジュールの概要

### 🔹 [`fft_utils.py`](docs/fft_utils.md)
- 高速フーリエ変換（FFT）
- スペクトログラムの可視化とピーク抽出
- ゼロパディング・ゼロフィリング

### 🔹 [`utils.py`](docs/utils.md)
- インデックス検索・部分抽出（1D/2D）
- 関数フィッティング補助（`scipy.optimize.curve_fit`）
- テキスト/CSV/Pickle データの読み書き

### 🔹 `plot_utils.py`（未記述）
- グラフ描画のスタイル統一や定型処理の簡略化

### 🔹 `data_loader.py`（未記述）
- CSV/JSON/TSV などの読み込みインターフェース

### 🔹 `constants.py`（未記述）
- 物理定数やよく使う単位系の定義（例：c, h, k_B, eV）

---

## 🔧 インストール方法
このリポジトリは `git submodule` で使用することを前提としています：

```bash
git submodule update --init --recursive
```

`requirements.txt` に必要なパッケージを記述しているため、以下でインストール可能です：

```bash
pip install -r requirements.txt
```

---

## 🏗 ディレクトリ構成（例）

```
common-py-tools/
├── README.md
├── fft_utils.py
├── utils.py
├── plot_utils.py
├── data_loader.py
├── constants.py
├── docs/
│   ├── fft_utils.md
│   └── utils.md
```

---

## 📄 ドキュメント
- [fft_utils.md](docs/fft_utils.md): FFT・スペクトログラム
- [utils.md](docs/utils.md): フィッティング・配列抽出・IO

---

## 🧪 実験プロジェクトとの連携方法
このライブラリは、実験ごとに構成されるプロジェクト（例：`research-dev/experiments/YYYY-MM-DD_myexp/`）に `git submodule` として導入され、特定バージョンに固定されます。

```bash
cd common-py-tools
git checkout v1.3.0
```

固定後、親リポジトリで以下のようにバージョンを固定：

```bash
git add common-py-tools
git commit -m "Fix common-py-tools to v1.3.0"
git push
```

---

## ✨ その他
- `tag_common.sh` スクリプトを使えば、バージョンタグ・ブランチ作成と親リポジトリへの反映を自動化できます。
- 各関数に対するサンプル使用例や Jupyter Notebook 化も今後対応予定です。

---

## 👨‍🔬 Maintainer
- Hiroki Tsusaka ＠ 研究用

---

ご要望・バグ報告・機能追加の希望があればお気軽に Issue をお寄せください！
