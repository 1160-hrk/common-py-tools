# 📦 `fft_utils.py` - FFT・スペクトログラムユーティリティ集

このモジュールは、時間軸信号に対して高速フーリエ変換（FFT）や逆FFT、スペクトログラムの計算などを行う関数群を提供します。
研究用途や分光データの解析などに有用です。

---

## ✅ 主な関数一覧

### 🔹 `fft_with_freq(x, y)`
- FFT と対応する周波数軸を返す。

### 🔹 `fft_positive_freq(x, y)`
- 正の周波数成分のみ抽出した FFT 結果を返す。

### 🔹 `zero_padding(x, len0pd, connection='lr')`
- 配列の両端・片側にゼロパディングを追加。

### 🔹 `zero_filling(x, len0fl, connection='lr')`
- 指定した長さに合わせてゼロで埋める（`zero_padding` を利用）。

### 🔹 `ifft_with_time(freq, fft_y)`
- 周波数軸から逆FFTを行い、時間軸と復元信号を返す。

---

## 🔊 スペクトログラム関連

### 🔹 `spectrogram(x, y, T, ...)`
- スライディング窓と FFT を用いたスペクトログラム（通常版）。

### 🔹 `spectrogram_fast(x, y, T, ...)`
- `numpy` によるバッチFFTを用いた高速版。

### 🔹 `spectrogram_vectorized(x, y, T, ...)`
- ベクトル化による最速版（Numba不使用）。

#### パラメータ共通：
- `x`：時間軸
- `y`：信号
- `T`：ウィンドウサイズ（サンプル数 or 時間）
- `unit_T`：単位（`'index'` or `'x'`）
- `window_type`：ウィンドウ関数（例：hamming, blackman）
- `step`：スライディングステップ幅
- `return_max_index`：最大強度周波数のインデックスを返すかどうか

---

## 🧪 使用例（main ブロック内）
- ガウシアンパルスに GDD（群遅延分散）を与えた信号を生成
- FFT スペクトルを表示
- 通常版と高速版でスペクトログラムを比較
- 差分の最大値も出力し、精度評価が可能

---

## 📈 可視化例
- `matplotlib` による：
  - 時間波形
  - 周波数スペクトル
  - スペクトログラム（等高線 + 最大成分）

---

## ✅ 想定用途
- 超短パルスの時間-周波数解析
- 分光データの前処理・特徴抽出
- 時系列データの局所周波数解析

---

## 📎 依存ライブラリ
- `numpy`
- `matplotlib`

`requirements.txt` 例：
```txt
numpy
matplotlib
```

---

ご質問や機能追加の希望があればお気軽にどうぞ！
