# 🧰 `utils.py` - データ処理・フィッティング補助関数集

このモジュールは、数値解析やデータ処理に役立つ関数群を提供します。
配列の一部抽出・ピーク検出・フィッティング・データファイルの読み書きなど、解析補助としてよく使われる処理をまとめています。

---

## ✅ 主な関数一覧

### 🔹 インデックス取得系
- `get_ind(array, values)`：配列内で最も近い値のインデックスを返す
- `get_ind_max_xi_xf(x, y, xi, xf)`：指定範囲内で最大値のインデックスを返す
- `get_ind_xi_xf(v, x, y, xi, xf)`：指定範囲内で値 `v` に最も近いインデックスを返す
- `get_inds_peak_xi_xf(v, x, y, xi, xf, **kwargs)`：`scipy.signal.find_peaks` を用いてピークインデックスを返す

### 🔹 サブ配列抽出系
- `get_subarray_1D(x, xi, xf)`：1次元配列の部分抽出
- `get_subarray_2D(x, y, xi, xf)`：2次元（x, y）配列の部分抽出

### 🔹 フィッティング
- `fitting_w_range(fit, x, y, xi, xf, p0, bounds, ...)`
  - 指定範囲でのフィッティングを実行
  - 初期値 `p0` や `bounds` 指定可能
  - 結果パラメータ・共分散・x範囲を返す

---

## 📁 ファイル読み書き

### 🔹 テキストファイル系
- `ndarray_from_txtfile(path, manner)`：任意区切りの `.txt` ファイルを読み込んで numpy 配列を返す
- `ndarray_from_csvfile(path)`：`.csv` ファイルを numpy 配列として読み込み

### 🔹 バイナリデータ
- `pickle_dump(obj, path)`：任意の Python オブジェクトを pickle 形式で保存
- `pickle_load(path)`：pickle 形式で保存されたオブジェクトを読み込み

---

## ✅ 想定用途
- スペクトルや時系列データの局所範囲抽出
- ピークの自動検出・位置推定
- フィッティングの範囲指定・パラメータ推定
- 実験データの読み込み・保存

---

## 🔧 依存ライブラリ
- `numpy`
- `scipy`
- `csv`, `pickle`

---

## 📎 例：フィッティング処理
```python
from common_py_tools.utils import fitting_w_range
from scipy.optimize import curve_fit

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

popt = fitting_w_range(gaussian, x, y, xi=1.0, xf=2.0, p0=[1, 1.5, 0.1])
```

---

何か他に追加したいユーティリティや、使用例があれば教えてください！
