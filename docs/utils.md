# utils.py

信号解析・前処理・ファイル読み書きなど、汎用的に使える関数群。

---

## 📌 関数一覧

### 🔍 インデックス・配列操作

- **`get_ind(array, values)`**  
  最近傍のインデックスを高速に検索。

- **`get_subarray_1D(x, xi, xf)`**  
  1次元配列から範囲を切り出す。

- **`get_subarray_2D(x, y, xi, xf)`**  
  2系列から範囲指定で切り出し（インデックスも返せる）。

- **`get_ind_max_xi_xf(x, y, xi, xf)`**  
  指定範囲内で `y` の最大値インデックスを取得。

- **`get_ind_xi_xf(v, x, y, xi, xf)`**  
  指定範囲で `y` が `v` に最も近い位置のインデックスを返す。

- **`get_inds_peak_xi_xf(v, x, y, xi, xf)`**  
  指定範囲でピークのインデックスを返す（`find_peaks` 使用）。

---

### ⚙ フィッティング

- **`fitting_w_range(fit, x, y, xi, xf, p0, bounds)`**  
  任意の範囲に制限して関数フィッティング。  
  `return_x`, `return_pcov` により戻り値を制御可能。

```python
from pyana import fitting_w_range, func
popt = fitting_w_range(func.gaussian_fwhm, x, y, xi=0.1, xf=0.5, p0=[1.0, 0.3, 0.1])
```

---

### 📁 ファイル I/O

- **`ndarray_from_txtfile(path, manner)`**  
  区切り文字指定でテキストファイルを 2列読み込み。

- **`ndarray_from_csvfile(path)`**  
  2列CSVファイルから NumPy 配列を読み込む。

- **`pickle_dump(obj, path)`**  
  任意のオブジェクトを pickle 形式で保存。

- **`pickle_load(path)`**  
  pickle バイナリを読み込み、元のオブジェクトを復元。

```python
from pyana import pickle_dump, pickle_load
pickle_dump(data, "backup.pkl")
restored = pickle_load("backup.pkl")
```

---

## 🔧 備考

- `get_ind`, `get_subarray_2D`, `fitting_w_range` などは、`SignalData` クラスの補助として使うことが多いです。
- `curve_fit` ベースのフィッティングでは初期値 `p0` の指定が重要です。
