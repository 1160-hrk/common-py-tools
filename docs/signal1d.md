# signal1d.py

## 📈 SignalData クラス

`SignalData` は、1次元の信号データに対してカーソルを用いた選択処理やフィッティング・変換・平滑化などの解析を簡単に行うためのクラスです。

---

## 🧱 初期化

```python
SignalData(x: np.ndarray, y: np.ndarray)
```

- `x`: 横軸（時間、波数、周波数など）
- `y`: 信号データ

---

## 🔧 基本機能

### `set_cursors(a, b)`
カーソル A, B を設定し、範囲選択を可能にします。

---

## 🎯 セグメント操作

### `get_segment()`
現在設定されたカーソル範囲に含まれる `(x, y)` データを返します。

---

## 📊 可視化

### `plot(ax=None)`
カーソル線を含む簡易的な信号プロットを表示します。

---

## 🧮 解析系メソッド

### `fit_segment(func, p0=None, bounds=(-inf, inf), return_curve=False)`
カーソル内のデータに任意の関数でフィッティングを行います。

- `func`: フィット関数（例：`gaussian`, `voigt` など）
- `p0`: 初期パラメータ
- `return_curve=True` でフィット結果も返します。

---

### `fft_segment()`
カーソル範囲の信号に対して FFT を行い、`(freq, fft_y)` を返します。

---

### `find_extrema_segment()`
最大・最小値およびそのインデックスと座標を返します（辞書形式）。

---

### `find_peaks_segment(**kwargs)`
カーソル範囲内のピーク検出を行い、インデックスを返します。

---

### `integrate_segment(method='simpson')`
カーソル内の信号を積分します（Simpson または Trapezoidal）。

---

### `detrend_segment(inplace=False)`
線形トレンド（傾き）を除去。`inplace=True` で元データを上書き可能。

---

### `baseline_subtract_segment(method='mean' or 'custom', value=None)`
ベースラインを除去（平均または指定値）。

---

### `smooth_segment(mode='moving_average' or 'savgol', window_size=5, polyorder=2)`
移動平均または Savitzky-Golay による平滑化。

---

### `resample_segment(num=None, step=None, method='linear')`
カーソル内データを等間隔で再補間します。

- `num`: 出力点数
- `step`: x軸ステップ幅
- `method`: `'linear'`, `'cubic'`, `'quadratic'` など

---

## 💾 データ入出力

### `to_csv(path)`
x, y の信号データを CSV に保存します。

### `from_csv(path)`
CSV ファイルから `SignalData` を生成します。

---

## ✅ 使用例

```python
from pyana.signal1d import SignalData
from pyana.func import gaussian

x = np.linspace(0, 10, 100)
y = gaussian(x, A=1, x0=5, sigma=1)

data = SignalData(x, y)
data.set_cursors(3, 7)
popt = data.fit_segment(gaussian, p0=[1, 5, 1])
```
