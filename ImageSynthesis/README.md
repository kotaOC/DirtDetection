# ImageSynthesis

重複して撮影した横長の金属部品画像を自動で位置合わせし、1枚の横長画像に合成するデスクトップアプリです。撮影順に並べた隣接画像から特徴点を検出し、RANSACで外れ値を除外して位置合わせします。重複部はフェザーブレンディングし、境界を自然につなぎます。

## 動作環境

- Python 3.9以上（3.11推奨）
- Windows / macOS / Linux
- PySide6（`requirements.txt`から自動でインストールされます）

## セットアップ

```bash
cd ImageSynthesis
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

パッケージとしてインストールする場合は `python -m pip install .` の後に `image-synthesis` でも起動できます。

## 使い方

1. ［画像を追加…］から2枚以上の画像を選びます。
2. リストが実際の撮影順（左から右、または右から左）になるよう、［↑］［↓］で調整します。
3. ［合成を実行］を押します。
4. プレビューを確認し、［結果を保存…］からPNG、JPEG、TIFF、BMP形式で保存します。

### 撮影のコツ

- 隣り合う画像に30～50%程度の重なりを持たせてください。
- カメラと部品の距離、ズーム、明るさをなるべく固定してください。
- 横移動を基本とし、上下移動や回転は小さくしてください。
- 光沢だけで模様がない面では特徴点が不足します。小さな傷・穴・刻印など、位置の手掛かりが重複部に入るよう撮影してください。

特徴点が不足した場合は位相相関による平行移動推定へ自動で切り替わります。位置合わせを信頼できない場合は、誤った巨大画像を出力せずエラーで停止します。

## テスト

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## ファイル構成

- `run.py`: GUI起動用スクリプト
- `image_synthesis/qt_app.py`: 高DPI対応Qt GUI（標準の起動先）
- `image_synthesis/app.py`: 旧Tkinter GUI（互換用）
- `image_synthesis/stitcher.py`: 読み込み、位置合わせ、ブレンディング、保存
- `tests/test_stitcher.py`: 合成処理と日本語パス保存の自動テスト
- `requirements.txt`: 実行時依存関係
- `pyproject.toml`: パッケージ設定
