# boatrace-ai Cloud Run

## 必要なモデルファイル

Google Drive の `MyDrive/boat_ai/v1.3/` から以下4ファイルを `models/` に入れます。

- `first_model.joblib`
- `second_conditional_model.joblib`
- `third_conditional_model.joblib`
- `feature_schema.json`

## GitHubへアップロード

このフォルダの中身をすべてGitHubリポジトリへアップロードします。

## Cloud Run

Cloud Runで「リポジトリから継続的にデプロイする」を選び、GitHubリポジトリを接続します。

推奨設定:

- リージョン: asia-northeast1
- 認証: 未認証の呼び出しを許可
- CPU: 1
- メモリ: 1GiB
- 最小インスタンス: 0
- 最大インスタンス: 1
- リクエストタイムアウト: 300秒

## 動作確認

- `/health`
- `/predict?race=蒲郡12R`
- 過去日: `/predict?race=蒲郡12R&date=2026-07-14`

## 注意

BOAT RACE公式サイトのHTML構造が変わると取得部分の修正が必要です。
