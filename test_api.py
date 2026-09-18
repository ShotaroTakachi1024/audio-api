import requests

# 1. あなたのRenderのURLに変更してください（末尾の /extract を忘れずに）
API_URL = "https://audio-api-te0h.onrender.com/extract"

# 2. テストで使いたいWAVファイルのパスを指定してください（このファイルと同じ階層に置くのが簡単です）
AUDIO_FILE_PATH = "sample.wav"

print(f"RenderのAPI ({API_URL}) に音声を送信中...")

try:
    with open(AUDIO_FILE_PATH, "rb") as f:
        # 'audio_file' というキーでファイルをPOST送信します
        files = {"audio_file": f}
        response = requests.post(API_URL, files=files)

    # サーバーからのレスポンスを確認
    if response.status_code == 200:
        features = response.json()
        print("\n✅ 抽出成功！")
        print(f"全 {len(features)} フレームの特徴量を抽出しました。")
        print("\n--- 最初の1フレームのデータ ---")
        print(features[0])
    else:
        print(f"\n❌ エラーが発生しました: ステータスコード {response.status_code}")
        print(response.text)

except FileNotFoundError:
    print(f"エラー: 指定されたファイル '{AUDIO_FILE_PATH}' が見つかりません。")
except Exception as e:
    print(f"予期せぬエラー: {e}")