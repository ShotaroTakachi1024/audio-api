# render_template を追加でインポートします
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS
import opensmile
import pandas as pd

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = '/tmp'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# openSMILEの初期化などはそのまま...
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

# ① アクセスされたときに index.html（画面）を表示する
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# ② 画面から送られてきた音声と評価を受け取る
@app.route('/submit', methods=['POST'])
def submit():
    # 1. 印象評価（アンケート回答）の取得
    satisfaction = request.form.get('satisfaction')
    
    # 2. WAVファイルの取得と特徴量抽出
    if 'audio_file' not in request.files:
        return "ファイルがありません", 400
    
    file = request.files['audio_file']
    if file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # 特徴量抽出 (先ほどまでの処理と同様)
            features_df = smile.process_file(filepath)
            os.remove(filepath)
            
            # 特徴量と評価結果を合わせて画面に返す
            # 実際はここで、ユーザーの回答と抽出した音響特徴量をCSVなどに保存する処理を書きます
            return jsonify({
                "message": "データを受け取りました！",
                "satisfaction_score": satisfaction,
                "extracted_frames": len(features_df)
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)