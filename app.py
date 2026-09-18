import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import opensmile
import pandas as pd

# テスト テスト テスト
app = Flask(__name__)
CORS(app) # Webアンケート画面など、外部からのAPI呼び出しを許可

# クラウド環境では /tmp が安全に書き込める一時フォルダです
app.config['UPLOAD_FOLDER'] = '/tmp'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# openSMILEの初期化（時系列 LLDs）
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

@app.route('/', methods=['GET'])
def index():
    return "API is running. POST /extract to extract features."

@app.route('/extract', methods=['POST'])
def extract():
    if 'audio_file' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400
    
    if file and file.filename.endswith('.wav'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # 特徴量抽出
            features_df = smile.process_file(filepath)
            features_df = features_df.reset_index()
            
            if 'file' in features_df.columns:
                features_df = features_df.drop(columns=['file'])
            if 'start' in features_df.columns:
                features_df['start'] = features_df['start'].dt.total_seconds()
            if 'end' in features_df.columns:
                features_df['end'] = features_df['end'].dt.total_seconds()
            
            features_list = features_df.to_dict(orient='records')
            os.remove(filepath)
            
            return jsonify(features_list)
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "WAV形式のファイルのみ対応しています"}), 400

if __name__ == '__main__':
    app.run(debug=True)