from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS
import opensmile
import pandas as pd

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = '/tmp'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # 1. 保存許可のチェック状態を取得
    # チェックされていれば 'yes'、されていなければ None になります
    allow_save = request.form.get('allow_save') == 'yes'
    
    # 2. 評価用CSVファイルの受け取り
    eval_csv = request.files.get('eval_csv')
    if eval_csv:
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], eval_csv.filename)
        eval_csv.save(csv_path)
        # TODO: ここで pd.read_csv(csv_path) などを使用して評価データを読み込む
    
    # 3. 複数WAVファイルの受け取りと処理
    audio_files = request.files.getlist('audio_files')
    processed_results = []
    
    for file in audio_files:
        if file.filename == '':
            continue
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # openSMILEで特徴量抽出
            features_df = smile.process_file(filepath)
            
            # 結果をリストに記録
            processed_results.append({
                "filename": file.filename,
                "frames_extracted": len(features_df)
            })
            
            # ★企業秘密への配慮（データ破棄ロジック）
            if not allow_save:
                # 許可がなければ、特徴量だけ抽出して即座にWAV本体を削除する
                os.remove(filepath)
            else:
                # 許可がある場合は保存しておく（※後述のRenderの仕様に注意）
                pass
                
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            processed_results.append({"filename": file.filename, "error": str(e)})

    # レスポンスを返す
    return jsonify({
        "message": "処理が完了しました",
        "saved_audio": allow_save,
        "processed_files": processed_results
    })

if __name__ == '__main__':
    app.run(debug=True)