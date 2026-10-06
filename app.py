from flask import Flask, request, jsonify, render_template
import os
import json
import uuid
import tempfile
import io
from datetime import datetime, timezone

import opensmile
import pandas as pd

from google.cloud import storage
from google.oauth2 import service_account
from werkzeug.utils import secure_filename


app = Flask(__name__)


# =========================================================
# 1. Google Cloud Storage
# =========================================================

def get_gcs_bucket():

    service_account_info = json.loads(
        os.environ["GCP_SERVICE_ACCOUNT_JSON"]
    )

    credentials = (
        service_account.Credentials.from_service_account_info(
            service_account_info
        )
    )

    client = storage.Client(
        credentials=credentials,
        project=service_account_info["project_id"]
    )

    bucket = client.bucket(
        os.environ["GCS_BUCKET_NAME"]
    )

    return bucket


def upload_bytes_to_gcs(
    data,
    object_name,
    content_type
):

    bucket = get_gcs_bucket()

    blob = bucket.blob(
        object_name
    )

    blob.upload_from_string(
        data,
        content_type=content_type
    )


def upload_file_to_gcs(
    local_path,
    object_name,
    content_type=None
):

    bucket = get_gcs_bucket()

    blob = bucket.blob(
        object_name
    )

    blob.upload_from_filename(
        local_path,
        content_type=content_type
    )


# =========================================================
# 2. openSMILE
# =========================================================

# ---------------------------------------------------------
# 88次元
# LightGBM / GAM の入力に使用
# ---------------------------------------------------------

smile_functionals = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


# ---------------------------------------------------------
# 時系列LLD
# 後ほどΔ特徴・ビブラート等に使用
# ---------------------------------------------------------

smile_lld = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)


# =========================================================
# 3. 特徴量抽出
# =========================================================

def extract_acoustic_features(
    wav_path
):

    # ---------------------------------------------
    # eGeMAPSv02 88次元 Functionals
    # ---------------------------------------------

    functionals_df = (
        smile_functionals.process_file(
            wav_path
        )
    )

    features = (
        functionals_df
        .iloc[0]
        .to_dict()
    )

    # ---------------------------------------------
    # LLD
    # 現時点ではフレーム数だけ記録
    #
    # 次のステップで、
    # F0 delta
    # Loudness delta
    # F0 stability
    # Vibrato
    # などを追加する
    # ---------------------------------------------

    lld_df = (
        smile_lld.process_file(
            wav_path
        )
    )

    frame_count = len(
        lld_df
    )

    return (
        features,
        frame_count
    )


# =========================================================
# 4. CSV読み込み確認
# =========================================================

def validate_csv(csv_bytes):

    # UTF-8をまず試す
    try:

        return pd.read_csv(
            io.BytesIO(csv_bytes),
            encoding="utf-8-sig"
        )

    except UnicodeDecodeError:

        # Excel等から出した日本語CSV対策
        return pd.read_csv(
            io.BytesIO(csv_bytes),
            encoding="cp932"
        )


# =========================================================
# 5. トップページ
# =========================================================

@app.route("/", methods=["GET"])
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# 6. GCS接続確認
# =========================================================

@app.route(
    "/gcs-test",
    methods=["GET"]
)
def gcs_test():

    try:

        upload_bytes_to_gcs(
            b"Render -> Google Cloud Storage connection OK",
            "test/render_connection.txt",
            "text/plain"
        )

        return jsonify({
            "success": True,
            "message":
                "GCSへの書き込みに成功しました"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# 7. 本処理
# =========================================================

@app.route(
    "/submit",
    methods=["POST"]
)
def submit():

    # -----------------------------------------------------
    # 特徴量・評価データの研究利用への同意
    # -----------------------------------------------------

    allow_feature_use = (
        request.form.get(
            "allow_feature_use"
        ) == "yes"
    )

    if not allow_feature_use:

        return jsonify({
            "success": False,
            "error":
                "音響特徴量および評価データの研究利用への同意が必要です。"
        }), 400


    # -----------------------------------------------------
    # WAV保存許可
    # -----------------------------------------------------

    allow_save = (
        request.form.get(
            "allow_save"
        ) == "yes"
    )


    # -----------------------------------------------------
    # 1回の送信を識別するID
    # -----------------------------------------------------

    batch_id = str(
        uuid.uuid4()
    )

    submitted_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )


    processed_results = []
    all_features = []


    # =====================================================
    # A. 印象評価CSV
    # =====================================================

    eval_csv = request.files.get(
        "eval_csv"
    )

    evaluation_path = None


    if (
        eval_csv
        and eval_csv.filename
    ):

        try:

            csv_bytes = (
                eval_csv.read()
            )

            # CSVとして読めるか確認
            df_eval = validate_csv(
                csv_bytes
            )

            evaluation_path = (
                f"evaluations/"
                f"{batch_id}/"
                f"evaluation.csv"
            )

            upload_bytes_to_gcs(
                csv_bytes,
                evaluation_path,
                "text/csv"
            )

        except Exception as e:

            return jsonify({
                "success": False,
                "error":
                    f"評価CSVの処理に失敗しました: {str(e)}"
            }), 400


    # =====================================================
    # B. WAV処理
    # =====================================================

    audio_files = (
        request.files.getlist(
            "audio_files"
        )
    )


    if len(audio_files) == 0:

        return jsonify({
            "success": False,
            "error":
                "WAVファイルがありません。"
        }), 400


    for index, file in enumerate(
        audio_files
    ):

        if (
            not file
            or file.filename == ""
        ):
            continue


        original_filename = (
            secure_filename(
                file.filename
            )
        )


        # ---------------------------------------------
        # 個人を直接識別しない研究用ID
        # ---------------------------------------------

        sample_id = (
            f"{batch_id}_"
            f"{index:04d}"
        )


        temp_path = None


        try:

            # =========================================
            # Render上へ一時保存
            # =========================================

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp_file:

                file.save(
                    temp_file
                )

                temp_path = (
                    temp_file.name
                )


            # =========================================
            # 音響特徴量抽出
            # =========================================

            (
                features,
                frame_count

            ) = extract_acoustic_features(
                temp_path
            )


            # =========================================
            # 特徴量データを記録
            # =========================================

            feature_row = {

                "batch_id":
                    batch_id,

                "sample_id":
                    sample_id,

                # CSVとの対応確認用
                # ファイル名に氏名等を含めない運用が望ましい
                "source_filename":
                    original_filename,

                **features
            }


            all_features.append(
                feature_row
            )


            # =========================================
            # WAV保存許可あり
            # =========================================

            audio_gcs_path = None


            if allow_save:

                audio_gcs_path = (
                    f"audio/"
                    f"{batch_id}/"
                    f"{sample_id}.wav"
                )

                upload_file_to_gcs(
                    temp_path,
                    audio_gcs_path,
                    "audio/wav"
                )


            processed_results.append({

                "sample_id":
                    sample_id,

                "filename":
                    original_filename,

                "frames_extracted":
                    frame_count,

                "audio_saved":
                    allow_save,

                "audio_path":
                    audio_gcs_path
            })


        except Exception as e:

            processed_results.append({

                "sample_id":
                    sample_id,

                "filename":
                    original_filename,

                "error":
                    str(e)
            })


        finally:

            # =========================================
            # ★非常に重要
            #
            # WAV保存許可の有無にかかわらず
            # Renderの一時ファイルは削除
            #
            # 保存許可あり
            # → GCSに保存済み
            #
            # 保存許可なし
            # → GCSには保存されない
            # =========================================

            if (
                temp_path
                and
                os.path.exists(
                    temp_path
                )
            ):

                os.remove(
                    temp_path
                )


    # =====================================================
    # C. features.csvを作る
    # =====================================================

    features_path = None


    if len(all_features) > 0:

        df_features = pd.DataFrame(
            all_features
        )


        # Excelで開いたときにも日本語対応しやすい
        features_csv = (
            df_features.to_csv(
                index=False
            ).encode(
                "utf-8-sig"
            )
        )


        features_path = (
            f"features/"
            f"{batch_id}/"
            f"features.csv"
        )


        upload_bytes_to_gcs(
            features_csv,
            features_path,
            "text/csv"
        )


    # =====================================================
    # D. 処理情報をmanifestとして保存
    # =====================================================

    manifest = {

        "batch_id":
            batch_id,

        "submitted_at":
            submitted_at,

        "allow_audio_save":
            allow_save,

        "number_of_audio_files":
            len(
                processed_results
            ),

        "number_of_successful_features":
            len(
                all_features
            ),

        "extractor": {

            "name":
                "Singing Evaluation Feature Extractor",

            "version":
                "1.0.0",

            "feature_set":
                "eGeMAPSv02",

            "functionals":
                True,

            "lld":
                True
        },

        "features_path":
            features_path,

        "evaluation_path":
            evaluation_path
    }


    manifest_path = (
        f"manifests/"
        f"{batch_id}/"
        f"manifest.json"
    )


    upload_bytes_to_gcs(

        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2
        ).encode(
            "utf-8"
        ),

        manifest_path,

        "application/json"
    )


    # =====================================================
    # E. 完了レスポンス
    # =====================================================

    return jsonify({

        "success":
            True,

        "message":
            "音響特徴量の抽出と保存が完了しました。",

        "batch_id":
            batch_id,

        "audio_saved":
            allow_save,

        "features_path":
            features_path,

        "evaluation_path":
            evaluation_path,

        "processed_files":
            processed_results
    })


# =========================================================
# ローカル開発用
# Renderではgunicorn利用推奨
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                10000
            )
        )
    )