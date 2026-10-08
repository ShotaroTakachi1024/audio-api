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

# eGeMAPSv02 LLDは約10ms刻み
FRAME_SHIFT_SEC = 0.01

# 10 / 25 / 50 / 100フレーム
# 約100 / 250 / 500 / 1000 ms
DYNAMIC_WINDOWS = [10, 25, 50, 100]

DYNAMIC_COLUMNS = {
    "f0": "F0semitoneFrom27.5Hz_sma3nz",
    "loudness": "Loudness_sma3",
    "mfcc1": "mfcc1_sma3",
    "f1": "F1frequency_sma3nz",
    "f2": "F2frequency_sma3nz",
}


def _get_contiguous_segments(mask):
    """
    Trueが連続している区間を取得する。
    F0などで無声音区間をまたいで計算しないために使用。
    """

    indices = np.where(mask)[0]

    if len(indices) == 0:
        return []

    split_points = np.where(
        np.diff(indices) > 1
    )[0] + 1

    return np.split(
        indices,
        split_points
    )


def _calculate_rolling_slopes(
    values,
    window_size,
    positive_only=False
):
    """
    指定窓幅ごとに局所的な線形回帰の傾きを求める。

    傾きの単位:
      F0       → semitone / sec
      F1, F2   → Hz / sec
      その他   → feature unit / sec
    """

    values = np.asarray(
        values,
        dtype=float
    )

    valid = np.isfinite(values)

    # F0・formantでは0は無効値として扱う
    if positive_only:
        valid &= values > 0

    segments = _get_contiguous_segments(
        valid
    )

    slopes = []

    # 時間軸
    x = (
        np.arange(window_size)
        * FRAME_SHIFT_SEC
    )

    x_centered = x - x.mean()

    denominator = np.sum(
        x_centered ** 2
    )

    for indices in segments:

        segment = values[indices]

        if len(segment) < window_size:
            continue

        for start in range(
            len(segment)
            - window_size
            + 1
        ):

            y = segment[
                start:
                start + window_size
            ]

            # x_centered の和は0なので
            # yの平均を引かなくても傾きを計算可能
            slope = (
                np.dot(
                    x_centered,
                    y
                )
                / denominator
            )

            slopes.append(
                slope
            )

    return np.asarray(
        slopes,
        dtype=float
    )


def extract_dynamic_features(
    lld_df
):

    dynamic_features = {}

    for short_name, column_name in (
        DYNAMIC_COLUMNS.items()
    ):

        if column_name not in lld_df.columns:
            continue

        values = (
            lld_df[column_name]
            .to_numpy(dtype=float)
        )

        # F0・formantは0を欠損相当として扱う
        positive_only = (
            short_name
            in {"f0", "f1", "f2"}
        )

        for window in DYNAMIC_WINDOWS:

            slopes = (
                _calculate_rolling_slopes(
                    values,
                    window,
                    positive_only
                )
            )

            duration_ms = int(
                window
                * FRAME_SHIFT_SEC
                * 1000
            )

            prefix = (
                f"{short_name}"
                f"_dynamic_{duration_ms}ms"
            )

            if len(slopes) == 0:

                dynamic_features[
                    f"{prefix}_mean_abs_slope"
                ] = np.nan

                dynamic_features[
                    f"{prefix}_std_slope"
                ] = np.nan

            else:

                # 動きの大きさ
                dynamic_features[
                    f"{prefix}_mean_abs_slope"
                ] = float(
                    np.mean(
                        np.abs(slopes)
                    )
                )

                # 動きのばらつき
                dynamic_features[
                    f"{prefix}_std_slope"
                ] = float(
                    np.std(slopes)
                )

    return dynamic_features

def extract_f0_stability_features(
    lld_df
):

    column = (
        "F0semitoneFrom27.5Hz_sma3nz"
    )

    if column not in lld_df.columns:

        return {}

    f0 = (
        lld_df[column]
        .to_numpy(dtype=float)
    )

    # 隣り合うフレームのF0差
    delta = np.diff(f0)

    # 両方とも有声音である場合のみ使用
    # 無声音を削除してからdiffすると、
    # 離れた有声音区間をつないでしまうのでNG
    valid_pair = (
        np.isfinite(f0[:-1])
        & np.isfinite(f0[1:])
        & (f0[:-1] > 0)
        & (f0[1:] > 0)
    )

    delta = delta[
        valid_pair
    ]

    if len(delta) == 0:

        return {
            "f0_delta_cent_mean_abs":
                np.nan,

            "f0_delta_cent_median_abs":
                np.nan,

            "f0_delta_cent_p90_abs":
                np.nan,

            "f0_stable_ratio_5cent":
                np.nan,

            "f0_stable_ratio_10cent":
                np.nan,

            "f0_stable_ratio_20cent":
                np.nan,
        }

    # openSMILEのF0はsemitoneなので
    # 1 semitone = 100 cents
    delta_cent = (
        np.abs(delta)
        * 100.0
    )

    return {

        # 10msごとのF0変化量
        "f0_delta_cent_mean_abs":
            float(
                np.mean(
                    delta_cent
                )
            ),

        "f0_delta_cent_median_abs":
            float(
                np.median(
                    delta_cent
                )
            ),

        "f0_delta_cent_p90_abs":
            float(
                np.percentile(
                    delta_cent,
                    90
                )
            ),

        # 安定しているフレームの割合
        "f0_stable_ratio_5cent":
            float(
                np.mean(
                    delta_cent <= 5
                )
            ),

        "f0_stable_ratio_10cent":
            float(
                np.mean(
                    delta_cent <= 10
                )
            ),

        "f0_stable_ratio_20cent":
            float(
                np.mean(
                    delta_cent <= 20
                )
            ),
    }

def extract_acoustic_features(
    wav_path
):

    # =============================================
    # eGeMAPSv02 88 Functionals
    # =============================================

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


    # =============================================
    # LLD
    # =============================================

    lld_df = (
        smile_lld.process_file(
            wav_path
        )
    )


    # =============================================
    # 動的特徴量
    # =============================================

    dynamic_features = (
        extract_dynamic_features(
            lld_df
        )
    )

    features.update(
        dynamic_features
    )


    # =============================================
    # F0安定度
    # =============================================

    stability_features = (
        extract_f0_stability_features(
            lld_df
        )
    )

    features.update(
        stability_features
    )


    return (
        features,
        len(lld_df)
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
                "1.1.0",

            "feature_set":
                "eGeMAPSv02",

            "functionals":
                True,

            "lld":
                True,

            "dynamic_features": {
                "targets": [
                    "F0",
                    "Loudness",
                    "MFCC1",
                    "F1",
                    "F2"
                ],
                "frame_shift_sec":
                    0.01,
                "window_frames": [
                    10,
                    25,
                    50,
                    100
                ]
            },

            "f0_stability": {
                "unit":
                    "cent",

                "threshold_cent": [
                    5,
                    10,
                    20
                ]
            }
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