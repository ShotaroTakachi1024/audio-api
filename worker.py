import io
import json
import os
import tempfile
import time
from datetime import datetime, timezone
import smtplib
from email.message import EmailMessage

import pandas as pd

from app import get_gcs_bucket, extract_acoustic_features


POLL_INTERVAL_SEC = 10


def upload_bytes(bucket, data, object_name, content_type):
    blob = bucket.blob(object_name)
    blob.upload_from_string(
        data,
        content_type=content_type,
    )
def send_result_email(
    collection_id,
    features_csv,
    total_files,
):

    # ==========================================
    # Render Environment Variables
    # ==========================================

    mail_username = os.environ[
        "MAIL_USERNAME"
    ]

    mail_app_password = os.environ[
        "MAIL_APP_PASSWORD"
    ]

    result_email_to = [
        address.strip()
        for address
        in os.environ[
            "RESULT_EMAIL_TO"
        ].split(",")
        if address.strip()
    ]


    # ==========================================
    # メール作成
    # ==========================================

    message = EmailMessage()

    message["From"] = mail_username

    message["To"] = ", ".join(
        result_email_to
    )

    message["Subject"] = (
        "【歌唱データ解析完了】"
        f"{collection_id}"
    )


    message.set_content(
        "歌唱データの解析が完了しました。\n\n"
        f"collection_id: {collection_id}\n"
        f"解析件数: {total_files}件\n\n"
        "音響特徴量CSVを添付します。"
    )


    # ==========================================
    # features.csv を添付
    # ==========================================

    message.add_attachment(
        features_csv,
        maintype="text",
        subtype="csv",
        filename=(
            f"features_"
            f"{collection_id}.csv"
        ),
    )


    # ==========================================
    # Gmail SMTPから送信
    # ==========================================

    with smtplib.SMTP(
        "smtp.gmail.com",
        587,
        timeout=60,
    ) as smtp:

        smtp.ehlo()

        smtp.starttls()

        smtp.ehlo()

        smtp.login(
            mail_username,
            mail_app_password,
        )

        smtp.send_message(
            message
        )


    print(
        f"[EMAIL SENT] "
        f"{result_email_to}",
        flush=True,
    )


def process_job(job_blob):
    bucket = get_gcs_bucket()

    job = json.loads(
        job_blob.download_as_text()
    )

    collection_id = job["collection_id"]
    total_files = int(job["total_files"])
    allow_audio_save = bool(
        job.get("allow_audio_save", False)
    )

    print(
        f"[START] collection={collection_id} "
        f"total_files={total_files} "
        f"allow_audio_save={allow_audio_save}",
        flush=True,
    )

    incoming_prefix = (
        f"incoming/{collection_id}/"
    )

    incoming_blobs = sorted(
        bucket.list_blobs(
            prefix=incoming_prefix
        ),
        key=lambda blob: blob.name
    )

    # 1音声ずつ処理。
    # 特徴量は feature_parts/ に即保存するので、
    # workerが途中で落ちても処理済み分を失いにくい。
    for audio_blob in incoming_blobs:

        basename = (
            audio_blob.name
            .rsplit("/", 1)[-1]
        )

        if "_" in basename:
            _, original_filename = (
                basename.split("_", 1)
            )
        else:
            original_filename = basename

        stem = os.path.splitext(
            basename
        )[0]

        part_path = (
            f"feature_parts/"
            f"{collection_id}/"
            f"{stem}.csv"
        )

        part_blob = bucket.blob(
            part_path
        )

        # 前回すでに特徴抽出済みなら再解析しない
        if not part_blob.exists():

            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav",
                    delete=False,
                ) as temp_file:
                    temp_path = temp_file.name

                audio_blob.download_to_filename(
                    temp_path
                )

                (
                    features,
                    frame_count,
                ) = extract_acoustic_features(
                    temp_path
                )

                row = {
                    "collection_id":
                        collection_id,
                    "sample_id":
                        stem,
                    "source_filename":
                        original_filename,
                    "frames_extracted":
                        frame_count,
                    **features,
                }

                part_csv = (
                    pd.DataFrame([row])
                    .to_csv(index=False)
                    .encode("utf-8-sig")
                )

                upload_bytes(
                    bucket,
                    part_csv,
                    part_path,
                    "text/csv",
                )

                print(
                    f"[FEATURE OK] {original_filename}",
                    flush=True,
                )

            finally:
                if (
                    temp_path
                    and os.path.exists(temp_path)
                ):
                    os.remove(temp_path)

        # 元音声を研究用に残す場合は audio/ へ移す。
        # 残さない場合は incoming/ から削除する。
        if allow_audio_save:

            final_audio_path = (
                f"audio/"
                f"{collection_id}/"
                f"{basename}"
            )

            final_blob = bucket.blob(
                final_audio_path
            )

            if not final_blob.exists():
                bucket.copy_blob(
                    audio_blob,
                    bucket,
                    new_name=final_audio_path,
                )

        audio_blob.delete()

    # ここまで来たら feature_parts を集約
    part_blobs = sorted(
        bucket.list_blobs(
            prefix=(
                f"feature_parts/"
                f"{collection_id}/"
            )
        ),
        key=lambda blob: blob.name
    )

    if len(part_blobs) < total_files:
        raise RuntimeError(
            f"特徴量が不足しています: "
            f"{len(part_blobs)} / {total_files}"
        )

    dfs = []

    for part_blob in part_blobs:
        part_bytes = (
            part_blob.download_as_bytes()
        )

        dfs.append(
            pd.read_csv(
                io.BytesIO(part_bytes),
                encoding="utf-8-sig",
            )
        )

    df_features = (
        pd.concat(
            dfs,
            ignore_index=True,
        )
        .sort_values(
            "sample_id"
        )
        .reset_index(
            drop=True
        )
    )

    features_csv = (
        df_features
        .to_csv(
            index=False
        )
        .encode(
            "utf-8-sig"
        )
    )

    features_path = (
        f"features/"
        f"{collection_id}/"
        f"features.csv"
    )

    upload_bytes(
        bucket,
        features_csv,
        features_path,
        "text/csv",
    )

    manifest = {
        "collection_id":
            collection_id,
        "status":
            "completed",
        "completed_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "number_of_audio_files":
            total_files,
        "number_of_feature_rows":
            len(df_features),
        "allow_audio_save":
            allow_audio_save,
        "features_path":
            features_path,
        "evaluation_path":
            job.get("evaluation_path"),
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
            "dynamic_features":
                True,
            "f0_stability":
                True,
        },
    }

    upload_bytes(
        bucket,
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        (
            f"manifests/"
            f"{collection_id}/"
            f"manifest.json"
        ),
        "application/json",
    )


    # ==========================================
    # 解析結果をメール送信
    # ==========================================

    email_marker_path = (
        f"jobs/emailed/"
        f"{collection_id}.json"
    )

    email_marker_blob = bucket.blob(
        email_marker_path
    )


    # Workerが再起動しても
    # 同じメールを二重送信しない
    if not email_marker_blob.exists():

        send_result_email(
            collection_id=
                collection_id,

            features_csv=
                features_csv,

            total_files=
                total_files,
        )


        # --------------------------------------
        # メール送信済みの記録
        # --------------------------------------

        email_sent_info = {

            "collection_id":
                collection_id,

            "sent_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "recipient":
                os.environ[
                    "RESULT_EMAIL_TO"
                ],
        }


        upload_bytes(
            bucket,

            json.dumps(
                email_sent_info,
                ensure_ascii=False,
                indent=2,
            ).encode(
                "utf-8"
            ),

            email_marker_path,

            "application/json",
        )


    completed_job = {

        **job,

        "status":
            "completed",

        "completed_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "features_path":
            features_path,

        "email_sent":
            True,
    }

    upload_bytes(
        bucket,

        json.dumps(
            completed_job,
            ensure_ascii=False,
            indent=2,
        ).encode(
            "utf-8"
        ),

        (
            f"jobs/completed/"
            f"{collection_id}.json"
        ),

        "application/json",
    )


    # ==========================================
    # 全処理が成功してからpendingを削除
    # ==========================================

    job_blob.delete()


    print(
        f"[DONE] collection={collection_id}",
        flush=True,
    )


def main():
    print(
        "Background worker started.",
        flush=True,
    )

    while True:
        try:
            bucket = get_gcs_bucket()

            pending_jobs = list(
                bucket.list_blobs(
                    prefix="jobs/pending/"
                )
            )

            if not pending_jobs:
                time.sleep(
                    POLL_INTERVAL_SEC
                )
                continue

            # 1 worker で順番に処理
            for job_blob in pending_jobs:
                try:
                    process_job(
                        job_blob
                    )
                except Exception as e:
                    print(
                        f"[JOB ERROR] "
                        f"{job_blob.name}: {e}",
                        flush=True,
                    )

                    # pending は残す。
                    # 一時的な失敗なら次回ポーリングで再試行する。
                    time.sleep(30)

        except Exception as e:
            print(
                f"[WORKER ERROR] {e}",
                flush=True,
            )
            time.sleep(30)


if __name__ == "__main__":
    main()
