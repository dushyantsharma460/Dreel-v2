from flask import Flask, render_template, request, redirect, url_for, flash, send_file, send_from_directory, abort
import uuid
import os
import io
import threading

from config import UPLOAD_FOLDER, SECRET_KEY, REELS_FOLDER

from services.image_service import (
    create_upload_directory,
    save_uploaded_images,
    create_input_file,
    save_description,
    save_uploaded_audio,
    detect_dominant_format,
    save_video_format,
    save_voice_gender,
)
from config import VIDEO_FORMAT_DESKTOP, VIDEO_FORMAT_MOBILE, VOICE_GENDER_CHOICES
from services.reel_service import list_reels, start_worker
from services.vision_service import get_quality_warnings
from services.analytics_service import get_dashboard_context
from services.ml_service import (
    enrich_reels_with_ml,
    get_model_summary,
    save_rating,
    train_engagement_model,
)
from services.db_service import init_db, get_reel_file

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY


def launch_worker() -> None:
    worker_thread = threading.Thread(
        target=start_worker,
        daemon=True,
        name="reel-worker",
    )
    worker_thread.start()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/create", methods=["GET", "POST"])
def create():
    myid = str(uuid.uuid1())

    if request.method == "POST":
        req_id = request.form.get("uuid")
        audio_option = request.form.get("audioOption")

        upload_dir = create_upload_directory(
            app.config["UPLOAD_FOLDER"],
            req_id,
        )

        if audio_option == "ai":
            description = request.form.get("text")
            if not description or not description.strip():
                flash("Please enter text for AI audio generation.", "danger")
                return render_template("create.html", myid=myid)
            save_description(upload_dir, description)

            voice_gender = request.form.get("voiceGender", "any")
            if voice_gender not in VOICE_GENDER_CHOICES:
                voice_gender = "any"
            save_voice_gender(upload_dir, voice_gender)

        elif audio_option == "upload":
            audio_file = request.files.get("audioFile")
            if not audio_file or audio_file.filename == "":
                flash("Please upload an audio file.", "danger")
                return render_template("create.html", myid=myid)
            save_uploaded_audio(audio_file, upload_dir)

        image_files = save_uploaded_images(
            request.files,
            upload_dir,
        )

        if not image_files:
            flash("Please upload at least one image.", "danger")
            return render_template("create.html", myid=myid)

        for warning in get_quality_warnings(upload_dir, image_files):
            flash(warning, "warning")

        create_input_file(
            upload_dir,
            image_files,
        )

        format_option = request.form.get("formatOption", "auto")
        if format_option == VIDEO_FORMAT_DESKTOP:
            video_format = VIDEO_FORMAT_DESKTOP
        elif format_option == VIDEO_FORMAT_MOBILE:
            video_format = VIDEO_FORMAT_MOBILE
        else:
            video_format = detect_dominant_format(upload_dir, image_files)

        save_video_format(upload_dir, video_format)

        flash(
            "Your reel is being generated. Images are auto-ranked by quality and synced to beats.",
            "success",
        )
        return redirect(url_for("gallery", processing=1))

    return render_template("create.html", myid=myid)


@app.route("/analytics")
def analytics():
    dashboard = get_dashboard_context()
    return render_template("analytics.html", **dashboard)


@app.route("/gallery")
def gallery():
    reels = enrich_reels_with_ml(list_reels())
    mobile_reels = [reel for reel in reels if reel["format"] == VIDEO_FORMAT_MOBILE]
    desktop_reels = [reel for reel in reels if reel["format"] == VIDEO_FORMAT_DESKTOP]
    processing = request.args.get("processing") == "1"
    return render_template(
        "gallery.html",
        reels=reels,
        mobile_reels=mobile_reels,
        desktop_reels=desktop_reels,
        processing=processing,
        model_metrics=get_model_summary(),
    )


@app.route("/reel-video/<reel_id>")
def reel_video(reel_id):
    """
    Stream a reel's video from the Postgres mirror (reel_files table) so
    the gallery plays it from the database instead of the static/reels/
    folder. Falls back to the file on disk if the DB copy is missing
    (e.g. Postgres was briefly down when this reel was generated).
    """

    file_row = get_reel_file(reel_id, "video")

    if file_row is not None:
        return send_file(
            io.BytesIO(file_row["file_data"]),
            mimetype=file_row["content_type"] or "video/mp4",
            download_name=file_row["filename"],
            conditional=True,
        )

    filename = f"{reel_id}.mp4"
    if not os.path.isfile(os.path.join(REELS_FOLDER, filename)):
        abort(404)

    return send_from_directory(REELS_FOLDER, filename, conditional=True)


@app.route("/rate", methods=["POST"])
def rate_reel():
    reel_id = request.form.get("reel_id", "").strip()

    if not reel_id:
        flash("Could not identify which reel to rate.", "danger")
        return redirect(url_for("gallery"))

    try:
        rating = int(request.form.get("rating", 0))
        save_rating(reel_id, rating)
        train_engagement_model()
        flash(f"Thanks! You rated this reel {rating}/5. Model retrained.", "success")
    except (ValueError, TypeError):
        flash("Please select a rating between 1 and 5.", "danger")

    return redirect(url_for("gallery"))


if __name__ == "__main__":
    DEBUG = True

    # app.debug is only set once app.run() executes, so checking it here
    # (before app.run) always read False and launched the worker twice
    # under the debug reloader (parent process + WERKZEUG_RUN_MAIN child).
    # Using the DEBUG constant directly makes the guard correct.
    if not DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        init_db()
        launch_worker()

    app.run(
        host="0.0.0.0",
        port=4600,
        debug=DEBUG,
    )
