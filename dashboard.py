import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = r"C:/Users/Priyalakshmi/Downloads/postings"
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def list_output_images():
    imgs = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.png')]
    return sorted(imgs)


@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_file_path = None
    top_rec_df = None
    skill_gap_df = None
    images = list_output_images()

    if request.method == "POST":
        if "resume" not in request.files:
            return redirect(request.url)
        file = request.files["resume"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_file_path)
            # Here you can call job_recommender_end2end.py logic
            # For simplicity, we assume outputs already exist in OUTPUT_FOLDER

            # Load CSVs for display
            top_rec_path = os.path.join(OUTPUT_FOLDER, "recommendations_hybrid_top_50.csv")
            skill_gap_path = os.path.join(OUTPUT_FOLDER, "skill_gap_top_50.csv")
            if os.path.exists(top_rec_path):
                top_rec_df = pd.read_csv(top_rec_path).head(20)  # first 20 for display
            if os.path.exists(skill_gap_path):
                skill_gap_df = pd.read_csv(skill_gap_path).head(20)

    return render_template("dashboard.html",
                           uploaded_file=uploaded_file_path,
                           top_rec_df=top_rec_df,
                           skill_gap_df=skill_gap_df,
                           images=images)


# Run app
if __name__ == "__main__":
    app.run(debug=True)
