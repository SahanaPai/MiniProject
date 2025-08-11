import os
import cv2
import numpy as np
from flask import Flask, render_template, redirect, url_for, request, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Model

# Initialize Flask app, SQLAlchemy, and LoginManager
app = Flask(__name__)
app.secret_key = "secret_key_for_session_management"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
UPLOAD_FOLDER = "static/uploads/"
RESULT_FOLDER = "static/results/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# Load models
mobilenet_model = load_model("pneumonia_detection_model.h5")
bactviral_model = load_model("VGG16_model.h5")

# User database model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/")
def home():
    if "user_id" in session:  # Check if the user is already logged in
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            session["user_id"] = user.id  # Store user ID in session
            flash("Login successful!")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials, please try again.")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256") # Secure password storage
        new_user = User(username=username, password=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Sign up successful, please login!")
            return redirect(url_for("login"))
        except:
            flash("Username already exists. Please choose a different username.")
    return render_template("signup.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/logout")
@login_required
def logout():
    session.pop("user_id", None)  # Remove user ID from session
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for("login"))

# Preprocess image
def preprocess_image(image_path, model_type="vgg16"):
    if model_type == "mobilenetv2":
        target_size = (128, 128)
    elif model_type == "vgg16":
        target_size = (224, 224)
    else:
        raise ValueError("Invalid model_type. Choose 'mobilenetv2' or 'vgg16'.")

    img = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Generate heatmap
def generate_heatmap(img_array, model, layer_name="block5_conv3"):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = np.dot(conv_outputs[0], weights)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Step 1: MobileNetV2 Prediction
    img_array_mobilenet = preprocess_image(file_path, model_type="mobilenetv2")
    first_model_prediction = mobilenet_model.predict(img_array_mobilenet)

    if first_model_prediction[0][0] > 0.5:  # Assuming 1: Pneumonia, 0: Normal
        # Step 2: VGG16 Model Prediction for Pneumonia Type
        img_array_vgg16 = preprocess_image(file_path, model_type="vgg16")
        second_model_prediction = bactviral_model.predict(img_array_vgg16)

        # Grad-CAM visualization
        heatmap = generate_heatmap(img_array_vgg16, bactviral_model, layer_name="block5_conv3")

        # Superimpose heatmap on the original image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224, 224))
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # Save the heatmap image
        heatmap_image_path = os.path.join(app.config["RESULT_FOLDER"], "heatmap_" + file.filename)
        cv2.imwrite(heatmap_image_path, superimposed_img)

        # Determine diagnosis based on VGG16 predictions
        bacterial_prob = second_model_prediction[0][0]  # Probability of bacterial pneumonia
        viral_prob = 1 - bacterial_prob  # Probability of viral pneumonia

        if bacterial_prob > 0.5:
            diagnosis = "Bacterial Pneumonia"
        elif viral_prob > 0.5:
            diagnosis = "Viral Pneumonia"
        else:
            diagnosis = "Pneumonia (Unspecified)"

        return render_template(
            "dashboard.html",
            diagnosis=diagnosis,
            image_url=url_for("static", filename="uploads/" + file.filename),
            heatmap_image_url=url_for("static", filename="results/" + os.path.basename(heatmap_image_path))
        )
    else:
        # Output for Normal Detection
        return render_template(
            "dashboard.html",
            diagnosis="Normal",
            image_url=url_for("static", filename="uploads/" + file.filename)
        )

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True)
