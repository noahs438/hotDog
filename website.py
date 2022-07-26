# Operating system library
import os

# Website libraries
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

# Linear algebra
import numpy as np

# Machine learning libraries
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf
import glob

# To fix "call Model.predict with eager mode enabled
tf.compat.v1.disable_eager_execution()

# Two categories
x = 'hotDog'
y = 'notHotDog'
# Static image locations
sample_x = 'static/hotDog.jpg'
sample_y = 'static/notHotDog.jpg'

# Where user uploads are stored
uploadFolder = 'static/uploads'

# Allowed files
allowedExtensions = {'png', 'jpg', 'jpeg', 'gif'}

# Create website object
app = Flask(__name__)


def load_model_from_file():
    # Set up machine learning session
    mySession = tf.compat.v1.Session()
    set_session(mySession)
    # Load the model from the saved_model.h5 file
    myModel = load_model('saved_model.h5')
    # Create the NN from that file
    myGraph = tf.compat.v1.get_default_graph()
    return mySession, myModel, myGraph


# Making sure ONLY images are uploaded
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowedExtensions


# Define the top level page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Initial webpage load
    if request.method == 'GET':
        return render_template('index.html', myX=x, myY=y, mySampleX=sample_x, mySampleY=sample_y)
    if request.method == 'POST':  # I.e. else
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser may submit an empty part
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If the upload is anything other than an image file
        if not allowed_file(file.filename):
            flash('Only files of type' + str(allowedExtensions) + 'are accepted')
            return redirect(request.url)
        # A file with the correct parameters are uploaded
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))


# Work on uploaded file
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    test_image = image.load_img(uploadFolder + '/' + filename, target_size=(150, 150))
    # Convert the image to an array
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    mySession = app.config['SESSION']
    myModel = app.config['MODEL']
    myGraph = app.config['GRAPH']

    with myGraph.as_default():
        set_session(mySession)
        result = myModel.predict(test_image)
        image_src = '/' + uploadFolder + '/' + filename
        # If the result is 0, then the uploaded image is category x, hotDog
        if result[0] < 0.5:
            answer = "<div class='col text-center'><img width='150' height='150' src='" + image_src + "' class='img-thumbnail' /><h4>guess:" + x + " " + str(
                result[0]) + "</h4></div><div class='col'></div><div class='w-100'></div>"
        # If the result is 1, then the uploaded image is category y, notHotDog
        else:
            answer = "<div class='col'></div><div class='col text-center'><img width='150' height='150' src='" + image_src + "' class='img-thumbnail' /><h4>guess:" + y + " " + str(
                result[0]) + "</h4></div><div class='w-100'></div>"
        results.append(answer)
        return render_template('index.html', myX=x, myY=y, mySampleX=sample_x, mySampleY=sample_y, len=len(results),
                               results=results)


def clear_uploads():
    files = glob.glob('static/uploads/*')
    for f in files:
        os.remove(f)


def main():
    (mySession, myModel, myGraph) = load_model_from_file()

    app.config['SECRET_KEY'] = 'secret key'

    app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = myGraph

    app.config['UPLOAD_FOLDER'] = uploadFolder
    # Upload limit (16mb)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    clear_uploads()
    app.run()


# Running list of results
results = []
del results[:]
results.clear()

# Launch everything
main()
