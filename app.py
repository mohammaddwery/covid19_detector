import cv2
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
from os import path
from datetime import datetime
from sklearn.metrics import accuracy_score

app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'
db = SQLAlchemy(app)


class ImagesModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(80))
    accuracy = db.Column(db.String(80))
    result = db.Column(db.String(80))
    createdAt = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


ma = Marshmallow(app)


class ImagesSchema(ma.Schema):
    class Meta:
        fields = ('id', 'url', 'accuracy', 'result', 'createdAt')


image_schema = ImagesSchema()
images_schema = ImagesSchema(many=True)

if not path.exists("./db.sqlite"):
    db.create_all(app=app)
    print('Created Database!')

model = load_model('model/classification_model.h5')
label_dict = {0: 'Covid19 Negative', 1: 'Covid19 Positive'}
img_size = 150


@app.route("/")
def index():
    # model.summary()
    return render_template("index.html")


@app.route('/images', methods=['GET'])
def get_images():
    tasks = ImagesModel.query.order_by(ImagesModel.id.desc())
    result = images_schema.dump(tasks)
    return jsonify(result)


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory('images', filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if not uploaded_file:
            response = {'error': {'message': 'Please upload a valid image'}}
            return jsonify(response)

        if uploaded_file.filename != '':
            file_name = uploaded_file.filename
            filename = secure_filename(file_name)
            uploaded_file.save('images/' + filename)

            img = image.load_img('images/' + filename, target_size=(150, 150))  # this is a PIL image
            x = image.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = np.expand_dims(x, axis=0)
            prediction = model.predict(x)
            # result = np.argmax(prediction, axis=1)[0]
            label = ''
            if prediction == 0:
                label = label_dict[1]
            else:
                label = label_dict[0]

            accuracy = float(np.max(prediction, axis=1)[0])
            #
            # train_preds = np.where(model.predict(x) > 0.5, 1, 0)
            # train_accuracy = accuracy_score(train_preds, train_preds)
            #
            # print(f'Train Accuracy : {train_accuracy:.4f}')

            print('result:', prediction, 'accuracy:', accuracy)

            image_model = ImagesModel(url=url_for('get_image', filename=filename), accuracy=accuracy, result=label)
            db.session.add(image_model)
            db.session.commit()

            response = image_schema.dump(ImagesModel.query.filter_by(id=image_model.id).first())

            # response = {'prediction': {'result': label, 'accuracy': accuracy}}

            return jsonify(response)


if __name__ == "__main__":
    # app.run(debug=True)
    app.debug = True
    app.run(host="0.0.0.0")
