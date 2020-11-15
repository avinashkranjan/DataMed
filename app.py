from flask import Flask, render_template, request, url_for, redirect, abort
import pickle
import numpy as np
from werkzeug import exceptions
from sklearn.linear_model import LogisticRegression
from keras.models import load_model


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    if request.method == 'POST':
        '''
        try:
            # database work
            return redirect(url_for('index')) 
        except exceptions.BadRequest:

        pass
        '''
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/predict/<disease>', methods=['POST', 'GET'])
def predict(disease):
    if disease == 'HeartFailure':
        if request.method == 'POST':
            try:
                float_features = [float(x) for x in request.form.values()]
                final = [np.array(float_features)]
                model = pickle.load(
                    open('ML/Heart_Failure/finalized_model.sav', 'rb'))
                prediction = model.predict_proba(final)
                output = '{0:.{1}f}'.format((prediction[0][1]*100), 2)
                parameter = []
                parameter.append(int(request.form["age"]))
                parameter.append(disease)
                [parameter.append("Yes") if prediction[0][1] >
                 0.5 else parameter.append("No")]
                parameter.append(output)
                return render_template('result-hrt.html', result=parameter)
            except exceptions.BadRequest:
                return abort(400)
            except ValueError:
                return abort(400)
        else:
            return redirect(url_for('index'))

    if disease == 'BreastCancer':
        if request.method == 'POST':
            try:
                float_features = [float(x) for x in request.form.values()]
                final = [np.array(float_features)]
                model = load_model("ML/BreastCancer/breast_cancer.h5")
                prediction = model.predict(final)
                output = '{0:.{1}f}'.format((prediction[0][0]*100), 2)
                parameter = {}
                parameter['Disease'] = disease
                parameter['Condition'] = ["Malignant" if prediction[0][0] >= 0.5 else "Benign"]
                parameter['Probability'] = output
                return render_template('result.html', result=parameter)
            except exceptions.BadRequest:
                return abort(400)
            except ValueError:
                return abort(400)
        else:
            return redirect(url_for('index'))
    else:
        abort(404)


if (__name__) == "__main__":
    app.run(debug=True)
