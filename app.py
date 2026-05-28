from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ============================================================
# LOAD MODEL DAN FILE
# ============================================================

model = pickle.load(
    open('model/knn_model.pkl', 'rb')
)

scaler = pickle.load(
    open('model/scaler.pkl', 'rb')
)

label_encoders = pickle.load(
    open('model/label_encoders.pkl', 'rb')
)

best_k = pickle.load(
    open('model/best_k.pkl', 'rb')
)

feature_columns = pickle.load(
    open('model/feature_columns.pkl', 'rb')
)

# ============================================================
# FUNCTION ENCODING
# ============================================================

def encode_input(column_name, value):

    le = label_encoders[column_name]

    return le.transform([value])[0]

# ============================================================
# CEK LABEL TARGET
# ============================================================

print("\n================ LABEL TARGET ================")

print(label_encoders['Risk'].classes_)

# ============================================================
# HOME
# ============================================================

@app.route('/')
def index():

    return render_template(
        'index.html',
        best_k=best_k
    )

# ============================================================
# PREDICT
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # ====================================================
        # INPUT USER
        # ====================================================

        age = int(request.form['age'])

        sex = request.form['sex']

        job = int(request.form['job'])

        housing = request.form['housing']

        saving = request.form['saving']

        checking = request.form['checking']

        credit_amount = float(
            request.form['credit_amount']
        )

        duration = int(
            request.form['duration']
        )

        purpose = request.form['purpose']

        # ====================================================
        # ENCODING MENGGUNAKAN LABEL ENCODER ASLI
        # ====================================================

        features = np.array([[

            age,

            encode_input('Sex', sex),

            job,

            encode_input('Housing', housing),

            encode_input('Saving accounts', saving),

            encode_input('Checking account', checking),

            credit_amount,

            duration,

            encode_input('Purpose', purpose)

        ]])

        # ====================================================
        # SCALING
        # ====================================================

        features_scaled = scaler.transform(
            features
        )

        # ====================================================
        # PREDIKSI
        # ====================================================

        prediction = model.predict(
            features_scaled
        )[0]

        probabilities = model.predict_proba(
            features_scaled
        )[0]

        confidence = round(
            np.max(probabilities) * 100,
            2
        )

        # ====================================================
        # LABEL TARGET ASLI
        # ====================================================

        risk_label = label_encoders['Risk'].inverse_transform(
            [prediction]
        )[0]

        # ====================================================
        # HASIL
        # ====================================================

        if risk_label.lower() == 'good':

            result = 'GOOD'

            status_color = 'success'

            description = 'LAYAK mendapatkan kredit.'

        else:

            result = 'BAD'

            status_color = 'danger'

            description = 'TIDAK LAYAK mendapatkan kredit.'

        # ====================================================
        # DEBUG TERMINAL
        # ====================================================

        print("\n================ INPUT USER ================")

        print(features)

        print("\n================ HASIL ================")

        print("Prediction :", prediction)

        print("Risk Label :", risk_label)

        print("Probabilities :", probabilities)

        print("Confidence :", confidence)

        # ====================================================
        # RENDER
        # ====================================================

        return render_template(

            'result.html',

            result=result,

            description=description,

            confidence=confidence,

            status_color=status_color,

            best_k=best_k,

            input_data={

                'Age': age,

                'Sex': sex,

                'Job': job,

                'Housing': housing,

                'Saving': saving,

                'Checking': checking,

                'Credit Amount': credit_amount,

                'Duration': duration,

                'Purpose': purpose
            }
        )

    except Exception as e:

        return f"""
        <h2>ERROR TERJADI</h2>
        <p>{str(e)}</p>
        """

# ============================================================
# RUN APP
# ============================================================

if __name__ == '__main__':

    app.run(
        debug=True
    )