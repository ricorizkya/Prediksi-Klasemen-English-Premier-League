import csv
from flask import Flask, request, render_template, flash, session, jsonify
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Fungsi untuk memproses file CSV


def process_csv_file(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        headers = next(csvreader)
        for row in csvreader:
            data.append(row)
    return headers, data

# Fungsi untuk melakukan prediksi dan menghitung akurasi


def predict_and_evaluate(file_path, data_input):
    dataset = pd.read_csv(file_path, delimiter=';', header=0)
    dataset = dataset.drop(labels="nama_tim", axis=1)

    enc = LabelEncoder()
    dataset['kualitas_pelatih'] = enc.fit_transform(
        dataset['kualitas_pelatih'].values)

    attr_dataset = dataset.drop(columns='status')
    cls_dataset = dataset['status']

    xtrain, xtest, ytrain, ytest = train_test_split(
        attr_dataset, cls_dataset, test_size=None, random_state=1)
    tree_dataset = DecisionTreeClassifier(random_state=1)
    tree_dataset.fit(xtrain, ytrain)

    ypred = tree_dataset.predict(data_input)

    # akurasi = accuracy_score(ytest, ypred)

    return ypred[0]


def predict_and_evaluate1(file_path):
    dataset = pd.read_csv(file_path, delimiter=';', header=0)

    # Menghapus kolom "nama_tim"
    dataset = dataset.drop(labels="nama_tim", axis=1)

    # Encoding kolom 'kualitas_pelatih'
    enc = LabelEncoder()
    dataset['kualitas_pelatih'] = enc.fit_transform(
        dataset['kualitas_pelatih'].values)

    attr_dataset = dataset.drop(columns='status')
    cls_dataset = dataset['status']

    xtrain, xtest, ytrain, ytest = train_test_split(
        attr_dataset, cls_dataset, test_size=0.9, random_state=1)
    tree_dataset = DecisionTreeClassifier(random_state=1)
    tree_dataset.fit(xtrain, ytrain)

    ypred = tree_dataset.predict(xtest)
    akurasi = accuracy_score(ytest, ypred)
    classification_report_str = classification_report(
        ytest, ypred, zero_division=1)

    return akurasi, classification_report_str


@app.route('/', methods=['GET', 'POST'])
def upload_csv():
    akurasi = 0
    classification_report_str = ""

    if request.method == 'POST':
        file = request.files['csv_file']
        if file and file.filename.endswith('.csv'):
            filename = file.filename
            file.save(os.path.join('uploads', filename))
            flash(f'File {filename} berhasil diunggah.', 'success')

            headers, data_rows = process_csv_file(
                os.path.join('uploads', filename))

            # Prediksi dan evaluasi hasil
            akurasi, classification_report_str = predict_and_evaluate1(
                os.path.join('uploads', filename))
        else:
            flash(
                f'File {file.filename} gagal diunggah. File yang diunggah harus berformat .csv.', 'error')
            headers, data_rows = [], []
    else:
        headers, data_rows = [], []

    return render_template('index.html', headers=headers, data_rows=data_rows, akurasi=akurasi*100, classification_report_str=classification_report_str)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        point_akhir = request.form.get('floatingPoint')
        jumlah_gol = request.form.get('floatingGol')
        jumlah_masuk = request.form.get('floatingBobol')
        rating_tim = request.form.get('floatingRating')
        kualitas_pelatih = request.form.get('floatingPelatih')
        nama_tim = request.form.get('floatingTim')

        # Load the uploaded CSV file and preprocess the input data for prediction
        data_input = [[point_akhir, jumlah_gol,
                       jumlah_masuk, rating_tim, kualitas_pelatih]]

        # # Perform the prediction using the C4.5 algorithm
        prediction_result = predict_and_evaluate(
            os.path.join('uploads', 'premierleague.csv'), data_input)

        # # Convert the prediction result to the corresponding class name
        if prediction_result == 'ucl':
            kelas = 'Liga Champions (UCL)'
        elif prediction_result == 'uecl':
            kelas = 'Liga Conference (UECL)'
        elif prediction_result == 'degradasi':
            kelas = 'Degradasi'
        elif prediction_result == 'no_europe':
            kelas = 'Tidak Bermain di Turnamen Eropa'
        else:
            kelas = 'Liga Eropa (UEL)'

        # # Return the prediction results as JSON
        return jsonify({
            'prediction_result': kelas,
            'nama_tim': nama_tim,
            'floatingPoint': point_akhir,
            'floatingGol': jumlah_gol,
            'floatingBobol': jumlah_masuk,
            'floatingRating': rating_tim,
            'floatingPelatih': kualitas_pelatih,
        })
        # return render_template('index.html', nama_tim=nama_tim, point_akhir=point_akhir, jumlah_gol=jumlah_gol, jumlah_masuk=jumlah_masuk, rating_tim=rating_tim, kualitas_pelatih=kualitas_pelatih)


if __name__ == '__main__':
    app.run(debug=True)
