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

# Fungsi untuk memproses file CSV klasemen


def process_csv_file(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        headers = next(csvreader)
        for row in csvreader:
            data.append(row)
    return headers, data

# Fungsi untuk memproses file CSV pertandingan


def process_csv_file_match(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        headers = next(csvreader)
        for row in csvreader:
            data.append(row)
    return headers, data

# Fungsi untuk melakukan prediksi dan menghitung akurasi klasemen


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
        attr_dataset, cls_dataset, test_size=0.2, random_state=1)
    tree_dataset = DecisionTreeClassifier(random_state=1)
    tree_dataset.fit(xtrain, ytrain)

    ypred = tree_dataset.predict(xtest)
    akurasi = accuracy_score(ytest, ypred)
    classification_report_str = classification_report(
        ytest, ypred, zero_division=1)

    return akurasi, classification_report_str

# Fungsi untuk melakukan prediksi dan menghitung akurasi pertandingan


def predict_and_evaluate_match(file_path):
    dataset = pd.read_csv(file_path, delimiter=';', header=0)

    # Menghapus kolom "nama_tim"
    dataset = dataset.drop(columns="pertandingan", axis=1)

    attr_dataset = dataset.drop(columns='hasil')
    cls_dataset = dataset['hasil']

    xtrain, xtest, ytrain, ytest = train_test_split(
        attr_dataset, cls_dataset, test_size=0.2, random_state=1)
    tree_dataset = DecisionTreeClassifier(criterion='entropy')
    tree_dataset.fit(xtrain, ytrain)

    ypred = tree_dataset.predict(xtest)
    akurasi = accuracy_score(ytest, ypred)
    classification_report_str = classification_report(
        ytest, ypred, zero_division=1)

    return akurasi, classification_report_str
# Fungsi untuk proses dataset klasemen


def predict_and_evaluate_match_input(file_path, data_input):
    dataset = pd.read_csv(file_path, delimiter=';', header=0)
    dataset = dataset.drop(labels="pertandingan", axis=1)

    attr_dataset = dataset.drop(columns='hasil')
    cls_dataset = dataset['hasil']

    xtrain, xtest, ytrain, ytest = train_test_split(
        attr_dataset, cls_dataset, test_size=0.9, random_state=1)
    tree_dataset = DecisionTreeClassifier(random_state=1)
    tree_dataset.fit(xtrain, ytrain)

    ypred = tree_dataset.predict(data_input)

    # akurasi = accuracy_score(ytest, ypred)

    return ypred[0]


@app.route('/', methods=['GET', 'POST'])
def upload_csv():
    akurasi = 0
    classification_report_str = ""

    if request.method == 'POST':
        file = request.files['csv_file']
        if file and file.filename.endswith('.csv'):
            filename = file.filename
            file.save(os.path.join('uploads/klasemen', filename))
            flash(f'File {filename} berhasil diunggah.', 'success')

            headers, data_rows = process_csv_file(
                os.path.join('uploads/klasemen', filename))

            # Prediksi dan evaluasi hasil
            akurasi, classification_report_str = predict_and_evaluate1(
                os.path.join('uploads/klasemen', filename))
        else:
            flash(
                f'File {file.filename} gagal diunggah. File yang diunggah harus berformat .csv.', 'error')
            headers, data_rows = [], []
    else:
        headers, data_rows = [], []

    return render_template('index.html', headers=headers, data_rows=data_rows, akurasi=akurasi*100, classification_report_str=classification_report_str)

# Fungsi untuk proses dataset pertandingan


@app.route('/prediksi', methods=['GET', 'POST'])
def upload_csv_match():
    akurasi = 0
    classification_report_str = ""

    if request.method == 'POST':
        file = request.files['csv_file']
        if file and file.filename.endswith('.csv'):
            filename = file.filename
            file.save(os.path.join('uploads/pertandingan', filename))
            flash(f'File {filename} berhasil diunggah.', 'success')

            headers, data_rows = process_csv_file_match(
                os.path.join('uploads/pertandingan', filename))

            # Prediksi dan evaluasi hasil
            akurasi, classification_report_str = predict_and_evaluate_match(
                os.path.join('uploads/pertandingan', filename))

            for row in data_rows:
                row[1] = 'Biasa' if row[1] == 0 else (
                    'Rata-Rata' if row[1] == 1 else 'Bagus')
                row[2] = 'Biasa' if row[2] == 0 else (
                    'Rata-Rata' if row[2] == 1 else 'Bagus')
                row[3] = 'Baru Melatih' if row[3] == 0 else (
                    'Berprestasi' if row[3] == 1 else 'Berpengalaman')
                row[4] = 'Baru Melatih' if row[4] == 0 else (
                    'Berprestasi' if row[4] == 1 else 'Berpengalaman')
                row[5] = 'Counter Attack' if row[5] == 0 else (
                    'Long Ball' if row[5] == 1 else ('Build Up' if row[4] == 2 else 'Tiki-Taka'))
                row[6] = 'Counter Attack' if row[6] == 0 else (
                    'Long Ball' if row[6] == 1 else ('Build Up' if row[5] == 2 else 'Tiki-Taka'))
                row[7] = 'Low Block' if row[7] == 0 else ('Man Marking' if row[6] == 1 else (
                    'High Pressing' if row[7] == 2 else 'Offside Trap'))
                row[8] = 'Low Block' if row[8] == 0 else ('Man Marking' if row[7] == 1 else (
                    'High Pressing' if row[8] == 2 else 'Offside Trap'))

        else:
            flash(
                f'File {file.filename} gagal diunggah. File yang diunggah harus berformat .csv.', 'error')
            headers, data_rows = [], []
    else:
        headers, data_rows = [], []

    return render_template('prediksi.html', headers=headers, data_rows=data_rows, akurasi=akurasi*100, classification_report_str=classification_report_str)


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
            os.path.join('uploads/klasemen', 'premierleague.csv'), data_input)

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


@app.route('/predict-match', methods=['POST'])
def predict_match():
    if request.method == 'POST':
        # Get the input data from the form
        nama_tim_home = request.form.get('floatingTimHome')
        nama_tim_away = request.form.get('floatingTimAway')
        rating_tim_home = request.form.get('floatingRatingTimHome')
        rating_tim_away = request.form.get('floatingRatingTimAway')
        kualitas_pelatih_home = request.form.get(
            'floatingKualitasPelatihHome')
        kualitas_pelatih_away = request.form.get(
            'floatingKualitasPelatihAway')
        menyerang_home = request.form.get('floatingMenyerangHome')
        menyerang_away = request.form.get('floatingMenyerangAway')
        bertahan_home = request.form.get('floatingBertahanHome')
        bertahan_away = request.form.get('floatingBertahanAway')

        # Load the uploaded CSV file and preprocess the input data for prediction
        data_input = [[rating_tim_home, rating_tim_away, kualitas_pelatih_home,
                       kualitas_pelatih_away, menyerang_home, menyerang_away, bertahan_home, bertahan_away]]

        # # Perform the prediction using the C4.5 algorithm
        prediction_result = predict_and_evaluate_match_input(
            os.path.join('uploads/pertandingan', 'match-dataset.csv'), data_input)

        if prediction_result == 'kalah':
            kelas = 'Kalah'
        elif prediction_result == 'imbang':
            kelas = 'Imbang'
        else:
            kelas = 'Menang'

        if rating_tim_home == '0':
            rating_tim_home = 'Biasa'
        elif rating_tim_home == '1':
            rating_tim_home = 'Rata-Rata'
        else:
            rating_tim_home = 'Bagus'

        if rating_tim_away == '0':
            rating_tim_away = 'Biasa'
        elif rating_tim_away == '1':
            rating_tim_away = 'Rata-Rata'
        else:
            rating_tim_away = 'Bagus'

        if kualitas_pelatih_home == '0':
            kualitas_pelatih_home = 'Baru Melatih'
        elif kualitas_pelatih_home == '1':
            kualitas_pelatih_home = 'Berprestasi'
        else:
            kualitas_pelatih_home = 'Berpengalaman'

        if kualitas_pelatih_away == '0':
            kualitas_pelatih_away = 'Baru Melatih'
        elif kualitas_pelatih_away == '1':
            kualitas_pelatih_away = 'Berprestasi'
        else:
            kualitas_pelatih_away = 'Berpengalaman'

        if menyerang_home == '0':
            menyerang_home = 'Counter Attack'
        elif menyerang_home == '1':
            menyerang_home = 'Long Ball'
        elif menyerang_home == '2':
            menyerang_home = 'Build Up'
        else:
            menyerang_home = 'Tiki-Taka'

        if menyerang_away == '0':
            menyerang_away = 'Counter Attack'
        elif menyerang_away == '1':
            menyerang_away = 'Long Ball'
        elif menyerang_away == '2':
            menyerang_away = 'Build Up'
        else:
            menyerang_away = 'Tiki-Taka'

        if bertahan_home == '0':
            bertahan_home = 'Low Block'
        elif bertahan_home == '1':
            bertahan_home = 'Man Marking'
        elif bertahan_home == '2':
            bertahan_home = 'High Pressing'
        else:
            bertahan_home = 'Offside Trap'

        if bertahan_away == '0':
            bertahan_away = 'Low Block'
        elif bertahan_away == '1':
            bertahan_away = 'Man Marking'
        elif bertahan_away == '2':
            bertahan_away = 'High Pressing'
        else:
            bertahan_away = 'Offside Trap'

        # Return the prediction results as JSON
        return jsonify({
            'prediction_result': kelas,
            'nama_tim_home': nama_tim_home,
            'nama_tim_away': nama_tim_away,
            'rating_tim_home': rating_tim_home,
            'rating_tim_away': rating_tim_away,
            'kualitas_pelatih_home': kualitas_pelatih_home,
            'kualitas_pelatih_away': kualitas_pelatih_away,
            'menyerang_home': menyerang_home,
            'menyerang_away': menyerang_away,
            'bertahan_home': bertahan_home,
            'bertahan_away': bertahan_away,
        })


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediksi')
def prediksi():
    return render_template('prediksi.html')


if __name__ == '__main__':
    app.run(debug=True)
