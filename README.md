## INFORMASI PROYEK
**Judul Proyek:**  
Analisis Sentimen Opini Publik di Media Sosial Terhadap Kebijakan PPKM sebagai Evaluasi Transparansi Pemerintah

**Nama Mahasiswa:**  
1. Iqbal Abdullah  
2. Rindy Cantika Agustina Putri  

**NIM:**  
1. 244311044  
2. 244311057  

- **Program Studi:** [Isi Program Studi]
- **Mata Kuliah:** Data Science
- **Dosen Pengampu:** Gus Nanang Syaifuddin, S.Kom., M.Kom.  
- **Tahun Akademik:** [Isi Tahun/Semester]  
- **Link GitHub Repository:** https://github.com/Adzer2605/id-ppkm-sentiment-analysis
- **Link Video Pembahasan:** [URL Video (YouTube/Drive)]  
- **SDG yang Dipilih:** SDG 16 – Perdamaian, Keadilan, dan Kelembagaan yang Tangguh  
- **Target SDG yang Relevan:** Target 16.6 – mengembangkan lembaga yang efektif, akuntabel, dan transparan di semua tingkat; serta Target 16.10 – menjamin akses publik terhadap informasi dan Target 16.7 – memastikan pengambilan keputusan yang responsif, inklusif, dan partisipatif

## 0. KERANGKA ML LIFECYCLE (L-M-O-P)

Laporan ini disusun mengikuti **ML Lifecycle** yang terdiri dari empat tahap berulang (iteratif):

| Tahap                  | Singkatan | Inti Pertanyaan                                                               | Bagian Laporan        |
| ---------------------- | --------- | ----------------------------------------------------------------------------- | --------------------- |
| **Learning Problem**   | **L**     | Apa target, objective, dan datanya?                                           | Bab 2, 3, 4, 5        |
| **Model Design**       | **M**     | Model family/arsitektur apa, hypothesis space, dan inductive bias/asumsi apa? | Bab 6 (6.x.1 – 6.x.3) |
| **Optimization**       | **O**     | Bagaimana model dilatih: loss, optimizer, tuning, regularisasi, konvergensi?  | Bab 6 (6.x.4 – 6.x.6) |
| **Predict & Evaluate** | **P**     | Seberapa baik prediksi model berdasarkan metrik evaluasi?                     | Bab 7                 |

```
   L (Learning Problem) ──► M (Model Design) ──► O (Optimization) ──► P (Predict & Evaluate)
          ▲                                                                   │
          └─────────────── iterasi / perbaikan berdasarkan hasil evaluasi ────┘
```

**Catatan untuk mahasiswa:** Setiap bagian diberi label **[L]**, **[M]**, **[O]**, atau **[P]**. Pastikan Anda dapat menjelaskan keterkaitan antar tahap, misalnya bagaimana hasil **P** mendorong perbaikan pada **L/M/O** (Bab 7.5).

---

## 1. LEARNING OUTCOMES

Pada proyek ini, mahasiswa diharapkan dapat:

1. **[L]** Mengidentifikasi masalah nyata dari **ide sendiri** yang selaras dengan salah satu tujuan **SDGs** ([Agenda 2030 – sdgs.un.org/2030agenda](https://sdgs.un.org/2030agenda)), lalu merumuskan problem statement (target, objective, data) secara jelas
2. **[L]** Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. **[L]** Melakukan data preparation yang sesuai dengan karakteristik dataset
4. **[M]** Merancang tiga model machine learning dengan justifikasi hypothesis space dan inductive bias (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. **[O]** Mengoptimasi model secara sistematis (loss function, optimizer, hyperparameter tuning, regularisasi)
6. **[P]** Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML dan menganalisis hasilnya
7. Melaporkan hasil eksperimen secara ilmiah dan sistematis
8. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
9. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW **[L]**

### 2.1 Latar Belakang

Pemberlakuan Pembatasan Kegiatan Masyarakat (PPKM) adalah kebijakan utama pemerintah Indonesia untuk menekan penyebaran COVID-19 sejak awal 2021. Kebijakan ini memengaruhi mobilitas, kegiatan ekonomi, ibadah, pendidikan, hingga layanan publik. Karena aturannya berubah-ubah (level 1–4, PPKM Mikro, PPKM Darurat, perpanjangan berkala), efektivitasnya sangat bergantung pada seberapa jelas pemerintah mengomunikasikan dasar keputusan dan seberapa baik publik memahaminya.

Media sosial, khususnya Twitter, menjadi kanal utama publik untuk bertanya, menyampaikan keluhan, dan merespons kebijakan secara langsung. Volume cuitan yang besar tidak mungkin dibaca manual oleh pembuat kebijakan. Pada dataset yang digunakan, volume cuitan memuncak pada Juli 2021 (9.603 cuitan), bertepatan dengan periode PPKM Darurat, yang menunjukkan bahwa perhatian publik bereaksi kuat terhadap perubahan kebijakan. Analisis sentimen otomatis dapat meringkas arah opini publik (positif, netral, negatif) sehingga lembaga pemerintah dapat mengevaluasi apakah kebijakan dan komunikasinya diterima.

Studi terdahulu menunjukkan bahwa analisis sentimen cuitan PPKM layak dilakukan dengan _machine learning_: Hansun dkk. (2022) menemukan sentimen netral dominan (58,07%), diikuti positif (27,12%) dan negatif (14,81%), lalu membangun model berbasis LSTM. Untuk bahasa Indonesia, model pra-latih IndoBERT (Wilie dkk., 2020) menyediakan representasi kontekstual yang relevan. Proyek ini membandingkan tiga pendekatan, yaitu Logistic Regression dengan TF-IDF, Linear SVM dengan TF-IDF, dan _fine-tuning_ IndoBERT.

**Referensi:**

> Hansun, S., Suryadibrata, A., Nurhasanah, R., & Fitra, J. (2022). Tweets sentiment on PPKM policy as a Covid-19 response in Indonesia. _Indian Journal of Computer Science and Engineering (IJCSE), 13_(1), 51–58.

> Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim, Z. Y., ... & Purwarianti, A. (2020). IndoNLU: Benchmark and resources for evaluating Indonesian natural language understanding. _Proceedings of AACL-IJCNLP 2020_, 843–857.

> United Nations. _Goal 16: Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels_. https://sdgs.un.org/goals/goal16

### 2.2 Keterkaitan dengan SDGs (WAJIB)

| Komponen | Pertanyaan Panduan | Jawaban Anda |
| --- | --- | --- |
| **SDG yang dipilih** | Goal nomor berapa dan apa namanya? | SDG 16 – Perdamaian, Keadilan, dan Kelembagaan yang Tangguh |
| **Target SDG** | Target (misal 3.4, 13.1) mana yang paling relevan? | Target 16.6 (lembaga yang efektif, akuntabel, dan transparan di semua tingkat) sebagai target utama, didukung Target 16.10 (akses publik terhadap informasi) dan Target 16.7 (pengambilan keputusan yang responsif dan partisipatif) |
| **Masalah nyata** | Masalah konkret apa yang diangkat? Siapa yang terdampak? | Opini publik terhadap kebijakan PPKM tersebar dalam ribuan cuitan sehingga sulit dipantau secara manual. Tanpa pemantauan yang sistematis, lembaga pemerintah sulit menilai apakah kebijakan dan komunikasinya dipahami dan diterima. Pihak terdampak: masyarakat yang menjalani PPKM, pelaku usaha, serta instansi pemerintah pusat dan daerah. |
| **Peran ML** | Mengapa ML/DL cocok? Apa keputusan atau proses yang dibantu? | Klasifikasi sentimen teks bahasa Indonesia dengan ML/DL mengotomatiskan pelabelan ribuan cuitan sehingga sentimen publik dapat dipantau dari waktu ke waktu. Proses yang dibantu adalah evaluasi penerimaan publik terhadap kebijakan dan komunikasinya. |
| **Indikator kontribusi** | Bagaimana hasil model berkontribusi pada target SDG (langsung/tidak langsung)? | Tidak langsung: proporsi sentimen positif, netral, dan negatif per periode menjadi indikator proksi respons publik yang dapat dipakai sebagai salah satu masukan evaluasi komunikasi kebijakan. Model tidak mengukur transparansi itu sendiri dan tidak menggantikan survei atau audit resmi. |
| **Pertimbangan etika** | Risiko bias, privasi, atau dampak salah prediksi? | Privasi: cuitan bersifat publik tetapi memuat _username_, sehingga mention dan URL dihapus, analisis dilakukan agregat, dan username tidak dipublikasikan. Bias: pengguna Twitter tidak mewakili seluruh masyarakat dan data didominasi akun berita/instansi (cenderung netral). Label juga tidak seluruhnya konsisten (ada teks identik berlabel berbeda). Dampak salah prediksi: sentimen negatif yang terlewat dapat membuat keluhan publik tidak terpantau, sehingga _recall_ kelas negatif dipantau. Hasil tidak digunakan untuk menilai atau menindak individu. |

**Sumber ide:** Ide berasal dari pengalaman masyarakat menjalani kebijakan PPKM selama pandemi dan pengamatan bahwa respons publik terhadap kebijakan terekspresikan luas di media sosial. Kemampuan lembaga memantau dan merespons opini publik berkaitan dengan akuntabilitas dan transparansi kelembagaan (SDG 16).

### 2.3 Ringkasan Alur Lifecycle Proyek

Pada tahap **L**, masalah dirumuskan sebagai klasifikasi teks multikelas (positif, netral, negatif) dengan dataset _Twitter Dataset PPKM_ dari Kaggle (23.644 cuitan), lalu data dipersiapkan (pembersihan URL dan mention, penghapusan duplikat, dan _splitting_ _stratified_ tanpa kebocoran data). Pada tahap **M**, dirancang tiga model: Logistic Regression dengan TF-IDF sebagai baseline, Linear SVM dengan TF-IDF sebagai model advanced, serta _fine-tuning_ IndoBERT sebagai model deep learning. Pada tahap **O**, model dilatih dan dioptimasi (Grid Search dengan _cross-validation_ untuk SVM; pengaturan learning rate, batch size, dan _weighted cross-entropy_ untuk IndoBERT). Pada tahap **P**, ketiga model dievaluasi pada test set dengan Macro F1-Score, Accuracy, Precision, dan Recall (terutama kelas negatif), disertai analisis kesalahan. Hasil evaluasi digunakan untuk satu iterasi perbaikan pada tahap L/M/O.

---

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING **[L]**

### 3.1 Problem Statements

1. Opini publik terhadap kebijakan PPKM tersebar dalam puluhan ribu cuitan sehingga tidak dapat dipantau secara manual; dibutuhkan model yang mengklasifikasikan sentimen (positif, netral, negatif) secara otomatis dan akurat.
2. Distribusi kelas sangat tidak seimbang (netral 74,9%, negatif 16,8%, positif 8,3%) sehingga model berisiko mengabaikan kelas minoritas, padahal sentimen negatif adalah sinyal terpenting bagi evaluasi respons publik.
3. Teks Twitter berbahasa Indonesia bersifat _noisy_ (URL, mention, hashtag, singkatan) dan memuat duplikat, sehingga diperlukan preprocessing yang tepat tanpa menimbulkan kebocoran data (_data leakage_).
4. Dibutuhkan model _deep learning_ berbasis representasi kontekstual yang dibandingkan secara adil dengan model klasik berbasis TF-IDF, serta metrik evaluasi yang sesuai untuk data tidak seimbang.

### 3.2 Formulasi Learning Problem (WAJIB)

| Komponen | Pertanyaan Panduan | Jawaban Anda |
| --- | --- | --- |
| **Target** | Apa yang diprediksi? Jenis tugas (klasifikasi biner/multikelas, regresi, forecasting, clustering, dll.)? | Label sentimen cuitan (kolom `sentiment`): **0 = positif, 1 = netral, 2 = negatif**. Jenis tugas: klasifikasi teks multikelas (3 kelas), _Natural Language Processing_. |
| **Objective** | Apa yang dioptimalkan/diminimalkan secara matematis (misal: cross-entropy, MSE)? Metrik keberhasilan utama? | Meminimalkan _cross-entropy_ berbobot kelas (log-loss + L2 pada Logistic Regression, hinge loss + L2 pada Linear SVM, weighted cross-entropy pada IndoBERT). Metrik utama: **Macro F1-Score** (karena kelas sangat tidak seimbang), dengan pendukung Accuracy, Precision, Recall per kelas (terutama Recall kelas negatif), dan Confusion Matrix. |
| **Data** | Sumber, ukuran, jenis, dan asumsi tentang data (i.i.d.? distribusi train vs test?) | Sumber: [Twitter Dataset PPKM (Kaggle, anggapurnama)](https://www.kaggle.com/datasets/anggapurnama/twitter-dataset-ppkm), berkas `INA_TweetsPPKM_Labeled_Pure.csv`. Jenis: teks berbahasa Indonesia. Ukuran: 23.644 baris × 4 kolom (`Date`, `User`, `Tweet`, `sentiment`), ±5,25 MB, periode Mei 2020 – Maret 2022. Asumsi: cuitan dianggap i.i.d. setelah duplikat dihapus, dengan catatan cuitan terkonsentrasi pada Juli–Agustus 2021 dan satu akun dapat menulis banyak cuitan, sehingga _splitting_ dilakukan _stratified_ setelah deduplikasi untuk mencegah kebocoran data. |

### 3.3 Goals

1. Membangun model klasifikasi sentimen tiga kelas pada cuitan PPKM dengan Accuracy > 80% dan Macro F1-Score ≥ 0,70 pada test set untuk model terbaik (target awal, dapat disesuaikan setelah EDA).
2. Mengukur dan membandingkan performa tiga pendekatan model: Logistic Regression + TF-IDF (baseline), Linear SVM + TF-IDF (advanced), dan IndoBERT (deep learning), dengan target model deep learning mengungguli baseline minimal +2 poin Macro F1-Score.
3. Menjaga Recall kelas negatif ≥ 0,60 pada model terbaik, serta menganalisis kesalahan prediksi dan perbandingan waktu training dan inferensi ketiga model.
4. Menghasilkan pipeline yang _reproducible_ (random seed 42, versi pustaka tercatat, kode di GitHub).
5. **Kontribusi SDG 16:** menyediakan alat bantu pemantauan sentimen publik terhadap kebijakan PPKM sebagai indikator proksi respons publik untuk mendukung kelembagaan yang akuntabel dan transparan (Target 16.6, didukung 16.10 dan 16.7).

### 3.4 Solution Approach **[M]**

#### **Model 1 – Baseline Model**

**Logistic Regression** dengan fitur TF-IDF (unigram dan bigram) dan `class_weight='balanced'`. Model ini dipilih sebagai pembanding dasar karena sederhana, cepat, dan mudah diinterpretasi (bobot kata per kelas dapat diinspeksi), serta kerap menjadi tolok ukur standar analisis sentimen.

#### **Model 2 – Advanced / ML Model**

**Linear Support Vector Machine (SVM)** dengan fitur TF-IDF yang sama, di-_tuning_ menggunakan Grid Search dan _stratified k-fold cross-validation_ (parameter C, rentang n-gram, `min_df`). Model ini dipilih karena efektif pada ruang fitur _sparse_ berdimensi tinggi dan mencari margin maksimum sehingga berbeda mendasar dari pendekatan probabilistik Logistic Regression.

#### **Model 3 – Deep Learning Model (WAJIB)**

**Fine-tuning IndoBERT** (`indobenchmark/indobert-base-p1`) (jenis data: teks, _pre-trained Transformer_). Model ini dipilih karena representasi kontekstualnya menangkap negasi dan konteks kalimat yang tidak tertangkap TF-IDF, sudah dipra-latih pada korpus bahasa Indonesia, dan cuitan yang pendek (rata-rata sekitar 21 kata) membuatnya layak dilatih di Google Colab atau Kaggle. Model dilatih minimal 10 epoch dengan _weighted cross-entropy_, plot loss dan metrik per epoch, serta evaluasi pada test set. Jika sumber daya GPU terbatas, alternatif cadangannya adalah _Embedding_ + Bi-LSTM.

---

## 4. DATA UNDERSTANDING **[L]**

### 4.1 Informasi Dataset

**Sumber Dataset:**  
[Sebutkan sumber: Kaggle, UCI ML Repository, atau sumber lain dengan URL]

**Deskripsi Dataset:**

- Jumlah baris (rows): [angka]
- Jumlah kolom (columns/features): [angka]
- Tipe data: [Tabular / Image / Text / Time Series / Audio / Video]
- Ukuran dataset: [MB/GB]
- Format file: [CSV / JSON / Images / TXT / etc.]

### 4.2 Deskripsi Fitur

Jelaskan setiap fitur/kolom yang ada dalam dataset.
**Contoh tabel:**
| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| id | Integer | ID unik data | 1, 2, 3 |
| age | Integer | Usia (tahun) | 25, 30, 45 |
| income | Float | Pendapatan (juta) | 5.5, 10.2 |
| category | Categorical | Kategori produk | A, B, C |
| text | String | Teks ulasan | "Produk bagus..." |
| image | Image | Citra 224x224 RGB | Array 224x224x3 |
| label | Categorical | Label target | 0, 1 atau "positif", "negatif" |

**[Buat tabel deskripsi fitur Anda di sini]**

### 4.3 Kondisi Data

Jelaskan kondisi dan permasalahan data:

- **Missing Values:** [Ada/Tidak, berapa persen?]
- **Duplicate Data:** [Ada/Tidak, berapa banyak?]
- **Outliers:** [Ada/Tidak, pada fitur apa?]
- **Imbalanced Data:** [Ada/Tidak, rasio kelas?]
- **Noise:** [Jelaskan jika ada]
- **Data Quality Issues:** [Jelaskan jika ada masalah lain]
- **Asumsi Data (i.i.d., distribusi train-test, potensi data leakage):** [Jelaskan]

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)

**Requirement:** Minimal 3 visualisasi yang bermakna dan insight-nya.
**Contoh jenis visualisasi yang dapat digunakan:**

- Histogram (distribusi data)
- Boxplot (deteksi outliers)
- Heatmap korelasi (hubungan antar fitur)
- Bar plot (distribusi kategori)
- Scatter plot (hubungan 2 variabel)
- Wordcloud (untuk text data)
- Sample images (untuk image data)
- Time series plot (untuk temporal data)
- Confusion matrix heatmap
- Class distribution plot

#### Visualisasi 1: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 2: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 3: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

---

## 5. DATA PREPARATION **[L]**

Bagian ini menjelaskan **semua** proses transformasi dan preprocessing data yang dilakukan.

### 5.1 Data Cleaning

**Aktivitas:**

- Handling missing values
- Removing duplicates
- Handling outliers
- Data type conversion
  **Contoh:**

```
Missing Values:
- Fitur 'age' memiliki 50 missing values (5% dari data)
- Strategi: Imputasi dengan median karena distribusi skewed
- Alasan: Median lebih robust terhadap outliers dibanding mean
```

**[Jelaskan langkah-langkah data cleaning yang Anda lakukan]**

### 5.2 Feature Engineering

**Aktivitas:**

- Creating new features
- Feature extraction
- Feature selection
- Dimensionality reduction

**[Jelaskan feature engineering yang Anda lakukan]**

### 5.3 Data Transformation

**Untuk Data Tabular:**

- Encoding (Label Encoding, One-Hot Encoding, Ordinal Encoding)
- Scaling (Standardization, Normalization, MinMaxScaler)

**Untuk Data Text:**

- Tokenization
- Lowercasing
- Removing punctuation/stopwords
- Stemming/Lemmatization
- Padding sequences
- Word embedding (Word2Vec, GloVe, fastText)

**Untuk Data Image:**

- Resizing
- Normalization (pixel values 0-1 atau -1 to 1)
- Data augmentation (rotation, flip, zoom, brightness, etc.)
- Color space conversion

**Untuk Time Series:**

- Creating time windows
- Lag features
- Rolling statistics
- Differencing

**[Jelaskan transformasi yang Anda lakukan]**

### 5.4 Data Splitting

**Strategi pembagian data:**

```
- Training set: [X]% ([jumlah] samples)
- Validation set: [X]% ([jumlah] samples) - jika ada
- Test set: [X]% ([jumlah] samples)
```

**Contoh:**

```
Menggunakan stratified split untuk mempertahankan distribusi kelas:
- Training: 80% (8000 samples)
- Test: 20% (2000 samples)
- Random state: 42 untuk reproducibility
```

> **Penting:** Test set hanya digunakan **sekali** pada tahap P (evaluasi akhir). Tuning pada tahap O hanya boleh memakai validation set / cross-validation. Fit scaler/encoder hanya pada data training untuk menghindari data leakage.

**[Jelaskan strategi splitting Anda dan alasannya]**

### 5.5 Data Balancing (jika diperlukan)

**Teknik yang digunakan:**

- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Undersampling
- Class weights
- Ensemble sampling

**[Jelaskan jika Anda melakukan data balancing]**

### 5.6 Ringkasan Data Preparation

**Per langkah, jelaskan:**

1. **Apa** yang dilakukan
   **[Jelaskan ]**
2. **Mengapa** penting
   **[Jelaskan Mengapa ?]**
3. **Bagaimana** implementasinya
   **[Jelaskan Bagaimana]**

---

## 6. MODELING **[M + O]**

> Setiap model (6.1, 6.2, 6.3) dijelaskan dalam dua fase: **Model Design [M]** (sub-bagian .1–.3) dan **Optimization [O]** (sub-bagian .4–.6).

### 6.1 Model 1 — Baseline Model

#### 6.1.1 Deskripsi Model **[M]**

**Nama Model:** [Nama model, misal: Logistic Regression]  
**Teori Singkat:**  
[Jelaskan secara singkat bagaimana model ini bekerja]  
**Alasan Pemilihan:**  
[Mengapa memilih model ini sebagai baseline?]

#### 6.1.2 Hypothesis Space & Inductive Bias **[M]**

- **Model family / arsitektur:** [misal: model linear]
- **Hypothesis space:** [himpunan fungsi yang dapat direpresentasikan, misal: batas keputusan linear]
- **Inductive bias / asumsi:** [misal: hubungan fitur-target linear, fitur independen]
- **Implikasi terhadap dataset Anda:** [apakah asumsi cocok dengan data?]

#### 6.1.3 Implementasi (Ringkas) **[M]**

```python
# Contoh kode (opsional, bisa dipindah ke GitHub)
from sklearn.linear_model import LogisticRegression

model_baseline = LogisticRegression(C=1.0, max_iter=100)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
```

#### 6.1.4 Optimization: Objective, Hyperparameter & Regularisasi **[O]**

**Objective/loss yang dioptimalkan:** [misal: log-loss + L2]  
**Parameter yang digunakan:**

```
[Tuliskan parameter penting, contoh:]
- C (regularization): 1.0
- solver: 'lbfgs'
- max_iter: 100
```

#### 6.1.5 Hasil Awal **[O → P]**

**[Tuliskan hasil awal pada validation set; evaluasi akhir pada test set dijelaskan di Bab 7]**

---

### 6.2 Model 2 — ML / Advanced Model

#### 6.2.1 Deskripsi Model **[M]**

**Nama Model:** [Nama model, misal: Random Forest / XGBoost]  
**Teori Singkat:**  
[Jelaskan bagaimana algoritma ini bekerja]

**Alasan Pemilihan:**  
[Mengapa memilih model ini?]

**Keunggulan:**

- [Sebutkan keunggulan]

**Kelemahan:**

- [Sebutkan kelemahan]

#### 6.2.2 Hypothesis Space & Inductive Bias **[M]**

- **Model family / arsitektur:** [misal: ensemble pohon keputusan]
- **Hypothesis space:** [fungsi non-linear piecewise, interaksi fitur otomatis, dll.]
- **Inductive bias / asumsi:** [misal: data tabular dengan struktur ambang/threshold]
- **Perbandingan dengan baseline:** [apa yang bisa dipelajari model ini yang tidak bisa baseline?]

#### 6.2.3 Implementasi (Ringkas) **[M]**

```python
# Contoh kode
from sklearn.ensemble import RandomForestClassifier

model_advanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
```

#### 6.2.4 Optimization: Hyperparameter Tuning **[O]**

**Parameter yang digunakan:**

```
[Tuliskan parameter penting, contoh:]
- n_estimators: 100
- max_depth: 10
- learning_rate: 0.1
- min_samples_split: 2
```

**Hyperparameter Tuning (WAJIB minimal 1 metode sederhana):**

- Metode: [Grid Search / Random Search / Bayesian Optimization]
- Search space: [rentang parameter yang dicoba]
- Skema validasi: [k-fold CV / validation set]
- Best parameters: [...]
- Skor validasi sebelum vs sesudah tuning: [...]

#### 6.2.5 Kontrol Overfitting **[O]**

[Jelaskan regularisasi/pembatasan kompleksitas yang dipakai, misal: max_depth, min_samples_leaf, early stopping pada boosting]

#### 6.2.6 Hasil Model **[O → P]**

**[Tuliskan hasil pada validation set; evaluasi akhir dijelaskan di Bab 7]**

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model **[M]**

**Nama Model:** [Nama arsitektur, misal: CNN / LSTM / MLP]

** (Centang) Jenis Deep Learning: **

- [ ] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
[Mengapa arsitektur ini cocok untuk dataset Anda?]

#### 6.3.2 Arsitektur Model, Hypothesis Space & Inductive Bias **[M]**

**Inductive bias arsitektur:** [misal: CNN → lokalitas & translation invariance; LSTM → ketergantungan sekuensial; MLP → tanpa asumsi struktur]  
**Hypothesis space:** [kapasitas model: jumlah layer, unit, parameter]

**Deskripsi Layer:**

[Jelaskan arsitektur secara detail atau buat tabel]

**Contoh:**

```
1. Input Layer: shape (224, 224, 3)
2. Conv2D: 32 filters, kernel (3,3), activation='relu'
3. MaxPooling2D: pool size (2,2)
4. Conv2D: 64 filters, kernel (3,3), activation='relu'
5. MaxPooling2D: pool size (2,2)
6. Flatten
7. Dense: 128 units, activation='relu'
8. Dropout: 0.5
9. Dense: 10 units, activation='softmax'

Total parameters: [jumlah]
Trainable parameters: [jumlah]
```

#### 6.3.3 Input & Preprocessing Khusus **[M]**

**Input shape:** [Sebutkan dimensi input]  
**Preprocessing khusus untuk DL:**

- [Sebutkan preprocessing khusus seperti normalisasi, augmentasi, dll.]

#### 6.3.4 Optimization: Konfigurasi Training **[O]**

**Training Configuration:**

```
- Optimizer: Adam / SGD / RMSprop
- Learning rate: [nilai]
- Loss function: [categorical_crossentropy / mse / binary_crossentropy / etc.]
- Metrics: [accuracy / mae / etc.]
- Batch size: [nilai]
- Epochs: [nilai]
- Validation split: [nilai] atau menggunakan validation set terpisah
- Callbacks: [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, etc.]
- Regularisasi: [Dropout / L2 / BatchNorm / augmentasi]
```

**Eksperimen optimasi (minimal 2 konfigurasi dibandingkan):**

| Eksperimen | Learning Rate | Batch Size | Regularisasi | Val Loss | Val Metric |
| ---------- | ------------- | ---------- | ------------ | -------- | ---------- |
| Exp-1      | [ ]           | [ ]        | [ ]          | [ ]      | [ ]        |
| Exp-2      | [ ]           | [ ]        | [ ]          | [ ]      | [ ]        |

#### 6.3.5 Implementasi (Ringkas) **[M + O]**

**Framework:** TensorFlow/Keras / PyTorch

```python
# Contoh kode TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

model_dl = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model_dl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model_dl.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)
```

#### 6.3.6 Training Process & Diagnosis Optimasi **[O]**

**Training Time:**  
[Sebutkan waktu training total, misal: 15 menit]

**Computational Resource:**  
[CPU / GPU, platform: Local / Google Colab / Kaggle]

**Training History Visualization:**

[Insert plot loss dan accuracy/metric per epoch]

**Contoh visualisasi yang WAJIB:**

1. **Training & Validation Loss** per epoch
2. **Training & Validation Accuracy/Metric** per epoch

**Analisis Training:**

- Apakah model mengalami overfitting? [Ya/Tidak, jelaskan]
- Apakah model sudah converge? [Ya/Tidak, jelaskan]
- Apakah perlu lebih banyak epoch? [Ya/Tidak, jelaskan]
- Apakah learning rate sesuai (loss stabil/osilasi/lambat)? [Jelaskan]

#### 6.3.7 Model Summary

```
[Paste model.summary() output atau rangkuman arsitektur]
```

---

## 7. EVALUATION **[P]**

### 7.1 Metrik Evaluasi

**Pilih metrik yang sesuai dengan jenis tugas:**

#### **Untuk Klasifikasi:**

- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean dari precision dan recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Visualisasi prediksi

#### **Untuk Regresi:**

- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² Score**: Koefisien determinasi
- **MAPE (Mean Absolute Percentage Error)**: Error dalam persentase

#### **Untuk NLP (Text Classification):**

- **Accuracy**
- **F1-Score** (terutama untuk imbalanced data)
- **Precision & Recall**
- **Perplexity** (untuk language models)

#### **Untuk Computer Vision:**

- **Accuracy**
- **IoU (Intersection over Union)** - untuk object detection/segmentation
- **Dice Coefficient** - untuk segmentation
- **mAP (mean Average Precision)** - untuk object detection

#### **Untuk Clustering:**

- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

#### **Untuk Recommender System:**

- **RMSE**
- **Precision@K**
- **Recall@K**
- **NDCG (Normalized Discounted Cumulative Gain)**

**Justifikasi pemilihan metrik:** [Mengapa metrik ini selaras dengan Objective di Bab 3.2? Mengapa accuracy saja cukup/tidak cukup?]

### 7.2 Hasil Evaluasi Model (pada Test Set)

> Semua prediksi dievaluasi pada **test set** yang tidak pernah dipakai saat training maupun tuning.

#### 7.2.1 Model 1 (Baseline)

**Metrik:**

```
[Tuliskan hasil metrik, contoh:]
- Accuracy: 0.75
- Precision: 0.73
- Recall: 0.76
- F1-Score: 0.74
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

#### 7.2.2 Model 2 (Advanced/ML)

**Metrik:**

```
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (Deep Learning)

**Metrik:**

```
- Accuracy: 0.89
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Tampilkan minimal 5 contoh prediksi benar dan salah]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model                   | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
| ----------------------- | -------- | --------- | ------ | -------- | ------------- | -------------- |
| Baseline (Model 1)      | 0.75     | 0.73      | 0.76   | 0.74     | 2s            | 0.01s          |
| Advanced (Model 2)      | 0.85     | 0.84      | 0.86   | 0.85     | 30s           | 0.05s          |
| Deep Learning (Model 3) | 0.89     | 0.88      | 0.90   | 0.89     | 15min         | 0.1s           |

**Visualisasi Perbandingan:**  
[Insert bar chart atau plot perbandingan metrik]

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   [Sebutkan model mana yang terbaik dan mengapa]

2. **Perbandingan dengan Baseline:**  
   [Jelaskan peningkatan performa dari baseline ke model lainnya]

3. **Trade-off:**  
   [Jelaskan trade-off antara performa vs kompleksitas vs waktu training]

4. **Error Analysis:**  
   [Jelaskan jenis kesalahan yang sering terjadi, kasus yang sulit diprediksi]

5. **Overfitting/Underfitting & Generalization Gap:**  
   [Bandingkan performa train vs validation vs test; analisis apakah model overfitting atau underfitting]

6. **Kesesuaian Inductive Bias:**  
   [Apakah asumsi tiap model (Bab 6.x.2) terbukti sesuai dengan data? Jelaskan dengan bukti hasil]

### 7.5 Umpan Balik Lifecycle (Iterasi L → M → O)

Jelaskan minimal **satu** iterasi perbaikan berdasarkan hasil evaluasi:

| Temuan dari Evaluasi [P]               | Tahap yang Diperbaiki | Tindakan                                 | Dampak pada Metrik |
| -------------------------------------- | --------------------- | ---------------------------------------- | ------------------ |
| [misal: recall kelas minoritas rendah] | [L] data              | [misal: class weight / SMOTE]            | [ ... ]            |
| [misal: gap train-val besar]           | [O] optimasi          | [misal: tambah dropout, early stopping]  | [ ... ]            |
| [misal: model terlalu sederhana]       | [M] desain            | [misal: tambah layer / ganti arsitektur] | [ ... ]            |

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Sebutkan model terbaik berdasarkan evaluasi]

**Alasan:**  
[Jelaskan mengapa model tersebut lebih unggul]

**Pencapaian Goals:**  
[Apakah goals di Section 3.3 tercapai? Jelaskan]

### 8.2 Key Insights

**Insight dari Data:**

- [Insight 1]
- [Insight 2]
- [Insight 3]

**Insight dari Modeling:**

- [Insight 1]
- [Insight 2]

### 8.3 Ringkasan per Tahap Lifecycle

| Tahap                      | Keputusan Kunci | Pelajaran |
| -------------------------- | --------------- | --------- |
| **L** – Learning Problem   | [ ... ]         | [ ... ]   |
| **M** – Model Design       | [ ... ]         | [ ... ]   |
| **O** – Optimization       | [ ... ]         | [ ... ]   |
| **P** – Predict & Evaluate | [ ... ]         | [ ... ]   |

### 8.4 Kontribusi Proyek

**Manfaat praktis:**  
[Jelaskan bagaimana proyek ini dapat digunakan di dunia nyata]

**Kontribusi terhadap SDGs:**  
[Jelaskan secara jujur dan terukur bagaimana hasil proyek mendukung target SDG yang dipilih (Bab 2.2), serta keterbatasan kontribusinya]

**Pembelajaran yang didapat:**  
[Jelaskan apa yang Anda pelajari dari proyek ini]

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**

- [ ] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [ ] Feature engineering lebih lanjut

**Model:**

- [ ] Mencoba arsitektur DL yang lebih kompleks
- [ ] Hyperparameter tuning lebih ekstensif
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**

- [ ] Membuat API (Flask/FastAPI)
- [ ] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**

- [ ] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [URL GitHub Anda]

**Repository harus berisi:**

- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

**Saran struktur folder (mengikuti lifecycle):**

```
project/
├── data/                 # [L] dataset (jangan upload jika besar)
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb      # [L]
│   ├── 02_data_preparation.ipynb          # [L]
│   ├── 03_model_design_baseline_ml.ipynb  # [M]
│   ├── 04_deep_learning_optimization.ipynb# [M+O]
│   └── 05_evaluation_comparison.ipynb     # [P]
├── src/                  # script modular
├── models/               # model tersimpan
├── reports/figures/
├── requirements.txt
└── README.md
```

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**

```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll
```

### 10.3 Reproducibility Checklist

- [ ] Random seed ditetapkan (misal 42) di semua library
- [ ] Notebook dapat dijalankan ulang dari awal sampai akhir
- [ ] Test set tidak dipakai saat training/tuning
- [ ] Versi library tercatat

---

## 11. TIMELINE PENGERJAAN (6 MINGGU)

Pengerjaan mengikuti urutan lifecycle L → M → O → P dengan satu iterasi perbaikan. Rincian tugas per minggu ada di file panduan terpisah: **Panduan_Tahapan_6_Minggu.md**.

| Minggu | Fokus Lifecycle            | Aktivitas                                                                                                                                               | Deliverable / Checkpoint                                                                     |
| ------ | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **1**  | **L** – Learning Problem   | Mahasiswa mengajukan ide sendiri selaras dengan satu SDG, pilih dataset, studi literatur, rumuskan problem statement, target, objective, data (Bab 1–3) | Formulir ide + proposal singkat 1–2 halaman; ide & dataset disetujui dosen. **Checkpoint 1** |
| **2**  | **L** – Data               | Data understanding, EDA (opsional), cleaning, transformasi, splitting (Bab 4–5)                                                                         | Notebook data preparation + repo GitHub dibuat                                               |
| **3**  | **M** – Model Design       | Rancang & latih Model 1 (baseline) dan Model 2 (advanced); jelaskan hypothesis space & inductive bias (Bab 6.1–6.2)                                     | Hasil baseline & advanced pada validation set. **Checkpoint 2**                              |
| **4**  | **M → O**                  | Tuning Model 2; desain arsitektur Model 3 (deep learning) dan training pertama (Bab 6.2.4, 6.3.1–6.3.5)                                                 | Model DL berjalan end-to-end (training awal)                                                 |
| **5**  | **O** – Optimization       | Optimasi Model 3: eksperimen learning rate/batch size/regularisasi, plot loss/metric, diagnosis konvergensi (Bab 6.3.4–6.3.7)                           | Model DL terlatih (≥10 epoch) + tabel eksperimen. **Checkpoint 3**                           |
| **6**  | **P** – Predict & Evaluate | Evaluasi test set, perbandingan 3 model, error analysis, iterasi perbaikan, kesimpulan, video, finalisasi repo (Bab 7–10)                               | Laporan final + GitHub + video + presentasi                                                  |

**Catatan untuk dosen:**

- Enam minggu memberi ruang tambahan untuk tahap Optimization, tempat mahasiswa biasanya paling banyak mengalami kendala (training deep learning, tuning, overfitting).
- Tiga checkpoint (akhir minggu 1, 3, dan 5) membantu mendeteksi keterlambatan lebih awal.

---

## 12. RUBRIK PENILAIAN (USULAN)

| Komponen                                                                          | Tahap | Bobot    |
| --------------------------------------------------------------------------------- | ----- | -------- |
| Orisinalitas ide mahasiswa & keterkaitan dengan SDGs (Bab 2.2)                    | L     | 10%      |
| Problem formulation (target, objective, data), latar belakang & referensi         | L     | 10%      |
| Data understanding & data preparation (tanpa data leakage)                        | L     | 10%      |
| Model design: baseline & advanced + justifikasi hypothesis space/inductive bias   | M     | 15%      |
| Model deep learning: arsitektur, training ≥10 epoch, plot loss/metric             | M + O | 15%      |
| Optimization: tuning hyperparameter, regularisasi, perbandingan eksperimen        | O     | 10%      |
| Evaluasi: metrik relevan, perbandingan 3 model, error analysis, iterasi lifecycle | P     | 20%      |
| Reproducibility: GitHub, README, requirements, struktur folder                    | –     | 5%       |
| Video pembahasan & kualitas penulisan laporan                                     | –     | 5%       |
| **Total**                                                                         |       | **100%** |

> Bobot dapat disesuaikan dengan kebijakan mata kuliah.

---

## LAMPIRAN A. SDGs SEBAGAI SUMBER IDE

Referensi resmi: [Transforming our world: the 2030 Agenda for Sustainable Development](https://sdgs.un.org/2030agenda)

**Aturan ide:**

- Ide **harus berasal dari mahasiswa**, bukan dari daftar contoh di bawah. Contoh hanya inspirasi.
- Pilih **satu** SDG utama dan sebutkan target (nomor target) yang relevan.
- Masalah harus nyata, spesifik, dan dapat dirumuskan sebagai tugas ML (klasifikasi, regresi, forecasting, dll.).
- Dataset harus tersedia (publik atau dikumpulkan sendiri secara etis) dan cukup untuk melatih tiga model, termasuk deep learning.
- Disarankan setiap mahasiswa/kelompok memiliki ide yang berbeda; dosen memeriksa di Checkpoint 1.

| SDG | Nama (Indonesia)                                   | Contoh arah ide ML (inspirasi)                                                  |
| --- | -------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1   | Tanpa Kemiskinan                                   | Prediksi tingkat kemiskinan wilayah dari data sosial-ekonomi (tabular)          |
| 2   | Tanpa Kelaparan                                    | Klasifikasi penyakit daun tanaman (image); prediksi hasil panen (tabular)       |
| 3   | Kehidupan Sehat dan Sejahtera                      | Deteksi penyakit dari data klinis atau citra medis; prediksi risiko diabetes    |
| 4   | Pendidikan Berkualitas                             | Prediksi risiko putus studi/performa siswa; analisis ulasan pembelajaran (teks) |
| 5   | Kesetaraan Gender                                  | Deteksi ujaran kebencian atau bias gender pada teks                             |
| 6   | Air Bersih dan Sanitasi Layak                      | Prediksi kelayakan/kualitas air minum (tabular)                                 |
| 7   | Energi Bersih dan Terjangkau                       | Forecasting konsumsi atau produksi energi (time series)                         |
| 8   | Pekerjaan Layak dan Pertumbuhan Ekonomi            | Klasifikasi lowongan kerja atau prediksi pengangguran                           |
| 9   | Industri, Inovasi, dan Infrastruktur               | Deteksi kerusakan jalan (image); predictive maintenance                         |
| 10  | Berkurangnya Kesenjangan                           | Pengenalan bahasa isyarat (image); analisis akses layanan publik                |
| 11  | Kota dan Permukiman yang Berkelanjutan             | Prediksi kualitas udara kota; prediksi kepadatan lalu lintas                    |
| 12  | Konsumsi dan Produksi yang Bertanggung Jawab       | Klasifikasi jenis sampah (image); prediksi limbah makanan                       |
| 13  | Penanganan Perubahan Iklim                         | Prediksi curah hujan/suhu; deteksi titik api atau kebakaran hutan               |
| 14  | Ekosistem Lautan                                   | Klasifikasi spesies ikan; deteksi sampah laut atau kesehatan terumbu karang     |
| 15  | Ekosistem Daratan                                  | Klasifikasi spesies flora/fauna; deteksi deforestasi dari citra                 |
| 16  | Perdamaian, Keadilan, dan Kelembagaan yang Tangguh | Deteksi hoaks/berita palsu (teks); deteksi fraud                                |
| 17  | Kemitraan untuk Mencapai Tujuan                    | Sistem rekomendasi kolaborasi/mitra; analisis data terbuka pemerintah           |

---

## LAMPIRAN B. FORMULIR PENGAJUAN IDE (DIKUMPULKAN PADA CHECKPOINT 1)

| Komponen                                                 | Isian Mahasiswa |
| -------------------------------------------------------- | --------------- |
| Nama / NIM                                               | [ ... ]         |
| Judul ide                                                | [ ... ]         |
| SDG & target yang dipilih                                | [ ... ]         |
| Masalah nyata yang diangkat                              | [ ... ]         |
| Tugas ML (Target & jenis tugas)                          | [ ... ]         |
| Objective & metrik utama                                 | [ ... ]         |
| Dataset (sumber, URL, ukuran, jenis data)                | [ ... ]         |
| Rencana tiga model (baseline / advanced / deep learning) | [ ... ]         |
| Risiko & pertimbangan etika (bias, privasi)              | [ ... ]         |
| Referensi awal (minimal 2)                               | [ ... ]         |

**Kriteria persetujuan dosen:**

- [ ] Ide berasal dari mahasiswa dan tidak duplikat
- [ ] Selaras jelas dengan satu SDG dan targetnya
- [ ] Dapat dirumuskan sebagai masalah ML dengan target yang jelas
- [ ] Dataset tersedia, legal/etis digunakan, dan cukup besar
- [ ] Layak untuk tiga model termasuk deep learning dalam 6 minggu
