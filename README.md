# Analisis Sentimen Opini Publik di Media Sosial Terhadap Kebijakan PPKM sebagai Evaluasi Transparansi Pemerintah Menggunakan Machine Learning dan Deep Learning

![SDG 16](https://img.shields.io/badge/UN_SDG-16.6_Peace,_Justice_&_Strong_Institutions-00689D?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Indonesian_NLP_Sentiment_Classification-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Checkpoint_1-Learning_Problem-success?style=for-the-badge)

Repository ini berisi dokumentasi, alur prapemrosesan bahasa alami (*Natural Language Processing*), dan eksperimen pemodelan untuk mengklasifikasikan sentimen (positif, netral, negatif) cuitan Twitter berbahasa Indonesia terkait kebijakan Pemberlakuan Pembatasan Kegiatan Masyarakat (PPKM). Hasil klasifikasi diposisikan sebagai indikator proksi respons publik terhadap kebijakan, untuk mendukung kelembagaan yang akuntabel dan transparan sesuai **Sustainable Development Goals (SDG) 16**.

---

## Identitas Pengusul
* **Nama:** Iqbal Abdullah (244311044) & Rindy Cantika Agustina Putri (244311057)
* **Mata Kuliah:** Data Science
* **Dosen Pengampu:** Gus Nanang Syaifuddin, S.Kom., M.Kom.

---

## Dokumen Deliverables (Minggu 1 — Checkpoint 1)
Sesuai dengan ketentuan **Minggu 1: [L] Learning Problem**, dokumen pengajuan ide dan proposal awal dapat diakses melalui tautan berikut:
1. 📄 **[Lampiran B: Formulir Pengajuan Ide](./docs/pengajuan-ide.md)**
2. 📑 **[Proposal Singkat (Bab 1 – Bab 3 Versi Awal)](./docs/learning-problem.md)**

---

## Ringkasan Learning Problem & Keterkaitan SDG

### 1. Kontribusi terhadap SDG 16 (Target 16.6, didukung 16.10 & 16.7)
Opini publik terhadap PPKM tersebar dalam ribuan cuitan sehingga sulit dipantau manual oleh lembaga pemerintah. Sistem klasifikasi sentimen ini dirancang sebagai alat bantu pemantauan respons publik: proporsi sentimen positif, netral, dan negatif per periode kebijakan dapat menjadi salah satu masukan evaluasi apakah kebijakan dan komunikasinya diterima. Kontribusinya **tidak langsung**: model tidak mengukur transparansi itu sendiri dan tidak menggantikan survei atau audit resmi.

### 2. Spesifikasi Learning Problem

| Komponen | Deskripsi Teknis |
| :--- | :--- |
| **Jenis Tugas ML** | *Supervised Learning — Multiclass Text Classification (NLP)* |
| **Target ($Y$)** | Kelas `0`: **Positif**<br>Kelas `1`: **Netral**<br>Kelas `2`: **Negatif** |
| **Data ($X$)** | Teks cuitan berbahasa Indonesia terkait PPKM (dibersihkan dari URL, mention, dan duplikat; diekstraksi menjadi vektor TF-IDF atau token sekuens) |
| **Objective** | Meminimalkan *cross-entropy* berbobot kelas dan memaksimalkan **Macro F1-Score**, dengan menjaga **Recall kelas Negatif** agar keluhan publik tidak banyak terlewat |
| **Metrik Evaluasi** | **Macro F1-Score** ($\ge 70\%$), **Accuracy** ($> 80\%$), **Recall kelas Negatif** ($\ge 60\%$), serta *Precision* & *Recall* per kelas dan *Confusion Matrix* |

---

## Dataset
* **Nama Dataset:** [Twitter Dataset PPKM](https://www.kaggle.com/datasets/anggapurnama/twitter-dataset-ppkm) (Kaggle, anggapurnama), berkas `INA_TweetsPPKM_Labeled_Pure.csv`.
* **Ukuran & Proporsi:** **23.644 baris** × 4 kolom (`Date`, `User`, `Tweet`, `sentiment`), ±5,25 MB, periode Mei 2020 – Maret 2022. Distribusi: Netral 17.706 (74,9%), Negatif 3.980 (16,8%), Positif 1.958 (8,3%). Berkas berformat tab (TSV) walaupun berekstensi `.csv`.
* **Catatan Kualitas Data:** terdapat 158 cuitan duplikat persis (2.125 jika URL diabaikan) dan sejumlah teks identik dengan label berbeda; duplikat dihapus **sebelum** pembagian data.
* **Strategi Pencegahan *Data Leakage*:** Pembagian data (*70% Train, 15% Validation, 15% Test*) dilakukan secara *stratified* setelah deduplikasi. `TfidfVectorizer` dan tokenizer hanya di-*fit* pada *Training Set*. *Test Set* hanya dipakai sekali pada evaluasi akhir.

---

## Rencana Arsitektur Tiga Model

| Tingkatan Model | Arsitektur / Metode | Input Fitur | Tujuan Eksperimen |
| :--- | :--- | :--- | :--- |
| **1. Baseline** | **Logistic Regression** (`class_weight='balanced'`) | Vektor **TF-IDF** (*Unigram & Bigram*) | Menetapkan batas bawah performa model linear dan menganalisis kata dengan bobot terbesar per kelas. |
| **2. Advanced** | **Linear SVM** + Grid Search (stratified k-fold CV) | Vektor **TF-IDF** (*Unigram & Bigram*) | Menguji pendekatan margin maksimum pada ruang fitur sparse berdimensi tinggi dan mengukur dampak tuning. |
| **3. Deep Learning** | **Fine-tuning IndoBERT (`indobert-base-p1`)** (cadangan: Embedding + Bi-LSTM) | *Token IDs* & *Attention Mask* | Memanfaatkan model Transformer pra-latih bahasa Indonesia untuk memahami konteks kalimat dua arah. |

---

## Struktur Direktori Repository

```text
├── data/                               # Dataset lokal (diabaikan oleh Git)
├── docs/                               # Dokumentasi laporan dan pengajuan ide
│   ├── 09_LAMPIRAN_B.md                # Deliverable 1: Formulir Lampiran B (Checkpoint 1)
│   └── EAS_Laporan_Proyek_ML_LMOP.md   # Deliverable 2: Proposal Singkat Bab 1 - Bab 3
├── notebooks/                          # Jupyter / Google Colab Notebooks eksperimen
│   ├── 01_eda_and_text_cleaning.ipynb  # Eksplorasi data, deduplikasi, pembersihan teks
│   ├── 02_baseline_logreg_tfidf.ipynb  # Model 1: TF-IDF + Logistic Regression
│   ├── 03_advanced_linear_svm.ipynb    # Model 2: TF-IDF + Linear SVM (Grid Search)
│   └── 04_dl_indobert_finetuning.ipynb # Model 3: Fine-tuning IndoBERT
├── src/                                # Modul fungsi pembersihan teks dan evaluasi metrik
├── .gitignore                          # Mengabaikan dataset dan bobot model besar
├── requirements.txt                    # Daftar pustaka Python (scikit-learn, transformers, torch)
└── README.md                           # Dokumentasi utama proyek
```
