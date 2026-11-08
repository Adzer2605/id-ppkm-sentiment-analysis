# PROPOSAL PROYEK MACHINE LEARNING (MINGGU 1)
**Judul:** Analisis Sentimen Opini Publik di Media Sosial Terhadap Kebijakan PPKM sebagai Evaluasi Transparansi Pemerintah  
**Pengusul:** Iqbal Abdullah (244311044) & Rindy Cantika Agustina Putri (244311057)

---

## BAB 1. PENDAHULUAN

### 1.1 Latar Belakang
Pemberlakuan Pembatasan Kegiatan Masyarakat (PPKM) adalah kebijakan utama pemerintah Indonesia untuk menekan penyebaran COVID-19 sejak awal 2021. Kebijakan ini memengaruhi mobilitas, kegiatan ekonomi, ibadah, pendidikan, hingga layanan publik. Karena aturannya berubah-ubah (level 1–4, PPKM Mikro, PPKM Darurat, perpanjangan berkala), efektivitasnya sangat bergantung pada seberapa jelas pemerintah mengomunikasikan dasar keputusan dan seberapa baik publik memahaminya.

Media sosial, khususnya Twitter, menjadi kanal utama publik untuk bertanya, menyampaikan keluhan, dan merespons kebijakan secara langsung. Volume cuitan yang besar tidak mungkin dibaca manual oleh pembuat kebijakan. Pada dataset yang digunakan, volume cuitan memuncak pada Juli 2021 (9.603 cuitan), bertepatan dengan periode PPKM Darurat, yang menunjukkan bahwa perhatian publik bereaksi kuat terhadap perubahan kebijakan. Analisis sentimen otomatis dapat meringkas arah opini publik (positif, netral, negatif) sehingga lembaga pemerintah dapat mengevaluasi apakah kebijakan dan komunikasinya diterima.

Oleh karena itu, penelitian ini mengembangkan dan membandingkan algoritma *Machine Learning* klasik, model margin maksimum, dan *Deep Learning* berbasis pemrosesan bahasa alami (*Natural Language Processing* / NLP) untuk mengklasifikasikan sentimen cuitan terkait PPKM secara akurat, termasuk pada kelas minoritas.

### 1.2 Rumusan Masalah (Problem Statements)
1. Opini publik terhadap PPKM tersebar dalam puluhan ribu cuitan sehingga tidak dapat dipantau manual; bagaimana mengklasifikasikan sentimen (positif, netral, negatif) secara otomatis dengan akurat?
2. Distribusi kelas sangat tidak seimbang (netral 74,9%, negatif 16,8%, positif 8,3%); bagaimana mencegah model mengabaikan kelas minoritas, terutama sentimen negatif yang paling penting bagi evaluasi respons publik?
3. Teks Twitter berbahasa Indonesia bersifat *noisy* (URL, mention, hashtag, singkatan) dan memuat duplikat; bagaimana merancang prapemrosesan yang tepat tanpa menimbulkan kebocoran data (*data leakage*)?
4. Bagaimana perbandingan performa model *Baseline* (*Logistic Regression* + TF-IDF), *Advanced* (*Linear SVM* + TF-IDF), dan *Deep Learning* (*IndoBERT*) pada tugas ini?

### 1.3 Tujuan Terukur (Goals)
1. Membangun *pipeline* prapemrosesan teks dan pemodelan klasifikasi tiga kelas yang bebas dari kebocoran data untuk ketiga arsitektur model.
2. Mencapai **Accuracy > 80%** dan **Macro F1-Score $\ge 70\%$** pada data uji (*test set*) untuk model terbaik.
3. Membuktikan bahwa model *deep learning* mengungguli baseline minimal **+2 poin Macro F1-Score** pada *test set* yang sama.
4. Mencapai **Recall kelas Negatif $\ge 60\%$** pada model terbaik sebagai kontribusi terhadap target SDG (pemantauan keluhan publik).
5. Menghasilkan proyek yang *reproducible* (*random seed* 42, versi pustaka tercatat, kode di GitHub).

---

## BAB 2. TINJAUAN PUSTAKA & KETERKAITAN SDGs

### 2.1 Studi Literatur
Hansun dkk. (2022) menganalisis cuitan tentang kebijakan PPKM dan menemukan sentimen netral dominan (58,07%), diikuti positif (27,12%) dan negatif (14,81%), lalu membangun model berbasis LSTM yang dideploy sebagai aplikasi web. Temuan ini menunjukkan bahwa distribusi sentimen PPKM cenderung condong ke netral, sejalan dengan karakteristik dataset proyek ini. Untuk bahasa Indonesia, Wilie dkk. (2020) melalui *benchmark* IndoNLU memperkenalkan model bahasa pra-latih **IndoBERT** yang menyediakan representasi kontekstual dua arah untuk teks berbahasa Indonesia, sehingga relevan sebagai dasar model *deep learning* pada proyek ini.

### 2.2 Keterkaitan dengan SDGs
Penelitian ini berkaitan dengan **Sustainable Development Goal (SDG) 16: Perdamaian, Keadilan, dan Kelembagaan yang Tangguh (*Peace, Justice and Strong Institutions*)**, khususnya:
* **Target 16.6 (utama):** *Mengembangkan lembaga yang efektif, akuntabel, dan transparan di semua tingkat.*
* **Target 16.10 (pendukung):** *Menjamin akses publik terhadap informasi dan melindungi kebebasan mendasar.*
* **Target 16.7 (pendukung):** *Memastikan pengambilan keputusan yang responsif, inklusif, partisipatif, dan representatif di setiap tingkatan.*

Peran *Machine Learning* adalah mengotomatiskan pelabelan sentimen ribuan cuitan sehingga respons publik dapat dipantau per periode kebijakan. Kontribusinya **tidak langsung**: proporsi sentimen positif, netral, dan negatif menjadi indikator proksi penerimaan kebijakan, bukan pengukuran transparansi itu sendiri. Keterbatasan: pengguna Twitter tidak mewakili seluruh masyarakat dan data didominasi akun berita/instansi.

---

## BAB 3. METODOLOGI & LEARNING PROBLEM

### 3.1 Deskripsi Dataset & Pertimbangan Etika
Penelitian ini menggunakan **Twitter Dataset PPKM** (Kaggle, anggapurnama), berkas `INA_TweetsPPKM_Labeled_Pure.csv`.
* **Ukuran & Karakteristik Data:** **23.644 baris** × 4 kolom (`Date`, `User`, `Tweet`, `sentiment`), ±5,25 MB, periode Mei 2020 – Maret 2022, rata-rata sekitar 21 kata per cuitan. Distribusi label: `1` Netral 17.706 (74,9%), `2` Negatif 3.980 (16,8%), `0` Positif 1.958 (8,3%).
* **Kondisi Data:** tidak ada *missing value*. Terdapat 158 cuitan duplikat persis (2.125 jika URL diabaikan) dan sejumlah teks identik berlabel berbeda. Duplikat dihapus sebelum pembagian data.
* **Pembagian Data & Pencegahan *Data Leakage*:** Dataset dibagi secara *stratified* menjadi 70% *Training*, 15% *Validation*, dan 15% *Testing* dengan `random_state=42`. TF-IDF dan tokenizer hanya di-*fit* pada *Training Set*. *Test Set* hanya dipakai sekali pada evaluasi akhir.
* **Asumsi Data:** cuitan dianggap i.i.d. setelah deduplikasi, namun asumsi ini hanya pendekatan karena cuitan terkonsentrasi pada Juli–Agustus 2021 dan satu akun dapat menulis banyak cuitan. Perubahan topik antar periode dicatat sebagai keterbatasan.
* **Pertimbangan Etika (Bias & Privasi):** Cuitan bersifat publik, namun mention dan URL dihapus, analisis dilakukan agregat, dan username tidak dipublikasikan. Hasil tidak dipakai untuk menilai atau menindak individu.

### 3.2 Perumusan Learning Problem

| Komponen | Spesifikasi Teknis |
| :--- | :--- |
| **Target ($Y$)** | Variabel multikelas (`sentiment`):<br>• $y = 0$: **Positif**<br>• $y = 1$: **Netral**<br>• $y = 2$: **Negatif** |
| **Objective** | Meminimalkan *cross-entropy* berbobot kelas selama pelatihan (log-loss + L2 untuk Logistic Regression, hinge loss + L2 untuk Linear SVM) dan memaksimalkan **Macro F1-Score** pada data uji, dengan penekanan pada **Recall kelas Negatif**. |
| **Data ($X$)** | Teks cuitan berbahasa Indonesia yang telah dibersihkan (URL, mention, duplikat), direpresentasikan sebagai vektor jarang TF-IDF (model ML) serta tensor *token IDs* & *attention mask* (model *Deep Learning*). |

### 3.3 Rencana Tiga Model Eksperimen
1. **Model 1 — Baseline: Logistic Regression dengan Fitur TF-IDF (Unigram–Bigram)**
   * **Alasan Pemilihan:** Tolok ukur dasar yang cepat dan transparan. Bobot koefisien per kata memungkinkan inspeksi kata yang paling berpengaruh pada tiap kelas. `class_weight='balanced'` dipakai untuk menangani ketidakseimbangan kelas.
2. **Model 2 — Advanced: Linear SVM dengan Fitur TF-IDF + Grid Search**
   * **Alasan Pemilihan:** SVM linear efektif pada ruang fitur sparse berdimensi tinggi dan mencari margin maksimum, sehingga berbeda mendasar dari pendekatan probabilistik Logistic Regression. Hyperparameter (C, rentang n-gram, `min_df`) dituning dengan *stratified k-fold cross-validation*.
3. **Model 3 — Deep Learning: Fine-tuning IndoBERT (atau Embedding + Bi-LSTM)**
   * **Alasan Pemilihan:** Representasi kontekstual menangkap negasi dan konteks kalimat yang tidak tertangkap TF-IDF. Model `indobenchmark/indobert-base-p1` dipra-latih pada korpus bahasa Indonesia, dan cuitan yang pendek serta ukuran dataset sedang layak dilatih di Google Colab/Kaggle. Dilatih minimal 10 epoch dengan *weighted cross-entropy*; Bi-LSTM menjadi cadangan bila sumber daya GPU terbatas.

---

## DAFTAR PUSTAKA
* Hansun, S., Suryadibrata, A., Nurhasanah, R., & Fitra, J. (2022). Tweets sentiment on PPKM policy as a Covid-19 response in Indonesia. *Indian Journal of Computer Science and Engineering (IJCSE)*, 13(1), 51–58.
* Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim, Z. Y., ... & Purwarianti, A. (2020). IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding. In *Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing* (pp. 843–857).

---

## TAUTAN PROYEK (DELIVERABLES)
* **Dataset Kaggle:** https://www.kaggle.com/datasets/anggapurnama/twitter-dataset-ppkm
* **Repository GitHub:** `https://github.com/Adzer2605/id-ppkm-sentiment-analysis`
