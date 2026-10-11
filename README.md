# Studio 2 — Klasifikasi Sampah · Kelompok 4

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap           |   NIM   | Kontribusi utama |
|----|------------------------|---------|------------------|
| 1  |Akhdan Arif Prayoga,    |122140099| Mengedit Readme  |
| 2  |Ribka Hana Josephine S. |123140103| Melihat Readme   |
| 3  |Hezkiel Rajani Aritonang|123140118| Membaca Readme   |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: EfficientNet-B2
- **Strategi pelatihan**: Learning Rate '0,001', Epochs '8', Optimizer 'Adam'
- **Data tambahan**: tidak ada
- **Ukuran input & preprocessing**: input 260x260 piksel (ukuran bawaan EfficientNet-B2).
  Training: RandomResizedCrop(260, scale 0,7-1,0), flip horizontal, rotasi ±15°,
  ColorJitter (brightness/contrast/saturation 0,2). Validasi/inferensi: Resize(288) lalu
  CenterCrop(260). Normalisasi mean [0,485; 0,456; 0,406], std [0,4785; 0,4733; 0,4743].

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | |0,9412 | |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

## Cara Reproduksi

Pelatihan dilakukan lewat notebook `EfficientNet_Trash_Classification_Training.ipynb`
di Google Colab (GPU). Langkahnya:

**1. Siapkan data di Google Drive**

Unggah ke folder `MyDrive/Studio2_DeepLearning/`:

- `studio2-sampah.zip` — dataset (berisi `train/` dan `val/`)
- `starter_kit-*.zip` — starter kit resmi (`train_baseline.py`, `evaluate.py`, dll.)

Struktur data setelah diekstrak: `<data>/<nama_kelas>/*.jpg`, dengan 5 kelas:
`botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, `styrofoam`.

**2. Instal dependensi**

```bash
pip install torch torchvision transformers scikit-learn pillow
```

**3. Jalankan notebook (sel berurutan)**

| Tahap | Sel | Yang dilakukan |
|-------|-----|----------------|
| Mount Drive & ekstrak | Import Dataset from Drive | `drive.mount`, ekstrak dataset + starter kit ke `/content/` |
| Cek data | (cek per kelas) | hitung jumlah gambar per kelas di `../train` |
| Baseline | Eksekusi train_baseline.py | `python -u train_baseline.py --data ../train --epochs 8` (ResNet-18, pembanding) |
| Inisialisasi B2 | Inisialisasi EfficientNet-B2 | `AutoModelForImageClassification.from_pretrained("google/efficientnet-b2", num_labels=5)` |
| Augmentasi | Persiapan Data dan Augmentasi | transform train (RandomResizedCrop 0,7–1,0, flip H, rotasi ±15°, ColorJitter 0,2) + val (Resize 288 → CenterCrop 260) |
| Training B2 | Fine-Tuning EfficientNet-B2 | 8 epoch, AdamW lr 1e-4 + CosineAnnealingLR, `CrossEntropyLoss(label_smoothing=0,1)`, batch 16; simpan bobot terbaik tiap Macro-F1 naik |
| Bikin submission | Pembuatan File Submission | tulis `submission/model.py` + `config.json`, copy `model_best.pth` → `model.pth` |
| Evaluasi & backup | EKSEKUSI CODE EVALUATE | `python evaluate.py --submission ./submission_b2 --data ./my_val`, lalu backup ke Drive |

**4. Evaluasi lokal (opsional, tanpa Colab)**

```bash
python evaluate.py --submission ./submission --data ./my_val --batch-size 32 --out hasil.json
```

`--submission` = folder berisi `model.py` + `model.pth`, `--data` = folder validasi
berstruktur `<data>/<nama_kelas>/*.jpg`. Skrip mencetak Macro-F1, akurasi, F1 per kelas,
dan confusion matrix.

## Struktur Repositori

```
submission/
  model.py                                   # get_model() dan get_transform() — TANPA unduh bobot
  config.json                                # konfigurasi arsitektur EfficientNet-B2 (num_labels=5)
  model.pth                                  # state_dict model terbaik (torch.save)
evaluate.py                                  # salinan skrip evaluasi resmi (dipakai asisten dosen)
EfficientNet_Trash_Classification_Training.ipynb  # notebook pelatihan (Colab): baseline ResNet-18 + fine-tune B2
README.md                                    # laporan ini
```

Kontrak submission: `get_model()` mengembalikan `torch.nn.Module` (arsitektur saja),
`get_transform()` = PIL.Image (RGB) → Tensor [3, 260, 260], output model logits
`[batch, 5]` dengan urutan kelas sesuai `CLASSES` di `evaluate.py`.
