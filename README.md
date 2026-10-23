# Studio 2 — Klasifikasi Sampah · Kelompok 4

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap           |   NIM   | Kontribusi utama |
|----|------------------------|---------|------------------|
| 1  |Akhdan Arif Prayoga,    |122140099| F2P              |
| 2  |Ribka Hana Josephine S. |123140103| Whale (VIP)      |
| 3  |Hezkiel Rajani Aritonang|123140118| FFA              |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: EfficientNet-B2
- **Strategi pelatihan**: Learning Rate '0,001', Epochs '8', Optimizer 'Adam'
- **Data tambahan**: tidak ada
- **Ukuran input & preprocessing**: 260×260 piksel (ukuran bawaan EfficientNet-B2).
  - **Training**: `RandomResizedCrop(260, scale=0,7–1,0)`, flip horizontal, rotasi ±15°, `ColorJitter` (brightness/contrast/saturation 0,2).
  - **Validasi & inferensi**: `Resize(288)` lalu `CenterCrop(260)` — sama dengan `get_transform()` di `submission/model.py`.
  - **Normalisasi**: mean `[0,485; 0,456; 0,406]`, std `[0,4785; 0,4733; 0,4743]` (bawaan EfficientNet-B2 dari Hugging Face).

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) |bc9e24a | 0,9412 | 0,9316 |
| Preview 2 (21 Okt) | | 0,9432 | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):

- **Tuning jumlah epoch**: percobaan awal pakai **15 epoch**, lalu diturunkan ke **8 epoch**
  supaya setara dengan model pembanding dosen (ResNet-18) yang juga dilatih 8 epoch —
  jadi perbandingannya apple-to-apple.
- **Hasil perbandingan**: pada jumlah epoch yang sama, Macro-F1 model dosen ≈ **0,92**,
  model kami (EfficientNet-B2) ≈ **0,94** — backbone EfficientNet-B2 unggul tipis di setting epoch setara.

## Cara Reproduksi

```bash
pip install torch torchvision transformers pillow
python evaluate.py --submission ./submission --data ./my_val
```

- `--data` = folder validasi berstruktur `<data>/<nama_kelas>/*.jpg`, 5 kelas:
  `botol_plastik`, `gelas_plastik`, `kertas_kardus`, `saset_kemasan`, `styrofoam`.
- Skrip mencetak Macro-F1, akurasi, F1 per kelas, dan confusion matrix.

Melatih ulang dari nol: jalankan `EfficientNet_Trash_Classification_Training.ipynb` di
Google Colab (GPU) — dataset `studio2-sampah.zip` + starter kit di folder Drive
`Studio2_DeepLearning/`, lalu eksekusi sel berurutan (8 epoch). Hasilnya folder
`submission/` (model.py, config.json, model.pth).

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
