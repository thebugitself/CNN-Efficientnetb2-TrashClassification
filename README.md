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

```bash
pip install -r requirements.txt
python train.py ...            # perintah melatih model
python evaluate.py --submission ./submission --data ./my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik
  requirements.txt  # (opsional) paket tambahan, mis. timm
evaluate.py         # salinan skrip evaluasi resmi
...                 # kode/notebook pelatihan kelompok
```
