# Earuna — Identifikasi Biometrik Telinga

Prototipe website dengan tampilan alur unggah dan hasil identifikasi seperti referensi desain. Frontend menggunakan HTML/CSS/JavaScript dan backend menggunakan FastAPI.

## Menjalankan

```powershell
cd "D:\My project\model-ear"
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload
```

.\.venv\Scripts\Activate.ps1

Buka `http://127.0.0.1:8000`. Model dibaca dari `backend/models/best_fusion_pca_aug.keras`.

## Artefak preprocessing dan nama kelas

Model fusion tidak dapat menerima citra mentah secara langsung. API sekarang mengikuti pipeline Colab secara persis:

1. Resize 224×224 dan `preprocess_input` MobileNetV2.
2. Ekstraksi GlobalAveragePooling MobileNetV2.
3. StandardScaler MobileNetV2 lalu PCA.
4. Ekstraksi GLCM 32 fitur lalu StandardScaler.
5. Gabungkan PCA + GLCM sesuai urutan training.

Salin model dan artefak berikut ke `backend/artifacts`:

```text
class_names.json
mobilenet_scaler.joblib
mobilenet_pca.joblib
glcm_scaler.joblib
```

Tambahkan blok berikut di bagian akhir notebook Colab setelah variabel training tersedia, lalu jalankan:

```python
import json
import joblib
from pathlib import Path

artifact_dir = Path("/content/earuna_artifacts")
artifact_dir.mkdir(exist_ok=True)

(artifact_dir / "class_names.json").write_text(
    json.dumps(class_names, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
joblib.dump(mobilenet_scaler, artifact_dir / "mobilenet_scaler.joblib")
joblib.dump(mobilenet_pca, artifact_dir / "mobilenet_pca.joblib")
joblib.dump(glcm_scaler, artifact_dir / "glcm_scaler.joblib")
```

Kemudian download folder `earuna_artifacts` dari Colab dan letakkan isinya ke `backend/artifacts`.
Nama kelas akan sama persis dengan `sorted(DATASET_DIR.iterdir())` pada notebook, sehingga tidak lagi menggunakan nama contoh hardcode.

Jika dataset tersedia lokal, nama kelas juga dapat dibaca dari subfolder dataset dengan environment variable:

```powershell
$env:EAR_DATASET_DIR = "D:\path\ke\Telinga"
uvicorn backend.main:app --reload
```

Tanpa artefak preprocessing, endpoint akan mengembalikan error yang jelas dan tidak menjalankan prediksi dengan fitur dummy.
