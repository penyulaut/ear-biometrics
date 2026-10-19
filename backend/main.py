import json
import io
import logging
import os
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from skimage.feature import graycomatrix, graycoprops


try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - reported clearly by the endpoint
    tf = None


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_fusion_pca_aug.keras"
ARTIFACTS_PATH = Path(__file__).resolve().parent / "artifacts"
DATASET_PATH = Path(
    os.getenv("EAR_DATASET_DIR", str(Path(__file__).resolve().parent / "dataset"))
)
FRONTEND_PATH = ROOT / "frontend"
CLASS_NAMES_PATH = ARTIFACTS_PATH / "class_names.json"
MOBILENET_SCALER_PATH = ARTIFACTS_PATH / "mobilenet_scaler.joblib"
MOBILENET_PCA_PATH = ARTIFACTS_PATH / "mobilenet_pca.joblib"
GLCM_SCALER_PATH = ARTIFACTS_PATH / "glcm_scaler.joblib"
IMG_SIZE = 224
BATCH_SIZE = 16
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPERTIES = ["contrast", "correlation", "energy", "homogeneity"]

model: Any = None
feature_model: Any = None
mobilenet_scaler: Any = None
mobilenet_pca: Any = None
glcm_scaler: Any = None
class_names: list[str] = []
model_error: str | None = None
preprocessing_error: str | None = None
logger = logging.getLogger(__name__)


def _without_quantization_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_quantization_config(item)
            for key, item in value.items()
            if key != "quantization_config"
        }
    if isinstance(value, list):
        return [_without_quantization_config(item) for item in value]
    return value


def _load_model_compatibly() -> Any:
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except (TypeError, ValueError) as first_error:
        if "quantization_config" not in str(first_error):
            raise
        logger.warning(
            "Model metadata uses an older quantization field; loading a normalized copy."
        )
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as handle:
            normalized_path = Path(handle.name)
        try:
            with (
                zipfile.ZipFile(MODEL_PATH) as source,
                zipfile.ZipFile(normalized_path, "w") as target,
            ):
                for entry in source.infolist():
                    content = source.read(entry.filename)
                    if entry.filename == "config.json":
                        config = json.loads(content.decode("utf-8"))
                        content = json.dumps(
                            _without_quantization_config(config)
                        ).encode("utf-8")
                    target.writestr(entry, content)
            return tf.keras.models.load_model(normalized_path, compile=False)
        finally:
            normalized_path.unlink(missing_ok=True)


def _load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return names
        raise ValueError(f"{CLASS_NAMES_PATH.name} harus berisi JSON array nama kelas.")

    if DATASET_PATH.exists():
        return sorted(path.name for path in DATASET_PATH.iterdir() if path.is_dir())

    raise FileNotFoundError(
        "Nama kelas belum tersedia. Tambahkan artifacts/class_names.json "
        "dari variabel class_names di Colab."
    )


def _load_preprocessing_artifacts() -> None:
    global class_names, mobilenet_scaler, mobilenet_pca, glcm_scaler
    class_names_error: Exception | None = None
    try:
        class_names = _load_class_names()
    except (FileNotFoundError, ValueError) as exc:
        class_names_error = exc
    required = {
        "mobilenet_scaler": MOBILENET_SCALER_PATH,
        "mobilenet_pca": MOBILENET_PCA_PATH,
        "glcm_scaler": GLCM_SCALER_PATH,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if class_names_error or missing:
        missing_files = [
            str(CLASS_NAMES_PATH.relative_to(ROOT)),
            *[
                str(required[name].relative_to(ROOT))
                for name in missing
            ],
        ]
        raise FileNotFoundError(
            "Artefak preprocessing belum tersedia: "
            + ", ".join(missing_files)
            + ". Jalankan blok export artefak di kode.txt pada Colab "
            "lalu salin file hasilnya ke backend/artifacts."
        )
    mobilenet_scaler = joblib.load(MOBILENET_SCALER_PATH)
    mobilenet_pca = joblib.load(MOBILENET_PCA_PATH)
    glcm_scaler = joblib.load(GLCM_SCALER_PATH)


def _load_feature_model() -> Any:
    global feature_model
    if feature_model is None:
        backbone = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        backbone.trainable = False
        feature_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=tf.keras.layers.GlobalAveragePooling2D(name="global_pool_pca")(
                backbone.output
            ),
            name="MobileNetV2_Feature_Extractor",
        )
    return feature_model


def extract_glcm_features(image_rgb: np.ndarray) -> np.ndarray:
    gray = np.asarray(Image.fromarray(image_rgb).convert("L"))
    gray_quantized = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(
        gray_quantized,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=32,
        symmetric=True,
        normed=True,
    )
    features: list[float] = []
    for prop in GLCM_PROPERTIES:
        features.extend(graycoprops(glcm, prop).flatten())
    return np.asarray(features, dtype=np.float32)


def image_to_features(image: Image.Image) -> np.ndarray:
    """Reproduce the Colab MobileNetV2 + PCA + GLCM fusion pipeline."""
    image_resized = image.convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS
    )
    image_array = np.asarray(image_resized, dtype=np.uint8)
    image_batch = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(image_array.astype(np.float32), axis=0)
    )
    mobilenet_features = _load_feature_model().predict(
        image_batch, batch_size=1, verbose=0
    )
    mobilenet_pca_features = mobilenet_pca.transform(
        mobilenet_scaler.transform(mobilenet_features)
    )
    glcm_features = extract_glcm_features(image_array).reshape(1, -1)
    glcm_scaled = glcm_scaler.transform(glcm_features)
    combined_features = np.concatenate(
        [mobilenet_pca_features, glcm_scaled], axis=1
    )
    expected_features = model.input_shape[-1]
    if combined_features.shape[1] != expected_features:
        raise ValueError(
            f"Fitur hasil preprocessing berjumlah {combined_features.shape[1]}, "
            f"model mengharapkan {expected_features}."
        )
    return combined_features


def load_model() -> None:
    global model, model_error, preprocessing_error
    if tf is not None and hasattr(tf, "keras") and MODEL_PATH.exists():
        try:
            model = _load_model_compatibly()
        except (OSError, TypeError, ValueError) as exc:
            model_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Unable to load ear identification model.")
        try:
            _load_preprocessing_artifacts()
        except (FileNotFoundError, ValueError, OSError) as exc:
            preprocessing_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Preprocessing artifacts are not ready: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Earuna Ear Identification API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": model_error,
        "preprocessing_ready": preprocessing_error is None,
        "preprocessing_error": preprocessing_error,
        "class_names": class_names,
    }


@app.get("/api/identities")
def identities() -> dict[str, list[dict[str, str]]]:
    try:
        names = class_names or _load_class_names()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "identities": [
            {
                "name": name,
                "initials": "".join(part[0] for part in name.split()[:2]).upper(),
                "identity_id": f"AUR-{index + 80128:06d}",
                "status": "Terdaftar",
            }
            for index, name in enumerate(names)
        ]
    }


@app.post("/api/identify")
async def identify(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File harus berupa gambar.")
    if model is None or preprocessing_error is not None:
        raise HTTPException(
            status_code=503,
            detail=preprocessing_error
            or "Model belum tersedia. Pastikan TensorFlow terpasang dan file model benar.",
        )
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Ukuran gambar maksimal 10 MB.")
    try:
        image = Image.open(io.BytesIO(content))
        features = image_to_features(image)
    except Exception as exc:
        logger.exception("Image preprocessing failed for %s.", file.filename)
        raise HTTPException(
            status_code=400,
            detail=f"Gambar tidak dapat diproses: {type(exc).__name__}.",
        ) from exc

    probabilities = np.asarray(model.predict(features, verbose=0))[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    return {
        "name": class_names[index] if index < len(class_names) else f"Identitas {index + 1}",
        "identity_id": f"AUR-{index + 80128:06d}",
        "confidence": round(confidence * 100, 1),
        "status": "Identitas terdaftar",
        "filename": file.filename or "foto-telinga.jpg",
        "model": MODEL_PATH.name,
    }


@app.get("/identitas", include_in_schema=False)
@app.get("/identitas/", include_in_schema=False)
def identity_page() -> FileResponse:
    return FileResponse(FRONTEND_PATH / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
