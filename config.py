from pydantic_settings import BaseSettings

CLASS_LABELS = ["healthy", "leaf_rust", "leaf_miner", "phoma", "cercospora"]
CLASS_LABELS_ES = {
    "healthy": "Sano",
    "leaf_rust": "Roya del cafeto",
    "leaf_miner": "Minador de la hoja",
    "phoma": "Mancha de Phoma",
    "cercospora": "Mancha de hierro (Cercospora)",
}
NUM_CLASSES = len(CLASS_LABELS)
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Settings(BaseSettings):
    hf_model_repo: str = "JuanAlvarezgh/coffee-leaf-vision"
    hf_model_filename: str = "model.pt"
    hf_metrics_filename: str = "metrics.json"
    log_level: str = "INFO"
    data_dir: str = "data"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    max_upload_mb: int = 10

    model_config = {"env_file": ".env"}


settings = Settings()
