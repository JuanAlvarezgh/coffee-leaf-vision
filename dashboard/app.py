"""Streamlit dashboard for CoffeeLeafVision."""

import base64
import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_LABELS, CLASS_LABELS_ES, settings  # noqa: E402

API_URL = f"http://{settings.api_host}:{settings.api_port}"
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

st.set_page_config(
    page_title="CoffeeLeafVision",
    page_icon=":material/eco:",
    layout="wide",
)
st.title("CoffeeLeafVision — Diagnóstico de Enfermedades del Café")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Diagnóstico",
        "Galería de muestras",
        "Comparación de modelos",
        "Acerca del Proyecto",
    ]
)


# ── Tab 1: Diagnóstico ───────────────────────────────────────────────────────
with tab1:
    st.header("Diagnóstico de hoja de café")
    st.markdown(
        "Sube una foto de una hoja de café (planta entera no, solo la hoja recortada). "
        "El modelo predice la enfermedad y muestra qué regiones activaron la decisión "
        "(Grad-CAM)."
    )

    uploaded = st.file_uploader("Imagen de hoja", type=["jpg", "jpeg", "png"])

    EJEMPLO_PATH = SAMPLES_DIR / "ejemplo_roya.jpg"

    gradcam_method = st.selectbox(
        "Método de explicabilidad",
        ["gradcam", "gradcam++", "eigencam"],
        help="Grad-CAM clásico, su versión refinada (++), o EigenCAM (sin clase target)",
    )

    # Si el usuario no sube nada, se usa una imagen de ejemplo ya cargada y se analiza
    # automáticamente, para que cualquier visitante vea de inmediato cómo funciona.
    imagen_bytes = None
    imagen_nombre = None
    imagen_tipo = None
    usando_ejemplo = False
    if uploaded is not None:
        imagen_bytes = uploaded.getvalue()
        imagen_nombre = uploaded.name
        imagen_tipo = uploaded.type
    elif EJEMPLO_PATH.exists():
        imagen_bytes = EJEMPLO_PATH.read_bytes()
        imagen_nombre = EJEMPLO_PATH.name
        imagen_tipo = "image/jpeg"
        usando_ejemplo = True

    if usando_ejemplo:
        st.info(
            "Mostrando un ejemplo precargado (roya del café). "
            "Sube tu propia foto arriba para analizar la tuya."
        )

    if imagen_bytes is not None:
        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.image(imagen_bytes, caption="Imagen original", use_column_width=True)

        with col_result, st.spinner("Evaluando..."):
            files = {"file": (imagen_nombre, imagen_bytes, imagen_tipo)}
            try:
                r = requests.post(
                    f"{API_URL}/api/v1/predict",
                    files=files,
                    params={"gradcam_method": gradcam_method},
                    timeout=60,
                )
            except Exception as e:
                st.error(f"No se pudo conectar a la API: {e}")
                r = None

        if r is not None and r.status_code == 200:
            body = r.json()
            with col_img:
                overlay_bytes = base64.b64decode(body["gradcam_overlay_base64"])
                st.image(
                    Image.open(io.BytesIO(overlay_bytes)), caption="Grad-CAM", use_column_width=True
                )

            with col_result:
                risk_color = "green" if body["predicted_class"] == "healthy" else "red"
                st.markdown(f"### Diagnóstico: :{risk_color}[{body['predicted_class_es']}]")
                st.metric("Confianza", f"{body['confidence']:.1%}")
                st.caption(f"Inferencia: {body['inference_ms']:.0f} ms")

                proba_df = pd.DataFrame(
                    [
                        {"Clase": CLASS_LABELS_ES[c], "Probabilidad": p}
                        for c, p in body["probabilities"].items()
                    ]
                ).sort_values("Probabilidad", ascending=True)
                fig = px.bar(proba_df, x="Probabilidad", y="Clase", orientation="h")
                fig.update_layout(xaxis=dict(range=[0, 1]), height=300)
                st.plotly_chart(fig, use_container_width=True)
        elif r is not None:
            st.error(f"Error de la API: {r.status_code} — {r.text}")


# ── Tab 2: Galería de muestras ───────────────────────────────────────────────
with tab2:
    st.header("Galería de Enfermedades del Café")
    st.markdown("Ejemplos representativos del dataset BRACOL + RoCoLe.")

    DISEASE_DESCRIPTIONS = {
        "healthy": ("Hoja sana, sin lesiones visibles.", "Mantener buenas prácticas de manejo."),
        "leaf_rust": (
            "Hongo *Hemileia vastatrix*. Pústulas amarillas-naranjas en el envés.",
            "Fungicidas cúpricos o sistémicos; podas de saneamiento.",
        ),
        "leaf_miner": (
            "Larva minadora (*Leucoptera coffeella*) hace galerías translúcidas.",
            "Control biológico con avispas parasitoides; insecticidas selectivos.",
        ),
        "phoma": (
            "Hongo *Phoma costarricensis*. Manchas necróticas circulares con halo.",
            "Mejorar drenaje y aireación; fungicidas preventivos.",
        ),
        "cercospora": (
            "Hongo *Cercospora coffeicola* (mancha de hierro). Manchas marrones con centro claro.",
            "Fertilización balanceada y fungicidas en floración.",
        ),
    }

    for cls in CLASS_LABELS:
        with st.expander(f"**{CLASS_LABELS_ES[cls]}** ({cls})", expanded=False):
            desc, treatment = DISEASE_DESCRIPTIONS[cls]
            st.markdown(f"**Descripción:** {desc}")
            st.markdown(f"**Manejo agronómico básico:** {treatment}")
            sample_paths = sorted(SAMPLES_DIR.glob(f"{cls}_*.jpg")) + sorted(
                SAMPLES_DIR.glob(f"{cls}_*.png")
            )
            if sample_paths:
                cols = st.columns(min(3, len(sample_paths)))
                for col, path in zip(cols, sample_paths[:3], strict=False):
                    col.image(str(path), caption=path.name)
            else:
                st.info(f"Coloca imágenes de muestra en {SAMPLES_DIR} (ver `samples/README.md`).")


# ── Tab 3: Comparación de modelos ────────────────────────────────────────────
with tab3:
    st.header("Comparación de las 4 Arquitecturas")
    st.markdown(
        "Resultados de 5-fold cross-validation sobre BRACOL + RoCoLe. "
        "El modelo con mejor F1 macro se promueve a producción."
    )

    try:
        info_resp = requests.get(f"{API_URL}/api/v1/model/info", timeout=5)
    except Exception as e:
        st.warning(f"API no disponible: {e}")
        info_resp = None

    if info_resp is not None and info_resp.status_code == 200:
        info = info_resp.json()
        st.subheader("Modelo en Producción")
        c1, c2, c3 = st.columns(3)
        c1.metric("Arquitectura", info["architecture"])
        c2.metric("Versión", info["model_version"])
        c3.metric("Tamaño", f"{info.get('size_mb', 0):.1f} MB")

        st.divider()
        st.subheader("Métricas (5-fold CV)")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Accuracy",
            f"{info.get('cv_accuracy_mean', 0):.4f}",
            help=f"± {info.get('cv_accuracy_std', 0):.4f} entre folds",
        )
        m2.metric(
            "F1 macro",
            f"{info.get('cv_f1_macro_mean', 0):.4f}",
            help=f"± {info.get('cv_f1_macro_std', 0):.4f} entre folds",
        )
        m3.metric(
            "AUC macro",
            f"{info.get('cv_auc_macro_mean', 0):.4f}",
            help=f"± {info.get('cv_auc_macro_std', 0):.4f} entre folds",
        )

        st.divider()
        st.caption(
            f"Latencia mediana CPU: {info.get('latency_median_ms', 0):.1f} ms · "
            f"Parámetros: {info.get('n_parameters', 0):,}"
        )

        st.divider()
        st.subheader("Comparación de las 4 arquitecturas (5-fold CV)")
        try:
            from huggingface_hub import hf_hub_download

            results_path = hf_hub_download(
                repo_id=settings.hf_model_repo,
                filename="results.json",
            )
            import json as _json

            with open(results_path) as _f:
                _results = _json.load(_f)

            rows = []
            for arch, data in _results.items():
                cv = data["cv_summary"]
                rows.append(
                    {
                        "Arquitectura": arch,
                        "Accuracy": f"{cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}",
                        "F1 macro": f"{cv['f1_macro']['mean']:.4f} ± {cv['f1_macro']['std']:.4f}",
                        "AUC macro": f"{cv['auc_macro_ovr']['mean']:.4f} ± {cv['auc_macro_ovr']['std']:.4f}",
                        "Latencia CPU (ms)": f"{data['latency_cpu']['median_ms']:.1f}",
                        "Tamaño (MB)": f"{data['size_mb']:.1f}",
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        except Exception as e:
            st.info(
                f"No se pudo cargar `results.json` desde HF Hub: {e}. "
                "Cuando subas el archivo con `scripts/upload_to_hf.py`, esta tabla se llena automáticamente."
            )


# ── Tab 4: Acerca del Proyecto ───────────────────────────────────────────────
with tab4:
    st.header("Acerca del Proyecto")
    st.markdown(
        "CoffeeLeafVision compara 4 arquitecturas deep learning (ResNet50, EfficientNetV2-S, "
        "MobileNetV3-Large, ViT-Small) para clasificar enfermedades del café con explicabilidad "
        "visual mediante Grad-CAM."
    )

    st.divider()
    st.subheader("Tecnologías")

    tech_stack = [
        ("Python", "https://cdn.simpleicons.org/python"),
        ("PyTorch", "https://cdn.simpleicons.org/pytorch"),
        ("timm", None),
        ("Albumentations", None),
        ("scikit-learn", "https://cdn.simpleicons.org/scikitlearn"),
        ("Pillow", None),
        ("FastAPI", "https://cdn.simpleicons.org/fastapi"),
        ("Streamlit", "https://cdn.simpleicons.org/streamlit"),
        ("Plotly", "https://cdn.simpleicons.org/plotly"),
        ("Hugging Face", "https://cdn.simpleicons.org/huggingface"),
        ("Docker", "https://cdn.simpleicons.org/docker"),
        ("GitHub Actions", "https://cdn.simpleicons.org/githubactions"),
    ]
    for i in range(0, len(tech_stack), 4):
        cols = st.columns(4)
        for col, (name, logo) in zip(cols, tech_stack[i : i + 4], strict=False):
            with col:
                if logo:
                    st.image(logo, width=48)
                st.markdown(f"**{name}**")

    st.divider()
    st.subheader("Sobre el creador")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.image("https://github.com/JuanAlvarezgh.png", width=160)
    with c2:
        st.markdown(
            """
            **Juan Alvarez**

            Estudiante de Ingeniería de Datos y Software en la **Universidad de San Buenaventura**.

            Líneas de trabajo: arquitectura de datos end-to-end, machine learning aplicado al
            sector financiero y APIs de producción.

            **Áreas de interés:** computer vision, MLOps, scoring crediticio, explicabilidad de
            modelos (SHAP, Grad-CAM).
            """
        )

    st.divider()
    st.subheader("Contacto")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image("https://cdn.simpleicons.org/github", width=32)
        st.markdown("[github.com/JuanAlvarezgh](https://github.com/JuanAlvarezgh)")
    with c2:
        st.image("https://cdn.simpleicons.org/gmail", width=32)
        st.markdown("[juanalvarezghcode@gmail.com](mailto:juanalvarezghcode@gmail.com)")
    with c3:
        st.image("https://api.iconify.design/logos:linkedin-icon.svg", width=32)
        st.markdown("[linkedin.com/in/juanalvarezgh](https://www.linkedin.com/in/juanalvarezgh)")
