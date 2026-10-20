# Image Detection Project

CLIP zero-shot image classification with Streamlit. Upload an image or use your webcam, set candidate labels, and get confidence scores.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Uses `openai/clip-vit-large-patch14`. For better results, use a short, focused list of labels.
