from pathlib import Path
keywords_c_file = Path("keywords_c.jsonl")
keywords_m_file = Path("keywords_m.jsonl")
import joblib

model_c_file   = Path("model_c.joblib")
model_m_file   = Path("model_m.joblib")

# -----------------------------
# Load or create+fit models (single clean block)
# -----------------------------
if model_c_file.exists() and model_m_file.exists():
    print("Loading saved models...")
    model_c = joblib.load(model_c_file)
    model_m = joblib.load(model_m_file)

topics = model_m.get_topics()
print(topics)