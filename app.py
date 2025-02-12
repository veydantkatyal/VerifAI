import gradio as gr
import pickle
import zipfile
import os
import numpy as np
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import requests

# ✅ Load Pre-Trained Models
def load_models():
    zip_path = "sentence_transformer_model.zip"
    extract_path = "sentence_transformer_model_final"

    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    with open("ai_text_detector_final (1).pkl", "rb") as f:
        clf = pickle.load(f)

    embedder = SentenceTransformer(extract_path)
    return clf, embedder

clf, embedder = load_models()

# ✅ AI Risk Scoring Function
def get_ai_risk_score(text):
    text_embedded = embedder.encode([text])

    # 🔹 Ensure the shape is (1, N)
    text_embedded = np.array(text_embedded).reshape(1, -1)

    ai_score = round(clf.predict_proba(text_embedded)[0][1] * 100, 2)
    return ai_score, text_embedded

# ✅ Google Fact-Checking API
GOOGLE_FACT_CHECK_API_KEY = "GOOGLE_FACT_CHECK_API"

def fact_check_google(text):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": text, "key": GOOGLE_FACT_CHECK_API_KEY}

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"❌ Error {response.status_code}: Unable to fetch data.", []

    data = response.json()
    if "claims" not in data or len(data["claims"]) == 0:
        return "⚠️ No verified fact-checks found.", []

    fact_check_results = []
    for claim in data["claims"]:
        claim_text = claim.get("text", "N/A")
        review = claim.get("claimReview", [{}])[0]
        publisher = review.get("publisher", {}).get("name", "Unknown Source")
        credibility = review.get("textualRating", "Not Rated")
        url = review.get("url", "#")

        fact_check_results.append(f"🔹 **Claim:** {claim_text}\n🔹 **Source:** {publisher}\n🔹 **Credibility:** {credibility}\n🔹 [More Info]({url})")

    return "✅ Fact-checking results found:", fact_check_results

# ✅ Hugging Face API for Explanation
HUGGINGFACE_API_KEY = "HUGGINGFACE_API_KEY"
HF_MODEL = "tiiuae/falcon-7b-instruct"

def generate_personalized_explanation(text, ai_score):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    Given the text: "{text}"

    - AI Risk Score: {ai_score}%
    - Classification: {"Fully Human-Written" if ai_score < 35 else "Mix of AI & Human" if 35 <= ai_score <= 60 else "Mostly AI-Generated" if ai_score < 85 else "Strongly AI-Generated"}

    Explain the classification with:
    1️⃣ **Why the AI classified it this way?**
    2️⃣ **How AI-generated text differs from human-written text?**
    3️⃣ **A real-world analogy.**
    Ensure responses are clear, concise, and avoid repetition.
    """

    data = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    response = requests.post(f"https://api-inference.huggingface.co/models/{HF_MODEL}", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "❌ Error: Could not generate explanation."

# ✅ SHAP & LIME Fixes
def explain_risk_score_with_visuals(text):
    text_embedded = embedder.encode([text])
    text_embedded = np.array(text_embedded).reshape(1, -1)

    # ✅ Get AI Risk Score
    ai_score, text_embedded = get_ai_risk_score(text)

    # ✅ SHAP Explanation (Fixed)
    reference_dataset = np.random.randn(10, text_embedded.shape[1])  # Background dataset
    explainer_shap = shap.Explainer(clf, reference_dataset, check_additivity=False)
    shap_values = explainer_shap(text_embedded)
    shap_values_mean = np.abs(shap_values.values).mean(axis=0)

    # ✅ LIME Explanation (Fixed)
    lime_exp = None
    if text_embedded.shape[0] > 0:
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=reference_dataset,  # Use a valid dataset
            mode="classification"
        )
        lime_exp = explainer_lime.explain_instance(text_embedded[0], clf.predict_proba)

    # ✅ Fact-Checking
    fact_check_status, fact_check_results = fact_check_google(text)

    # ✅ Hugging Face Explanation
    ai_explanation = generate_personalized_explanation(text, ai_score)

    # ✅ AI Risk Score Visualization (Bar Chart)
    fig1, ax = plt.subplots(figsize=(4, 2))
    sns.barplot(x=["Human", "AI"], y=[100 - ai_score, ai_score], palette=["blue", "red"], ax=ax)
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"AI Risk Score: {ai_score}%")

    # ✅ SHAP Summary Plot (Fixed)
    fig2, ax = plt.subplots(figsize=(6, 4))
    shap.summary_plot(shap_values, text_embedded, show=False)
    ax.set_title(f"SHAP Explanation (Risk Score: {ai_score}%)")

    # ✅ SHAP Feature Importance Bar Plot (Fixed)
    fig3, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=shap_values_mean, y=np.arange(len(shap_values_mean)), palette="coolwarm", ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature Index")
    ax.set_title(f"SHAP Feature Impact - AI Risk Score {ai_score}%")

    # ✅ Feature Heatmap for AI/Human Influence (Fixed)
    fig4, ax = plt.subplots(figsize=(10, 5))
    heatmap_values = np.abs(shap_values.values).reshape(1, -1)
    sns.heatmap(heatmap_values, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title(f"Feature Heatmap - AI Risk Score {ai_score}%")

    return ai_score, fig1, fig2, fig3, fig4, ai_explanation, fact_check_status, "\n".join(fact_check_results)

# ✅ Create Gradio Interface
iface = gr.Interface(
    fn=explain_risk_score_with_visuals,
    inputs=gr.Textbox(lines=3, placeholder="Enter text..."),
    outputs=[
        gr.Textbox(label="AI Risk Score"),
        gr.Plot(label="AI Risk Score Chart"),
        gr.Plot(label="SHAP Summary"),
        gr.Plot(label="SHAP Feature Importance"),
        gr.Plot(label="Feature Heatmap"),
        gr.Textbox(label="AI Explanation"),
        gr.Textbox(label="Fact-Checking Results"),
        gr.Textbox(label="Fact-Check Sources"),
    ],
    title="📝 AI Text Detection & Explainability",
    description="Detect AI-generated text, analyze with SHAP & LIME, get AI explanation, and fact-check claims.",
)

# ✅ Launch Gradio App
if __name__ == "__main__":
    iface.launch(debug=True, share=True)
