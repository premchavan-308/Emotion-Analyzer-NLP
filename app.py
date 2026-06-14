import streamlit as st
import pickle
import time

# ==========================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Emotion Analytics Engine", 
    page_icon="🧠", 
    layout="centered"
)

# Custom CSS to elevate UI presentation and override flat default styles
st.markdown("""
    <style>
    /* Main container background tweaks */
    .main {
        background-color: #0f1116;
    }
    
    /* Elegant Title Styling */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin-bottom: 5px;
    }
    
    /* Styled Results Card Container */
    .result-card {
        background-color: #1e293b;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        margin-top: 20px;
    }
    
    /* Metric styling customization */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important; /* Tech Cyan */
    }
    
    /* Footer section */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #64748b;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL CORE ASSET LOADING
# ==========================================
@st.cache_resource
def load_nlp_assets():
    # Automatically unpacks your saved dict file
    with open('emotion_assets.pkl', 'rb') as f:
        assets = pickle.load(f)
    return assets['vectorizer'], assets['model']

try:
    vectorizer, model = load_nlp_assets()
except FileNotFoundError:
    st.error("🚨 Critical Asset Missing: 'emotion_assets.pkl' was not found in this folder. Please verify your notebook export cell run.")
    st.stop()

# ==========================================
# 3. EXACT LABEL DICTIONARY INDEX MAPPING
# ==========================================
emotion_dictionary = {
    0: "😢 Sadness / Melancholy",
    1: "😊 Joy / Happiness",
    2: "🥰 Love / Affection",
    3: "😡 Anger / Frustration",
    4: "😨 Fear / Anxiety",
    5: "😲 Surprise / Astonishment"
}

# ==========================================
# 4. APP INTERFACE LAYOUT BUILD
# ==========================================
st.title("🧠 Core Emotion Analytics System")
st.markdown("<p style='color:#94a3b8; font-size:1.1rem; margin-bottom: 25px;'>Production-Ready Text Classification Pipeline using an optimized <b>TF-IDF + Logistic Regression</b> architecture.</p>", unsafe_allow_html=True)

# Dashboard Summary Badge metrics
col_meta1, col_meta2 = st.columns(2)
with col_meta1:
    st.markdown("<div style='background-color:#1e293b; border: 1px solid #334155; padding:10px; border-radius:8px; text-align:center;'><small style='color:#94a3b8;'>VAL ACCURACY BASELINE</small><br><strong style='color:#10b981; font-size:1.2rem;'>88.78% (Nearly 90%)</strong></div>", unsafe_allow_html=True)
with col_meta2:
    st.markdown("<div style='background-color:#1e293b; border: 1px solid #334155; padding:10px; border-radius:8px; text-align:center;'><small style='color:#94a3b8;'>CLASSIFICATION STRATEGY</small><br><strong style='color:#38bdf8; font-size:1.2rem;'>Multi-Class Softmax</strong></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input Box
user_text = st.text_area(
    "Text Input String For Analysis:", 
    placeholder="Type an entry or thought here to test semantic vector evaluation...", 
    height=140
)

# Execution Flow
if st.button("Evaluate Text Matrices", type="primary", use_container_width=True):
    if not user_text.strip():
        st.toast("⚠️ Input buffer is empty. Please enter text strings first.", icon="❗")
    else:
        # Subtle processing element for standard fluid UI presentation
        with st.spinner("Processing text arrays and optimizing matrix fields..."):
            time.sleep(0.35)
            
            # Vector transform & classification execution pipeline
            transformed_vector = vectorizer.transform([user_text])
            raw_prediction = model.predict(transformed_vector)[0]
            prediction_int = int(raw_prediction) # Explicit validation cast for dictionary index mapping
            
            # Extract prediction percentages
            probabilities = model.predict_proba(transformed_vector)[0]
            max_confidence = max(probabilities) * 100
            
        # Display Styled Output Display block
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0; color:#f8fafc;'>📈 Evaluation Matrix Metrics</h3>", unsafe_allow_html=True)
        
        # Pull friendly label string safely mapping from structural integer index
        final_emotion_label = emotion_dictionary.get(prediction_int, f"Unknown ID Variant ({prediction_int})")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"<small style='color:#94a3b8;'>PREDICTED CATEGORY</small><br><h2 style='color:#f8fafc; margin-top:0;'>{final_emotion_label}</h2>", unsafe_allow_html=True)
        with col_res2:
            st.metric(label="Model Confidence Score", value=f"{max_confidence:.2f}%")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive distribution layout expansion block
        with st.expander("📊 View Complete Dimensional Class Distributions"):
            st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Normalized continuous output value spread generated via Logistic Regression coefficients:</p>", unsafe_allow_html=True)
            
            classes_arr = model.classes_
            
            # Map values dynamically across loops safely
            for index, model_class in enumerate(classes_arr):
                current_class_idx = int(model_class)
                friendly_class_name = emotion_dictionary.get(current_class_idx, f"Class {current_class_idx}")
                class_percentage = probabilities[index] * 100
                
                # Render clean structural layout rows
                st.write(f"**{friendly_class_name}**")
                st.progress(int(class_percentage))
                st.markdown(f"<p style='text-align:right; margin-top:-15px; color:#38bdf8; font-size:0.85rem; font-weight:bold;'>{class_percentage:.2f}%</p>", unsafe_allow_html=True)

# ==========================================
# 5. FOOTER STAMP
# ==========================================
st.markdown("""
    <div class='footer'>
        <hr style='border-color: #334155;'>
        SPPU Computer Engineering Portfolio Framework | NLP Analysis Pipeline Module<br>
        <small style='color:#475569;'>Built with Streamlit & Scikit-Learn Engine</small>
    </div>
""", unsafe_allow_html=True)