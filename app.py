import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import joblib
import PyPDF2
import docx
import os
from tempfile import TemporaryDirectory
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Set page config
st.set_page_config(page_title="Resume Screening App", page_icon="📄", layout="wide")

# Custom CSS for dark-themed UI
st.markdown("""
<style>
    .main { background-color: #1e1e1e; padding: 20px; border-radius: 10px; color: #e0e0e0; }
    .stButton>button { background-color: #007bff; color: #e0e0e0; border-radius: 5px; border: none; padding: 10px 20px; }
    .stButton>button:hover { background-color: #0056b3; }
    .stFileUploader { background-color: #2a2a2a; border: 1px solid #444; border-radius: 5px; padding: 10px; color: #e0e0e0; }
    .stSidebar { background-color: #252525; }
    h1, h2, h3 { color: #007bff; font-family: 'Arial', sans-serif; }
    .stAlert { border-radius: 5px; background-color: #333; color: #e0e0e0; }
    .stDataFrame { background-color: #2a2a2a; color: #e0e0e0; }
    .stProgress > div > div > div > div { background-color: #007bff; }
    .css-1d391kg { color: #e0e0e0; }
    .css-1v3fvcr { background-color: #1e1e1e; }
    .stMarkdown { color: #e0e0e0; }
    .stSelectbox > div > div { background-color: #2a2a2a; color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📄 Advanced Resume Screening with NLP")
st.markdown("Classify resumes into job categories with cutting-edge NLP and ML. Upload a CSV dataset to train and PDF/DOCX resumes to analyze with advanced insights.")

# Sidebar for model training
with st.sidebar:
    st.header("Model Configuration")
    st.subheader("Train Model")
    train_file = st.file_uploader("Upload training CSV", type=["csv"])
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                if train_file is None:
                    st.error("Upload a CSV file.")
                    st.stop()
                df_train = pd.read_csv(train_file)
                if 'Resume' not in df_train.columns or 'Category' not in df_train.columns:
                    st.error("CSV needs 'Resume' and 'Category' columns.")
                    st.stop()

                # Preprocess data
                le = LabelEncoder()
                df_train['Category'] = le.fit_transform(df_train['Category'])
                
                # Text preprocessing
                ps = PorterStemmer()
                stop_words = set(stopwords.words('english'))
                
                @st.cache_data
                def preprocess_text(text, for_wordcloud=False):
                    if not text or pd.isna(text):
                        return ""
                    text = text.lower()
                    text = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z\s]', '', text)).strip()
                    words = nltk.word_tokenize(text)
                    if for_wordcloud:
                        # Less aggressive filtering for word cloud
                        words = [word for word in words if word not in stop_words and len(word) > 2]
                    else:
                        words = [ps.stem(word) for word in words if word not in stop_words]
                    return ' '.join(words) if words else ""
                
                df_train['Processed_Resume'] = df_train['Resume'].apply(lambda x: preprocess_text(x))
                
                # Vectorize text
                tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                X = tfidf.fit_transform(df_train['Processed_Resume']).toarray()
                y = df_train['Category']
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                
                # Evaluate
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Save model
                if os.path.exists('model.joblib'):
                    st.warning("Overwriting existing model.")
                joblib.dump(clf, 'model.joblib')
                joblib.dump(tfidf, 'tfidf.joblib')
                joblib.dump(le, 'label_encoder.joblib')
                
                unique_labels = le.inverse_transform(np.unique(y_test))
                st.success(f"Model trained! Accuracy: {accuracy:.2f}")
                st.write(classification_report(y_test, y_pred, target_names=unique_labels))
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

# Main content
st.header("Resume Classification & Insights")

# Load model
try:
    clf = joblib.load('model.joblib')
    tfidf = joblib.load('tfidf.joblib')
    le = joblib.load('label_encoder.joblib')
    model_loaded = True
except:
    model_loaded = False
    st.warning("No trained model. Train one in the sidebar.")

if model_loaded:
    uploaded_files = st.file_uploader("Upload resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    
    if uploaded_files:
        results = []
        progress_bar = st.progress(0)
        all_texts = []
        
        with TemporaryDirectory() as tmpdirname:
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    if uploaded_file.size > 10 * 1024 * 1024:
                        st.error(f"{uploaded_file.name} exceeds 10MB.")
                        continue
                    
                    file_content = ""
                    file_name = uploaded_file.name
                    
                    if uploaded_file.type == "application/pdf":
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                file_content += text
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        doc = docx.Document(uploaded_file)
                        for para in doc.paragraphs:
                            file_content += para.text + "\n"
                    
                    ps = PorterStemmer()
                    stop_words = set(stopwords.words('english'))
                    
                    def preprocess_text(text, for_wordcloud=False):
                        if not text or pd.isna(text):
                            return ""
                        text = text.lower()
                        text = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z\s]', '', text)).strip()
                        words = nltk.word_tokenize(text)
                        if for_wordcloud:
                            words = [word for word in words if word not in stop_words and len(word) > 2]
                        else:
                            words = [ps.stem(word) for word in words if word not in stop_words]
                        return ' '.join(words) if words else ""
                    
                    processed_text = preprocess_text(file_content)
                    wordcloud_text = preprocess_text(file_content, for_wordcloud=True)
                    text_vector = tfidf.transform([processed_text]).toarray()
                    prediction = clf.predict(text_vector)
                    predicted_category = le.inverse_transform(prediction)[0]
                    confidence = clf.predict_proba(text_vector)[0].max()
                    
                    results.append({
                        "Filename": file_name,
                        "Predicted Category": predicted_category,
                        "Confidence Score": round(confidence, 2),
                        "Raw Text": file_content[:500] + "..." if len(file_content) > 500 else file_content,
                        "Word Count": len(file_content.split())
                    })
                    all_texts.append(wordcloud_text)
                except Exception as e:
                    st.error(f"Error processing {file_name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
        
        if results:
            results_df = pd.DataFrame(results)
            
            # Tabs for results and insights
            tab1, tab2 = st.tabs(["Classification Results", "Advanced Insights"])
            
            with tab1:
                st.subheader("Classification Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Category distribution
                st.subheader("Category Distribution")
                category_counts = results_df['Predicted Category'].value_counts()
                fig = px.bar(x=category_counts.index, y=category_counts.values, color=category_counts.index,
                             title="Distribution of Predicted Categories",
                             labels={'x': 'Category', 'y': 'Count'},
                             color_discrete_sequence=px.colors.qualitative.Bold)
                fig.update_layout(paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font_color="#e0e0e0")
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='resume_classification_results.csv',
                    mime='text/csv'
                )
            
            with tab2:
                st.subheader("Advanced Insights")
                
                # Insight 1: Skill Clustering
                st.markdown("### Skill Clustering")
                st.write("Resumes are grouped by similar skills using K-Means clustering.")
                if len(all_texts) > 1:
                    X_cluster = tfidf.transform(all_texts).toarray()
                    kmeans = KMeans(n_clusters=min(4, len(all_texts)), random_state=42)
                    clusters = kmeans.fit_predict(X_cluster)
                    results_df['Cluster'] = clusters
                    fig_cluster = px.scatter(results_df, x='Word Count', y='Confidence Score', color='Cluster',
                                             hover_data=['Filename', 'Predicted Category'],
                                             title="Resume Skill Clusters",
                                             color_continuous_scale=px.colors.sequential.Viridis)
                    fig_cluster.update_layout(paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font_color="#e0e0e0")
                    st.plotly_chart(fig_cluster, use_container_width=True)
                else:
                    st.warning("Need more resumes for clustering.")
                
                # Insight 2: Confidence Score Distribution
                st.markdown("### Confidence Score Distribution")
                fig_conf = px.histogram(results_df, x='Confidence Score', color='Predicted Category',
                                        title="Distribution of Prediction Confidence Scores",
                                        nbins=20, color_discrete_sequence=px.colors.qualitative.Bold)
                fig_conf.update_layout(paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font_color="#e0e0e0")
                st.plotly_chart(fig_conf, use_container_width=True)
                
                # Insight 3: Temporal Skill Trends
                st.markdown("### Skill Trends Across Categories")
                st.write("Top skills per category based on word frequency.")
                category_skills = {}
                for category in results_df['Predicted Category'].unique():
                    category_texts = results_df[results_df['Predicted Category'] == category]['Raw Text']
                    all_words = []
                    for text in category_texts:
                        words = [word for word in nltk.word_tokenize(text.lower()) if word not in stop_words and word.isalpha()]
                        all_words.extend(words)
                    category_skills[category] = Counter(all_words).most_common(5)
                
                for category, skills in category_skills.items():
                    st.write(f"**{category}**: {', '.join([f'{skill} ({count})' for skill, count in skills])}")
                
                # Insight 4: Resume Length Analysis
                st.markdown("### Resume Length Analysis")
                st.write("Correlation between resume word count and predicted category.")
                fig_length = px.box(results_df, x='Predicted Category', y='Word Count',
                                    title="Resume Length by Category",
                                    color='Predicted Category',
                                    color_discrete_sequence=px.colors.qualitative.Bold)
                fig_length.update_layout(paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font_color="#e0e0e0")
                st.plotly_chart(fig_length, use_container_width=True)
                
                # Insight 5: Word Cloud
                st.markdown("### Skill Word Cloud")
                all_text = " ".join(all_texts)
                if all_text.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='#1e1e1e', color_func=lambda *args, **kwargs: "#007bff").generate(all_text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)
                else:
                    st.warning("No valid words found for word cloud. Try uploading resumes with more text content.")

# Dataset format info
st.sidebar.header("Dataset Format")
st.sidebar.markdown("CSV with 'Resume' (text) and 'Category' (job role) columns. Example:")
st.sidebar.code("""
Resume,Category
"Python developer with 5 years in Django...",Software Engineer
"Data analyst skilled in SQL...",Data Scientist
""")