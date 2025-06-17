Resume Screening App
A powerful Streamlit application for classifying resumes into job categories using Natural Language Processing (NLP) and Machine Learning (ML). Features a modern dark-themed UI and advanced insights like skill clustering, confidence score distribution, skill trends, resume length analysis, and word clouds.
Features

Resume Classification: Classifies PDF/DOCX resumes into categories (e.g., Software Engineer, Data Scientist, Marketing, Project Manager) using a RandomForest model.
Dark-Themed UI: Sleek, professional interface with interactive Plotly charts and tabbed layout for results and insights.
Advanced Insights:
Skill Clustering: Groups resumes by similar skills using K-Means clustering, visualized as a scatter plot.
Confidence Score Distribution: Shows prediction reliability with a histogram of confidence scores per category.
Skill Trends: Identifies top skills per category (e.g., "Python" for Data Scientist) based on word frequency.
Resume Length Analysis: Correlates word count with categories via a box plot.
Word Cloud: Visualizes frequent skills with a blue-on-dark word cloud, robust against empty text.


Dataset: Includes resume_dataset.csv with 100 balanced records (25 per category) for training.
Robustness: Handles file size limits (10MB), corrupt files, and empty text with clear error messages.
Exportable Results: Download classification results as CSV.

Prerequisites

Python 3.10 or higher
Git
Text-rich PDF/DOCX resumes for classification
Internet connection for initial NLTK resource downloads

Setup

Clone the Repository:
git clone https://github.com/<your-username>/resume-screening-app.git
cd resume-screening-app


Install Dependencies:
pip install -r requirements.txt

Dependencies are listed in requirements.txt:

streamlit==1.31.1
pandas==2.2.2
numpy==1.26.4
nltk==3.8.1
scikit-learn==1.5.2
joblib==1.4.2
PyPDF2==3.0.1
python-docx==1.1.2
wordcloud==1.9.3
matplotlib==3.9.2
plotly==5.24.1


Run the App:
streamlit run app.py

Open http://localhost:8501 in your browser (tested as of June 17, 2025, 08:42 PM IST).


Usage

Train the Model:

In the sidebar, upload resume_dataset.csv (included in the repository).
Click "Train Model" to train the RandomForest classifier.
Expected accuracy: ~0.85-0.90 with the provided 100-record dataset.


Classify Resumes:

Upload PDF or DOCX resumes (ensure they contain readable text, not scanned images).
View results in the "Classification Results" tab, including a table with filename, predicted category, confidence score, and word count, plus a category distribution chart.


Explore Insights:

In the "Advanced Insights" tab, analyze:
Skill Clustering: Groups of resumes with similar skills.
Confidence Scores: Reliability of predictions.
Skill Trends: Top skills per category (e.g., "Agile" for Project Manager).
Resume Length: Word count differences across categories.
Word Cloud: Visual of frequent skills.


Insights help HR identify skill trends, candidate groups, and prediction reliability.


Download Results:

Click "Download Results as CSV" to export the classification table.



Dataset

File: resume_dataset.csv
Details: 100 records, balanced across four categories (25 each):
Software Engineer
Data Scientist
Marketing
Project Manager


Format:Resume,Category
"Python developer with 5 years in Django...",Software Engineer
"Data analyst skilled in SQL...",Data Scientist
...


Source: Synthetically generated based on Kaggle resume datasets and job descriptions.
Note: For production, use larger datasets (e.g., Kaggle's "Resume Dataset" with 13,000+ records).


Troubleshooting

Word Cloud Error: If "No valid words found for word cloud" appears, ensure uploaded resumes have readable text (not scanned images). Use OCR tools like Tesseract for scanned PDFs.
Empty Results: Verify that PDF/DOCX files are not empty or corrupt. Test with text-heavy resumes mentioning skills like "Python" or "SEO".
Dependency Issues: Use the exact versions in requirements.txt to avoid compatibility problems.
Model Training Failure: Ensure resume_dataset.csv has Resume and Category columns and no missing values.
Performance: For large datasets, increase max_features in TfidfVectorizer or deploy on a cloud server.


