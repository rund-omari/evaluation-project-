# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# =========================
df = pd.read_csv("student_data.csv")  

score_cols = [
    'logical_reasoning', 'memory_recall', 'problem_solving',
    'analytical_thinking', 'abstract_thinking', 'critical_evaluation',
    'mathematical_reasoning', 'decision_making', 'comprehension',
    'spatial_intelligence', 'coding_score'
]

for col in score_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
    df[col].fillna(df[col].mean(), inplace=True)

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'].fillna(df['age'].mean(), inplace=True)

df['gender'] = df['gender'].map({'Male':0, 'Female':1})
df = pd.get_dummies(df, columns=['university_major'])

le = LabelEncoder()
df['target_track'] = le.fit_transform(df['target_track'])

df['thinking_score'] = df[['analytical_thinking', 'abstract_thinking', 'critical_evaluation']].mean(axis=1)
df['logic_math_score'] = df[['logical_reasoning','mathematical_reasoning','problem_solving']].mean(axis=1)

feature_cols = score_cols + ['age', 'gender', 'thinking_score', 'logic_math_score']
one_hot_cols = [col for col in df.columns if 'university_major_' in col]
feature_cols += one_hot_cols

X = df[feature_cols]
y = df['target_track']


# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# =========================
model = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train_res, y_train_res)

# =========================
joblib.dump(model, "track_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(X_train_res.columns.tolist(), "feature_columns.pkl")

print("Model, LabelEncoder and feature columns saved successfully.")
