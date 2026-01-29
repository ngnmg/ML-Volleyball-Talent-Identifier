import pandas as pdfrom sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# تابع برای آموزش مدل‌ها
def train_models(data):
    # جدا کردن X و y
    X = data.drop("Talent", axis=1)
    y = data["Talent"]

    # تقسیم داده
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # تعریف مدل‌ها
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    # آموزش مدل‌ها و ذخیره
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        # ذخیره مدل
        with open(f"../models/{name.replace(' ', '_')}.pkl", "wb") as f:
            pickle.dump(model, f)
    
    return trained_models, X_test, y_test
