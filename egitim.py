from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv("veriseti.csv")
y = df["Etiket"]
X = df.drop("Etiket", axis=1)

# eğitim ve test olarak 2 ye böl
Xegt, Xtst, Yegt, Ytst = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("std", StandardScaler()), 
    ("sinif", LogisticRegression())
    ])


# modeli eğitim verisetini kullanarak eğit
pipeline.fit(Xegt, Yegt)
# modelin test setindeki tahminlerini al
Y_model = pipeline.predict(Xtst)
dogruluk_orani = accuracy_score(Ytst, Y_model)
print(f"Doğruluk Oranı = {dogruluk_orani}")

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
