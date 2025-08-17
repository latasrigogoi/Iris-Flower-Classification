import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

candidates = {
    "LogReg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]),
    "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "SVC": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))]),
    "RF": RandomForestClassifier(n_estimators=200, random_state=42)
}

best_name, best_score, best_model = None, -1.0, None
for name, model in candidates.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    mean_score = scores.mean()
    print(f"{name} CV accuracy: {mean_score:.4f}")
    if mean_score > best_score:
        best_name, best_score, best_model = name, mean_score, model

print(f"\nSelected model: {best_name} (CV accuracy ~ {best_score:.4f})")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Optional quick plot
plt.figure()
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y, alpha=0.8)
plt.xlabel(iris.feature_names[0]); plt.ylabel(iris.feature_names[1])
plt.title("Iris (two-feature view)")
plt.show()
