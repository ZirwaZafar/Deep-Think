from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Model creation
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Model fitting
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
