from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample data loading
data = # load your dataset here

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data.features, data.target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
