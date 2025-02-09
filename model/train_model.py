import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model(model_path="model.joblib"):
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate (for demonstration, print accuracy)
    accuracy = model.score(X_test, y_test)
    print(f"Model trained. Test Accuracy: {accuracy:.2f}")
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()

