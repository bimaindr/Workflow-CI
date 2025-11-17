import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(filepath):
    """Load data yang sudah di-preprocessing"""
    df = pd.read_csv(filepath)
    
    # Encode target
    if df['price_category'].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(df['price_category'])
    else:
        y = df['price_category'].values
    
    # Pisahkan features dan target
    columns_to_drop = ['price_category']
    if 'selling_price' in df.columns:
        columns_to_drop.append('selling_price')
    
    X = df.drop(columns=columns_to_drop)
    
    return X, y


def train_models_with_autolog(X_train, X_test, y_train, y_test):
    """Melatih berbagai model dengan MLflow autolog"""
    
    # Definisi model-model yang akan dilatih
    models = {
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(kernel='rbf', random_state=42),
        'GaussianNB': GaussianNB()
    }
    
    results = []
    
    # Train setiap model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Enable autolog untuk scikit-learn
        mlflow.sklearn.autolog()
        
        with mlflow.start_run(run_name=model_name):
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Log additional metrics
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)
            
            # Simpan hasil
            results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # Disable autolog setelah selesai
    mlflow.sklearn.autolog(disable=True)
    
    return pd.DataFrame(results)


def main():
    """Fungsi utama untuk menjalankan training"""
    # Load preprocessed data
    filepath = "CAR DETAILS_FROM_CAR_DEKHO_preprocessing.csv"
    X, y = load_preprocessed_data(filepath)
    print(f"Data loaded: {X.shape[0]} rows, {X.shape[1]} features\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models with autolog
    results_df = train_models_with_autolog(X_train, X_test, y_train, y_test)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Best model
    best_model = results_df.loc[results_df['Accuracy'].idxmax()]
    print("\n" + "="*60)
    print(f"Best Model: {best_model['Model']}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"F1-Score: {best_model['F1-Score']:.4f}")
    print("="*60)
    
    print("\n✓ Training completed!")


if __name__ == "__main__":

    main()
