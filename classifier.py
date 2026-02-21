from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
import streamlit as st
import xgboost as xgb
import pandas as pd

class Classifier:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model=None
    def train_simple(self):
        with st.status("Training Base XGBoost Classifier...", expanded=False) as status:
            st.write("Initializing Model...")
            self.model = xgb.XGBClassifier(
                tree_method="hist",
                random_state=self.random_state
            )
            self.model.fit(self.X_train, self.y_train)
            st.write("Fitting Data to Model...")
            status.update(label=f"Training Completed!", state="complete", expanded=False)
        score = self.model.score(self.X_test, self.y_test)
        st.success(f"Completed training baseline model with {score:.2%} accuracy")
        return score
    def train_cv(self, param_dist, n_splits=5, n_iter=10):
        with st.status("Tunning Model Hyperparameters...", expanded=False) as status:
            st.write("Initializing Cross-Validation strategy...")
            cv_strategy = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state
            )
            st.write(f"Searching across {n_iter} combinations...")
            search = RandomizedSearchCV(
                estimator=xgb.XGBClassifier(tree_method='hist'),
                param_distributions=param_dist,
                cv=cv_strategy,
                n_iter=n_iter,
                n_jobs=-1,
                random_state=self.random_state
            )
            search.fit(self.X_train, self.y_train)
            status.update(label=f"Tuning Complete! Best Score: {search.best_score_:.4f}", state="complete", expanded=False)  
            self.model = search.best_estimator_
        st.success(f"Optimized model found with {search.best_score_:.2%} accuracy")
        return search
    def get_feature_list(self) -> pd.DataFrame:
        if self.model is None:
            st.write("Failed to")
            return None
        importance_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return importance_df
