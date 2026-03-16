# g_train_meta_logistic_regression.py

"""
Train stacking Logistic Regression meta learner

Meta purpose
-------------
Combine OOF predictions from three base models (OCT0, OCT1, OCTA3)
Meta feature vector: X_meta = [oct0_feat, oct1_feat, octa3_feat]

Meta loss
---------
Weighted Binary Cross Entropy + L2 regularization
α implemented via class_weight="balanced"

Input
-----
outputs/meta_dataset/<meta_tag>/
    meta_train_oof.csv
    meta_dataset_info.json

Output
------
outputs/meta_model/<meta_tag>/
    meta_model.pkl
    meta_coefficients.csv
    meta_metrics.json
    meta_training_config.json
    meta_roc_curve.png
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT="/data/Irene/SwinTransformer/Swin_Meta"
META_ROOT=os.path.join(PROJECT_ROOT,"outputs","meta_dataset")
MODEL_ROOT=os.path.join(PROJECT_ROOT,"outputs","meta_model")

# Create output directory
def ensure_dir(path):
    os.makedirs(path,exist_ok=True)

# Plot ROC curve
def plot_roc(y_true,prob,save_path):
    fpr,tpr,_=roc_curve(y_true,prob)
    auc=roc_auc_score(y_true,prob)
    plt.figure(figsize=(6,6))
    plt.plot(fpr,tpr,label=f"Stacking LR (AUC={auc:.4f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stacking Meta ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path,dpi=200)
    plt.close()

# Main training procedure
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--meta_tag",required=True)
    parser.add_argument("--cv",type=int,default=5)
    args=parser.parse_args()

    # Load meta dataset
    meta_dir=os.path.join(META_ROOT,args.meta_tag)
    meta_csv=os.path.join(meta_dir,"meta_train_oof.csv")
    info_json=os.path.join(meta_dir,"meta_dataset_info.json")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(meta_csv)

    df=pd.read_csv(meta_csv)

    # Prepare meta features
    features=["oct0_feat","oct1_feat","octa3_feat"]
    X=df[features].values
    y=df["y_true"].values

    n_samples=len(X)
    pos_ratio=float(np.mean(y))
    print("Meta samples:",n_samples)
    print("Positive ratio:",pos_ratio)

    # Build pipeline (scaling + LR)
    pipeline=Pipeline([
        ("scaler",StandardScaler()),
        ("lr",LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000
        ))
    ])

    # Hyperparameter search
    param_grid={"lr__C":[0.001,0.01,0.1,1.0,10.0]}
    cv=StratifiedKFold(n_splits=args.cv,shuffle=True,random_state=42)

    grid=GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    # Train meta learner
    grid.fit(X,y)
    best_model=grid.best_estimator_
    print("Best parameters:",grid.best_params_)

    # Evaluate meta model
    prob=best_model.predict_proba(X)[:,1]
    auc=roc_auc_score(y,prob)
    acc=accuracy_score(y,prob>0.5)

    # Prepare output directory
    out_dir=os.path.join(MODEL_ROOT,args.meta_tag)
    ensure_dir(out_dir)

    # Save trained model
    joblib.dump(best_model,os.path.join(out_dir,"meta_model.pkl"))

    # Save modality weights
    coef=best_model.named_steps["lr"].coef_[0]
    coef_df=pd.DataFrame({
        "feature":features,
        "coef":coef,
        "abs_coef":np.abs(coef)
    }).sort_values("abs_coef",ascending=False)
    coef_df.to_csv(os.path.join(out_dir,"meta_coefficients.csv"),index=False)

    # Save ROC curve
    plot_roc(y,prob,os.path.join(out_dir,"meta_roc_curve.png"))

    # Save metrics
    metrics={
        "auc":float(auc),
        "accuracy":float(acc),
        "samples":int(n_samples),
        "positive_ratio":pos_ratio,
        "best_params":grid.best_params_,
        "cv_folds":args.cv
    }
    with open(os.path.join(out_dir,"meta_metrics.json"),"w") as f:
        json.dump(metrics,f,indent=2)

    # Save training configuration
    if os.path.exists(info_json):
        with open(info_json) as f:
            dataset_info=json.load(f)
    else:
        dataset_info={}

    config={
        "meta_tag":args.meta_tag,
        "meta_dataset":meta_csv,
        "features":features,
        "dataset_info":dataset_info,
        "grid_search":param_grid
    }
    with open(os.path.join(out_dir,"meta_training_config.json"),"w") as f:
        json.dump(config,f,indent=2)

    print("Saved meta model to:",out_dir)

if __name__=="__main__":
    main()