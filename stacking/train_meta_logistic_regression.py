# train_meta_logistic_regression.py

"""
Train stacking Logistic Regression meta learner

Meta purpose
------------
Combine OOF predictions from three base models using logistic regression stacking.

Meta feature vector
-------------------
X_meta = [oct0_feat, oct1_feat, octa3_feat]

Meta loss function
------------------
Weighted binary cross entropy with L2 regularization
L = -1/N Σ α_i[y log(p)+(1-y)log(1-p)] + λ/2||w||²

Where α_i = class weight. This compensates residual class imbalance in OOF predictions.

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
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT="/data/Irene/SwinTransformer/Swin_Meta"
META_ROOT=os.path.join(PROJECT_ROOT,"outputs","meta_dataset")
MODEL_ROOT=os.path.join(PROJECT_ROOT,"outputs","meta_model")


def ensure_dir(path):
    os.makedirs(path,exist_ok=True)


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


def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("--meta_tag",required=True)
    parser.add_argument("--C",type=float,default=1.0)
    args=parser.parse_args()

    meta_dir=os.path.join(META_ROOT,args.meta_tag)
    meta_csv=os.path.join(meta_dir,"meta_train_oof.csv")
    info_json=os.path.join(meta_dir,"meta_dataset_info.json")

    if not os.path.exists(meta_csv):
        raise FileNotFoundError(meta_csv)

    df=pd.read_csv(meta_csv)
    features=["oct0_feat","oct1_feat","octa3_feat"]
    X=df[features].values
    y=df["y_true"].values
    n_samples=len(X)
    pos_ratio=float(np.mean(y))

    print("Meta samples:",n_samples)
    print("Positive ratio:",pos_ratio)

    classes=np.unique(y)
    
    # Automatically calculates and balances weights
    class_weights=compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )

    class_weight_dict=dict(zip(classes,class_weights))

    print("Class weights:",class_weight_dict)

    model=LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight=class_weight_dict,
        max_iter=2000,
        C=args.C
    )

    model.fit(X,y)
    prob=model.predict_proba(X)[:,1]

    auc=roc_auc_score(y,prob)
    acc=accuracy_score(y,prob>0.5)

    out_dir=os.path.join(MODEL_ROOT,args.meta_tag)
    ensure_dir(out_dir)

    joblib.dump(
        model,
        os.path.join(out_dir,"meta_model.pkl")
    )

    coef=pd.DataFrame({
        "feature":features,
        "coef":model.coef_[0]
    })

    coef.to_csv(
        os.path.join(out_dir,"meta_coefficients.csv"),
        index=False
    )

    plot_roc(
        y,
        prob,
        os.path.join(out_dir,"meta_roc_curve.png")
    )

    metrics={

        "auc":float(auc),
        "accuracy":float(acc),
        "samples":int(n_samples),
        "positive_ratio":pos_ratio,
        "class_weight":class_weight_dict,
        "solver":"lbfgs",
        "penalty":"l2",
        "C":args.C,
        "intercept":float(model.intercept_[0])
    }

    with open(
        os.path.join(out_dir,"meta_metrics.json"),
        "w"
    ) as f:
        json.dump(metrics,f,indent=2)

    if os.path.exists(info_json):
        with open(info_json) as f:
            dataset_info=json.load(f)
    else:
        dataset_info={}

    config={
        "meta_tag":args.meta_tag,
        "meta_dataset":meta_csv,
        "features":features,
        "dataset_info":dataset_info
    }

    with open(
        os.path.join(out_dir,"meta_training_config.json"),
        "w"
    ) as f:
        json.dump(config,f,indent=2)

    print("Saved meta model to:",out_dir)

if __name__=="__main__":
    main()