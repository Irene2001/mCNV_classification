# train_meta_logistic_regression.py

"""
Train stacking logistic regression meta learner.

Input:
meta_dataset.csv

Output:
meta_model.pkl
meta_metrics.json
meta_coefficients.csv
roc_curve.png
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve
import matplotlib.pyplot as plt


PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"

META_ROOT = os.path.join(PROJECT_ROOT,"outputs","meta_dataset")
MODEL_ROOT = os.path.join(PROJECT_ROOT,"outputs","meta_model")


def ensure_dir(p):
    os.makedirs(p,exist_ok=True)


def plot_roc(y,prob,save):

    fpr,tpr,_ = roc_curve(y,prob)

    plt.figure()

    plt.plot(fpr,tpr,label="Stacking LR")
    plt.plot([0,1],[0,1],'--')

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.title("Meta ROC")

    plt.legend()

    plt.savefig(save,dpi=200)

    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name",required=True)
    parser.add_argument("--meta_tag",required=True)

    parser.add_argument("--C",type=float,default=1.0)

    args = parser.parse_args()

    meta_csv = os.path.join(
        META_ROOT,
        args.model_name,
        args.meta_tag,
        "meta_dataset.csv"
    )

    df = pd.read_csv(meta_csv)

    X = df[[
        "oct0_feat",
        "oct1_feat",
        "octa3_feat"
    ]].values

    y = df["y_true"].values

    print("Training samples:",len(X))

    model = LogisticRegression(

        penalty="l2",
        solver="lbfgs",

        class_weight="balanced",

        max_iter=1000,

        C=args.C
    )

    model.fit(X,y)

    prob = model.predict_proba(X)[:,1]

    auc = roc_auc_score(y,prob)

    acc = accuracy_score(y,prob>0.5)

    print("AUC:",auc)
    print("ACC:",acc)

    out_dir = os.path.join(
        MODEL_ROOT,
        args.model_name,
        args.meta_tag
    )

    ensure_dir(out_dir)

    joblib.dump(
        model,
        os.path.join(out_dir,"meta_model.pkl")
    )

    coef = pd.DataFrame({

        "feature":[
            "oct0_feat",
            "oct1_feat",
            "octa3_feat"
        ],

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

        "C":args.C,

        "intercept":float(model.intercept_[0]),

        "samples":int(len(X))
    }

    with open(os.path.join(out_dir,"meta_metrics.json"),"w") as f:
        json.dump(metrics,f,indent=2)

    print("Saved model to:",out_dir)


if __name__=="__main__":
    main()