# calibrate_oof_predictions.py

"""
Temperature Scaling calibration for OOF predictions.

Compatible with:
train_singlemode_oof.py
build_meta_dataset.py
train_meta_logistic_regression.py

Terminal
-------
python training/calibrate_oof_predictions.py \
    --model_name swin_tiny \
    --modality OCT0 \
    --run_tag BS16_EP100_LR1e-06_WD0.01_FULL_FINETUNE_FL0.11_0.89_2_WSon_1_2.9
    
runtag:
ls outputs/oof_predictions/swin_tiny/OCT0
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = "/data/Irene/SwinTransformer/Swin_Meta"

OOF_ROOT = os.path.join(PROJECT_ROOT,"outputs","oof_predictions")

MODALITIES=["OCT0","OCT1","OCTA3"]


def ensure_dir(p):
    os.makedirs(p,exist_ok=True)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def fit_temperature(logits,labels,device):

    logits_t=torch.tensor(logits,dtype=torch.float32,device=device).view(-1,1)
    labels_t=torch.tensor(labels,dtype=torch.float32,device=device).view(-1,1)

    nll=nn.BCEWithLogitsLoss()

    with torch.no_grad():
        before=float(nll(logits_t,labels_t).item())

    best_T=1.0
    best_nll=before

    for init_T in [0.7,1.0,1.3,1.8,2.5]:

        logT=nn.Parameter(torch.tensor([np.log(init_T)],device=device))
        opt=optim.LBFGS([logT],lr=0.5,max_iter=60)

        def closure():

            opt.zero_grad()
            T=torch.exp(logT).clamp(1e-3,10)
            loss=nll(logits_t/T,labels_t)
            loss.backward()

            return loss

        try:

            opt.step(closure)

            with torch.no_grad():

                T=float(torch.exp(logT).item())
                after=float(nll(logits_t/T,labels_t).item())

            if after<best_nll:

                best_T=T
                best_nll=after

        except RuntimeError:
            continue

    return best_T,before,best_nll


def apply_calibration(df,T):

    out=df.copy()

    z=out["logit_uncal"].values.astype(float)

    z_calib=z/T

    if "prob_uncal" not in out.columns:
        out["prob_uncal"]=sigmoid(z)

    out["temperature"]=T
    out["logit_calib"]=z_calib
    out["prob_calib"]=sigmoid(z_calib)

    return out


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--model_name",required=True)
    parser.add_argument("--modality",required=True,choices=MODALITIES)
    parser.add_argument("--run_tag",required=True)

    args=parser.parse_args()

    oof_dir=os.path.join(
        OOF_ROOT,
        args.model_name,
        args.modality,
        args.run_tag
    )

    input_csv=os.path.join(oof_dir,"all_folds_oof.csv")

    df=pd.read_csv(input_csv)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds=sorted(df["fold"].unique())

    parts=[]
    fold_summary=[]

    for fold in folds:

        fold_df=df[df["fold"]==fold].copy()

        logits=fold_df["logit_uncal"].values
        labels=fold_df["y_true"].values

        T,before,after=fit_temperature(logits,labels,device)

        fold_calib=apply_calibration(fold_df,T)

        parts.append(fold_calib)

        fold_summary.append({
            "fold":int(fold),
            "temperature":float(T),
            "nll_before":float(before),
            "nll_after":float(after),
            "samples":int(len(fold_df))
        })

    merged=pd.concat(parts)

    merged=merged.sort_values(["fold","exam_key"])

    out_csv=os.path.join(oof_dir,"all_folds_oof_calibrated.csv")

    merged.to_csv(out_csv,index=False)

    summary={

        "model_name":args.model_name,
        "modality":args.modality,
        "run_tag":args.run_tag,
        "calibration_method":"temperature_scaling",
        "feature_source":"logit_uncal",
        "folds":fold_summary
    }

    with open(os.path.join(oof_dir,"calibration_summary.json"),"w") as f:
        json.dump(summary,f,indent=2)

    print("Saved:",out_csv)


if __name__=="__main__":
    main()