#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import subprocess

def evaluate_SEM(model):
    command = '/home/prashant/anaconda3/envs/rstudio/bin/Rscript'
    path2script = '/home/prashant/Documents/all-relations/gp-all-relation/src/R/evaluate.R'
    # model = 'ind60 =~ x1 + x2  \n \
    # dem60 =~ x3 \n \
    # dem65 =~ x4 + x5 \n \
    # dem60 ~ ind60 \n \
    # dem65 ~ ind60 + dem60\n \
    # '
    args = [model]
    cmd = [command, path2script] + args
    x = subprocess.check_output(cmd, universal_newlines=True)
    return x



