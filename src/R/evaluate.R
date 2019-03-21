# initialize
setwd("/home/prashant/Documents/all-relations/gp-all-relation/src/R")
library('lavaan')
library(readr)

# read data
dataset <- read_csv('../../data/g.csv')

# read model as an argument
model <- commandArgs(trailingOnly = TRUE)

# fit the SEM model and return fit indices
fit <- sem(model, data=dataset)
print(fitMeasures(fit, c("cfi","tli", "aic", "bic", "rmsea")))
