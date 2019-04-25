# initialize
setwd("/home/prashant/Documents/all-relations/evosem/src/R")
library('lavaan')
library(readr)

# read data
dataset <- read_csv('../../data/g.csv')

# read model as an argument
model <- commandArgs(trailingOnly = TRUE)


# fit the SEM model and return fit indices
tryCatch({
    fit <- sem(model, data=dataset)
    print(fitMeasures(fit, c("cfi","tli", "aic", "bic", "rmsea")))
}, warning = function(w) {
    print('warning occured')
}, error = function(e) {
    print('error occured')
}, finally = {

})