# ----------------------------------------------------------------------------
# file: vadis_simulation_code.R
# author: Jason Grafmiller
# date: 2023-06-30
# description:
# Code for reproducing the simulation studies for validating the VADIS method
# ----------------------------------------------------------------------------

pacman::p_load(
  here,        # for file path management
  tidyverse,   # for data wrangling and plotting: dplyr, ggplot2, etc.
  data.table,  # for data.tables (*much* faster data wrangling)
  knitr,       # for working with markdown docs

  patchwork,   # for combining ggplots
  ggrepel,     # for nicely annotating points with text
  ggdendro,    # for tree dendrograms in {ggplot2}
  dendextend,  # supplement to {ggdendro}
  extrafont,   # for fonts in figures
  ggpubr,      # for ggplot themes
  ggsci,       # for ggplot color themes
  plotly,      # for interactive plots

  hypr,        # for calculating custom factor contrast codings
  lme4,        # for mixed-effects models
  lmerTest,    # significance tests for GLMMs
  optimx,      # for lme4 optimizer
  MuMIn,       # for r.squaredGLMM
  car,         # for recoding
  performance, # for evaluating model
  parameters,  # for summarizing model
  effects,     # get partial effects from models
  Hmisc,       # for somers2()

  ranger,      # for fast random forests
  phangorn,    # for neighborNets
  VADIS        # for VADIS analysis
)

library(rgl)
library(kableExtra)

theme_set(theme_minimal())


# Simulating stepwise variety data ----------------------------------------

# Here we create simulate 'variety' dataset with gradually diverging constraint weightings. In step one, varieties are identical, and are incrementally less so with each step, as in @fig-stepwise-effects.

seed <- 43214; # random seed
n_obs_per_var <- 2000; # Number of observations in datasets
n_intercepts <- 15; # Number of distinct baseline variant proportion

# set values for predictor weights in the data
cfs <- c(
  bin1 = 2,
  bin2 = -1.38,
  bin3 = -.56,
  cont1 = -.03,
  cont2 = .3,
  cont3 = .69,
  cont4 = 1.1,
  cat1b = -1.2,
  cat1c = -.1
)

set.seed(seed)
# create distinct intercepts of reasonable values on the log odds scale
intercepts <- rnorm(n_intercepts, -.1, .5)

n_preds <- length(cfs) + 3 # Number of predictors in simulation

s <- seq(0, .5, .05) # create varying degrees of similarity

stepwise_data_list <- vector("list")
for (n in seq_along(s)){ ## loop through scale steps
  j <- s[n]
  data_list <- vector("list")
  for(i in seq_along(intercepts)){
    intercept <- intercepts[i]
    dataset <- data.table(
      bin1 = rbinom(n_obs_per_var, 1, .35) %>% as.factor(),
      bin2 = rbinom(n_obs_per_var, 1,  .75) %>% as.factor(),
      bin3 = rbinom(n_obs_per_var, 1, .6) %>% as.factor(),
      cont1 = rnorm(n_obs_per_var, 100, 5),
      cont2 = rnorm(n_obs_per_var, 10, 2),
      cont3 = rf(n_obs_per_var, 1, 10),
      cont4 = rpois(n_obs_per_var, 2),
      cat1 = sample(c("a", "b", "c"), n_obs_per_var, replace = T,
                    prob = c(.6, .3, .1)) %>% factor(),
      NOISE_1 = rnorm(n_obs_per_var, 0, 5),
      NOISE_2 = rpois(n_obs_per_var, 2),
      NOISE_3 = rbinom(n_obs_per_var, 1, .4) %>% as.factor()
    )
    ## We sample one value for each coefficient from a normal distribution with
    ## a mean at the values in `cfs` and a standard deviation equivalent to
    ## j * the mean, where j increases from 0, to .1, to .2, ..., 1.
    ## So the most similar case samples its first coefficient from N(2.3, 0),
    ## while the most diverse case samples its first coefficient from N(2.3, 2.3).
    ## This results in 11 sets of ten varieties whose grammars should be
    ## increasingly different from one another.
    var_y <- intercept +
      rnorm(1, cfs[1], abs(cfs[1])*j) * dataset[, bin1==1] +
      rnorm(1, cfs[2], abs(cfs[2])*j) * dataset[, bin2==1] +
      rnorm(1, cfs[3], abs(cfs[3])*j) * dataset[, bin3==1] +
      rnorm(1, cfs[4], abs(cfs[4])*j) * dataset[, cont1] +
      rnorm(1, cfs[5], abs(cfs[5])*j) * dataset[, cont2] +
      rnorm(1, cfs[6], abs(cfs[6])*j) * dataset[, cont3] +
      rnorm(1, cfs[7], abs(cfs[7])*j) * dataset[, cont4] +
      rnorm(1, cfs[8], abs(cfs[8])*j) * dataset[, cat1=="b"] +
      rnorm(1, cfs[9], abs(cfs[9])*j) * dataset[, cat1=="c"]

    dataset[, ':=' (Name = rep(paste0("var_", LETTERS[i], j),
                               n_obs_per_var) %>% factor(),
                    log.odds = var_y)]
    dataset[, Prob := 1 / (1 + exp(-log.odds))]
    dataset[, Variant := ifelse(rbinom(n_obs_per_var, 1, Prob), "A", "B") %>% factor()]
    dataset[, scale := j]
    dataset[, step := paste0("step_", n-1)]
    dataset[, speaker := letters[i]]
    data_list[[i]] <- dataset
  }
  # names(data_list) <- letters[1:15]
  stepwise_data_list[[n]] <- rbindlist(data_list)
}
stepwise_dataset <- rbindlist(stepwise_data_list)

saveRDS(stepwise_data_list, here("data", "simulation_dataset_stepped.rds"))


# Simulating individual 'speaker' data ------------------------------------

# Here we create speaker datasets.

seed <- 43214; # random seed
n_obs_per_var <- 2000; # Number of observations in datasets
n_intercepts <- 15; # Number of distinct baseline variant proportion

set.seed(seed)
intercepts <- rnorm(n_intercepts, 0, .5)
n_preds <- length(coefs) + 3 # Number of predictors in simulation
samps <- sample(c(1.2, .8), 9, replace = T)

speaker_dataset <- data.table()
for(i in seq_along(intercepts)){
  intercept <- intercepts[i]
  dataset <- data.table(
    bin1 = rbinom(n_obs_per_var, 1, .3) %>% as.factor(),
    bin2 = rbinom(n_obs_per_var, 1,  .75) %>% as.factor(),
    bin3 = rbinom(n_obs_per_var, 1, .6) %>% as.factor(),
    cont1 = rnorm(n_obs_per_var, 100, 5),
    cont2 = rnorm(n_obs_per_var, 10, 2),
    cont3 = rf(n_obs_per_var, 1, 10),
    cont4 = rpois(n_obs_per_var, 2),
    cat1 = sample(c("a", "b", "c"), n_obs_per_var, replace = T,
                  prob = c(.6, .3, .1)) %>% factor(),
    NOISE_1 = rnorm(n_obs_per_var, 0, 5),
    NOISE_2 = rpois(n_obs_per_var, 2),
    NOISE_3 = rbinom(n_obs_per_var, 1, .4) %>% as.factor()
  )

  dataset2 <- copy(dataset)
  dataset3 <- copy(dataset)
  dataset4 <- copy(dataset)
  dataset5 <- copy(dataset)

  # Deriving outcome from formula
  var1_yi <- intercept +
    2 * dataset[, bin1==1] +
    -1.38 * dataset[, bin2==1] +
    .56 * dataset[, bin3==1] +
    -.05 * dataset[, cont1] +
    .3  * dataset[, cont2] +
    .69  * dataset[, cont3] +
    1.8  * dataset[, cont4] +
    -1.4   * dataset[, cat1=="b"] +
    -.69   * dataset[, cat1=="c"]

  var2_yi <- intercept +
    -2   * dataset2[, bin1==1] +
    1.38 * dataset2[, bin2==1] +
    -.56 * dataset2[, bin3==1] +
    .05 * dataset2[, cont1] +
    -.3  * dataset2[, cont2] +
    -.69  * dataset2[, cont3] +
    -1.8  * dataset2[, cont4] +
    1.4   * dataset2[, cat1=="b"] +
    .69   * dataset2[, cat1=="c"]

  var3_yi <- intercept +
    .4   * dataset3[, bin1==1] +
    -1 * dataset3[, bin2==1] +
    1 * dataset3[, bin3==1] +
    .03 * dataset3[, cont1] +
    -.3  * dataset3[, cont2] +
    .9  * dataset3[, cont3] +
    .4  * dataset3[, cont4] +
    2.4   * dataset3[, cat1=="b"] +
    -.9   * dataset3[, cat1=="c"]

  # randomly adjust values of var3 by 20% larger or smaller
  var4_yi <- intercept +
    samps[1] * .4   * dataset4[, bin1==1] +
    samps[2] * -1 * dataset4[, bin2==1] +
    samps[3] * 1 * dataset4[, bin3==1] +
    samps[4] * .03 * dataset4[, cont1] +
    samps[5] * -.3  * dataset4[, cont2] +
    samps[6] * .9  * dataset4[, cont3] +
    samps[7] * .4  * dataset4[, cont4] +
    samps[8] * 2.4   * dataset4[, cat1=="b"] +
    samps[9] * -.9   * dataset4[, cat1=="c"]

  # randomly sample values around 0
  var5_yi <- intercept +
    2 * dataset5[, bin1==1] +
    -1 * dataset5[, bin2==1] +
    1 * dataset5[, bin3==1] +
    .05 * dataset5[, cont1] +
    -.3  * dataset5[, cont2] +
    samps[6] * .9  * dataset5[, cont3] +
    -.69  * dataset5[, cont4] +
    -1.4 * dataset5[, cat1=="b"] +
    samps[9] * -.9 * dataset5[, cat1=="c"]

  # Generating outcome based on formula
  paste0("var", 1:4, letters[i])

  dataset[, ':=' (Label = rep(paste0("var1", letters[i]),
                              n_obs_per_var) %>% factor(),
                  log.odds = var1_yi)]
  dataset[, Prob := 1 / (1 + exp(-log.odds))]
  dataset[, Variant := ifelse(rbinom(n_obs_per_var, 1, Prob),
                              "variant_A", "variant_B") %>% factor()]

  dataset2[, ':=' (Label = rep(paste0("var2", letters[i]),
                               n_obs_per_var) %>% factor(),
                   log.odds = var2_yi)]
  dataset2[, Prob := 1 / (1 + exp(-log.odds))]
  dataset2[, Variant := ifelse(rbinom(n_obs_per_var, 1, Prob),
                               "variant_A", "variant_B") %>% factor()]

  dataset3[, ':=' (Label = rep(paste0("var3", letters[i]),
                               n_obs_per_var) %>% factor(),
                   log.odds = var3_yi)]
  dataset3[, Prob := 1 / (1 + exp(-log.odds))]
  dataset3[, Variant := ifelse(rbinom(n_obs_per_var, 1, Prob),
                               "variant_A", "variant_B") %>% factor()]

  dataset4[, ':=' (Label = rep(paste0("var4", letters[i]),
                               n_obs_per_var) %>% factor(),
                   log.odds = var4_yi)]
  dataset4[, Prob := 1 / (1 + exp(-log.odds))]
  dataset4[, Variant := ifelse(rbinom(n_obs_per_var, 1, Prob),
                               "variant_A", "variant_B") %>% factor()]

  dataset5[, ':=' (Label = rep(paste0("var5", letters[i]),
                               n_obs_per_var) %>% factor(),
                   log.odds = var5_yi)]
  dataset5[, Prob := 1 / (1 + exp(-log.odds))]
  dataset5[, Variant := ifelse(rbinom(n_obs_per_var, 1, Prob),
                               "variant_A", "variant_B") %>% factor()]

  speaker_dataset <- rbindlist(list(speaker_dataset,
                                    dataset,
                                    dataset2,
                                    dataset3,
                                    dataset4,
                                    dataset5))
}

speaker_dataset[, Variety := gsub("[a-z]$", "", Label)]
speaker_dataset[, Speaker := gsub("var[0-9]", "", Label)]

saveRDS(as.data.frame(speaker_dataset), here("data", "simulation_datasets_speakers.rds"))



# GLM modeling ------------------------------------------------------------

# Define formula for simple GLMs.

fmla <- Variant ~ bin1 + bin2 + bin3 + cat1 + cont1 + cont2 + cont3 +
  cont4 + NOISE_1 + NOISE_2 + NOISE_3

# Fit models for stepwise datasets.

# create function for modeling multiple datasets
fit_glm_models <- function(fmla, data, split_by = NULL, progress = TRUE,
                           save_path = format(Sys.time(), here("data", "glm_list_%Y-%m-%d_%Hh-%Mm-%Ss.rds"))
){

  if (is.data.frame(data)){
    main_data_list <- split(data, data[, split_by], drop = T)
  } else main_data_list <- data

  # function to standardize continuous predictors (see Gelman, 2008)
  std <- function(x){
    if(is.factor(x) & nlevels(x) == 2) {
      as.numeric(x) - mean(as.numeric(x))
    } else if (is.numeric(x)) {
      (x - mean(x))/(2*sd(x))
    } else x
  }

  pb <- txtProgressBar(min = 0, max = length(main_data_list), style = 3)

  main_glm_list <- vector("list")
  for(i in seq_along(main_data_list)){
    df <- as.data.frame(main_data_list[[i]])

    df[all.vars(fmla)[-1]] <- lapply(df[all.vars(fmla)[-1]], std)

    main_glm_list[[i]] <- glm(fmla, data = df, family = binomial)

    if (progress) {
      setTxtProgressBar(pb, i)
    }
  }
  names(main_glm_list) <- names(main_data_list)

  saveRDS(main_glm_list, save_path)
  close(pb)
  return(main_glm_list)
}


# split the data
stepwise_data_list <- split(stepwise_dataset,
                            list(stepwise_dataset$step, stepwise_dataset$speaker),
                            drop = T)

# run the function
glm_list_stepped <- fit_glm_models(
  fmla,
  stepwise_data_list,
  here("data", "simulation_glm_list_stepped.rds")
)

# Fit models for speakers.

speaker_data_list <- split(speaker_dataset, list(speaker_dataset$Variety, speaker_dataset$Speaker),
                           sep = "_")

glm_list_speakers <- fit_glm_models(
  fmla,
  speaker_data_list,
  here("data", "simulation_glm_list_speakers.rds")
)


# Random forest modeling --------------------------------------------------

rf_list_stepwise <- vector("list")

# create function for modeling multiple datasets
fit_rf_models <- function(fmla, data, split_by = NULL, progress = TRUE,
                          save_path = format(Sys.time(), here("data", "glm_list_%Y-%m-%d_%Hh-%Mm-%Ss.rds"))
){

  if (is.data.frame(data)){
    main_data_list <- split(data, data[, split_by], drop = T)
  } else main_data_list <- data

  pb <- txtProgressBar(min = 0, max = length(main_data_list), style = 3)

  main_rf_list <- vector("list")
  for(i in seq_along(main_data_list)){
    df <- as.data.frame(main_data_list[[i]])

    rf <- ranger::ranger(
      fmla,
      data = df,
      num.tree = 1000L,
      write.forest = FALSE, # no need to save forest objects
      respect.unordered.factors = "partition",
      probability = TRUE,
      importance = "permutation"
    )
    main_rf_list[[i]] <- rf

    if (progress) {
      setTxtProgressBar(pb, i)
    }
  }
  names(main_rf_list) <- names(main_data_list)

  saveRDS(main_rf_list, save_path)
  close(pb)
  return(main_rf_list)
}

# run the function
rf_list_stepped <- fit_rf_models(
  fmla,
  stepwise_data_list,
  here("data", "simulation_rf_list_stepped.rds")
)

rf_list_speakers <- fit_rf_models(
  fmla,
  speaker_data_list,
  here("data", "simulation_rf_list_speakers.rds")
)


# VADIS lines -------------------------------------------------------------

speaker_line1 <- VADIS::vadis_line1(glm_list_speakers, path = F)
speaker_line2 <- VADIS::vadis_line2(glm_list_speakers, path = F)
speaker_line3 <- VADIS::vadis_line3(rf_list_speakers, path = F)
