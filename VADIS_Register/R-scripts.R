#this file contains the scripts used in the study "Variation-based Distance and Similarity Modeling: A new way to measure  distances between registers"

#loading reqiured packages 
library(tidyverse) # for data wrangling
library(lme4) # for regression models
library(party) # for random forests
library(phangorn) # for neighborNets
library(car) # for re-coding
library(VADIS)
library(ade4)
library(brms)
library(knitr)

#PVs
## step0: data reading & recoding & spliting
pv_data <- read.csv("PVs.csv")
pv_data$Register <- car::recode(pv_data$Register, "
                             'online'  = 'online'; 
                             'spok.informal'  = 'SpokenInformal'; 
                             'spok.formal' = 'SpokenFormal'; 
                             'writ.informal'  = 'WrittenInformal';
                             'writ.formal' = 'WrittenFormal'
                             ")
summary(pv_data) 
### split dataset according to register
pv_data_list <- split(pv_data, pv_data$Register, drop = TRUE) # drop unused levels
names(pv_data_list)

## Step1: defining the most important factors
f1 <- Response ~ 
  DirObjWordLength +
  DirObjDefiniteness +
  DirObjGivenness +
  DirObjConcreteness +
  DirObjThematicity +
  DirectionalPP +
  Semantics +
  Surprisal.P

pv_data$Variety=as.factor(pv_data$Variety)
pv_data$DirObjHead=as.factor(pv_data$DirObjHead)
pv_data$Response=as.factor(pv_data$Response)
pv_data$DirObjLettLength=as.numeric(pv_data$DirObjLettLength)
pv_data$DirObjDefiniteness=as.factor(pv_data$DirObjDefiniteness)
pv_data$DirObjGivenness=as.factor(pv_data$DirObjGivenness)
pv_data$DirObjConcreteness=as.factor(pv_data$DirObjConcreteness)
pv_data$DirObjThematicity=as.numeric(pv_data$DirObjThematicity)
pv_data$DirectionalPP=as.factor(pv_data$DirectionalPP)
pv_data$Semantics=as.factor(pv_data$Semantics)
pv_data$Surprisal.P=as.numeric(pv_data$Surprisal.P)

##step2: fitting the glmer models
###formula
f2 <- update(f1, .~. + (1|Variety) + (1|Verb) + (1|Particle))
f2
###model fitting
pv_glmer_list <- vector("list")
for (i in seq_along(pv_data_list)){
  pv_d <- pv_data_list[[i]]
  pv_d$Response=as.factor(pv_d$Response) # added this step to avoid error report
  # standardize the model inputs, excluding the response and random effects
  pv_d_std <- stand(pv_d, cols = f2) 
  # fit the model
  pv_glmer_list[[i]] <- glmer(f2, data = pv_d_std, family = binomial, 
                              # set optimizer controls to help convergence 
                              control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e7)))
  rm(pv_d, pv_d_std) # remove datasets
}

names(pv_glmer_list) <- names(pv_data_list)

###checking fits

summary_stats(pv_glmer_list, pv_data_list) %>% 
  round(3)

##step3: the first line of evidence - constraint significance
pv_signif_line <- vadis_line1(pv_glmer_list, path = FALSE)
pv_signif_line$signif.table
pv_signif_line$distance.matrix %>% 
  round(3)
pv_signif_line$similarity.scores %>% 
  arrange(desc(Similarity))
write.csv(pv_signif_line$signif.table, 
          file = "pv_line1_significance_table.csv")
write.csv(as.matrix(pv_signif_line$distance.matrix), 
          file = "pv_line1_distance_matrix.csv")
write.csv(pv_signif_line$similarity.scores, 
          file = "pv_line1_similarity_scores.csv")

##step4: the second line of evidence -constraint strength
pv_coef_line <- vadis_line2(pv_glmer_list, path = FALSE)
pv_coef_line$coef.table %>% 
  round(3)
pv_coef_line$distance.matrix %>% 
  round(3)
pv_coef_line$similarity.scores %>% 
  arrange(desc(Similarity))

##step5: Fitting CRF models
library(tuneRanger)
library(mlr)

tune_df <- data.frame(matrix(NA, ncol =5, nrow = 5))
names(tune_df) <- c("mtry", "min.node.size", "sample.fraction", "auc", "exec.time")
f1

for (i in seq_along(pv_data_list)){
  pv_d <- pv_data_list[[i]][, all.vars(f1)]
  pv_d$DirObjGivenness=as.factor(pv_d$DirObjGivenness) ### added to to prevent error
  pv_d$DirObjWordLength=as.numeric(pv_d$DirObjWordLength)
  pv_d$DirObjDefiniteness=as.factor(pv_d$DirObjDefiniteness)
  pv_d$DirObjConcreteness=as.factor(pv_d$DirObjConcreteness)
  pv_d$DirObjThematicity=as.numeric(pv_d$DirObjThematicity)
  pv_d$DirectionalPP=as.factor(pv_d$DirectionalPP)
  pv_d$Semantics=as.factor(pv_d$Semantics)
  pv_d$Surprisal.P=as.numeric(pv_d$Surprisal.P)
  pv_task <- makeClassifTask(data = pv_d, target = "Response")
  
  result <- tuneRanger(pv_task, measure = list(auc), num.trees = 1000, 
                       num.threads = 4, iters = 80, show.info = F)
  
  tune_df[i,] <- result$recommended.pars
}
rownames(tune_df) <- names(pv_data_list)

write.csv(tune_df, 
          file = "pv_crf_tune_df.csv")

### fitting CRF models to per register and alternation
#####using cforest() function
library(party)
rf_list <- vector("list")
set.seed(123)
for (i in seq_along(pv_data_list)){
  pv_d <- pv_data_list[[i]]
  pv_d$Response<-factor(pv_d$Response)
  pv_d$DirObjGivenness=as.factor(pv_d$DirObjGivenness)
  pv_d$DirObjWordLength=as.numeric(pv_d$DirObjWordLength)
  pv_d$DirObjDefiniteness=as.factor(pv_d$DirObjDefiniteness)
  pv_d$DirObjConcreteness=as.factor(pv_d$DirObjConcreteness)
  pv_d$DirObjThematicity=as.numeric(pv_d$DirObjThematicity)
  pv_d$DirectionalPP=as.factor(pv_d$DirectionalPP)
  pv_d$Semantics=as.factor(pv_d$Semantics)
  pv_d$Surprisal.P=as.numeric(pv_d$Surprisal.P)
  # fit the random forest and add it to the list
  rf_list[[i]] <- cforest(
    f1, 
    data = pv_d,
    controls = cforest_unbiased(ntree = 1000, mtry = 3)
  )
}

names(rf_list) <- names(pv_data_list)

####check fitting 
##### split subset data by register
library(tidyverse)
online<-pv_data %>% filter(Register == "online") 
SpokenInformal<-pv_data %>% filter(Register == "SpokenInformal") 
SpokenFormal<-pv_data %>% filter(Register == "SpokenFormal") 
WrittenInformal<-pv_data %>% filter(Register == "WrittenInformal") 
WrittenFormal<-pv_data %>% filter(Register == "WrittenFormal")

#####online
set.seed(116)
crf_online <- cforest(f1, 
                      data = online,
                      control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.online = varimp(crf_online, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_online))[c(FALSE, TRUE)]
library(Hmisc)
somerssmallcrf <- somers2(prob2.rf, as.numeric(online$Response) - 1)
somerssmallcrf["C"]

#####SpokenInformal
set.seed(117)
crf_SpokenInformal <- cforest(f1, 
                              data = SpokenInformal,
                              control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.SpokenInformal = varimp(crf_SpokenInformal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_SpokenInformal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(SpokenInformal$Response) - 1)
somerssmallcrf["C"]

#####SpokenFormal
set.seed(118)
crf_SpokenFormal <- cforest(f1, 
                            data = SpokenFormal,
                            control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.SpokenFormal = varimp(crf_SpokenFormal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_SpokenFormal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(SpokenFormal$Response) - 1)
somerssmallcrf["C"]

#####WrittenInformal
set.seed(119)
crf_WrittenInformal <- cforest(f1, 
                               data = WrittenInformal,
                               control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.WrittenInformal = varimp(crf_WrittenInformal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_WrittenInformal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(WrittenInformal$Response) - 1)
somerssmallcrf["C"]

#####WrittenFormal
set.seed(120)
crf_WrittenFormal <- cforest(f1, 
                             data = WrittenFormal,
                             control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.WrittenFormal = varimp(crf_WrittenFormal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_WrittenFormal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(WrittenFormal$Response) - 1)
somerssmallcrf["C"]

###step6: the third line of evidence
pv_varimp_line <- vadis_line3(rf_list, path = FALSE, conditional = FALSE)

pv_varimp_line$varimp.table %>% 
  round(3)
pv_varimp_line$rank.table
pv_varimp_line$distance.matrix %>% 
  round(3)
pv_varimp_line$similarity.scores %>% 
  arrange(desc(Similarity))

##step7: fusing three lines of evidence
pv_mean_sims <- data.frame(
  pv_line1 = pv_signif_line$similarity.scores[,1], # get only the values in the 2nd column
  pv_line2 = pv_coef_line$similarity.scores[,1],
  pv_line3 = pv_varimp_line$similarity.scores[,1],
  row.names = names(pv_data_list)
)
pv_mean_sims$mean <- rowMeans(pv_mean_sims)
round(pv_mean_sims, 3)
mean(pv_mean_sims$mean)

pv_fused_dist <- analogue::fuse(pv_signif_line$distance.matrix, 
                                pv_coef_line$distance.matrix, 
                                pv_varimp_line$distance.matrix)
round(pv_fused_dist, 3)

####visualization
library(plotly)
library(magrittr)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(analogue)
library(phangorn)
library(vegan)

input<-pv_fused_dist
fit <- cmdscale(input, eig = TRUE, k = 2)
fit.df <- as.data.frame(fit[[1]])
names(fit.df) <- c("x","y")
fit.df$Register <- rownames(fit.df) %>% as.factor
theme_mr = theme_set(theme_light())
theme_mr = theme_update(axis.text = element_text(size = rel(1.), color="black"),
                        axis.ticks = element_line(colour = "grey90", linewidth = 0.25),
                        axis.title = element_text(size=rel(0.9)),
                        panel.border = element_rect(color = "black"),
                        strip.background=element_rect(fill="grey95", color="black"), 
                        strip.text.x=element_text(color="black"))

dev.new(width=2, height=2)
ggplot(fit.df, aes(x,y)) +
  geom_text_repel(aes(label=Register), size=7, box.padding = unit(0.5, "lines"), segment.colour = "grey65") + # change text size here
  geom_point(size=5) + # change dot size here
  labs(y= "MDS Dimension 2", x="MDS Dimension 1") +
  theme(axis.title = element_text(size=15))

#genitive
##step0: data reading & spliting
gen_data <- read.csv("Genitives.csv")

gen_data$REGISTER <- car::recode(gen_data$REGISTER, " ## adapt as necessary
                             'online'  = 'online'; 
                             'SI'  = 'SpokenInformal'; 
                             'SF' = 'SpokenFormal'; 
                             'WI'  = 'WrittenInformal';
                             'WF' = 'WrittenFormal'
                             ")
summary(gen_data)

gen_data_list <- split(gen_data, gen_data$REGISTER, drop = TRUE) # drop unused levels
names(gen_data_list)

##step 1
f1 <- RESPONSE ~ 
  POR_ANIMACY_2 +
  POR_LENGTH_WORDS +
  PUM_LENGTH_WORDS +
  POR_NP_EXPR_TYPE_3 +
  POR_FINAL_SIBILANCY +
  PREVIOUS_CHOICE +
  SEM_REL_2 +
  POR_HEAD_FREQ

gen_data$VARIETY=as.factor(gen_data$VARIETY)
gen_data$POR_ANIMACY_2=as.factor(gen_data$POR_ANIMACY_2)
gen_data$RESPONSE=as.factor(gen_data$RESPONSE)
gen_data$POR_LENGTH_WORDS=as.numeric(gen_data$POR_LENGTH_WORDS)
gen_data$PUM_LENGTH_WORDS=as.numeric(gen_data$PUM_LENGTH_WORDS)
gen_data$POR_NP_EXPR_TYPE_3=as.factor(gen_data$POR_NP_EXPR_TYPE_3)
gen_data$POR_FINAL_SIBILANCY=as.factor(gen_data$POR_FINAL_SIBILANCY)
gen_data$PREVIOUS_CHOICE=as.factor(gen_data$PREVIOUS_CHOICE)
gen_data$SEM_REL_2=as.factor(gen_data$SEM_REL_2)
gen_data$POR_HEAD_FREQ=as.numeric(gen_data$POR_HEAD_FREQ)

##step 2
f2 <- update(f1, .~. + (1|VARIETY))
f2

gen_glmer_list <- vector("list")
for (i in seq_along(gen_data_list)){
  gen_d <- gen_data_list[[i]]
  gen_d$RESPONSE=as.factor(gen_d$RESPONSE)
  gen_d_std <- stand(gen_d, cols = f2) 
  gen_glmer_list[[i]] <- glmer(f2, data = gen_d_std, family = binomial, 
                               control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e7)))
  rm(gen_d, gen_d_std)
}

names(gen_glmer_list) <- names(gen_data_list)

###checking fits

summary_stats(gen_glmer_list, gen_data_list) %>% 
  round(3)

##step 3
gen_signif_line <- vadis_line1(gen_glmer_list, path = FALSE)
gen_signif_line$signif.table
gen_signif_line$distance.matrix %>% 
  round(3)
gen_signif_line$similarity.scores %>% 
  arrange(desc(Similarity))
write.csv(gen_signif_line$signif.table, 
          file = "gen_line1_significance_table.csv")
write.csv(as.matrix(gen_signif_line$distance.matrix), 
          file = "gen_line1_distance_matrix.csv")
write.csv(gen_signif_line$similarity.scores, 
          file = "gen_line1_similarity_scores.csv")

##step 4
gen_coef_line <- vadis_line2(gen_glmer_list, path = FALSE)
gen_coef_line$coef.table %>% 
  round(3)
gen_coef_line$distance.matrix %>% 
  round(3)
gen_coef_line$similarity.scores %>% 
  arrange(desc(Similarity))

## step 5
library(tuneRanger)
library(mlr)

tune_df <- data.frame(matrix(NA, ncol =5, nrow = 5))
names(tune_df) <- c("mtry", "min.node.size", "sample.fraction", "auc", "exec.time")
f1

for (i in seq_along(gen_data_list)){
  gen_d <- gen_data_list[[i]][, all.vars(f1)]
  gen_d$POR_ANIMACY_2=as.factor(gen_d$POR_ANIMACY_2)
  gen_d$RESPONSE=as.factor(gen_d$RESPONSE)
  gen_d$POR_LENGTH_WORDS=as.numeric(gen_d$POR_LENGTH_WORDS)
  gen_d$PUM_LENGTH_WORDS=as.numeric(gen_d$PUM_LENGTH_WORDS)
  gen_d$POR_NP_EXPR_TYPE_3=as.factor(gen_d$POR_NP_EXPR_TYPE_3)
  gen_d$POR_FINAL_SIBILANCY=as.factor(gen_d$POR_FINAL_SIBILANCY)
  gen_d$PREVIOUS_CHOICE=as.factor(gen_d$PREVIOUS_CHOICE)
  gen_d$SEM_REL_2=as.factor(gen_d$SEM_REL_2)
  gen_d$POR_HEAD_FREQ=as.numeric(gen_d$POR_HEAD_FREQ)
  gen_task <- makeClassifTask(data = gen_d, target = "RESPONSE")
  
  # Tuning process (takes around 1 minute); Tuning measure is the Area Under the Curve
  result <- tuneRanger(gen_task, measure = list(auc), num.trees = 1000, 
                       num.threads = 4, iters = 80, show.info = F)
  
  tune_df[i,] <- result$recommended.pars
}
rownames(tune_df) <- names(gen_data_list)

write.csv(tune_df, 
          file = "gen_crf_tune_df.csv")

###using cforest() function
library(party)
rf_list <- vector("list")
set.seed(123)
for (i in seq_along(gen_data_list)){
  gen_d <- gen_data_list[[i]]
  gen_d$RESPONSE=as.factor(gen_d$RESPONSE)
  gen_d$POR_ANIMACY_2=as.factor(gen_d$POR_ANIMACY_2)
  gen_d$POR_LENGTH_WORDS=as.numeric(gen_d$POR_LENGTH_WORDS)
  gen_d$PUM_LENGTH_WORDS=as.numeric(gen_d$PUM_LENGTH_WORDS)
  gen_d$POR_NP_EXPR_TYPE_3=as.factor(gen_d$POR_NP_EXPR_TYPE_3)
  gen_d$POR_FINAL_SIBILANCY=as.factor(gen_d$POR_FINAL_SIBILANCY)
  gen_d$PREVIOUS_CHOICE=as.factor(gen_d$PREVIOUS_CHOICE)
  gen_d$SEM_REL_2=as.factor(gen_d$SEM_REL_2)
  gen_d$POR_HEAD_FREQ=as.numeric(gen_d$POR_HEAD_FREQ)
  rf_list[[i]] <- cforest(
    f1, 
    data = gen_d,
    controls = cforest_unbiased(ntree = 1000, mtry = 3)
  )
}

names(rf_list) <- names(gen_data_list)

###check fitting 
#### split subset data by Variety
library(tidyverse)
online<-gen_data %>% filter(REGISTER == "online") 
SpokenInformal<-gen_data %>% filter(REGISTER == "SpokenInformal") 
SpokenFormal<-gen_data %>% filter(REGISTER == "SpokenFormal") 
WrittenInformal<-gen_data %>% filter(REGISTER == "WrittenInformal") 
WrittenFormal<-gen_data %>% filter(REGISTER == "WrittenFormal")

f1 <- RESPONSE ~ 
  POR_ANIMACY_2 +
  POR_LENGTH_WORDS +
  PUM_LENGTH_WORDS +
  POR_NP_EXPR_TYPE_3 +
  POR_FINAL_SIBILANCY +
  PREVIOUS_CHOICE +
  SEM_REL_2 +
  POR_HEAD_FREQ

####online
set.seed(116)
crf_online <- cforest(f1, 
                      data = online,
                      control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.online = varimp(crf_online, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_online))[c(FALSE, TRUE)]
library(Hmisc)
somerssmallcrf <- somers2(prob2.rf, as.numeric(online$RESPONSE) - 1)
somerssmallcrf["C"]

####SpokenInformal
set.seed(117)
crf_SpokenInformal <- cforest(f1, 
                              data = SpokenInformal,
                              control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.SpokenInformal = varimp(crf_SpokenInformal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_SpokenInformal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(SpokenInformal$RESPONSE) - 1)
somerssmallcrf["C"]

####SpokenFormal
set.seed(118)
crf_SpokenFormal <- cforest(f1, 
                            data = SpokenFormal,
                            control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.SpokenFormal = varimp(crf_SpokenFormal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_SpokenFormal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(SpokenFormal$RESPONSE) - 1)
somerssmallcrf["C"]

####WrittenInformal
set.seed(119)
crf_WrittenInformal <- cforest(f1, 
                               data = WrittenInformal,
                               control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.WrittenInformal = varimp(crf_WrittenInformal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_WrittenInformal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(WrittenInformal$RESPONSE) - 1)
somerssmallcrf["C"]

####WrittenFormal
set.seed(120)
crf_WrittenFormal <- cforest(f1, 
                             data = WrittenFormal,
                             control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.WrittenFormal = varimp(crf_WrittenFormal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_WrittenFormal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(WrittenFormal$RESPONSE) - 1)
somerssmallcrf["C"]

##step 6
gen_varimp_line <- vadis_line3(rf_list, path = FALSE, conditional = FALSE)

gen_varimp_line$varimp.table %>% 
  round(3)
gen_varimp_line$rank.table
gen_varimp_line$distance.matrix %>% 
  round(3)
gen_varimp_line$similarity.scores %>% 
  arrange(desc(Similarity))

##step7
gen_mean_sims <- data.frame(
  gen_line1 = gen_signif_line$similarity.scores[,1], # get only the values in the 2nd column
  gen_line2 = gen_coef_line$similarity.scores[,1],
  gen_line3 = gen_varimp_line$similarity.scores[,1],
  row.names = names(gen_data_list)
)
gen_mean_sims$mean <- rowMeans(gen_mean_sims)
round(gen_mean_sims, 3)
mean(gen_mean_sims$mean)

gen_fused_dist <- analogue::fuse(gen_signif_line$distance.matrix, 
                                 gen_coef_line$distance.matrix, 
                                 gen_varimp_line$distance.matrix)
round(gen_fused_dist, 3)

###visualization
input<-gen_fused_dist
fit <- cmdscale(input, eig = TRUE, k = 2)
fit.df <- as.data.frame(fit[[1]])
names(fit.df) <- c("x","y")
fit.df$Register <- rownames(fit.df) %>% as.factor
theme_mr = theme_set(theme_light())
theme_mr = theme_update(axis.text = element_text(size = rel(1.), color="black"),
                        axis.ticks = element_line(colour = "grey90", linewidth = 0.25),
                        axis.title = element_text(size=rel(0.9)),
                        panel.border = element_rect(color = "black"),
                        strip.background=element_rect(fill="grey95", color="black"), 
                        strip.text.x=element_text(color="black"))

dev.new(width=2, height=2)
ggplot(fit.df, aes(x,y)) +
  geom_text_repel(aes(label=Register), size=7, box.padding = unit(0.5, "lines"), segment.colour = "grey65") + # change text size here
  geom_point(size=5) + # change dot size here
  labs(y= "MDS Dimension 2", x="MDS Dimension 1") +
  theme(axis.title = element_text(size=15))

#dative 
##step 0
dat_data <- read.csv("Datives.csv")

dat_data$Register <- car::recode(dat_data$Register, " ## adapt as necessary
                             'online'  = 'online'; 
                             'SI'  = 'SpokenInformal'; 
                             'SF' = 'SpokenFormal'; 
                             'WI'  = 'WrittenInformal';
                             'WF' = 'WrittenFormal'
                             ")
summary(dat_data)

#### split dataset according to register
dat_data_list <- split(dat_data, dat_data$Register, drop = TRUE) # drop unused levels
names(dat_data_list)

##step 1
#identifying the most important factors 
f1 <- Resp ~ 
  logWeightRatio +
  RecPron +
  ThemeBinComplexity +
  ThemeHeadFreq +
  ThemePron +
  ThemeDefiniteness +
  RecGivenness +
  RecHeadFreq

dat_data$Variety=as.factor(dat_data$Variety)
dat_data$logWeightRatio=as.numeric(dat_data$logWeightRatio)
dat_data$Resp=as.factor(dat_data$Resp)
dat_data$RecPron=as.factor(dat_data$RecPron)
dat_data$ThemeBinComplexity=as.factor(dat_data$ThemeBinComplexity)
dat_data$ThemeHeadFreq=as.numeric(dat_data$ThemeHeadFreq)
dat_data$ThemePron=as.factor(dat_data$ThemePron)
dat_data$ThemeDefiniteness=as.factor(dat_data$ThemeDefiniteness)
dat_data$RecGivenness=as.factor(dat_data$RecGivenness)
dat_data$RecHeadFreq=as.numeric(dat_data$RecHeadFreq)

##step 2
f2 <- update(f1, .~. + (1|Variety) + (1|Verb))
f2

dat_glmer_list <- vector("list")
for (i in seq_along(dat_data_list)){
  dat_d <- dat_data_list[[i]]
  dat_d$Resp=as.factor(dat_d$Resp)
  # standardize the model inputs, excluding the response and random effects
  dat_d_std <- stand(dat_d, cols = f2) 
  # fit the model
  dat_glmer_list[[i]] <- glmer(f2, data = dat_d_std, family = binomial, 
                               # set optimizer controls to help convergence 
                               control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e7)))
  rm(dat_d, dat_d_std) # remove datasets
}
#the optimization setting might cause warnings 

names(dat_glmer_list) <- names(dat_data_list)

####checking fits

summary_stats(dat_glmer_list, dat_data_list) %>% 
  round(3)

##step 3
dat_signif_line <- vadis_line1(dat_glmer_list, path = FALSE)
dat_signif_line$signif.table
dat_signif_line$distance.matrix %>% 
  round(3)
dat_signif_line$similarity.scores %>% 
  arrange(desc(Similarity))
write.csv(dat_signif_line$signif.table, 
          file = "dat_line1_significance_table.csv")
write.csv(as.matrix(dat_signif_line$distance.matrix), 
          file = "dat_line1_distance_matrix.csv")
write.csv(dat_signif_line$similarity.scores, 
          file = "dat_line1_similarity_scores.csv")

## step4
dat_coef_line <- vadis_line2(dat_glmer_list, path = FALSE)
dat_coef_line$coef.table %>% 
  round(3)
dat_coef_line$distance.matrix %>% 
  round(3)
dat_coef_line$similarity.scores %>% 
  arrange(desc(Similarity))

##step 5
library(tuneRanger)
library(mlr)

tune_df <- data.frame(matrix(NA, ncol =5, nrow = 5))
names(tune_df) <- c("mtry", "min.node.size", "sample.fraction", "auc", "exec.time")
f1

for (i in seq_along(dat_data_list)){
  dat_d <- dat_data_list[[i]][, all.vars(f1)]
  dat_d$logWeightRatio=as.numeric(dat_d$logWeightRatio)
  dat_d$Resp=as.factor(dat_d$Resp)
  dat_d$RecPron=as.factor(dat_d$RecPron)
  dat_d$ThemeBinComplexity=as.factor(dat_d$ThemeBinComplexity)
  dat_d$ThemeHeadFreq=as.numeric(dat_d$ThemeHeadFreq)
  dat_d$ThemePron=as.factor(dat_d$ThemePron)
  dat_d$ThemeDefiniteness=as.factor(dat_d$ThemeDefiniteness)
  dat_d$RecGivenness=as.factor(dat_d$RecGivenness)
  dat_d$RecHeadFreq=as.numeric(dat_d$RecHeadFreq)
  dat_task <- makeClassifTask(data = dat_d, target = "Resp")
  result <- tuneRanger(dat_task, measure = list(auc), num.trees = 1000, 
                       num.threads = 4, iters = 80, show.info = F)
  
  tune_df[i,] <- result$recommended.pars
}
rownames(tune_df) <- names(dat_data_list)

write.csv(tune_df, 
          file = "dat_crf_tune_df.csv")

### fitting CRF models to per register and alternation
###using cforest() function
library(party)
rf_list <- vector("list")
set.seed(123)
for (i in seq_along(dat_data_list)){
  dat_d <- dat_data_list[[i]]
  dat_d$logWeightRatio=as.numeric(dat_d$logWeightRatio)
  dat_d$Resp=as.factor(dat_d$Resp)
  dat_d$RecPron=as.factor(dat_d$RecPron)
  dat_d$ThemeBinComplexity=as.factor(dat_d$ThemeBinComplexity)
  dat_d$ThemeHeadFreq=as.numeric(dat_d$ThemeHeadFreq)
  dat_d$ThemePron=as.factor(dat_d$ThemePron)
  dat_d$ThemeDefiniteness=as.factor(dat_d$ThemeDefiniteness)
  dat_d$RecGivenness=as.factor(dat_d$RecGivenness)
  dat_d$RecHeadFreq=as.numeric(dat_d$RecHeadFreq)
  # fit the random forest and add it to the list
  rf_list[[i]] <- cforest(
    f1, 
    data = dat_d,
    controls = cforest_unbiased(ntree = 1000, mtry = 3)
  )
}

names(rf_list) <- names(dat_data_list)

###check fitting 
##### split subset data by Variety
library(tidyverse)
online<-dat_data %>% filter(Register == "online") 
SpokenInformal<-dat_data %>% filter(Register == "SpokenInformal") 
SpokenFormal<-dat_data %>% filter(Register == "SpokenFormal") 
WrittenInformal<-dat_data %>% filter(Register == "WrittenInformal") 
WrittenFormal<-dat_data %>% filter(Register == "WrittenFormal")

####online
set.seed(116)
crf_online <- cforest(f1, 
                      data = online,
                      control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.online = varimp(crf_online, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_online))[c(FALSE, TRUE)]
library(Hmisc)
somerssmallcrf <- somers2(prob2.rf, as.numeric(online$Resp) - 1)
somerssmallcrf["C"]

####SpokenInformal
set.seed(117)
crf_SpokenInformal <- cforest(f1, 
                              data = SpokenInformal,
                              control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.SpokenInformal = varimp(crf_SpokenInformal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_SpokenInformal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(SpokenInformal$Resp) - 1)
somerssmallcrf["C"]

####SpokenFormal
set.seed(118)
crf_SpokenFormal <- cforest(f1, 
                            data = SpokenFormal,
                            control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.SpokenFormal = varimp(crf_SpokenFormal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_SpokenFormal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(SpokenFormal$Resp) - 1)
somerssmallcrf["C"]

####WrittenInformal
set.seed(119)
crf_WrittenInformal <- cforest(f1, 
                               data = WrittenInformal,
                               control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.WrittenInformal = varimp(crf_WrittenInformal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_WrittenInformal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(WrittenInformal$Resp) - 1)
somerssmallcrf["C"]

####WrittenFormal
set.seed(120)
crf_WrittenFormal <- cforest(f1, 
                             data = WrittenFormal,
                             control = cforest_unbiased (ntree=1000, mtry=3))
forest.varimp.WrittenFormal = varimp(crf_WrittenFormal, conditional = FALSE) 
prob2.rf <- unlist(treeresponse(crf_WrittenFormal))[c(FALSE, TRUE)]
somerssmallcrf <- somers2(prob2.rf, as.numeric(WrittenFormal$Resp) - 1)
somerssmallcrf["C"]

##step6
dat_varimp_line <- vadis_line3(rf_list, path = FALSE, conditional = FALSE)

dat_varimp_line$varimp.table %>% 
  round(3)
dat_varimp_line$rank.table
dat_varimp_line$distance.matrix %>% 
  round(3)
dat_varimp_line$similarity.scores %>% 
  arrange(desc(Similarity))

##step7
#combining the 3 lines
dat_mean_sims <- data.frame(
  dat_line1 = dat_signif_line$similarity.scores[,1], # get only the values in the 2nd column
  dat_line2 = dat_coef_line$similarity.scores[,1],
  dat_line3 = dat_varimp_line$similarity.scores[,1],
  row.names = names(dat_data_list)
)
dat_mean_sims$mean <- rowMeans(dat_mean_sims)
round(dat_mean_sims, 3)
mean(dat_mean_sims$mean)

dat_fused_dist <- analogue::fuse(dat_signif_line$distance.matrix, 
                                 dat_coef_line$distance.matrix, 
                                 dat_varimp_line$distance.matrix)
round(dat_fused_dist, 3)

###visualization
library(ggplot2)
input<-dat_fused_dist
fit <- cmdscale(input, eig = TRUE, k = 2)
fit.df <- as.data.frame(fit[[1]])
names(fit.df) <- c("x","y")
fit.df$Register <- rownames(fit.df) %>% as.factor
theme_mr = theme_set(theme_light())
theme_mr = theme_update(axis.text = element_text(size = rel(1.), color="black"),
                        axis.ticks = element_line(colour = "grey90", linewidth = 0.25),
                        axis.title = element_text(size=rel(0.9)),
                        panel.border = element_rect(color = "black"),
                        strip.background=element_rect(fill="grey95", color="black"), 
                        strip.text.x=element_text(color="black"))

dev.new(width=2, height=2)
ggplot(fit.df, aes(x,y)) +
  geom_text_repel(aes(label=Register), size=7, box.padding = unit(0.5, "lines"), segment.colour = "grey65") + # change text size here
  geom_point(size=5) + # change dot size here
  labs(y= "MDS Dimension 2", x="MDS Dimension 1") +
  theme(axis.title = element_text(size=15))


#similarity coefficients
##pv similarity socres across three lines 
pv_line1_mean<-mean(pv_signif_line$similarity.scores$Similarity)
print(pv_line1_mean)
pv_line2_mean<-mean(pv_coef_line$similarity.scores$Similarity)
print(pv_line2_mean)
pv_line3_mean<-mean(pv_varimp_line$similarity.scores$Similarity)
print(pv_line3_mean)

gen_line1_mean<-mean(gen_signif_line$similarity.scores$Similarity)
print(gen_line1_mean)
gen_line2_mean<-mean(gen_coef_line$similarity.scores$Similarity)
print(gen_line2_mean)
gen_line3_mean<-mean(gen_varimp_line$similarity.scores$Similarity)
print(gen_line3_mean)

dat_line1_mean<-mean(dat_signif_line$similarity.scores$Similarity)
print(dat_line1_mean)
dat_line2_mean<-mean(dat_coef_line$similarity.scores$Similarity)
print(dat_line2_mean)
dat_line3_mean<-mean(dat_varimp_line$similarity.scores$Similarity)
print(dat_line3_mean)

##similarity coefficients acorss lines of evidence and alternations 
pv_mean_list <- c(pv_line1_mean, pv_line2_mean, pv_line3_mean)
gen_mean_list <- c(gen_line1_mean, gen_line2_mean, gen_line3_mean)
dat_mean_list <- c(dat_line1_mean, dat_line2_mean, dat_line3_mean)

mean_sims <- data.frame(
  PV = c(pv_mean_list, mean(pv_mean_list)),
  Genitive = c(gen_mean_list, mean(gen_mean_list)),
  Dative = c(dat_mean_list, mean(dat_mean_list)),
  row.names = c("line1", "line2", "line3", "mean")
)

##Print the mean similarity data frame
print(mean_sims)

##calculating Core Grammar 
pv_mean <- mean_sims["mean", "PV"]
gen_mean <- mean_sims["mean", "Genitive"]
dat_mean <- mean_sims["mean", "Dative"]
core_grammar <- mean(c(pv_mean, gen_mean, dat_mean))
print(core_grammar)

#mapping out distances between registers
##fusing fused distance matricces across alternations
dfused_total_all<-analogue::fuse(pv_fused_dist,
                                 gen_fused_dist,
                                 dat_fused_dist)
round(dfused_total_all, 3)

##visualizaiton of the fused lines of evidence based on all data
library(ggplot2)
input<-dfused_total_all
fit <- cmdscale(input, eig = TRUE, k = 2)
fit.df <- as.data.frame(fit[[1]])
names(fit.df) <- c("x","y")
fit.df$Register <- rownames(fit.df) %>% as.factor
theme_mr = theme_set(theme_light())
theme_mr = theme_update(axis.text = element_text(size = rel(1.), color="black"),
                        axis.ticks = element_line(colour = "grey90", linewidth = 0.25),
                        axis.title = element_text(size=rel(0.9)),
                        panel.border = element_rect(color = "black"),
                        strip.background=element_rect(fill="grey95", color="black"), 
                        strip.text.x=element_text(color="black"))

dev.new(width=2, height=2)
ggplot(fit.df, aes(x,y)) +
  geom_text_repel(aes(label=Register), size=7, box.padding = unit(0.5, "lines"), segment.colour = "grey65") + # change text size here
  geom_point(size=5) + # change dot size here
  labs(y= "MDS Dimension 2", x="MDS Dimension 1") +
  theme(axis.title = element_text(size=15))


