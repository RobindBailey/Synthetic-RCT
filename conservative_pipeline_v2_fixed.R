
# Conservative Synthetic Generation Pipeline with Jittered Residuals

library(tidyverse)
library(caret)
library(randomForest)
library(lme4)

set.seed(42)

sample_sizes <- c(20, 50, 100, 200, 500)
numDatasets <- 10
models <- c("BoostedTree", "RandomForest", "GAN")
variables <- c("WI","BAI","BDI","MCQHA","HCQ")
real_coef <- c(-26.2, -16.0, -7.3, -15.2, -17.6)
real_SE <- c(2.8, 2.9, 2.6, 2.6, 2.3)
real_g <- c(3.18, 2.50, 1.28, 2.67, 1.81)
real_CI_low <- real_coef - 1.96 * real_SE
real_CI_high <- real_coef + 1.96 * real_SE
means_MCT_post <- c(24.5,5.2,3.6,17.0,55.4)
sd_MCT_post <- c(7.3,6.5,4.0,4.3,9.8)
means_WL_post <- c(50.7,21.2,10.9,32.2,73.0)
sd_WL_post <- c(8.3,5.6,6.5,6.3,9.0)

results_list <- list()

generate_gan_samples <- function(dataTbl, n) {
  group_labels <- dataTbl$Group
  data_post <- dataTbl$Post
  mean_MCT <- mean(data_post[group_labels == 1])
  std_MCT <- sd(data_post[group_labels == 1])
  mean_WL <- mean(data_post[group_labels == 0])
  std_WL <- sd(data_post[group_labels == 0])
  MCT_fake <- rnorm(n, mean_MCT, std_MCT)
  WL_fake <- rnorm(n, mean_WL, std_WL)
  return(c(MCT_fake, WL_fake))
}

row_counter <- 1

for (n_per_group in sample_sizes) {
  for (model_type in models) {
    for (v in seq_along(variables)) {

      Group <- c(rep(1, n_per_group), rep(0, n_per_group))
      Post <- c(rnorm(n_per_group, means_MCT_post[v], sd_MCT_post[v]),
                rnorm(n_per_group, means_WL_post[v], sd_WL_post[v]))
      dataTbl <- data.frame(Group = Group, Post = Post)

      coef_syn <- numeric(numDatasets)
      p_syn <- numeric(numDatasets)
      CI_syn <- matrix(0, nrow = numDatasets, ncol = 2)
      g_syn <- numeric(numDatasets)

      for (ds in 1:numDatasets) {
        if (model_type == "BoostedTree" && n_per_group < 50) next  # Skip too-small BoostedTree

        if (model_type == "BoostedTree") {
          control <- trainControl(method="cv", number=5)
          model <- train(Post ~ Group,
                         data = dataTbl,
                         method = "gbm",
                         trControl = control,
                         verbose = FALSE,
                         tuneGrid = expand.grid(
                           n.trees = 10,
                           interaction.depth = 1,
                           shrinkage = 0.1,
                           n.minobsinnode = 2
                         ),
                         preProcess = c("center", "scale"))
          preds <- predict(model, dataTbl)
        } else if (model_type == "RandomForest") {
          control <- trainControl(method="cv", number=5)
          model <- train(Post ~ Group, data=dataTbl, method="rf",
                         trControl=control,
                         tuneLength=1, preProcess=c("center", "scale"))
          preds <- predict(model, dataTbl)
        } else if (model_type == "GAN") {
          preds <- generate_gan_samples(dataTbl, n_per_group)
        }

        residuals <- dataTbl$Post - preds
        jitter <- rnorm(length(residuals), 0, sd(residuals) * 0.25)
        synthetic_Post <- preds + 1.1 * sample(residuals) + jitter

        mdl_syn_ds <- lm(synthetic_Post ~ Group, data = dataTbl)
        coef_syn[ds] <- coef(mdl_syn_ds)["Group"]
        p_syn[ds] <- summary(mdl_syn_ds)$coefficients["Group", "Pr(>|t|)"]
        conf_int <- confint(mdl_syn_ds, level=0.95)["Group",]
        CI_syn[ds,] <- conf_int

        diff_mean <- mean(synthetic_Post[Group==0]) - mean(synthetic_Post[Group==1])
        pooled_sd <- sqrt(((n_per_group-1)*var(synthetic_Post[Group==1]) +
                          (n_per_group-1)*var(synthetic_Post[Group==0])) / (2*n_per_group - 2))
        d <- diff_mean / pooled_sd
        correction <- 1 - (3 / (4*(2*n_per_group - 2) - 1))
        g_syn[ds] <- d * correction
      }

      if (length(coef_syn[!is.na(coef_syn)]) == 0) next  # Skip if model completely failed

      mean_coef_syn <- mean(coef_syn, na.rm=TRUE)
      pooled_SE <- sd(coef_syn, na.rm=TRUE)
      Z_synthetic <- (mean_coef_syn - real_coef[v]) / pooled_SE
      Z_real <- (mean_coef_syn - real_coef[v]) / real_SE[v]
      synth_CI_low <- mean(CI_syn[,1], na.rm=TRUE)
      synth_CI_high <- mean(CI_syn[,2], na.rm=TRUE)

      overlap_low <- max(real_CI_low[v], synth_CI_low)
      overlap_high <- min(real_CI_high[v], synth_CI_high)
      overlap_amount <- max(0, overlap_high - overlap_low)
      real_CI_range <- real_CI_high[v] - real_CI_low[v]
      CI_overlap_pct <- (overlap_amount / real_CI_range) * 100

      direction_agreement <- sign(real_coef[v]) == sign(mean_coef_syn)
      real_sig <- real_CI_low[v] > 0 || real_CI_high[v] < 0
      synth_sig <- mean(p_syn, na.rm=TRUE) < 0.05
      full_decision_agreement <- direction_agreement && (real_sig == synth_sig)
      estimate_distance <- max(0, max(real_CI_low[v] - mean_coef_syn, mean_coef_syn - real_CI_high[v]))

      results_list[[row_counter]] <- data.frame(
        SampleSize = n_per_group,
        Model = model_type,
        Variable = variables[v],
        RealCoef = real_coef[v],
        SynthMeanCoef = mean_coef_syn,
        DirectionAgree = direction_agreement,
        RealSig = real_sig,
        SynthSig = synth_sig,
        FullDecisionAgree = full_decision_agreement,
        EstimateDistFromCI = estimate_distance,
        Z_Synthetic = Z_synthetic,
        Z_Synth_Accept = abs(Z_synthetic) < 1.96,
        Z_Real = Z_real,
        Z_Real_Accept = abs(Z_real) < 1.96,
        CI_OverlapPct = CI_overlap_pct,
        Real_g = real_g[v],
        SynthMean_g = mean(g_syn, na.rm=TRUE),
        g_Diff = abs(real_g[v] - mean(g_syn, na.rm=TRUE))
      )
      row_counter <- row_counter + 1
    }
  }
}

results_table <- do.call(rbind, results_list)
write.csv(results_table, "conservative_replicability_results_jittered.csv", row.names = FALSE)

lme <- lmer(g_Diff ~ SampleSize + Model + (1|Variable), data = results_table)
summary(lme)
anova(lme)
