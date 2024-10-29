if (!require(ggplot2)) install.packages("ggplot2")
if (!require(lme4)) install.packages("lme4")
if (!require(MuMIn)) install.packages("MuMIn")
if (!require(dplyr)) install.packages("dplyr")
if (!require(lmerTest)) install.packages("car")
if (!require(lmerTest)) install.packages("lmerTest")

library(ggplot2)
library(lme4)
library(MuMIn)
library(dplyr)
library(car)
library(lmerTest)


args <- commandArgs(trailingOnly = TRUE)
aoa_and_predictors_path <- args[1]
regression_data_path <- args[2]
result_data_path <- args[5]

aoa_and_predictors <- read.csv(aoa_and_predictors_path)

# scale the child_aoa with min max scaling
aoa_and_predictors$child_aoa <- (aoa_and_predictors$child_aoa - 16) / (30 - 16)

aoa_and_predictors <- aoa_and_predictors %>%
  filter(lexical_class != "other", )

cat("\nLength of the data:", nrow(aoa_and_predictors), "\n")


thresholds <- c(0.07)
tasks <- unique(aoa_and_predictors$task)
datasets <- unique(aoa_and_predictors$dataset)

num_unique_seeds <- length(unique(aoa_and_predictors$seed))
table_data <- data.frame(model = character(), dataset = character(), task = character(), threshold = numeric(), num_words = numeric(), m1_adj_r2 = numeric(), reduced_adj_r2 = numeric(), singular = logical(), stringsAsFactors = FALSE)
result_data <- data.frame(model = character(), dataset = character(), task = character(), threshold = numeric(), adj_r2=numeric(), reduced_adj_r2 = numeric(), lrt = numeric(), singular = logical(), stringsAsFactors = FALSE)

for (dataset in datasets){
  for (task in tasks) {
    for (threshold in thresholds) {

      # Filter data for intrinsic task and current threshold
      lm_df <- aoa_and_predictors[
        aoa_and_predictors$task == task &
        aoa_and_predictors$threshold == threshold &
        aoa_and_predictors$model == "gpt2" &
        aoa_and_predictors$dataset == dataset,
        ]

      if  ( nrow(lm_df) == 0) {
        next
      }

      mean_aoa <- mean(lm_df$aoa_x)
      std_aoa <- sd(lm_df$aoa_x)

      lower_bound <- mean_aoa - 3 * std_aoa
      upper_bound <- mean_aoa + 3 * std_aoa

      lm_df <- lm_df %>%
        filter(aoa_x >= lower_bound & aoa_x <= upper_bound)

      lm_df <- lm_df %>%
        group_by(word) %>%
        filter(n_distinct(seed) == 3) %>%
        ungroup()
      
      lm_df <- lm_df %>%
        select(word, aoa_x, child_aoa, seed) %>%
        na.omit() %>%
        distinct()

      m <- lmer(child_aoa ~ aoa_x + (1 | seed), data = lm_df)
      m_reduced <- lm(child_aoa ~ aoa_x, data = lm_df)


      marginal_r2 <- r.squaredGLMM(m)[1]
      conditional_r2 <- r.squaredGLMM(m)[2]
 


      model_summary <- summary(m)
      num_words <- length(unique(lm_df$word))
      anova_results <- anova(m, m_reduced)

      m_reduced_adj_rsquared <- summary(m_reduced)$adj.r.squared
      table_data <- rbind(table_data, data.frame(model = "gpt2", dataset = dataset, task = task, threshold = threshold, num_words = num_words, m1_adj_r2 = marginal_r2, reduced_adj_r2=m_reduced_adj_rsquared, singular = isSingular(m), stringsAsFactors = FALSE))
      result_data <- rbind(result_data, data.frame(model = "gpt2", dataset = dataset, task = task, threshold = threshold, adj_r2 = marginal_r2, reduced_adj_r2=m_reduced_adj_rsquared, lrt = anova_results$`Pr(>Chisq)`[2], singular = isSingular(m), stringsAsFactors = FALSE))
    }
  }
}

write.csv(table_data, file = regression_data_path, row.names = FALSE)
write.csv(result_data, file = result_data_path, row.names = FALSE)