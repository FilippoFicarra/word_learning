if (!require(ggplot2)) install.packages("ggplot2")
if (!require(lme4)) install.packages("lme4")
if (!require(MuMIn)) install.packages("MuMIn")
if (!require(dplyr)) install.packages("dplyr")
if (!require(car)) install.packages("car")
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
acquired_words_path <- args[3]
converged_words_path <- args[4]
result_data_path <- args[5]

if (length(args) < 5) {
  cat("Please provide the path to the aoa_and_predictors file, regression_data file, acquired_words file, converged_words file and result_data file")
  quit()
}

# Load the data
aoa_and_predictors <- read.csv(aoa_and_predictors_path)
cat("\nLength of the data:", nrow(aoa_and_predictors), "\n")

aoa_and_predictors <- aoa_and_predictors %>%
  filter(lexical_class != "other", )

tasks <- unique(aoa_and_predictors$task)
datasets <- unique(aoa_and_predictors$dataset)



thresholds <- c(0.07)

table_data <- data.frame(model = character(), dataset = character(), task = character(), threshold = numeric(), num_words = numeric(), LF = numeric(), Co = numeric(), NC = numeric(), MLU = numeric(), LC = numeric(), Full = numeric(), reduced = numeric(), max_vif=numeric(), stringsAsFactors = FALSE)
acquired_word_data <- data.frame(model = character(), dataset = character(), task = character(), threshold = numeric(), word = character(), frequency = numeric(), aoa = numeric(), rank = numeric(), lexical_class = character(),  stringsAsFactors = FALSE)
result_data <- data.frame(model = character(), dataset = character(), task = character(), threshold = numeric(), adj_r2=numeric(), reduced_adj_r2 = numeric(), singular = logical(), seed_variance = numeric(),  bic = numeric(), aic = numeric(),bic_reduced = numeric(), aic_reduced = numeric(), stringsAsFactors = FALSE)
converged_words <- data.frame(model = character(), dataset = character(), task = character(), threshold = numeric(), num_words = numeric(), stringsAsFactors = FALSE)


for (dataset in datasets){
  for (task in tasks) {
    for (threshold in thresholds) {

      lm_df <- aoa_and_predictors[
        aoa_and_predictors$task == task &
        aoa_and_predictors$threshold == threshold &
        aoa_and_predictors$model == "gpt2" &
        aoa_and_predictors$dataset == dataset,
      ]
      num_unique_seeds <- length(unique(lm_df$seed))

      lm_df <- lm_df %>%
        select(word, aoa_x, mlu, concreteness, n_chars, log_frequency, frequency, lexical_class, threshold, seed) %>%
        na.omit() %>%
        distinct()

      d <- lm_df %>%
        select(word) %>%
        na.omit() %>%
        distinct()

      lm_df$lexical_class <- factor(lm_df$lexical_class)
      vif_values <- vif(lm(aoa_x ~ log_frequency + n_chars + concreteness + mlu + lexical_class, data = lm_df))


      mean_aoa <- mean(lm_df$aoa_x)
      std_aoa <- sd(lm_df$aoa_x)

      lower_bound <- mean_aoa - 3 * std_aoa
      upper_bound <- mean_aoa + 3 * std_aoa
      lm_df <- lm_df %>%
        filter(aoa_x >= lower_bound & aoa_x <= upper_bound)
      
      lm_df <- lm_df %>%
        group_by(word) %>%
        filter(n_distinct(seed) == num_unique_seeds) %>%
        ungroup()

      num_words <- length(unique(lm_df$word))
      converged_words <- rbind(converged_words, data.frame(
        model = "gpt2", 
        dataset = dataset, 
        task = task, 
        threshold = threshold, 
        num_words = num_words,
        stringsAsFactors = FALSE
      ))

      m1 <- lmer(aoa_x ~ log_frequency + (1 | seed), data = lm_df)
      m2 <- lmer(aoa_x ~ concreteness + (1 | seed), data = lm_df)
      m3 <- lmer(aoa_x ~ n_chars + (1 | seed), data = lm_df)
      m4 <- lmer(aoa_x ~ mlu + (1 | seed), data = lm_df)
      m5 <- lmer(aoa_x ~ lexical_class + (1 | seed), data = lm_df)
      m6 <- lmer(aoa_x ~ log_frequency + concreteness + n_chars + mlu + lexical_class + (1 | seed), data = lm_df)
      m7 <- lmer(aoa_x ~ concreteness + n_chars + mlu + lexical_class + (1 | seed), data = lm_df)


      # models reduced by removing random effects
      m1_reduced <- lm(aoa_x ~ log_frequency, data = lm_df)
      m2_reduced <- lm(aoa_x ~ concreteness, data = lm_df)
      m3_reduced <- lm(aoa_x ~ n_chars, data = lm_df)
      m4_reduced <- lm(aoa_x ~ mlu, data = lm_df)
      m5_reduced <- lm(aoa_x ~ lexical_class, data = lm_df)
      m6_reduced <- lm(aoa_x ~ log_frequency + concreteness + n_chars + mlu + lexical_class, data = lm_df)
      m7_reduced <- lm(aoa_x ~ concreteness + n_chars + mlu + lexical_class, data = lm_df)

      models <- list(m1, m2, m3, m4, m5, m6, m7)      
      models_reduced <- list(m1_reduced, m2_reduced, m3_reduced, m4_reduced, m5_reduced, m6_reduced, m7_reduced)
      num_words <- length(unique(lm_df$word))


      for (i in 1:length(models)) {
        model_summary <- summary(models[[i]])

        variance <- as.data.frame(VarCorr(models[[i]])) %>%
          filter(grp == "seed") %>%
          select(vcov)

        bic <- BIC(models[[i]])
        aic <- AIC(models[[i]])

        bic_reduced <- BIC(models_reduced[[i]])
        aic_reduced <- AIC(models_reduced[[i]])

        reduced_adj_r2 <- summary(models_reduced[[i]])$adj.r.squared


        result_data <- rbind(result_data, data.frame(
          model = paste("m", i, sep=""), 
          dataset = dataset, 
          task = task, 
          threshold = threshold, 
          adj_r2 = r.squaredGLMM(models[[i]])[1],
          reduced_adj_r2 = reduced_adj_r2,
          singular = isSingular(models[[i]]),
          seed_variance = variance$vcov,
          bic = bic,
          aic = aic,
          bic_reduced = bic_reduced,
          aic_reduced = aic_reduced,
          stringsAsFactors = FALSE
        ))

      }

      adj_r2_values <- numeric(length(models))
      for (i in 1:length(models)) {
        adj_r2_values[i] <- summary(models_reduced[[i]])$adj.r.squared
      }

      table_data <- rbind(table_data, data.frame(
        model = "gpt2", 
        dataset = dataset, 
        task = task, 
        threshold = threshold, 
        num_words = num_words,
        LF = adj_r2_values[1], 
        Co = adj_r2_values[2], 
        NC = adj_r2_values[3], 
        MLU = adj_r2_values[4], 
        LC = adj_r2_values[5], 
        Full = adj_r2_values[6], 
        reduced = adj_r2_values[7], 
        max_vif = max(vif_values),
        stringsAsFactors = FALSE
      ))

      lm_df$frequency <- format(lm_df$frequency, scientific = TRUE)

      top_words <- lm_df %>%
        arrange(aoa_x) %>%
        head(10) %>%
        select(word, frequency, aoa_x, lexical_class) %>%
        mutate(
          model = "gpt2",
          dataset = dataset,
          task = task,
          threshold = threshold, 
          rank = 1:10
        )
      
      last_words <- lm_df %>%
        arrange(desc(aoa_x)) %>%
        head(10) %>%
        select(word, frequency, aoa_x, lexical_class) %>%
        mutate(
          model = "gpt2",
          dataset = dataset,
          task = task,
          threshold = threshold,
          rank = -1:-10
        )

      acquired_word_data <- rbind(acquired_word_data, top_words, last_words)
    }
  }
}

write.csv(table_data, regression_data_path, row.names = FALSE)
write.csv(acquired_word_data, acquired_words_path, row.names = FALSE)
write.csv(result_data, result_data_path, row.names = FALSE)
write.csv(converged_words, converged_words_path, row.names = FALSE)