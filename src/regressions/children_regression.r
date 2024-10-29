# Load necessary libraries
# install.packages("lme4")
# install.packages("MuMIn")
# install.packages("dplyr")
# install.packages("car")
# install.packages("optimx")
# install.packages("brms")

library(lme4)
library(MuMIn)
library(dplyr)
library(car)
library(optimx)
library(brms)
library(ggplot2)


args <- commandArgs(trailingOnly = TRUE)
aoa_and_predictors_path <- args[1]
regression_data_path <- args[2]
acquired_words_path <- args[3]

if (length(args) < 3) {
  cat("Please provide the path to the aoa_and_predictors file, regression_data file and acquired_words file")
  quit()
}

# Load the data
aoa_and_predictors <- read.csv(aoa_and_predictors_path)
cat("\n\nLength of the data:", nrow(aoa_and_predictors), "\n")


table_data <- data.frame(num_words = numeric(), model = character(), adjusted_r2=numeric(), f_statistic = numeric(), p_value = numeric(), stringsAsFactors = FALSE)
acquired_word_data <- data.frame(word = character(), frequency = numeric(), aoa = numeric(), rank = numeric(), lexical_class = character(), stringsAsFactors = FALSE)


aoa_and_predictors <- aoa_and_predictors %>%
  filter(lexical_class != "other", )

min_step <- 16
max_step <- 30

# min max scale the child_aoa
aoa_and_predictors$child_aoa <- (aoa_and_predictors$child_aoa - min_step) / (max_step - min_step)

# Select relevant columns
child_df <- aoa_and_predictors %>%
  select(word, child_aoa, child_log_frequency, lexical_class, n_chars, concreteness, child_mlu) %>%
  na.omit() %>%
  distinct()

mean_child_aoa <- mean(child_df$child_aoa)
std_child_aoa <- sd(child_df$child_aoa)

# Filter out rows where child_aoa is more than 3 standard deviations away from the mean
lower_bound <- mean_child_aoa - 3 * std_child_aoa
upper_bound <- mean_child_aoa + 3 * std_child_aoa
child_df <- child_df %>%
  filter(child_aoa > lower_bound & child_aoa < upper_bound)

# Check for multicollinearity
vif_values <- vif(lm(child_aoa ~ child_log_frequency + n_chars + concreteness + child_mlu + C(lexical_class), data = child_df))

control=lmerControl(check.conv.singular = .makeCC(action = "ignore",  tol = 1e-4))

child_df$child_frequency <- 10^child_df$child_log_frequency
child_df$child_frequency <- format(child_df$child_frequency, scientific = TRUE)



m1 <- lm(child_aoa ~ child_log_frequency , data = child_df)
m2 <- lm(child_aoa ~ concreteness , data = child_df)
m3 <- lm(child_aoa ~ n_chars , data = child_df)
m4 <- lm(child_aoa ~ child_mlu , data = child_df)
m5 <- lm(child_aoa ~ C(lexical_class) , data = child_df)
m6 <- lm(child_aoa ~ child_log_frequency + n_chars + concreteness + child_mlu + C(lexical_class), data = child_df)
m7 <- lm(child_aoa ~ n_chars + concreteness + child_mlu + C(lexical_class), data = child_df)

models <- list(m1, m2, m3, m4, m5, m6, m7)

for (i in 1:length(models)) {
  print(paste0("Model m", i))
  print(summary(models[[i]]))
}

# Calculate adjusted R^2 for each model
adjusted_r_squared <- function(model, data) {
  r_squared <- summary(model)$r.squared
  n <- nrow(data)
  p <- length(model$coefficients) - 1
  
  adjusted_marginal_r2 <- 1 - ((1 - r_squared) * (n - 1)) / (n - p - 1)
  
  return(adjusted_marginal_r2)
}

r2s <- lapply(models, adjusted_r_squared, data = child_df)

adjusted_r2_values <- numeric(length(models))

p_value_m1 <- summary(m1)$coefficients[2, 4]
p_value_m2 <- summary(m2)$coefficients[2, 4]
p_value_m3 <- summary(m3)$coefficients[2, 4]
p_value_m4 <- summary(m4)$coefficients[2, 4]
p_value_m5 <- summary(m5)$coefficients[2, 4]
p_value_m6 <- summary(m6)$coefficients[6, 4]
p_value_m7 <- summary(m7)$coefficients[5, 4]

p_value_m <- c(p_value_m1, p_value_m2, p_value_m3, p_value_m4, p_value_m5, p_value_m6, p_value_m7)

for (i in 1:length(models)) {
  adjusted_r2_values[i] <- r2s[[i]]

  table_data <- rbind(table_data, data.frame(
      num_words = length(unique(child_df$word)),
      model = paste0("m", i),
      adjusted_r2 = adjusted_r2_values[i], 
      f_statistic = summary(models[[i]])$fstatistic[1],
      p_value = p_value_m[i],
      stringsAsFactors = FALSE
    )
  )
}

top_words <- child_df %>%
  arrange(child_aoa) %>%
  head(10) %>%
  select(word, child_frequency, child_aoa, lexical_class) %>%
  mutate(
    rank = 1:10
  )

last_words <- child_df %>%
  arrange(desc(child_aoa)) %>%
  head(10) %>%
  select(word, child_frequency, child_aoa, lexical_class) %>%
  mutate(
    rank = -1:-10
  )

acquired_word_data <- rbind(acquired_word_data, top_words, last_words)
cat("\n\n", "num_words:", length(unique(child_df$word)), "\n\n")


write.csv(table_data, regression_data_path, row.names = FALSE)
write.csv(acquired_word_data, acquired_words_path, row.names = FALSE)