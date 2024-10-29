## code taken and adapted from https://github.com/tylerachang/word-acquisition-language-models/

if (!require(dplyr)) {
    install.packages("dplyr")
}
if (!require(ggplot2)) {
    install.packages("ggplot2")
}
if (!require(plyr)) {
    install.packages("plyr")
}
if (!require(tidyr)) {
    install.packages("tidyr")
}
if (!require(scales)) {
    install.packages("scales")
}
if (!require(stats)) {
    install.packages("stats")
}
if (!require(stringr)) {
    install.packages("stringr")
}
if (!require(lmtest)) {
    install.packages("lmtest")
}
if (!require(gridExtra)) {
    install.packages("gridExtra")
}
if (!require(car)) {
    install.packages("car")
}
if (!require(cowplot)) {
    install.packages("cowplot")
}


# Load dplyr
library(dplyr)
library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(scales)
library(stats)
library(stringr)
library(lmtest)
library(gridExtra)
library(car)
library(cowplot)


# Get child dataframes.
get_child_proportion_df <- function(proportion_file) {
  child_proportion_df <- read.csv(file=proportion_file)
  child_proportion_df <- child_proportion_df %>% gather(Month, Proportion, X16:X30)
  child_proportion_df$Month <- as.numeric(sub('X', '', child_proportion_df$Month))
  child_proportion_df$definition <- as.factor(child_proportion_df$definition)
  child_proportion_df <- child_proportion_df %>% dplyr::select(definition, Month, Proportion)
  child_proportion_df <- setNames(child_proportion_df, c("Token", "Month", "Proportion"))
  # Note: these tokens are not cleaned, so some may not be the same as the smooth child AoA data.
  # This is okay for now because this data is only used to generate plots.
  return(child_proportion_df)
}

get_child_aoa_df <- function(aoa_file, childes_file="") {
  child_aoa_df <- read.table(file=aoa_file,
                             encoding="UTF-8", header=TRUE, sep="\t", fill=TRUE, quote="")
  child_aoa_df$language <- as.factor(child_aoa_df$language)
  child_aoa_df <- child_aoa_df %>% filter(child_aoa_df$language == "English (American)" &
                                          child_aoa_df$measure == "produces")
  # Get info about each word.
  # Note: UniLemma is only relevant for multilingual analyses.
  word_info_df <- child_aoa_df %>% dplyr::select(CleanedSingle, lexical_class, uni_lemma)
  word_info_df <- setNames(word_info_df, c("Token", "LexicalClass", "UniLemma"))
  word_info_df$LexicalClass <- as.factor(word_info_df$LexicalClass)
  word_info_df$UniLemma <- stringr::word(word_info_df$UniLemma, 1)
  word_info_df$NChars <- nchar(word_info_df$Token)
  word_info_df <- unique(word_info_df)
  # Get AoA for each word.
  child_aoa_df <- child_aoa_df %>% dplyr::select(CleanedSingle, aoa)
  # For AoA, average over all data for each token/word.
  child_aoa_df <- unique(child_aoa_df) # Remove duplicates.
  child_aoa_df <- aggregate(child_aoa_df$aoa, by=list(child_aoa_df$CleanedSingle), FUN=mean)
  child_aoa_df <- setNames(child_aoa_df, c("Token", "ChildAoA"))
  child_aoa_df$Token <- as.factor(child_aoa_df$Token)
  child_aoa_df <- merge(child_aoa_df, word_info_df, by.x="Token", by.y="Token")
  if (childes_file != "") {
    # Add additional fields from CHILDES.
    childes_df <- read.table(file=childes_file, encoding="UTF-8",
                             header=TRUE, sep="\t", fill=TRUE, quote="")
    childes_df <- childes_df %>% dplyr::select(word, word_count, mean_sent_length)
    childes_df <- setNames(childes_df, c("Token", "ChildesCount", "ChildMLU"))
    childes_df$Token <- tolower(childes_df$Token)
    childes_df <- childes_df %>% filter(childes_df$Token != '')
    total_childes_tokens <- sum(childes_df$ChildesCount)
    childes_df$ChildFreq <- (childes_df$ChildesCount*1000.0)/total_childes_tokens
    childes_df$ChildLogFreq <- log(childes_df$ChildFreq)
    child_aoa_df <- merge(child_aoa_df, childes_df, by.x="Token", by.y="Token")
  }
  return(child_aoa_df)
}


# Add concreteness data to a dataframe.
add_concreteness <- function(original_df, concreteness_file, merge_field="Token") {
  concreteness_df <- read.table(file=concreteness_file, encoding="UTF-8",
                                header=TRUE, sep="\t", fill=TRUE, quote="")
  concreteness_df <- concreteness_df %>% dplyr::select(Word, Conc.M)
  concreteness_df <- setNames(concreteness_df, c("Token", "Concreteness"))
  original_df <- merge(original_df, concreteness_df, by.x=merge_field, by.y="Token", all.x=TRUE)
  # cat("  Imputing ", sum(is.na(original_df$Concreteness)), " concreteness values.\n", sep="")
  original_df$Concreteness[is.na(original_df$Concreteness)] <-
    mean(original_df$Concreteness, na.rm=TRUE) # Replace NA.
  return(original_df)
}




# Run the linear regressions AoA analysis.
# For convenience, returns the dataframe of predictors and AoA data.
run_regressions <- function(is_lm=TRUE, child_aoa_file="", childes_file="",
                            concreteness_file="", lm_aoa_file="", lm_data_stats_file="",
                            print_analyses=FALSE, quadratic_logfreq=FALSE) {
  quadratic <- function(formula) { # Add a quadratic log-frequency term to the formula.
    return(str_replace_all(formula, "LogFreq", "poly(LogFreq,2)"))
  }
  # The child data is required to obtain lexical class data.
  child_aoa_df <- get_child_aoa_df(child_aoa_file, childes_file=childes_file)

  print(child_aoa_df$LexicalClass %>% unique())

  # print the number of unique words
    cat("Number of unique words: ", nrow(child_aoa_df), "\n")
  regression_df <- data.frame()
  if (is_lm) {
    lm_aoa_df <- get_lm_aoa_df(lm_aoa_file, lm_data_stats_file=lm_data_stats_file)
    combined_aoa_df <- merge(lm_aoa_df, child_aoa_df, by.x="Token", by.y="Token")
    regression_df <- combined_aoa_df %>% dplyr::select(
      Token, LmAoA, LmLogFreq, LmMLU, NChars, LexicalClass)
  } else {
    regression_df <- child_aoa_df %>% dplyr::select(
      Token, ChildAoA, ChildLogFreq, ChildMLU, NChars, LexicalClass)
  }
  regression_df <- setNames(regression_df,
                            c("Token", "AoA", "LogFreq", "MLU", "NChars", "LexicalClass"))
  regression_df <- add_concreteness(regression_df, concreteness_file)
  
  # Correlations.
  if (print_analyses) {
    cat("  Correlations:\n")
    corr_df <- regression_df %>% dplyr::select(LogFreq, NChars, Concreteness, MLU)
    print(cor(corr_df, method="pearson"))
    cat("\n")
  }
  
  # Log-frequency alone regression.
  formula_logfreq <- "AoA ~ LogFreq"
  if (quadratic_logfreq) { formula_logfreq <- quadratic(formula_logfreq) }
  logfreq_reg <- lm(formula_logfreq, data=regression_df)
  cat("  LogFreq R^2: ", summary(logfreq_reg)$adj.r.squared, "\n", sep="")

  formula_concreteness <- "AoA ~ Concreteness"
    concreteness_reg <- lm(formula_concreteness, data=regression_df)
    cat("  Concreteness R^2: ", summary(concreteness_reg)$adj.r.squared, "\n", sep="")

  formula_nchars <- "AoA ~ NChars"
    nchars_reg <- lm(formula_nchars, data=regression_df)
    cat("  NChars R^2: ", summary(nchars_reg)$adj.r.squared, "\n", sep="")

  formula_mlu <- "AoA ~ MLU"
    mlu_reg <- lm(formula_mlu, data=regression_df)
    cat("  MLU R^2: ", summary(mlu_reg)$adj.r.squared, "\n", sep="")

  formula_lexical_class <- "AoA ~ LexicalClass"
    lexical_class_reg <- lm(formula_lexical_class, data=regression_df)
    cat("  LexicalClass R^2: ", summary(lexical_class_reg)$adj.r.squared, "\n", sep="")


  
  # Run regressions.
  predictors <- c("LogFreq", "MLU", "NChars", "Concreteness", "LexicalClass")
  formula_predictors <- paste(predictors, collapse=" + ")
  formula_with_predictor <- paste("AoA ~ ", formula_predictors, sep="")
  if (quadratic_logfreq) { formula_with_predictor <- quadratic(formula_with_predictor) }
  reg_with_predictor <- lm(formula_with_predictor, data=regression_df)
    cat("  Full model R^2: ", summary(reg_with_predictor)$adj.r.squared, "\n", sep="")
  if (print_analyses) {
    cat("  VIFs:\n")
    print(vif(reg_with_predictor))
    cat("\n")
  }
}

child_aoa_file <- "src/wordbank/data/child_aoa_chang.tsv"
childes_file <- "src/wordbank/data/childes_eng-na.tsv"

# run regression for child data
run_regressions(is_lm=FALSE, child_aoa_file=child_aoa_file, childes_file=childes_file,
                concreteness_file="src/wordbank/data/concreteness_data.tsv",
                print_analyses=TRUE, quadratic_logfreq=TRUE)