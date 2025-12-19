################################################################################
# FILE 02: METHOD 1 - OLS & FAMA-FRENCH APPROACH
# ============================================================================
# Description: Implementation of Ordinary Least Squares (OLS) estimation
#              This method is a foundation to evaluate advanced methods
#
# Mathematical model:
# y = X * beta + epsilon
# - y: Asset excess return vector (n x 1)
# - X: Risk factor matrix (n x p) [Mkt-RF, SMB, HML]
# - beta: Sensitivity vector (p x 1)
#
# Analytical solution:
# beta_hat = (X'X)^(-1) X'y
# This solution minimizes the sum of squared residuals ||y - X*beta||^2
################################################################################

library(tidyverse)
library(broom)

# ============================================================================
# 1. LOADING PREPARED DATA
# ============================================================================
clean_data <- readRDS("data/clean_data.rds")

# Variable definition
factors <- c("Mkt_RF", "SMB", "HML")
tickers <- setdiff(names(clean_data), c("date", factors, "RF"))

cat(sprintf("Starting OLS analysis for %d assets...\n", length(tickers)))

# ============================================================================
# 2. ALGEBRAIC DEMONSTRATION
# ============================================================================
# To demonstrate understanding of underlying mechanisms, we calculate
# the coefficients for one asset (e.g., NVDA) without using `lm()`

cat("\n--- ALGEBRAIC DEMONSTRATION (NVDA) ---\n")

# Construction of Design Matrix X (with Intercept)
# The intercept (alpha) captures abnormal performance unexplained by factors
X_matrix <- clean_data %>%
  select(all_of(factors)) %>%
  mutate(Intercept = 1) %>%
  select(Intercept, everything()) %>%
  as.matrix()

# Response Vector y (NVDA Excess Return)
y_vector <- clean_data$NVDA

# Calculation of cross-product matrices
XtX <- t(X_matrix) %*% X_matrix  # Gram Matrix (Fisher Information)
Xty <- t(X_matrix) %*% y_vector  # Projection of y onto factor space

# Solving the normal system: (X'X) * beta = X'y
# Direct inversion via solve() assumes X'X is invertible
beta_algebraic <- solve(XtX, Xty)

print(beta_algebraic)
cat("Note: These values must be identical to those from the lm() function.\n")

# ============================================================================
# 3. GENERALIZED ESTIMATION (LM FUNCTION)
# ============================================================================
# We apply the model to the entire investment universe,
# and use `lm()` to benefit from statistical diagnostics (t-stats, p-values)

results_list <- list()

for (ticker in tickers) {
  # Dynamic formula construction: Ticker ~ Mkt_RF + SMB + HML
  # Using backticks to handle names with special characters (e.g., BRK-B)
  formula_str <- paste0("`", ticker, "` ~ Mkt_RF + SMB + HML")
  
  model <- lm(as.formula(formula_str), data = clean_data)
  
  # Clean extraction of results with broom::tidy
  # We retrieve: estimate (beta), std.error, statistic (t-stat), p.value
  model_stats <- tidy(model) %>%
    mutate(ticker = ticker) %>%
    select(ticker, term, estimate, std.error, statistic, p.value)
  
  # Adding R-squared (Goodness of fit)
  # R^2 represents the share of variance explained by the factor model
  r_squared <- summary(model)$r.squared
  model_stats$r_squared <- r_squared
  
  results_list[[ticker]] <- model_stats
}

# Merging results into a single table
ols_results <- bind_rows(results_list)

# ============================================================================
# 4. ANALYSIS AND INTERPRETATION OF RESULTS
# ============================================================================
# Identification of "Offensive" vs "Defensive" assets
# Market Beta > 1: Amplifies market movements (Offensive/Risky)
# Market Beta < 1: Dampens market movements (Defensive)

market_betas <- ols_results %>%
  filter(term == "Mkt_RF") %>%
  arrange(desc(estimate))

cat("\n--- Top 5 Assets most sensitive to Market (High Beta) ---\n")
print(head(market_betas, 5))

cat("\n--- Top 5 Assets least sensitive to Market (Low Beta) ---\n")
print(tail(market_betas, 5))

# ============================================================================
# 5. SAVING RESULTS
# ============================================================================
if (!file.exists("results")) dir.create("results")

saveRDS(ols_results, "results/method1_ols_results.rds")
write_csv(ols_results, "results/method1_ols_results.csv")

cat("\nOLS analysis completed. Results saved in 'results/method1_ols_results.csv'.\n")
