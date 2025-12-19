################################################################################
# FILE 03: METHOD 2 - REGULARIZATION (RIDGE & LASSO)
# ============================================================================
# Description: Handling multicollinearity and variable selection via
#              regularization techniques (Coefficient norm penalization)
#
# Problem:
# In the tech sector, assets are highly correlated.
# The (X'X) matrix becomes ill-conditioned (close to singularity),
# this makes the OLS estimator unstable (explosive variance)
#
# Solutions:
# 1. Ridge (Tikhonov): We minimize ||y - Xb||^2 + lambda * ||b||_2^2
#    -> Shrinks coefficients towards 0, stabilizes inversion
# 2. Lasso: We minimize ||y - Xb||^2 + lambda * ||b||_1
#    -> Forces some coefficients to exactly 0 (Variable Selection)
################################################################################

if (!require("glmnet")) install.packages("glmnet")
library(tidyverse)
library(glmnet)

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================
clean_data <- readRDS("data/clean_data.rds")

# Objective: Explain NVDA returns by those of its peers
# p (number of predictors) is high and correlated
target_ticker <- "NVDA"
predictors <- setdiff(names(clean_data), c("date", "RF", target_ticker))

# Matrix construction for glmnet (requires matrix, not data.frame)
X_vars <- clean_data %>%
  select(all_of(predictors)) %>%
  as.matrix()

y_var <- clean_data[[target_ticker]]

cat(sprintf("Dimensions of X matrix for Ridge/Lasso: %d observations x %d predictors\n", nrow(X_vars), ncol(X_vars)))

# ============================================================================
# 2. RIDGE REGRESSION (L2 PENALIZATION)
# ============================================================================
# The Ridge estimator is given by: beta_ridge = (X'X + lambda*I)^(-1) X'y
# Adding lambda*I to the diagonal makes the matrix invertible

cat("\n--- Optimization of Lambda Parameter (Ridge) ---\n")

# Cross-Validation (K-Fold) to find optimal lambda
# We seek the lambda that minimizes out-of-sample Mean Squared Error (MSE)
set.seed(123) # For reproducibility
cv_ridge <- cv.glmnet(X_vars, y_var, alpha = 0) # alpha = 0 for ridge

best_lambda_ridge <- cv_ridge$lambda.min
cat(sprintf("Optimal Lambda (Ridge): %.4f\n", best_lambda_ridge))

# Fitting the final model with optimal lambda
ridge_model <- glmnet(X_vars, y_var, alpha = 0, lambda = best_lambda_ridge)

# ============================================================================
# 3. LASSO REGRESSION (L1 PENALIZATION)
# ============================================================================
# Lasso induces sparsity. It acts as an automatic selector of relevant variables
# amid noise

cat("\n--- Optimization of Lambda Parameter (Lasso) ---\n")

set.seed(123)
cv_lasso <- cv.glmnet(X_vars, y_var, alpha = 1) # alpha = 1 for lasso

best_lambda_lasso <- cv_lasso$lambda.min
cat(sprintf("Optimal Lambda (Lasso): %.4f\n", best_lambda_lasso))

lasso_model <- glmnet(X_vars, y_var, alpha = 1, lambda = best_lambda_lasso)

# ============================================================================
# 4. COEFFICIENT COMPARISON AND INTERPRETATION
# ============================================================================
# We compare coefficients obtained by Ridge and Lasso to identify the real
# drivers of NVDA performance

# Coefficient extraction (removing intercept for comparison)
ridge_coefs <- as.matrix(coef(ridge_model))[-1, 1]
lasso_coefs <- as.matrix(coef(lasso_model))[-1, 1]

comparison_df <- data.frame(
  Predictor = names(ridge_coefs),
  Ridge = ridge_coefs,
  Lasso = lasso_coefs
) %>%
  arrange(desc(abs(Lasso))) # Sort by lasso importance

cat("\n--- Top 10 Drivers of NVDA according to Lasso (Variable Selection) ---\n")
print(head(comparison_df, 10))

# If a Lasso coefficient is 0, the variable is considered irrelevant
# Ridge never sets to 0, but strongly reduces influence

# ============================================================================
# 5. GENERATION OF THE "RIDGE TRACE"
# ============================================================================
# The Ridge Trace visualizes coefficient evolution as a function of Lambda
# It shows how regularization stabilizes estimates

ridge_fit_full <- glmnet(X_vars, y_var, alpha = 0)
ridge_trace_data <- as.data.frame(as.matrix(t(ridge_fit_full$beta)))
ridge_trace_data$lambda <- ridge_fit_full$lambda

# ============================================================================
# 6. SAVING RESULTS
# ============================================================================
if (!file.exists("results")) dir.create("results")

saveRDS(list(ridge = ridge_model, lasso = lasso_model, comparison = comparison_df), "results/method2_regularization.rds")
write_csv(comparison_df, "results/method2_coefficients.csv")
write_csv(ridge_trace_data, "results/method2_ridge_trace.csv")

cat("\nRegularization analysis completed. Results saved.\n")
