################################################################################
# FILE 04c: FINAL OPTIMIZATION (LAGRANGE & RESULTS)
# ============================================================================
# Description: Analytical solution of the portfolio optimization problem 
#              via the Lagrange Multiplier method 
#
# The Optimization Problem:
# Minimize   1/2 * w' * Sigma * w  (Variance)
# Under constraints:
#   1. w' * mu = target (Target return)
#   2. w' * 1 = 1       (Budget)
#
# Solution (Lagrange) :
# Linear system A * z = b
# where z = [w, lambda1, lambda2]'
################################################################################

library(tidyverse)
library(xtable)

# ============================================================================
# 1. DATA SELECTION AND LOADING
# ============================================================================
clean_data <- readRDS("data/clean_data.rds")

# List of assets for the final portfolio
target_tickers <- c("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "AMD", "QCOM")

# Verification of the existence of assets in the database 
available_tickers <- intersect(target_tickers, names(clean_data))
missing_tickers <- setdiff(target_tickers, names(clean_data))

if (length(missing_tickers) > 0) {
  warning(sprintf("Warning: The following assets are missing and will be ignored: %s", 
                  paste(missing_tickers, collapse = ", ")))
}

if (length(available_tickers) < 2) {
  stop("Error: Not enough assets available for optimization.")
}

# Ticker List Update
tickers <- available_tickers

# Construction of the yield matrix
returns_mat <- clean_data %>% 
  select(all_of(tickers)) %>% 
  as.matrix()

# Annualized parameters (Assumption: 252 trading days)
mu <- colMeans(returns_mat) * 252       # Vector of expected returns
Sigma <- cov(returns_mat) * 252         # Variance-covariance matrix
N <- length(tickers)

cat(sprintf("Optimization on %d assets: %s\n", N, paste(tickers, collapse=", ")))

# ============================================================================
# 2. ANALYTICAL SOLUTION (LAGRANGE)
# ============================================================================
# We are looking for the minimum variance portfolio for a target return of 33%

target_return <- 0.33 # 33% per year

# Construction of the linear matrix system for Lagrange
# [ Sigma   mu   1 ] [ w ]   [ 0 ]
# [ mu'     0    0 ] [ l1] = [ target ]
# [ 1'      0    0 ] [ l2]   [ 1 ]

# Blocks of matrix A
top_block <- cbind(Sigma, mu, rep(1, N))
mid_block <- c(mu, 0, 0)
bot_block <- c(rep(1, N), 0, 0)

A_lagrange <- rbind(top_block, mid_block, bot_block)
b_vector <- c(rep(0, N), target_return, 1)

# System resolution A * x = b
solution <- tryCatch({
  solve(A_lagrange, b_vector)
}, error = function(e) {
  stop("Error during the inversion of the Lagrangian matrix. Check the collinearity of the assets.")
})

# Extraction of optimal weights (the first N elements)
optimal_weights <- solution[1:N]
names(optimal_weights) <- tickers

# ============================================================================
# 3. GENERATING RESULTS (LATEX TABLE)
# ============================================================================
# Creating a clean DataFrame for export
df_res <- data.frame(
  Asset = names(optimal_weights),
  Weight = optimal_weights,
  Allocation = paste0(round(optimal_weights * 100, 2), " \\%")
) %>% 
  filter(abs(Weight) > 0.001) %>% # Filtering of negligible weights (<0.1%)
  arrange(desc(Weight)) %>%
  select(Asset, Allocation)

# Generation of LaTeX code via xtable
latex_table <- xtable(df_res, 
                      caption = "Optimal Portfolio Allocation (Target 33\\% - Short Selling Allowed)", 
                      label = "tab:optimal_allocation")

# Saved in the results folder
if (!file.exists("results")) dir.create("results")
print(latex_table, 
      file = "results/optimal_weights.tex", 
      include.rownames = FALSE, 
      floating = FALSE, 
      sanitize.text.function = function(x){x}) # Prevents escape of LaTeX characters (e.g.: \%)

cat("Generated LaTeX Table: results/optimal_weights.tex\n")

# ============================================================================
# 4. VISUALIZATION (EFFICIENT FRONTIER)
# ============================================================================
if (!file.exists("figures")) dir.create("figures")
png("figures/efficient_frontier.png", width = 800, height = 600)

# Simulation of random portfolios to visualize all the possibilities
set.seed(123)
n_sim <- 2000
sim_risk <- numeric(n_sim)
sim_ret <- numeric(n_sim)

for(i in 1:n_sim) {
  w_sim <- rnorm(N)
  w_sim <- w_sim / sum(w_sim) # Normalization (sum = 1, short sale possible)
  
  sim_ret[i] <- sum(w_sim * mu)
  sim_risk[i] <- sqrt(t(w_sim) %*% Sigma %*% w_sim)
}

# Plot of simulated portfolios
plot(sim_risk, sim_ret, 
     col = "grey", pch = 19, cex = 0.5,
     xlab = "Risk (Annualized Volatility)", 
     ylab = "Expected Return (Annualized)",
     main = "Optimal Portfolio Positioning (Lagrange)")

# Added the calculated optimal point
opt_risk <- sqrt(t(optimal_weights) %*% Sigma %*% optimal_weights)
opt_ret <- sum(optimal_weights * mu)

points(opt_risk, opt_ret, col = "red", pch = 19, cex = 2)
text(opt_risk, opt_ret, labels = sprintf(" Target %.0f%%", target_return*100), pos = 4, col = "red", font = 2)

grid()
dev.off()

cat("Generated graph: figures/efficient_frontier.png\n")
cat("Success: Optimization complete\n")
