################################################################################
# FILE 04: METHOD 3 - STOCHASTIC ANALYSIS AND OPTIMIZATION
# ============================================================================
# Description: Unified script with advanced probabilistic approaches
#
# Structure:
# A. Regime Dynamics (Markov Chains)
#    - Modeling Bull/Bear transitions
#    - Calculation of stationary distribution (pi * P = pi)
#
# B. Risk Simulation (Monte Carlo + Cholesky)
#    - Generation of correlated trajectories via Cholesky decomposition (Sigma = LL')
#    - Estimation of Value-at-Risk (VaR) at 95%
#
# C. Optimal Strategy (Max Sharpe Ratio)
#    - Constrained quadratic optimization (Lagrangian)
#    - Construction of the Efficient Frontier
#
# Limitations:
# - Normality Assumption: Returns are assumed Gaussian (no Fat Tails)
# - Stationarity: Parameters (mu, Sigma) are assumed constant over the period
################################################################################
# ============================================================================
# 0. CONFIGURATION AND LOADING
# ============================================================================

library(tidyverse)
library(ggplot2)
library(markovchain) # For Markov chain analysis
library(quadprog)    # For quadratic optimization (QP)
library(xtable)      # For LaTeX export

clean_data <- readRDS("data/clean_data.rds")

# Selection of asset universe (Tech Tickers only)
tech_tickers <- setdiff(names(clean_data), c("date", "Mkt_RF", "SMB", "HML", "RF"))

# Returns Matrix (n x p)
returns_matrix <- clean_data %>% 
  select(all_of(tech_tickers)) %>% 
  as.matrix()

# ============================================================================
# PART A: MARKET REGIME MODELING (MARKOV)
# ============================================================================
# Objective: To quantify the probability of switching between a Bull and a Bear market
# and to determine the long-term equilibrium state

cat("\n---- PART A: REGIME DYNAMICS (MARKOV) ----\n")

# A.1 Construction of the Reference Index (Tech Market Proxy)
# We use the equal-weighted average of sector returns.
clean_data <- clean_data %>%
  mutate(
    Tech_Index_Return = rowMeans(select(., all_of(tech_tickers))),
    State = ifelse(Tech_Index_Return >= 0, "Bull", "Bear") # Binary discretization
  )

# A.2 Transition Preparation (t -> t+1)
market_states <- clean_data %>%
  select(date, Tech_Index_Return, State) %>%
  mutate(
    Next_State = lead(State) # State at t+1
  ) %>%
  drop_na()

# A.3 Estimation of Transition Matrix P
# P_ij = P(X_{t+1} = j | X_t = i)
transition_counts <- table(market_states$State, market_states$Next_State)
P_matrix <- prop.table(transition_counts, margin = 1)

cat("Stochastic Transition Matrix P:\n")
print(P_matrix)

# A.4 Calculation of Stationary Distribution (pi)
# Theorem: If the chain is irreducible and aperiodic, there exists a unique vector pi
# such that pi * P = pi. It is the left eigenvector associated with eigenvalue 1

eigen_decomp <- eigen(t(P_matrix))
idx_unit <- which.min(abs(eigen_decomp$values - 1)) # Find lambda ~ 1
eigen_vec <- Re(eigen_decomp$vectors[, idx_unit])
stationary_dist <- eigen_vec / sum(eigen_vec) # Normalization (Sum = 1)
names(stationary_dist) <- rownames(P_matrix)

cat("\n--- Stationary Distribution (Long Term Probability) ---\n")
print(stationary_dist)

# Saving
saveRDS(list(P=P_matrix, pi=stationary_dist), "results/method3_markov.rds")

# ============================================================================
# PART B: STOCHASTIC SIMULATION (MONTE CARLO)
# ============================================================================
# Objective: Simulate 10,000 possible futures to assess extreme risk (VaR)
# Method: Multivariate Geometric Brownian Motion dP = mu*P*dt + sigma*P*dW
# Correlation: Induced by Cholesky factorization of the covariance matrix

cat("\n---- PART B: MONTE CARLO SIMULATION (CHOLESKY) ----\n")

# B.1 Model Parameters
Sigma <- cov(returns_matrix) # Historical Covariance Matrix
mu <- colMeans(returns_matrix) # Mean Returns Vector

# B.2 Cholesky Factorization
# We seek L such that Sigma = L * L^T
# This allows transforming independent white noise Z into correlated noise X = L*Z
L <- t(chol(Sigma)) # R returns U (Upper), we take the transpose to get L (Lower)

# B.3 Simulation Configuration
n_sims <- 10000      # Number of trajectories
n_days <- 252        # Horizon (1 trading year)
n_assets <- length(tech_tickers)
initial_capital <- 100
weights <- rep(1/n_assets, n_assets) # Equal-weighted portfolio for simulation

final_values <- numeric(n_sims)
set.seed(123) # Reproducibility

cat("Launching 10,000 simulations (Engine: Cholesky + GBM)...\n")

# B.4 Simulation Loop
for (i in 1:n_sims) {
  # 1. Generation of Gaussian white noise Z ~ N(0, I)
  Z <- matrix(rnorm(n_assets * n_days), nrow = n_assets, ncol = n_days)
  
  # 2. Introduction of correlation structure
  shocks <- L %*% Z
  
  # 3. Calculation of simulated returns (Drift + Diffusion)
  sim_returns <- shocks + mu
  
  # 4. Portfolio-level aggregation
  sim_simple <- exp(sim_returns) - 1
  
  # 5. Value trajectory (Compounded integration)
  # Convert log-return portfolio series to simple-return series before compounding
  port_daily_ret <- as.vector(t(weights) %*% sim_returns)
  port_simple <- exp(port_daily_ret) - 1
  cum_path <- cumprod(1 + port_simple) * initial_capital
  final_values[i] <- tail(cum_path, 1)
}

# B.5 Value-at-Risk (VaR) Calculation
# The 95% VaR corresponds to the 5% quantile of the final value distribution.
VaR_level <- 0.05
VaR_value <- quantile(final_values, probs = VaR_level)
VaR_loss <- initial_capital - VaR_value

cat(sprintf("\nVaR 95%% (Estimated Max Loss with 95%% confidence): €%.2f\n", VaR_loss))

# ============================================================================
# PART C: PORTFOLIO OPTIMIZATION (MAX SHARPE)
# ============================================================================
# Objective: To find allocation w* that maximizes the ratio (mu_p - rf) / sigma_p
# This is the "Tangency Portfolio" of Modern Portfolio Theory (Markowitz)

cat("\n---- PART C: QUADRATIC OPTIMIZATION (MAX SHARPE) ----\n")

# C.1 Input Preparation (Annualization)
# Optimization is generally done on annualized parameters
mu_annual <- colMeans(returns_matrix) * 252
Sigma_annual <- cov(returns_matrix) * 252

# C.2 Resolution Function (Quadratic Programming)
# Problem: Min w'Sigma w  s.t. w'mu = target, sum(w) = 1, w >= 0
solve_portfolio <- function(target_return, Sigma, mu) {
  n <- length(mu)
  # Dmat Matrix (2*Sigma) for standard quadratic form 1/2 x'Dx
  Dmat <- 2 * Sigma
  dvec <- rep(0, n) # No linear term in objective function (variance)
  
  # Constraints:
  # 1. Return = target
  # 2. Sum of weights = 1
  # 3. Weights >= 0 (Long only)
  Amat <- cbind(mu, rep(1, n), diag(n))
  bvec <- c(target_return, 1, rep(0, n))
  
  # The first 2 constraints are equalities
  sol <- solve.QP(Dmat, dvec, Amat, bvec, meq = 2)
  return(sol$solution)
}

# C.3 Efficient Frontier Construction
# We sweep a range of target returns to trace the optimal curve
targets <- seq(min(mu_annual), max(mu_annual), length.out = 50)
frontier_res <- list()

for (t in targets) {
  try({
    w <- solve_portfolio(t, Sigma_annual, mu_annual)
    risk <- sqrt(t(w) %*% Sigma_annual %*% w)
    sharpe <- t / risk
    frontier_res[[length(frontier_res)+1]] <- data.frame(Return=t, Risk=risk, Sharpe=sharpe, Weights=I(list(w)))
  }, silent = TRUE)
}
df_frontier <- bind_rows(frontier_res)

# C.4 Optimal Portfolio Extraction (Max Sharpe)
best_idx <- which.max(df_frontier$Sharpe)
best_port <- df_frontier[best_idx, ]
best_weights <- unlist(best_port$Weights)
names(best_weights) <- colnames(returns_matrix)

# Filtering significant weights (> 0.1%)
final_allocation <- data.frame(Ticker = names(best_weights), Weight = best_weights) %>%
  filter(Weight > 0.001) %>%
  arrange(desc(Weight))

cat("\n=== OPTIMAL ALLOCATION (MAX SHARPE) ===\n")
print(final_allocation)
cat(sprintf("Expected Sharpe Ratio: %.4f\n", best_port$Sharpe))

# C.5 LaTeX Export
# Automatic table generation for the report
latex_table <- xtable(final_allocation, caption = "Optimal Tangency Portfolio Allocation")
print(latex_table, file = "results/optimal_strategy.tex", include.rownames = FALSE)

cat("\nComplete analysis finished. Files generated.\n")
