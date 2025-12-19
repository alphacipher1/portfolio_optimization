################################################################################
# FILE 04c_optimization_frontier.R
# ============================================================================
# Objective: 
# Generate 3 Strategies on the Efficient Frontier
# 
# Method: 
# Quadratic Programming
################################################################################

library(tidyverse)
library(quadprog)
library(xtable)

# ---- 1. Data Prep ----
clean_data <- readRDS("data/clean_data.rds")
# Asset selection (Lasso Top + Defensive for balance)
tickers <- c("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "AMD", "WMT", "JPM", "COST")
# Ensure we only select columns that exist in clean_data
available_tickers <- intersect(tickers, colnames(clean_data))
if(length(available_tickers) < length(tickers)) {
  warning("Some tickers were not found in the dataset: ", paste(setdiff(tickers, available_tickers), collapse=", "))
}
tickers <- available_tickers

returns_mat <- clean_data %>% select(all_of(tickers)) %>% as.matrix()

mu <- colMeans(returns_mat) * 252       # Annual Returns
Sigma <- cov(returns_mat) * 252         # Annual Covariance
N <- length(tickers)

# ---- 2. Quadratic Solver Function ----
# Min 1/2 w'Sw s.t. sum(w)=1, w>=0, return=target
solve_portfolio <- function(target_ret) {
  Dmat <- Sigma
  dvec <- rep(0, N)
  
  # Constraints: Sum=1, Ret=Target, LongOnly
  # quadprog solve.QP solves: min(-d^T b + 1/2 b^T D b) with A^T b >= b0
  # We want A^T w = bvec (for equality) and A^T w >= bvec
  
  # Constraints:
  # 1. sum(w) = 1
  # 2. mu'w = target_ret
  # 3. w >= 0  => I_N w >= 0
  
  Amat <- cbind(rep(1, N), mu, diag(N))
  bvec <- c(1, target_ret, rep(0, N))
  
  # meq=2 means the first 2 constraints are strict equalities
  tryCatch({
    sol <- solve.QP(Dmat, dvec, Amat, bvec, meq = 2)
    return(sol$solution)
  }, error = function(e) return(rep(NA, N)))
}

# ---- 3. Calculation of the 3 Strategies ----

# A. Shield (Minimum Volatility - We relax the return constraint)
# For MinVol, we only want Sum=1 and w>=0. We remove the mu constraint
Amat_min <- cbind(rep(1, N), diag(N))
bvec_min <- c(1, rep(0, N))
sol_min <- solve.QP(Sigma, rep(0, N), Amat_min, bvec_min, meq=1)
w_min <- sol_min$solution

# B. Balanced (Aim for the Tech sector average, approx 25%)
w_bal <- solve_portfolio(0.25)

# C. Growth (Aim high, approx 35%)
w_agg <- solve_portfolio(0.35)

# ---- 4. Consolidation of Results ----
format_w <- function(w) {
  if(any(is.na(w))) return(rep(NA, length(w)))
  w <- round(w, 4); w[w < 0.001] <- 0
  return(w) # Keep numeric for risk calculations
}

# Creation of the DataFrame
df_strat <- data.frame(Asset = tickers)
df_strat$Shield <- format_w(w_min)
df_strat$Balanced <- format_w(w_bal)
df_strat$Growth <- format_w(w_agg)

# Calculation of Metrics (Risk/Return) for each strategy
calc_metrics <- function(w) {
  if(any(is.na(w))) return(c(NA, NA, NA))
  ret <- sum(w * mu)
  risk <- sqrt(as.numeric(t(w) %*% Sigma %*% w))
  sharpe <- ret / risk
  return(c(ret, risk, sharpe))
}

met_min <- calc_metrics(df_strat$Shield)
met_bal <- calc_metrics(df_strat$Balanced)
met_agg <- calc_metrics(df_strat$Growth)

# Addition of metric rows to the table
# First format the weights as %
df_disp <- df_strat %>% mutate(across(-Asset, ~ paste0(round(. * 100, 1), "%")))

# Summary rows
row_ret <- c("Annual Return", paste0(round(c(met_min[1], met_bal[1], met_agg[1])*100, 1), "%"))
row_ris <- c("Risk (Volatility)", paste0(round(c(met_min[2], met_bal[2], met_agg[2])*100, 1), "%"))
row_shp <- c("Sharpe Ratio", round(c(met_min[3], met_bal[3], met_agg[3]), 2))

df_final <- rbind(df_disp, row_ret, row_ris, row_shp)

# LaTeX Save
print(xtable(df_final, caption = "Comparison of the 3 Optimal Strategies", label="tab:strategies_comparison"),
      file = "results/strategies_comparison.tex", include.rownames = FALSE, floating = FALSE)

# ---- 5. Graph Visualization ----
png("figures/efficient_frontier_final.png", width=800, height=600)

# Simulation for the background
set.seed(123)
sim_w <- matrix(runif(N*5000), ncol=N); sim_w <- sim_w/rowSums(sim_w)
risk_s <- sqrt(diag(sim_w %*% Sigma %*% t(sim_w)))
ret_s <- sim_w %*% mu

plot(risk_s, ret_s, col="grey90", pch=19, xlim=c(min(risk_s)*0.9, max(risk_s)*1.1), ylim=c(min(ret_s)*0.9, max(ret_s)*1.1),
     xlab="Risk (Volatility)", ylab="Expected Return",
     main="Efficient Frontier: The Strategic Choice")

# Addition of the 3 points
if(!any(is.na(met_min))) {
  points(met_min[2], met_min[1], col="green", pch=19, cex=2.5)
  text(met_min[2], met_min[1], " SHIELD\n(Min Risk)", pos=1, col="darkgreen", font=2)
}

if(!any(is.na(met_bal))) {
  points(met_bal[2], met_bal[1], col="orange", pch=19, cex=2.5)
  text(met_bal[2], met_bal[1], " BALANCED\n(Best Sharpe)", pos=3, col="darkorange", font=2)
}

if(!any(is.na(met_agg))) {
  points(met_agg[2], met_agg[1], col="red", pch=19, cex=2.5)
  text(met_agg[2], met_agg[1], " GROWTH\n(Max Return)", pos=4, col="darkred", font=2)
}

# Frontier line (approximation for visual purposes)
ord <- order(risk_s)
risk_s_ord <- risk_s[ord]
ret_s_ord <- ret_s[ord]

dev.off()

print("Success: Strategies generated.")