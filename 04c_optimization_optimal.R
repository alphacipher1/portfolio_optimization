################################################################################
# FILE 04c_optimization_optimal.R
# ============================================================================
# Objective: 
# Find the optimal strategy (Max Sharpe Ratio)
#
# Constraints: 
# Sum = 100%, No Short Selling
#
# Limitation: 
# This model assumes Gaussian distribution of returns
# Real-world Fat Tails are not captured by standard Cholesky
################################################################################

library(tidyverse)
library(quadprog)
library(xtable)

# ---- 1. Data Prep ----
clean_data <- readRDS("data/clean_data.rds")
# Broad selection to give the algorithm choice
tickers <- c("NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "AMD", "WMT", "JPM", "COST", "ORCL", "META")

# Ensure we only select columns that exist in clean_data
available_tickers <- intersect(tickers, colnames(clean_data))
if(length(available_tickers) < length(tickers)) {
  warning("Some tickers were not found in the dataset: ", paste(setdiff(tickers, available_tickers), collapse=", "))
}
tickers <- available_tickers

returns_mat <- clean_data %>% select(all_of(tickers)) %>% as.matrix()

mu <- colMeans(returns_mat) * 252
Sigma <- cov(returns_mat) * 252
N <- length(tickers)

# ---- 2. Solver Function (Min Variance for a given Return) ----
solve_for_target <- function(target) {
  Dmat <- Sigma
  dvec <- rep(0, N)
  
  # Constraints: Sum=1, Return=Target, LongOnly
  # quadprog solve.QP solves: min(-d^T b + 1/2 b^T D b) with A^T b >= b0
  # We want A^T w = bvec (for equality) and A^T w >= bvec (for inequality)
  
  # Constraints:
  # 1. sum(w) = 1 (Equality)
  # 2. mu'w = target (Equality)
  # 3. w >= 0 (Inequality)
  
  Amat <- cbind(rep(1, N), mu, diag(N))
  bvec <- c(1, target, rep(0, N))
  
  tryCatch({
    # meq=2 : The first 2 constraints (Sum, Return) are equalities
    sol <- solve.QP(Dmat, dvec, Amat, bvec, meq = 2)
    w <- sol$solution
    return(list(w = w, risk = sqrt(as.numeric(t(w) %*% Sigma %*% w))))
  }, error = function(e) return(NULL))
}

# ---- 3. Efficient Frontier Scan (To find Max Sharpe) ----
# Test 100 targets between the min and max asset return
# We solve for target returns that are feasible
# The feasible range is between the min return of the assets (or global min var portfolio) and max return of assets
targets <- seq(min(mu), max(mu), length.out = 100)
frontier_res <- list()

for(t in targets) {
  res <- solve_for_target(t)
  if(!is.null(res)) {
    frontier_res[[length(frontier_res)+1]] <- data.frame(
      Return = t,
      Risk = res$risk,
      Sharpe = t / res$risk, # Assumption Rf=0 for simplification
      Weights = I(list(res$w))
    )
  }
}
df_frontier <- bind_rows(frontier_res)

# ---- 4. Selection of the Winner (Max Sharpe) ----
# This is the "Optimal" strategy
best_idx <- which.max(df_frontier$Sharpe)
best_strat <- df_frontier[best_idx, ]
best_weights <- unlist(best_strat$Weights)
names(best_weights) <- tickers

# Cleanup for display (Sum = 100%)
final_weights <- round(best_weights, 4)
final_weights[final_weights < 0.001] <- 0
final_weights <- final_weights / sum(final_weights)

# ---- 5. LaTeX Table Generation ----
df_res <- data.frame(
  Asset = names(final_weights),
  Weight = final_weights
) %>%
  filter(Weight > 0) %>% # Only show what we buy
  arrange(desc(Weight)) %>%
  mutate(Allocation = paste0(round(Weight * 100, 2), " \\%")) %>%
  select(Asset, Allocation)

# Add Total
df_res <- rbind(df_res, c("\\textbf{TOTAL}", "\\textbf{100.00 \\%}"))

# Strategy metrics
metrics_caption <- paste0(
  "Optimal Portfolio (Max Sharpe). ",
  "Expected Return: ", round(best_strat$Return*100, 1), "\\%, ",
  "Risk: ", round(best_strat$Risk*100, 1), "\\%"
)

print(xtable(df_res, caption = metrics_caption, label="tab:optimal_strategy"),
      file = "results/optimal_strategy.tex",
      include.rownames = FALSE, sanitize.text.function = identity, floating = FALSE)

# ---- 6. Final Graph ----
png("figures/optimal_frontier.png", width=800, height=600)
plot(df_frontier$Risk, df_frontier$Return, type="l", col="blue", lwd=2,
     xlab="Risk (Volatility)", ylab="Expected Return",
     main="Optimization: Search for the Tangent Point (Max Sharpe)",
     xlim=c(min(df_frontier$Risk)*0.9, max(df_frontier$Risk)*1.1),
     ylim=c(min(df_frontier$Return)*0.9, max(df_frontier$Return)*1.1))
grid()

# Optimal Point
points(best_strat$Risk, best_strat$Return, col="red", pch=19, cex=2)
text(best_strat$Risk, best_strat$Return,
     paste0(" OPTIMUM\n(Sharpe=", round(best_strat$Sharpe, 2), ")"),
     pos=4, col="red", font=2)

dev.off()

print("Success: Optimal Strategy generated.")