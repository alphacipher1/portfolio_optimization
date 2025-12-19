################################################################################
# FILE 04b : MONTE CARLO SIMULATION
# ============================================================================
################################################################################

library(tidyverse)
library(ggplot2)

# Load data
clean_data <- readRDS("data/clean_data.rds")
tech_tickers <- setdiff(names(clean_data), c("date", "Mkt_RF", "SMB", "HML", "RF"))
returns_matrix <- clean_data %>% select(all_of(tech_tickers)) %>% as.matrix()

# 1. Covariance Matrix
Sigma <- cov(returns_matrix)

# 2. Cholesky Decomposition
# Sigma = L * L^T
# R's chol() returns Upper Triangular U such that t(U) %*% U = Sigma
# We want L such that L %*% t(L) = Sigma. So L = t(U).
L <- t(chol(Sigma))

# 3. Simulation Parameters
n_sims <- 10000
n_days <- 252 # 1 year
n_assets <- length(tech_tickers)
initial_capital <- 100

# Weights: Equal weighted
weights <- rep(1/n_assets, n_assets)

# Storage
final_values <- numeric(n_sims)
# Store first 100 paths for plotting
plot_paths <- matrix(NA, nrow = n_days + 1, ncol = 100)

set.seed(123)

# Drift (Mean historical daily return)
mu <- colMeans(returns_matrix)

print("Starting Monte Carlo Simulation...")

# 4. Simulation Loop
for (i in 1:n_sims) {
  # Generate Z: n_assets x n_days
  Z <- matrix(rnorm(n_assets * n_days), nrow = n_assets, ncol = n_days)
  
  # Correlated shocks: L*Z
  shocks <- L %*% Z
  
  # Simulated Log returns (mu + shocks)
  sim_logret <- sweep(shocks, 1, mu, "+")
  
  # Convert to simple returns for portfolio arithmetic
  sim_simple <- exp(sim_logret) - 1
  
  # Portfolio daily simple return (daily rebalanced)
  port_simple <- as.vector(t(weights) %*% sim_simple)
  
  # Compound correctly
  cum_path <- initial_capital * cumprod(1 + port_simple)
  cum_path <- c(initial_capital, cum_path) # Add start point
  
  final_values[i] <- cum_path[length(cum_path)]
  
  if (i <= 100) {
    plot_paths[, i] <- cum_path
  }
}

# 5. VaR Calculation (Value at Risk)
# 5% as worst case
VaR_level <- 0.05
sorted_values <- sort(final_values)
cutoff_index <- floor(VaR_level * n_sims)
VaR_value <- sorted_values[cutoff_index]
VaR_loss <- initial_capital - VaR_value

print(paste("VaR 95% (1 Year):", round(VaR_loss, 2)))

# 6. Plotting with ggplot2
# Prepare data for ggplot
plot_df <- as.data.frame(plot_paths)
plot_df$Day <- 0:n_days
plot_df_long <- pivot_longer(plot_df, cols = -Day, names_to = "Simulation", values_to = "Value")

p_mc <- ggplot(plot_df_long, aes(x = Day, y = Value, group = Simulation)) +
  geom_line(alpha = 0.1, color = "blue") +
  geom_hline(yintercept = initial_capital, linetype = "solid", color = "black", size = 1) +
  geom_hline(yintercept = VaR_value, linetype = "dashed", color = "red", size = 1) +
  annotate("text", x = n_days, y = VaR_value, label = paste("VaR 95%:", round(VaR_value, 2)), vjust = -1, color = "red") +
  labs(title = "Monte Carlo Simulation (100 Trajectories via Cholesky)",
       subtitle = paste("VaR 95% (Potential Loss) =", round(VaR_loss, 2)),
       x = "Trading Days",
       y = "Portfolio Value (Base 100)") +
  theme_minimal()

ggsave("figures/monte_carlo_paths.png", plot = p_mc, width = 8, height = 6)

# Save results
saveRDS(list(VaR_95_loss = VaR_loss, L = L, final_values = final_values), "results/monte_carlo_results.rds")

