################################################################################
# FILE 05: VISUALIZATION AND TABLE PRODUCTION
# ============================================================================
# Description: This script generates the Figures and Tables
#
# Objectives:
# 1. Descriptive Statistics (Distribution Moments)
# 2. Sensitivity Analysis (Market Betas).
# 3. Regularization Visualization (Ridge Trace)
# 4. Efficient Frontier Illustration (Markowitz)
################################################################################

library(tidyverse)
library(ggplot2)
library(moments) # For Skewness (Asymmetry) and Kurtosis (Flattening) calculation
library(glmnet)  # For Ridge Trace
library(reshape2)

# Creating output directories if non-existent
if (!file.exists("figures")) dir.create("figures")
if (!file.exists("results")) dir.create("results")

# Loading cleaned data
clean_data <- readRDS("data/clean_data.rds")
tickers <- setdiff(names(clean_data), c("date", "Mkt_RF", "SMB", "HML", "RF"))

# ============================================================================
# 1. TABLE 1: DESCRIPTIVE STATISTICS
# ============================================================================
# Calculation of 4 statistical elements to validate normality hypothesis:
# - Mean (Expected Return)
# - Volatility (Standard Deviation, risk measure)
# - Skewness (Distribution bias, negative = crash risk)
# - Kurtosis (Distribution tails, high = Fat Tails)

cat("Generating Table 1: Descriptive Statistics...\n")

stats_df <- clean_data %>%
  select(all_of(tickers)) %>%
  summarise(across(everything(), list(
    Mean = ~ mean(.),
    Volatility = ~ sd(.),
    Skewness = ~ skewness(.),
    Kurtosis = ~ kurtosis(.)
  ))) %>%
  pivot_longer(cols = everything(), names_to = "Stat", values_to = "Value") %>%
  separate(Stat, into = c("Ticker", "Metric"), sep = "_") %>%
  pivot_wider(names_from = Metric, values_from = Value) %>%
  mutate(Sharpe = Mean / Volatility) %>%     # Simplified Sharpe Ratio (Rf=0 for relative comparison)
  filter(Ticker %in% c("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"))

# CSV Export
write_csv(stats_df, "results/tab1_desc_stats.csv")
print(stats_df)

# ============================================================================
# 2. FIGURE 1: MARKET BETA DISTRIBUTION (OLS)
# ============================================================================
# Visualization of systematic risk exposure
# Assets with Beta > 1 are "aggressive", Beta < 1 are "defensive"

cat("Generating Figure 1: Market Betas...\n")

betas <- numeric(length(tickers))
names(betas) <- tickers

for(tick in tickers) {
  # Simple regression R_i = alpha + beta * R_mkt + epsilon
  model <- lm(clean_data[[tick]] ~ clean_data$Mkt_RF)
  betas[tick] <- coef(model)[2]
}

beta_df <- data.frame(Ticker = names(betas), Beta = betas)

p_beta <- ggplot(beta_df, aes(x = reorder(Ticker, Beta), y = Beta, fill = Beta > 1)) +
  geom_bar(stat = "identity") +
  coord_flip() + # Horizontal bars for ticker readability
  scale_fill_manual(values = c("gray", "firebrick")) + # Red for Beta > 1 (Risky)
  labs(
    title = "Market Sensitivity (Beta)",
    subtitle = "OLS Estimation on daily data (2020-2025)",
    x = "Asset",
    y = "Market Beta"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("figures/fig1_market_betas.png", plot = p_beta, width = 8, height = 6)

# ============================================================================
# 3. FIGURE 2: RIDGE TRACE (COEFFICIENT CONVERGENCE)
# ============================================================================
# This shows L2 penalization effect on coefficients
# and how increasing Lambda reduces estimator variance

cat("Generating Figure 2: Ridge Trace...\n")

# Example on AAPL explained by Fama-French factors
y <- clean_data$AAPL
X <- clean_data %>% select(Mkt_RF, SMB, HML) %>% as.matrix()

ridge_fit <- glmnet(X, y, alpha = 0)

# Direct saving of base R plot
png("figures/fig2_ridge_trace.png", width = 800, height = 600)
plot(ridge_fit, xvar = "lambda", label = TRUE)
title("Ridge Trace: Coefficient Convergence to 0", line = 2.5)
dev.off()

# ============================================================================
# 4. FIGURE 3: EFFICIENT FRONTIER (SIMULATION)
# ============================================================================
# Visualization of Risk-Return trade-off by Monte Carlo simulation
# of Random Portfolios

cat("Generating Figure 3: Efficient Frontier (Simulation)...\n")

n_ports <- 5000
returns_mat <- clean_data %>% select(all_of(tickers))
mean_ret <- colMeans(returns_mat) * 252 # Annualization
cov_mat <- cov(returns_mat) * 252       # Annualization

port_returns <- numeric(n_ports)
port_risk <- numeric(n_ports)
sharpe_ratio <- numeric(n_ports)

set.seed(123)
for (i in 1:n_ports) {
  # Normalized random weights
  w <- runif(length(tickers))
  w <- w / sum(w)
  
  # Portfolio Return and Risk
  port_returns[i] <- sum(w * mean_ret)
  port_risk[i] <- sqrt(t(w) %*% cov_mat %*% w)
  sharpe_ratio[i] <- port_returns[i] / port_risk[i]
}

sim_df <- data.frame(Risk = port_risk, Return = port_returns, Sharpe = sharpe_ratio)

p_frontier <- ggplot(sim_df, aes(x = Risk, y = Return, color = Sharpe)) +
  geom_point(alpha = 0.5, size = 1) +
  scale_color_viridis_c() +
  labs(
    title = "Efficient Frontier (Simulation)",
    subtitle = "5000 Random Portfolios",
    x = "Risk (Annualized Volatility)",
    y = "Expected Return (Annualized)"
  ) +
  theme_minimal()

ggsave("figures/fig3_efficient_frontier_sim.png", plot = p_frontier, width = 8, height = 6)

cat("\nFigure generation completed.\n")
