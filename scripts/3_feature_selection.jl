# SFU ECON832 Final
# Spring 2024
# Analysis of feature selection for the final project

# Preliminaries -------------------------------------------------------------------

# Activate the package environment

using Pkg

Pkg.activate("final_env")

# Load packages

using TidierData
using DataFrames
using TidierFiles
using TidierPlots
using HypothesisTests

# Load data

# Cleaned calibration data with all features

calibration_cleaned = @chain read_csv("data/output/cleaned_calibration.csv") begin
        @clean_names()
end

# Data preparation -------------------------------------------------------------------

# Quick data preparation for the feature selection analysis

# Filter for experiment 1 data 

df = @chain calibration_cleaned begin
    @filter(experiment_1 == 1)
    @mutate(choice = if_else(b == 1, "B", "A"))
end 

dropmissing(df)

# Analysis of risk features -------------------------------------------------------------------

# Correlate risk features with the outcome variable to observe the relationship and justify their inclusion in the DFNN

## Expected values ---------------------------------------------------------------

cor(df.hb, df.b)

cor(df.ha, df.b)

# Do a scatter plot of the risk features against the outcome variable (using TidierPlots)

ggplot(df) +
    geom_boxplot(@aes(x = choice, y = hb)) +
    labs(title = "Boxplot of E[B] against Having chosen B",
         x = "Choice of lottery B or A",
         y = "E[B]")

# No particular preference if we don't control for any other variables. No reason to include these features in the DFNN unless they work with other variables.

# Number of outcomes of the lotteries ---------------------------------------------------------------

# Correlate number of lottery outcomes with the outcome variable to observe the relationship and justify their inclusion in the DFNN

cor(df.lotnumb, df.b) # Stronger

cor(df.lotnuma, df.b) # Negative, small number

# Perform correlation test 

pvalue(CorrelationTest(df.lotnumb, df.b)) # Statistically significant

pvalue(CorrelationTest(df.lotnuma, df.b)) # Statistically significant

# Do a scatter plot of the number of lottery outcomes of b against the outcome variable (using TidierPlots)

ggplot(df) +
    geom_boxplot(@aes(x = choice, y = lotnumb)) +
    labs(title = "Boxplot of Number of Outcomes of B against Having chosen B",
         x = "Choice of lottery B or A",
         y = "Number of Outcomes of B")

# Correlation between lotteries ---------------------------------------------------------------

# Correlate the correlation between lotteries with the outcome variable to observe the relationship and justify their inclusion in the DFNN

cor(df.corr, df.b) 

# Perform correlation test

pvalue(CorrelationTest(df.corr, df.b)) # Statistically significant

# Do a scatter plot of the correlation between lotteries against the outcome variable (using TidierPlots)

obs_by_corr = 
@chain df begin
    @group_by(choice, corr)
    @summarize(number = n())
    @mutate(corr = as_string(corr))
    @ungroup()
end

ggplot(obs_by_corr) +
    geom_col(@aes(x = corr, y = number, colour = choice, group = choice), position = "dodge", stat = "identity") +
    labs(title = "Number of Observations by Correlation between Lotteries",
         x = "Correlation between Lotteries",
         y = "Number of Observations") +
    scale_colour_manual(values = ["#F8766D", "#00BFC4"])
