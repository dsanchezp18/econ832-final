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
using Plots

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

# Descriptive statistics -------------------------------------------------------------------

# Get the percentage of choices for each lottery -- this will be used to determine the baseline accuracy

df_choice = @chain df begin
    @group_by(choice)
    @summarize(number = n())
    @mutate(percentage = number / sum(number) * 100)
    @ungroup()
end

# Get the percentage within each location

df_choice_by_set = @chain df begin
    @group_by(location_rehovot, choice)
    @summarize(number = n())
    @mutate(percentage = number / sum(number) * 100)
    @ungroup()
end

# Get the percentage by gender

df_choice_by_gender = @chain df begin
    @group_by(gender, choice)
    @summarize(number = n())
    @mutate(percentage = number / sum(number) * 100)
    @ungroup()
end

# Percentages by set 

df_choice_by_set = @chain df begin
    @group_by(choice, set)
    @summarize(number = n())
    @mutate(percentage = number / sum(number) * 100)
    @ungroup()
end

# Analysis of risk features -------------------------------------------------------------------

# Correlate risk features with the outcome variable to observe the relationship and justify their inclusion in the DFNN

## Expected values ---------------------------------------------------------------

# Correlate expected values with the outcome variable to observe the relationship and justify their inclusion in the DFNN

cor(df.hb, df.b)

cor(df.ha, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.hb, df.b)) # Not statistically significant

pvalue(CorrelationTest(df.ha, df.b)) # Statistically significant

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

lotnumb_boxplot =
    ggplot(df) +
    geom_boxplot(@aes(x = choice, y = lotnumb)) +
    labs(title = "Choice of lottery against number of B lottery outcomes",
         x = "Choice of lottery B or A",
         y = "Number of Outcomes of B")

lotnumb_boxplot

ggsave(lotnumb_boxplot, "figures/lotnumb_boxplot.png", scale = 1.5)

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

## Ambiguity -------------------------------------------------------------------

# Correlate ambiguity with the outcome variable to observe the relationship and justify its inclusion in the DFNN

cor(df.amb, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.amb, df.b)) # Not statistically significant

## Probabilities -------------------------------------------------------------------

# Correlate probabilities with the outcome variable to observe the relationship and justify their inclusion in the DFNN

cor(df.p_hb, df.b)

cor(df.p_ha, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.p_hb, df.b)) # Not statistically significant

pvalue(CorrelationTest(df.p_ha, df.b)) # Not statistically significant

## Low payoffs -------------------------------------------------------------------

# Correlate low payoffs with the outcome variable to observe the relationship and justify their inclusion in the DFNN

cor(df.lb, df.b)

cor(df.la, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.lb, df.b)) # Statistically significant

pvalue(CorrelationTest(df.la, df.b)) # Statistically significant

# Do a scatter plot of the low payoffs against the outcome variable (using TidierPlots)

ggplot(df) +
    geom_boxplot(@aes(x = choice, y = lb)) +
    labs(title = "Boxplot of Low Payoffs of B against Having chosen B",
         x = "Choice of lottery B or A",
         y = "Low Payoffs of B")

## Shape of the distribution -------------------------------------------------------------------

# Bar plot of the shape of the distribution of the B lottery against the outcome variable

df_by_lotshape  = 
@chain df begin
    @group_by(choice, lotshapeb)
    @summarize(number = n())
    @ungroup()
end

barplot_by_lotshape =
ggplot(df_by_lotshape) +
    geom_col(@aes(x = lotshapeb, y = number, colour = choice), position = "dodge") + 
    labs(x = "Shape of the Distribution of B",
         y = "Number of Observations") +
    scale_colour_manual(values = ["#F8766D", "#00BFC4"])

barplot_by_lotshape

ggsave(barplot_by_lotshape, "figures/barplot_by_lotshape.png", scale = 1.2)

# A symmetric distribution greatly favours B

## Payoff features -------------------------------------------------------------------

# Correlate the payoff the subject got from the the choice in the current trial

cor(df.payoff, df.payoff)

CorrelationTest(df.b, df.payoff)

# The Payoff provided by B

cor(df.b, df.bpay)

CorrelationTest(df.b, df.bpay)

# The Payoff provided by A

cor(df.b, df.apay)

CorrelationTest(df.b, df.apay)

# Do a boxplot of the payoff obtained by the subject against the choice of lottery

payoff_boxplot = 
ggplot(df) +
    geom_boxplot(@aes(x = choice, y = payoff)) +
    labs(title = "Boxplot of Payoff against Choice of Lottery",
         x = "Choice of lottery B or A",
         y = "Payoff")

payoff_boxplot

# Payoff of b against the outcome variable

payoff_b_boxplot =
ggplot(df) +
    geom_boxplot(@aes(x = choice, y = bpay)) +
    labs(title = "Boxplot of Payoff of B against Having chosen B",
         x = "Choice of lottery B or A",
         y = "Payoff of B")

payoff_b_boxplot