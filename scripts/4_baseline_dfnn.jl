# SFU ECON832 Final
# Spring 2024
# Script to train the deep feedforward neural network (DFNN)

# Preliminaries -------------------------------------------------------------------

# Activate the package environment

using Pkg

Pkg.activate("final_env")

# Load packages

using TidierData
using DataFrames
using TidierFiles
using Flux
using MLBase
using Random
using LinearAlgebra
using Parameters: @with_kw

# Load data

# Calibration data (clean)

calibration_cleaned = @chain read_csv("data/output/cleaned_calibration.csv") begin
        @clean_names()
    end

@glimpse(calibration_cleaned)

# Data -------------------------------------------------------------------

## Selecting variables -------------------------------------------------------------------

# Select the relevant variables related to lotteries

df = @chain calibration_cleaned begin
    @filter(experiment_1 == 1)
    @select(subjid, location_rehovot, gender_female, age, # Demographics
            shape_a_symm, shape_a_rskew, shape_a_lskew, shape_b_symm, shape_b_rskew, shape_b_lskew, # Lottery shapes
            lotnuma, lotnumb, p_ha, p_hb, ha, hb, amb, corr, trial, button,  # Other variables related to the lottery
            payoff, forgone, apay, bpay, # payoff variables
            b )# Outcome variable
end

## Test-train split -------------------------------------------------------------------

# Test-train split of 80-20

Random.seed!(593)

# Use slice_sample with prop = 0.8 to produce the training data

df_train = @chain df begin
    @slice_sample(prop = 0.8, replace = false)
end

# Use an antijoin to produce the testing data

df_test = @chain df begin
    @anti_join(df_train)
end

# Verify the number of observations in the training and testing data (should comply the 80-20 rule)

nrow(df_train)/nrow(df) 

nrow(df_test)/nrow(df)

# DFNN -------------------------------------------------------------------

# Train a DFNN with risk preference variables and demographic variables
# Outcome variable is b

## Feature engineering -------------------------------------------------------------------

# Need to prepare the data for the model
# 1. Separate features from the outcome
# 2. Normalize features (mean 0, sd 1)
# 3. Transpose the matrix of features to have observations as columns

@with_kw mutable struct Args
    lr::Float64 = 0.5
end

# Separate data in X and Y

features = Matrix(train_simple_dfnn[:, Not(:b)])

outcome = train_simple_dfnn.b # No need to hot encode the outcome variable since it is binary

# Standardize continuous variables, mean of 0 and sd 1

X = transpose(Flux.normalise(features, dims = 2))

# Transpose the outcome vector

Y = transpose(outcome)

# Data

data = [(X, Y)]

## Model training -------------------------------------------------------------------

# Define your model

model = Flux.Chain(
    Dense(3, 64, Flux.relu),
    Dense(64, 64, Flux.relu),
    Dense(64, 1, Flux.sigmoid)
)

# Define loss function based on MSE

loss(X, Y) = Flux.mse(model(X), Y)

# Define optimizer

opt = ADAM(0.001, (0.9, 0.8))

# Train model 

Flux.train!(loss, Flux.params(model), data, opt)

# Model evaluation -------------------------------------------------------------------

# Loss

loss(X, Y)

# Predictions

predictions_train = model(X)

# Convert predictions to binary

predictions_train_binary = predictions_train .> 0.5

# Confusion matrix# Convert boolean arrays to integer arrays with class labels starting from 1
Y_int = Int.(Y .> 0.5) .+ 1
predictions_train_binary_int = Int.(predictions_train_binary) .+ 1

# Calculate confusion matrix

confusion_matrix_train = confusmat(2, vec(Y_int), vec(predictions_train_binary_int))

# Accuracy

accuracy_train = sum(diag(confusion_matrix_train)) / sum(confusion_matrix_train)

# Testing the model -------------------------------------------------------------------

# Extract X and Y from calibration_test

features_test = Matrix(test_simple_dfnn[:, Not(:b)])

outcome_test = test_simple_dfnn.b

X_test = transpose(Flux.normalise(features_test, dims = 2))

Y_test = transpose(outcome_test)

# Execute the model on the test data

loss(X_test, Y_test)

# Compute the confusion matrix for the test data

predictions_test = model(X_test)

predictions_test_binary = predictions_test .> 0.5

Y_test_int = Int.(Y_test .> 0.5) .+ 1

predictions_test_binary_int = Int.(predictions_test_binary) .+ 1

confusion_matrix_test = confusmat(2, vec(Y_test_int), vec(predictions_test_binary_int))

accuracy_test = sum(diag(confusion_matrix_test)) / sum(confusion_matrix_test)