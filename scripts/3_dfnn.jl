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

# Load data

# Calibration data

calibration_cleaned = @chain read_csv("data/output/cleaned_calibration.csv") begin
        @clean_names()
    end

@glimpse(calibration_cleaned)

# Data preparation -------------------------------------------------------------------

# Select relevant variables for models

# Simple DFNN (only demographic variables for set 1)

simple_dfnn_df = @chain calibration_cleaned begin
    @filter(set == 1)
    @select(location_rehovot, gender_female, age, b)
end

@glimpse(simple_dfnn_df)

# Simple DFNN -------------------------------------------------------------------

# Train a DFNN with only demographic variables: location, gender, age
# Outcome variable is b

## Feature engineering -------------------------------------------------------------------

# Need to prepare the data for the model
# 0. Separate the data into features (X) and outcome (Y)
# 1. Normalize features
# 2. Transpose the matrix of features to have observations as columns

# Separate data in X and Y

features = Matrix(simple_dfnn_df[:, Not(:b)])

outcome = simple_dfnn_df.b # No need to hot encode the outcome variable since it is binary

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

opt = ADAM(0.01)

# Train model 

Flux.train!(loss, Flux.params(model), data, opt)
