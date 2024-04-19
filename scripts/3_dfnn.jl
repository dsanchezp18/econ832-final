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

calibration_raw = @chain read_csv("data/calibration/All estimation raw data.csv") begin
        @clean_names()
    end

@glimpse(calibration_raw)

# Data preparation -------------------------------------------------------------------

# Select relevant variables for models

# Simple DFNN (only demographic variables for set 1)

simple_dfnn_df = @chain calibration_raw begin
    @filter(set == 1)
    @select(location, gender, age, b)
end

@glimpse(simple_dfnn_df)

# Simple DFNN -------------------------------------------------------------------

# Train a DFNN with only demographic variables: location, gender, age
# Outcome variable is b

## Feature engineering -------------------------------------------------------------------

# Need to prepare the data for the model
# 0. Separate the data into features (X) and outcome (Y)
# 1. One hot encode categorical variables
# 2. Normalize continuous variables
# 3. Transpose the matrix of features to have observations as columns

# Separate data in X and Y

features = Matrix(simple_dfnn_df[:, Not(:b)])

outcome = simple_dfnn_df.b

# One hot encode categorical variables

categorical_features_df = @chain simple_dfnn_df[:, Not(:b)] begin
    @select(where(is_string))
end

categorical_features = Matrix(categorical_features_df)

one_hot_features = Flux.onehotbatch(categorical_features[:, 1], unique(categorical_features[:,1]))

# Standardize continuous variables, mean of 0 and sd 1

continuous_features_df = @chain simple_dfnn_df[:, Not(:b)] begin
    @select(where(is_number))
end

normalized_features = Flux.normalise(Matrix(continuous_features_df), dims = 2)

# Join the one hot encoded and normalized features

X = hcat(one_hot_features, normalized_features)

# Trnaspose the matrix of features



## Model training -------------------------------------------------------------------

# Define your model

model = Flux.Chain(
    Dense(3, 64, Flux.relu),
    Dense(64, 64, Flux.relu),
    Dense(64, 1, Flux.sigmoid)
)

# Define your loss function based on MSE

loss(X, Y) = Flux.mse(model(X), Y)

# Define your optimizer

opt = ADAM(0.01)

# Feature engineering
# Assuming that simple_dfnn_df is your DataFrame and :location, :gender, :age are your input columns and :b is your output column
X = transpose(Matrix(simple_dfnn_df[!, [:location, :gender, :age]]))
Y = simple_dfnn_df.b

data = [(X, Y)]

# Train model 

train!(loss, Flux.params(model), data, opt)
