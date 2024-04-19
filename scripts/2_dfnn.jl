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
    @select(subjid, location, gender, age, b)
end

@glimpse(simple_dfnn_df)

# Simple DFNN -------------------------------------------------------------------

# Train a DFNN with only demographic variables: location, gender, age
# Outcome variable is b