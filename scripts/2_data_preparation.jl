# SFU ECON832 Final
# Spring 2024
# Data preparation for deep feedforward neural networks (DFNNs) (feature engineering)

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

# Prepare data for DFNNs