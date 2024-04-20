# SFU ECON832 Final
# Spring 2024
# Data preparation for deep feedforward neural networks (DFNNs)

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
# Categorical variables should be made into binary variables

cleaned_calibration_df = @chain calibration_raw begin
    @mutate(subjid = as_string(subjid),
            location_rehovot = if_else(location == "Rehovot", 1, 0),
            gender_female = if_else(gender == "F", 1, 0))
end

# Exporting -------------------------------------------------------------------

# Export the cleaned calibration data

write_csv(cleaned_calibration_df, "data/output/cleaned_calibration.csv")
