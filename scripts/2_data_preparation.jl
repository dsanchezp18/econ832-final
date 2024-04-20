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
            gender_female = if_else(gender == "F", 1, 0),
            set_1 = if_else(set == 1, 1, 0),
            set_2 = if_else(set == 2, 1, 0),
            set_3 = if_else(set == 3, 1, 0),
            set_4 = if_else(set == 4, 1, 0),
            set_5 = if_else(set == 5, 1, 0),
            set_6 = if_else(set == 6, 1, 0),
            experiment_1 = if_else(set in (5,6), 1, 0), #CPC18 experiment using sets 5 and 6
            button_A = if_else(button == "A", 1, 0), # Potential attention check
            shape_A_symm = if_else(lotshapea = "Symm", 1, 0), # Lot shape A Symm
            shape_B_symm = if_else(lotshapeb = "Symm", 1, 0), # Lot shape B Symm
            shape_A_rskew = if_else(lotshapea = "R-skew", 1, 0), # Lot shape A R-skew
            shape_B_rskew = if_else(lotshapeb = "R-skew", 1, 0), # Lot shape B R-skew
            shape_A_lskew = if_else(lotshapea = "L-skew", 1, 0), # Lot shape A L-skew
            shape_B_lskew = if_else(lotshapeb = "L-skew", 1, 0), # Lot shape B L-skew
    )
end

# Exporting -------------------------------------------------------------------

# Export the cleaned calibration data

write_csv(cleaned_calibration_df, "data/output/cleaned_calibration.csv")
