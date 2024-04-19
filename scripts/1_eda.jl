# SFU ECON832 Final
# Spring 2024
# Julia Project Setup Script

# Preliminaries -------------------------------------------------------------------

# Activate the package environment

using Pkg

Pkg.activate("final_env")

# Load packages

using TidierData
using DataFrames
using TidierFiles

# Load data 

# Calibration data

calibration_raw = @chain read_csv("data/calibration/All estimation raw data.csv") begin
        @clean_names()
        @arrange(set)
    end

# Exploratory data analysis (EDA) -------------------------------------------------------

# Glimpse

@glimpse(calibration_raw)

# How many observations (rows) per subjectid 

obs_per_subject = @chain calibration_raw begin
    @group_by(subjid)
    @summarize(n_obs = n())
    @arrange(n_obs)
end

# Count how many sets per subjid (should be just 1)

sets_per_subject = @chain calibration_raw begin
    @group_by(subjid, set)
    @summarize(n_sets = n())
    @ungroup()
    @select(-n_sets)
    @group_by(subjid)
    @summarize(n_sets = n())
end

# Get set 1 data 

set1_df = @chain calibration_raw begin
    @filter(set == 1)
end

# Unique values of the GameID variable

@chain set1_df begin
    @select(gameid) 
    @distinct()
    @arrange(gameid)
end