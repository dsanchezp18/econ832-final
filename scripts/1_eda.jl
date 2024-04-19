# SFU ECON832 Final
# Spring 2024
# Julia Project Setup Script

# Preliminaries -------------------------------------------------------------------

# Load packages

using TidierData
using TidierFiles

# Load data 

# Calibration data

calibration_raw = read_csv("data/calibration/All estimation raw data.csv")

# Exploratory data analysis (EDA) -------------------------------------------------------

# Glimpse

@glimpse(calibration_raw)