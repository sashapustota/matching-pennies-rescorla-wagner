# Rescorla Wagner for the Matching Pennies task

This repository contains code and resources for modelling the dynamics of a matching-pennies game using the Rescorla-Wagner model, implemented with Stan and R. 

## Overview
This project implements and validates the Rescorla-Wagner (RW) model, a widely used theory in associative learning, within the context of the matching pennies task. The RW model explains how individuals update their expectations based on prediction errors, and it's applied here to analyze competitive interactions in the matching pennies game.

## Contents
- **RW Model Implementation**: The RW model is implemented in Stan (with R). The R file is called `rescorla_wagner.rmd` and the Stan file is called `rescorla_wagner.stan`.
- **Data Handling**: The project uses simulated data and includes data generation and manipulation required for model fitting and validation.
- **Model Validation**: Model quality is assessed using various techniques, including predictive checks and inspection of R-hat values and trace plots.
