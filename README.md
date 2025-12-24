# Premier League Match Predictor

A python-based machine learning project that predicts the result of a Premier League match using the Random Forest Classifier.

## Overview

This project takes data from the current season, processes it, and predicts future match results. The workflow includes:

1.  **Data Cleaning**: parsing match results and cleaning missing values from `pl2526.csv`.
2.  **Feature Engineering**: creating insightful features like:
    *   Recent form (Last 3, 5, and 10 games)
    *   Points and goal averages
    *   Point difference
    *   Home advantage
    *   Win rates
3.  **Model Training**: splitting the data into training and test sets (30% test / 70% Train) and training a **Random Forest Classifier**.

## Installation

1.  Make sure you have Python installed.
2.  Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to use

Run the predictor script:

```bash
python pl_predictor.py
```

Follow the on-screen prompts to enter the **Home Team** and **Away Team** (e.g., `Arsenal`, `Man Utd`) to see the predicted outcome (H -> Home team wins / D -> Draw / A -> Away team wins), probabilities, and comparison charts.
