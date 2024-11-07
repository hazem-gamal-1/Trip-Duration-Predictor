# Trip Duration Predictor

## Project Overview
The Trip Duration Predictor aims to estimate the duration of taxi trips based on various features, including geographic coordinates, time of day, and other contextual information. Utilizing machine learning models, specifically Ridge regression, the project analyzes historical trip data to identify patterns and predict future trip durations accurately.

## Table of Contents
- [Key Insights](#key-insights)
- [Data Preparation](#data-preparation)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)

## Key Insights
1. **Distribution of Trip Durations**: Trip durations are highly skewed, with a long tail indicating outliers due to unusual conditions.
2. **Time of Day Variability**: Shorter trip durations during peak traffic hours contrast with longer durations during off-peak hours.
3. **Day of the Week Impact**: Weekdays show shorter trip durations compared to weekends, attributed to routine trips.
4. **Distance vs. Duration Correlation**: A strong positive correlation exists between trip duration and distance traveled.
5. **Pickup and Dropoff Location Effects**: Central locations often lead to shorter distances but may have longer durations due to congestion.
6. **Route Patterns**: Frequently traveled routes show consistent trip durations, while lesser-used routes exhibit greater variability.
7. **Passenger Count Relationship**: No significant relationship between the number of passengers and trip duration was observed.
8. **Hourly Trip Duration Variation**: Early morning trips are faster, while peak hours see longer durations due to congestion.
9. **Outlier Analysis**: The dataset includes outliers representing unusual trip conditions, meriting further investigation.

## Data Preparation
The data is loaded from CSV files, and several preprocessing steps are performed, including feature extraction and engineering to enrich the dataset with relevant features such as distance, time-based features, and more.

## Feature Engineering
Features are created using geographical data (latitude and longitude) to calculate:
- Haversine distance
- Bearing between pickup and dropoff locations
- Dummy Manhattan distance

Time features are also extracted, including the day of the month, day of the week, month, hour, and day of the year.

## Modeling
Ridge regression is implemented as the primary modeling technique, with hyperparameters specified for optimal performance.

## Evaluation
The model is evaluated using R² and RMSE metrics to assess its performance on training and testing datasets.

## Results
| Metric            | Training R²          | Training RMSE         | Validation R²        | Validation RMSE      |
|-------------------|----------------------|-----------------------|-----------------------|----------------------|
| **Values**        | 0.6315               | 0.4824                | 0.6312                | 0.4858               |
