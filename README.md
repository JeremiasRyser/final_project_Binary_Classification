# final_project_Binary-Classification-of-Insurance-Cross-Selling

This repository contains code and resources for the Kaggle Playground Series competition on predicting customer responses to automobile insurance offers.

## Overview
Welcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: The objective of this competition is to predict which customers respond positively to an automobile insurance offer.

Link: https://www.kaggle.com/competitions/playground-series-s4e7

## Data
Dataset Description
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Health Insurance Cross Sell Prediction Data dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Variables description

Feature Name	Type	Description
id	(continous)	Unique identifier for the Customer.
Age	(continous)	Age of the Customer.
Gender	(dichotomous)	Gender of the Customer.
Driving_License	(dichotomous)	0 for customer not having DL, 1 for customer having DL.
Region_Code	(nominal)	Unique code for the region of the customer.
Previously_Insured	(dichotomous)	0 for customer not having vehicle insurance, 1 for customer having vehicle insurance.
Vehicle_Age	(nominal)	Age of the vehicle.
Vehicle_Damage	(dichotomous)	Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past.
Annual_Premium	(continous)	The amount customer needs to pay as premium in the year.
Policy_Sales_Channel	(nominal)	Anonymized Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.
Vintage	(continous)	Number of Days, Customer has been associated with the company.
Response (Dependent Feature)	(dichotomous)	1 for Customer is interested, 0 for Customer is not interested.

## Files
- train.csv - the training dataset; Response is the binary target
- test.csv - the test dataset; your objective is to predict the probability of Response for each row
- sample_submission.csv - a sample submission file in the correct format


## Project Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for data exploration, cleaning, feature engineering, and modeling.
- `scripts/`: Python scripts for data preparation, model training, and evaluation.
- `streamlit_app/`: Streamlit app for interactive data and model visualization.
- `submissions/`: Submission files for Kaggle.
- `reports/`: Final report and documentation.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/JeremiasRyser/final_project_Binary-Classification-of-Insurance-Cross-Selling.git

