# Estimating the Impact of 401(k) Participation on Financial Assets Using Double Machine Learning Methods

This repository contains the code and analysis for estimating the treatment effect of 401(k) eligibility and participation on net total financial assets (net_tfa) using Double Machine Learning (DoubleML) techniques. The analysis utilizes various machine learning models to ensure robustness and reliability of the treatment effect estimation.

## Abstract
This study employs Double Machine Learning (DoubleML) techniques to estimate the treatment effect of 401(k) eligibility and participation on net total financial assets (net_tfa). Utilizing the 401(k) dataset, we prepare the data by ensuring appropriate data types and performing exploratory data analysis. Various machine learning models, including Lasso, Random Forest, Decision Trees, XGBoost, and HistGradientBoosting, are used to fit DoubleML models such as DoubleMLPLR, DoubleMLIRM, and DoubleMLIIVM. Our findings indicate that the treatment effect estimated using HistGradientBoosting is significantly stronger compared to other models. This can be attributed to its superior model fit, balanced bias-variance trade-off, robustness to noise and outliers, enhanced flexibility, and effective interaction modeling. The results highlight the importance of selecting appropriate models for reliable treatment effect estimation and provide valuable insights into the impact of 401(k) participation on financial wealth.

## Introduction
This repository details the analysis conducted using Double Machine Learning (DoubleML) methods to estimate the treatment effect of 401(k) eligibility and participation on net total financial assets (net_tfa). Various machine learning models were employed to understand the robustness and reliability of the treatment effect estimation.

## Summary of the Code

### Setting up the Environment
- Import necessary libraries such as `numpy`, `pandas`, `doubleml`, `sklearn`, `xgboost`, `matplotlib`, and `seaborn`.
- Configure the plotting environment using `seaborn`.

### Fetching and Preparing the Data
- Load the 401(k) dataset using `fetch_401K` from the `doubleml` package.
- Ensure the correct data types for certain columns (`nifa`, `net_tfa`, `tw`, `inc`).

### Exploratory Data Analysis (EDA)
- Display the first few rows and descriptive statistics of the dataset.
- Create and save bar plots for 401(k) eligibility (`e401`) and participation (`p401`).
- Create and save KDE plots for `net_tfa` distribution based on `e401`.
- Calculate the unconditional average predictive effect (APE) of 401(k) eligibility on accumulated assets. The unconditional APE of `e401` is about 19559. Among the 3682 individuals that are eligible, 2594 decided to participate in the program. The unconditional APE of `p401` is about 27372.

### Preparing the Data for DoubleML Analysis
- Define basic and flexible feature sets for the analysis.
- Create `DoubleMLData` objects for both basic and flexible feature sets, including IV specifications for 401(k) participation.

### Defining Machine Learning Models
Set up different machine learning models, including Lasso, Random Forest, Decision Trees, XGBoost, and HistGradientBoosting.

### DoubleML Model Fitting and Summary Collection
- Fit `DoubleMLPLR`, `DoubleMLIRM`, and `DoubleMLIIVM` models using the specified machine learning models.
- Collect and store summaries of each model fit.

### Plotting the Results
- Create and save error bar plots for the estimated coefficients and confidence intervals of each model type (`PLR`, `IRM`, `IIVM`).
- Combine summaries into a single DataFrame and plot them together for a comparative overview.

### Explanation of HistGradientBoosting's Stronger Treatment Effect
- Discuss potential reasons why HistGradientBoosting shows a stronger estimated treatment effect compared to other models.
- Provide a comprehensive explanation suitable for inclusion in a report.

## Data Preparation and Exploration

The 401(k) dataset provides detailed information on individuals' financial status and 401(k) participation. After loading the dataset, appropriate data types for crucial columns were ensured to facilitate accurate analysis.

### Exploratory Data Analysis (EDA)
To gain initial insights, exploratory data analysis was performed:
- **Bar Plots**: Visualize the distribution of 401(k) eligibility (`e401`) and participation (`p401`).
- **KDE Plots**: Plot the kernel density estimates of `net_tfa` based on `e401` to understand the distribution of financial assets among eligible and non-eligible individuals.

### Data Preparation for DoubleML
The treatment effect of `e401` on net total financial assets was analyzed using the following model:
\[ Y = D\alpha + f(X)'\beta + \epsilon \]
where \( f(X) \) is a dictionary applied to the raw regressors. \( X \) contains variables on marital status, two-earner status, defined benefit pension status, IRA participation, home ownership, family size, education, age, and income.

Two sets of features were defined:
- **Basic Features**: Age, income, education, family size, marital status, dual earners, defined benefit plan, personal IRA, and home ownership.
- **Flexible Features**: Interaction terms and orthogonal polynomial features of degree 2 for age, income, education, and family size.

Using these features, `DoubleMLData` objects were created for both basic and flexible feature sets, including IV specifications to account for 401(k) participation.

### Machine Learning Models
Several machine learning models were employed to fit the DoubleML methods:
- Lasso (L1 regularization)
- Random Forest
- Decision Trees
- XGBoost
- HistGradientBoosting

### DoubleML Model Fitting and Summaries
`DoubleMLPLR`, `DoubleMLIRM`, and `DoubleMLIIVM` models were fitted using the specified machine learning models, and their summaries were collected. These models estimate the treatment effect while accounting for potential confounders.

## Results
The estimated treatment effects and their confidence intervals were plotted for each model type. The plots provide a visual representation of the coefficients and their 95% confidence intervals, highlighting the differences across models.

## Explanation of HistGradientBoosting's Stronger Treatment Effect
The analysis shows that the treatment effect estimated using HistGradientBoosting is much stronger compared to other ML methods due to several factors:
1. **Superior Model Fit**: Captures underlying patterns in the data more effectively.
2. **Balanced Bias-Variance Trade-off**: Reduces both overfitting and underfitting.
3. **Robustness to Noise and Outliers**: Minimizes impact of outliers and noise.
4. **Enhanced Flexibility**: Greater flexibility in capturing complex relationships.
5. **Effective Interaction Modeling**: Automatically captures interaction terms more effectively.

## Conclusion
The DoubleML analysis, enhanced by various machine learning models, provides a comprehensive understanding of the treatment effect of 401(k) participation. Among the models, HistGradientBoosting stands out for its robust estimation, highlighting its potential in capturing complex relationships in the data. This analysis underscores the importance of selecting appropriate models for reliable treatment effect estimation.

## Files in the Repository
- `401k_analysis.ipynb`: Jupyter notebook containing the complete analysis.
- `images/`: Directory containing the generated plots.
- `README.md`: This readme file.

## Requirements
To run the analysis, ensure you have the following packages installed:
- numpy
- pandas
- doubleml
- scikit-learn
- xgboost
- matplotlib
- seaborn

## Usage
1. Clone the repository.
2. Navigate to the directory.
3. Open `401k_analysis.ipynb` in Jupyter Notebook.
4. Run the notebook cells to reproduce the analysis.
