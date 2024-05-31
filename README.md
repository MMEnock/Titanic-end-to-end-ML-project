# Titanic Survival Prediction

This project predicts the survival of passengers on the Titanic using machine learning techniques. The dataset is provided by Kaggle's "Titanic - Machine Learning from Disaster" competition.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Model Used](#model-used)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [Conclusion](#conclusion)
11. [Contributing](#contributing)
12. [License](#license)

## Introduction
This project aims to predict the survival of passengers aboard the Titanic using the Random Forest Regressor. The model analyzes various features such as age, gender, and ticket class to make predictions.

## Dataset
The dataset is obtained from Kaggle and includes information about Titanic passengers.

- **Training set:** Used to train the model.
- **Test set:** Used to evaluate the model's performance.

For more details, visit the [Kaggle Titanic competition page](https://www.kaggle.com/c/titanic).

## Features
The dataset includes the following features:
- **PassengerId:** Unique ID for each passenger
- **Survived:** Survival (0 = No, 1 = Yes)
- **Pclass:** Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name:** Name of the passenger
- **Sex:** Gender of the passenger
- **Age:** Age of the passenger
- **SibSp:** Number of siblings/spouses aboard the Titanic
- **Parch:** Number of parents/children aboard the Titanic
- **Ticket:** Ticket number
- **Fare:** Passenger fare
- **Cabin:** Cabin number
- **Embarked:** Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Project Structure
The project directory is structured as follows:
- `data/`: Contains the dataset files
- `notebooks/`: Jupyter notebooks with data analysis and model training
- `src/`: Source code for data processing, feature engineering, and model implementation
- `models/`: Saved models
- `results/`: Evaluation results and visualizations
- `README.md`: Project documentation

## Installation
To run this project, you need Python 3.x and the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/MMEnock/Titanic-end-to-end-ML-project.git
cd titanic-survival-prediction
```

2. Navigate to the `notebooks/` directory and open the Jupyter notebooks to explore data analysis and model training:
```bash
jupyter notebook
```

3. Run the scripts in the `src/` directory to preprocess data, train models, and evaluate results.

## Model Used
The primary model used in this project is the **Random Forest Regressor**, which is effective for handling complex datasets and capturing non-linear relationships.

## Evaluation
The model was evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Results
The Random Forest Regressor achieved an accuracy of 78% on the test set. Detailed results and evaluation metrics can be found in the `Jupyter notebook/` directory.

## Conclusion
This project demonstrates the application of the Random Forest Regressor to predict survival on the Titanic. It highlights the importance of feature selection, data preprocessing, and model evaluation in building an effective predictive model.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify the above README file as per your specific project details and requirements.
