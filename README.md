# Obesity Risk Prediction Project

## Project Summary

This project uses a dataset from **Kaggle Playground Series S4E2** to predict the risk of obesity. The goal is to classify individuals into different categories of obesity risk based on their age, gender, physical activity, and eating habits.

### Project Structure

- `src/obesity_risk.py`
- `notebook/Obesity_Risk.ipynb`


### Target Variable: **NObeyesdad**
- **Insufficient Weight**: 0
- **Normal Weight**: 1
- **Obesity Type I**: 2
- **Obesity Type II**: 3
- **Obesity Type III**: 4
- **Overweight Level I**: 5
- **Overweight Level II**: 6

## Dataset

The dataset includes both numerical and categorical features:
- **Numerical**: Age, Height, Weight, BMI
- **Categorical**: Gender, Transportation Method, Physical Activity

## Project Steps

1. **Data Preprocessing and Analysis**: The dataset is loaded, cleaned, and checked for missing values. Target labels are transformed into numbers.
2. **Feature Engineering**: New features are created based on the existing data. For example, BMI is calculated and added to the dataset.
3. **Modeling and Evaluation**:
   - The data is split into training and testing sets. Various classifiers (KNN, CART, Random Forest, GBM, LightGBM, CatBoost) are used to make predictions.
   - Models are evaluated using metrics like accuracy, precision, recall, and F1-score.
4. **Hyperparameter Tuning**: The best parameters for each model are found using RandomizedSearchCV.

## Classifiers Used
- **KNN** (K-Nearest Neighbors)
- **SVC** (Support Vector Classifier)
- **CART** (Decision Tree)
- **RF** (Random Forest)
- **GBM** (Gradient Boosting Machine)
- **XGBoost**
- **LightGBM**
- **CatBoost**

## Model Performance

The models are evaluated using different metrics. After tuning the parameters, the best model was **LightGBM**. It performed the best compared to other models.

### Metrics Used:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/aysecnkci/ObesityRisk-ML-Modeling.git


2. Navigate to the `src` directory and run the `obesity_risk.py` file:
   ```bash
   cd src
   python obesity_risk.py
   ```

3. Alternatively, you can explore the project step-by-step in the Jupyter Notebook:
   ```bash
   jupyter notebook notebook/Obesity_Risk.ipynb
   ```

4. The project requires the following Python libraries, which you can install using:
   ```bash
   pip install -r requirements.txt
   ```

## Results

After tuning the hyperparameters, the best model was **CatBoost**, which achieved high accuracy. Key results from the analysis include:

- People with low physical activity are at higher risk of obesity.
- Gender and transportation method are important features for predicting obesity risk.

## Future Work

- **Add More Data**: Including genetic and mental health data could improve the model's accuracy and predictive power.
- **Deep Learning Models**: Neural networks and other deep learning techniques could enhance the model’s performance further.

## Contributing

If you want to contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a pull request.

## License

This project is licensed under the **Apache License**. See the [LICENSE](./LICENSE) file for more details.

## Resources

- [Kaggle Playground Series S4E2](https://www.kaggle.com/competitions/playground-series-s4e2/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## Contact

If you have any questions about this project, feel free to contact me: 

- [GitHub Profile](https://github.com/aysecnkci)
- [LinkedIn Profile](https://www.linkedin.com/in/aysearslancanakci)
```


