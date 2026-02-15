# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Random Forest Classifier trained to predict whether an individual earns >50K per year using the Census Income dataset. The model is implemented in scikit‑learn and deployed via FastAPI. Supporting artifacts include model.pkl, encoder.pkl, and lb.pkl.

## Intended Use
This model is intended for educational purposes within the Udacity MLOps course. It demonstrates data processing, model training, deployment, and slice‑based evaluation. It should not be used for real hiring, lending, or financial decisions.

## Training Data
The model was trained on the U.S. Census Income dataset. Categorical features were one‑hot encoded and the target variable was binarized.

## Evaluation Data
A held‑out test split from the same dataset was used for evaluation, with identical preprocessing applied.

## Metrics
The model was evaluated using precision, recall, and F1 score.

- Precision: 0.7419  
- Recall: 0.6384  
- F1 Score: 0.6863  

Slice‑based metrics were also computed for categorical features, revealing variation in performance across demographic groups.

## Ethical Considerations
The dataset includes sensitive attributes such as race, sex, and marital status. Predictions may reflect historical bias. The model should not be used for real‑world decision‑making or any scenario affecting people’s livelihoods.

## Caveats and Recommendations
Performance varies across demographic slices, especially in underrepresented groups. The model is for demonstration only. Future improvements could include hyperparameter tuning, fairness analysis, and bias mitigation techniques.