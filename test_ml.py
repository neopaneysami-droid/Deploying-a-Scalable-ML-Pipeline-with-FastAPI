import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference

# TODO: implement the first test. Change the function name and input as needed
def test_process_data_shapes():
    """
    Test that process_data returns X and y with the correct number of rows.
    """
    data = pd.DataFrame({
        "age": [25, 40],
        "workclass": ["Private", "Self-emp-not-inc"],
        "fnlgt": [226802, 89814],
        "education": ["11th", "Bachelors"],
        "education-num": [7, 13],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Machine-op-inspct", "Exec-managerial"],
        "relationship": ["Own-child", "Husband"],
        "race": ["Black", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 50],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })

    X, y, encoder, lb = process_data(
        data,
        categorical_features=[
            "workclass", "education", "marital-status",
            "occupation", "relationship", "race", "sex",
            "native-country"
        ],
        label="salary",
        training=True
    )

    assert X.shape[0] == 2
    assert len(y) == 2


# TODO: implement the second test. Change the function name and input as needed
def test_train_model_returns_model():
    """
    Test that train_model returns a trained model object.
    """
    data = pd.DataFrame({
        "age": [25, 40],
        "workclass": ["Private", "Self-emp-not-inc"],
        "fnlgt": [226802, 89814],
        "education": ["11th", "Bachelors"],
        "education-num": [7, 13],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Machine-op-inspct", "Exec-managerial"],
        "relationship": ["Own-child", "Husband"],
        "race": ["Black", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 50],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })

    X, y, encoder, lb = process_data(
        data,
        categorical_features=[
            "workclass", "education", "marital-status",
            "occupation", "relationship", "race", "sex",
            "native-country"
        ],
        label="salary",
        training=True
    )

    model = train_model(X, y)
    assert model is not None


# TODO: implement the third test. Change the function name and input as needed
def test_inference_output_length():
    """
    Test that inference returns predictions with the same number of rows as X.
    """
    data = pd.DataFrame({
        "age": [25, 40],
        "workclass": ["Private", "Self-emp-not-inc"],
        "fnlgt": [226802, 89814],
        "education": ["11th", "Bachelors"],
        "education-num": [7, 13],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Machine-op-inspct", "Exec-managerial"],
        "relationship": ["Own-child", "Husband"],
        "race": ["Black", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [0, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 50],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    })

    X, y, encoder, lb = process_data(
        data,
        categorical_features=[
            "workclass", "education", "marital-status",
            "occupation", "relationship", "race", "sex",
            "native-country"
        ],
        label="salary",
        training=True
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == 2

