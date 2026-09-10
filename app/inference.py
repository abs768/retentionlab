"""One adapter shared by all prediction and monitoring paths."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from app.schemas import RAW_FEATURES, Customer, Prediction


def customers_to_frame(customers: Sequence[Customer]) -> pd.DataFrame:
    rows = [customer.model_dump(include=set(RAW_FEATURES)) for customer in customers]
    frame = pd.DataFrame(rows, columns=list(RAW_FEATURES))
    # Explicit float dtype keeps an all-missing TotalCharges batch numeric.
    for feature in ("MonthlyCharges", "TotalCharges"):
        frame[feature] = frame[feature].astype(float)
    return frame


def checked_probabilities(model, frame: pd.DataFrame) -> np.ndarray:
    matrix = np.asarray(model.predict_proba(frame), dtype=float)
    if matrix.shape != (len(frame), 2) or not np.isfinite(matrix).all():
        raise ValueError("Model returned invalid probabilities")
    if ((matrix < 0) | (matrix > 1)).any() or not np.allclose(matrix.sum(axis=1), 1):
        raise ValueError("Model returned invalid probabilities")
    return matrix[:, 1]


def predict_customers(service, customers: Sequence[Customer]) -> list[Prediction]:
    probabilities = checked_probabilities(service.model, customers_to_frame(customers))
    return [
        Prediction(
            customer_id=customer.customerID,
            churn_probability=float(probability),
            model_version=service.metadata["model_version"],
        )
        for customer, probability in zip(customers, probabilities, strict=True)
    ]
