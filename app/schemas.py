"""Validated raw IBM Telco inputs. Identifiers are response metadata only."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import MAX_BATCH_SIZE
from retentionlab.config import EXPECTED_COLUMNS, ID_COLUMN, TARGET_COLUMN

RAW_FEATURES = tuple(c for c in EXPECTED_COLUMNS if c not in (ID_COLUMN, TARGET_COLUMN))
YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    customerID: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: Annotated[int, Field(ge=0)]
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: NonNegative
    TotalCharges: NonNegative | None

    @field_validator("SeniorCitizen", mode="before")
    @classmethod
    def integer_binary(cls, value):
        if type(value) is not int:
            raise ValueError("SeniorCitizen must be integer 0 or 1")
        return value

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def raw_total_charges(cls, value):
        # The source CSV encodes this numeric field as text; blank means missing.
        if isinstance(value, str):
            return float(value.strip()) if value.strip() else None
        return value


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    customers: Annotated[list[Customer], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class RankRequest(BatchRequest):
    capacity_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    top_n: Annotated[int, Field(ge=0, le=MAX_BATCH_SIZE)] | None = None

    @model_validator(mode="after")
    def capacity_is_unambiguous(self):
        if (self.capacity_fraction is None) == (self.top_n is None):
            raise ValueError("Supply exactly one of capacity_fraction or top_n")
        if self.top_n is not None and self.top_n > len(self.customers):
            raise ValueError("top_n cannot exceed the number of customers")
        return self


class Prediction(BaseModel):
    customer_id: str | None
    churn_probability: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    model_version: str


class BatchPrediction(BaseModel):
    predictions: list[Prediction]
    total_customers: int


class RankedCustomer(Prediction):
    rank: int
    selected_for_outreach: bool


class RankingResult(BaseModel):
    customers: list[RankedCustomer]
    total_customers: int
    selected_count: int
