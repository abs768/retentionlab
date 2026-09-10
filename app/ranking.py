"""Capacity-constrained predictive prioritization; no causal treatment claim."""

import math
from decimal import Decimal

from app.schemas import Prediction, RankedCustomer, RankingResult, RankRequest


def rank_predictions(predictions: list[Prediction], request: RankRequest) -> RankingResult:
    # Decimal avoids spurious extra selections from binary floating point (e.g. 0.07 * 100).
    count = request.top_n
    if count is None:
        count = math.ceil(Decimal(str(request.capacity_fraction)) * len(predictions))
    # Python's stable sort preserves input order for exactly tied probabilities.
    ordered = sorted(predictions, key=lambda prediction: -prediction.churn_probability)
    return RankingResult(
        customers=[
            RankedCustomer(
                **prediction.model_dump(), rank=index + 1, selected_for_outreach=index < count
            )
            for index, prediction in enumerate(ordered)
        ],
        total_customers=len(predictions),
        selected_count=count,
    )
