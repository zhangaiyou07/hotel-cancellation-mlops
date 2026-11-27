"""Request and response formats of the model API (the "contract" with the frontend and monitoring)."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BookingRequest(BaseModel):
    """One hotel booking, described only with information known when the booking is created.

    Fields without a default are required. The other fields fall back to the most common value
    in the training data, so the frontend can send a short form.
    """

    model_config = ConfigDict(extra="forbid")  # unknown fields are rejected with a clear error

    # optional id so monitoring can join the prediction with the true outcome later
    booking_id: str | None = Field(default=None, max_length=64)

    # --- required
    hotel: Literal["City Hotel", "Resort Hotel"]
    arrival_date: date = Field(description="Planned arrival date, e.g. 2017-06-10")
    lead_time: int = Field(ge=0, le=800, description="Days between booking and arrival")
    adr: float = Field(ge=0, lt=1000, description="Average daily rate in EUR")
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    market_segment: Literal[
        "Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"
    ]

    # --- optional (defaults = most common value in the training data)
    stays_in_weekend_nights: int = Field(default=1, ge=0, le=30)
    stays_in_week_nights: int = Field(default=2, ge=0, le=60)
    adults: int = Field(default=2, ge=0, le=60)
    children: int = Field(default=0, ge=0, le=10)
    babies: int = Field(default=0, ge=0, le=10)
    meal: Literal["BB", "HB", "FB", "SC"] = "BB"
    country: str = Field(default="PRT", min_length=2, max_length=3, description="ISO country code")
    distribution_channel: Literal["TA/TO", "Direct", "Corporate", "GDS"] = "TA/TO"
    customer_type: Literal["Transient", "Transient-Party", "Contract", "Group"] = "Transient"
    reserved_room_type: Literal["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"] = "A"
    is_repeated_guest: Literal[0, 1] = 0
    previous_cancellations: int = Field(default=0, ge=0, le=50)
    previous_bookings_not_canceled: int = Field(default=0, ge=0, le=100)
    days_in_waiting_list: int = Field(default=0, ge=0, le=400)
    agent: int | None = Field(default=None, ge=0, description="Travel agent id, empty if none")
    company: int | None = Field(default=None, ge=0, description="Company id, empty if none")


class TopFactor(BaseModel):
    feature: str
    value: str | int | float | None  # the booking's value for this feature
    impact: float  # > 0 pushes towards cancellation, < 0 towards a kept booking (log-odds)


class PredictionResponse(BaseModel):
    booking_id: str | None
    cancel_probability: float = Field(ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    model_version: str
    top_factors: list[TopFactor] = Field(default_factory=list)  # empty if the model cannot be explained
