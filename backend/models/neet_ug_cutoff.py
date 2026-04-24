"""
NEET UG 2025 cutoff table model.
"""

from typing import Optional

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from database.connection import Base


class NeetUg2025Cutoff(Base):
    __tablename__ = "neet_ug_2025_cutoffs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    state: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    air_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    state_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    college_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    college_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    institution_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    college_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    course: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    sub_category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    seat_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    quota: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    domicile: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, index=True)
    eligibility: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    round: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
