import time
from typing import List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from decimal import Decimal, ROUND_HALF_UP

# ------------------------
# Core logic 
# ------------------------
def round_1dp_half_up(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

def weighted_mean(gc: np.ndarray) -> Tuple[float, float]:
    """
    gc: Nx2 numpy array -> [grade, credit]
    returns: (credit-weighted mean grade, total credits)
    """
    if gc.size == 0:
        return np.nan, 0.0

    grades = gc[:, 0].astype(float)
    credits = gc[:, 1].astype(float)
    total_credits = float(credits.sum())
    if total_credits == 0:
        return np.nan, 0.0

    mean = float(np.dot(grades, credits) / total_credits)
    return mean, total_credits

def cw_median_js_equivalent(grades_and_credits):
    """
    Exact Python equivalent of the JavaScript CWMedian(),
    assuming input is a list of (grade, credits) pairs.

    Example input:
        [(16.4, 15), (12.0, 20), (18.0, 30)]
    """
    grades_and_credits = list(grades_and_credits)  # make a copy as a list


    # JS sorts IN PLACE by grade
    grades_and_credits.sort(key=lambda x: x[0])

    # Sum credits
    credits = 0
    for _, c in grades_and_credits:
        credits += c

    # Middle credit number
    credit_total = (credits + 1) / 2

    def is_int(x):
        return x == int(x)

    if is_int(credit_total):
        # Integer midpoint case
        credits = 0
        for grade, c in grades_and_credits:
            credits += c
            if credits >= credit_total:
                return grade
    else:
        # Non-integer midpoint
        credits = 0
        for i in range(len(grades_and_credits)):
            grade, c = grades_and_credits[i]
            credits += c

            # EXACT JS condition
            if ((credit_total - 0.5) == credits) and ((credit_total + 0.5) > credits):
                # average current + next grade
                return (grade + grades_and_credits[i + 1][0]) / 2.0

            elif credits > credit_total:
                return grade

def weighted_median_by_expansion_5credits(
    gc: List[Tuple[float, float]],  # allow floats coming from numpy
    *,
    average_middle_two: bool = True,
) -> float:
    expanded: List[float] = []

    for grade, credits in gc:
        if credits is None or float(credits) <= 0:
            continue

        # Convert to integer credits safely (handles 15.0 from numpy)
        credits_int = int(round(float(credits)))

        # Enforce multiples of 5 (after conversion)
        if credits_int % 5 != 0:
            raise ValueError(f"Credits must be multiples of 5 (got {credits}).")

        repetitions = credits_int // 5
        expanded.extend([float(grade)] * repetitions)

    if not expanded:
        return float("nan")

    expanded.sort()
    n = len(expanded)

    if n % 2 == 1:
        return expanded[n // 2]

    lower = expanded[(n // 2) - 1]
    upper = expanded[n // 2]

    if average_middle_two:
        return (lower + upper) / 2.0
    return lower


def weighted_median(gc, double_check: bool = False) -> float:

    median1 = weighted_median_by_expansion_5credits(gc)
    
    if double_check:
        median2 = cw_median_js_equivalent(gc)
        if median1 != median2:
            raise ValueError(
                f"Weighted median mismatch: {median1} (expansion) != {median2} (JS equivalent)"
            )
    return median1


CLASS_ORDER = {
    "Not of Honours standard": 5,
    "Third (III)": 4,
    "Lower Second (II.2)": 3,
    "Upper Second (II.1)": 2,
    "First (I)": 1,
}

def classify_degree(mean: float, median: float) -> str:
    if np.isnan(mean) or np.isnan(median):
        return "Not of Honours standard"

    mean = round_1dp_half_up(mean)
    median = round_1dp_half_up(median)
    # First
    if mean >= 16.5:
        return "First (I)"
    elif 16.0 <= mean <= 16.4:
        return "First (I)" if median >= 16.5 else "Upper Second (II.1)"

    # Upper Second
    elif 13.5 <= mean <= 15.9:
        return "Upper Second (II.1)"
    elif 13.0 <= mean <= 13.4:
        return "Upper Second (II.1)" if median >= 13.5 else "Lower Second (II.2)"

    # Lower Second
    elif 10.5 <= mean <= 12.9:
        return "Lower Second (II.2)"
    elif 10.0 <= mean <= 10.4:
        return "Lower Second (II.2)" if median >= 10.5 else "Third (III)"

    # Third / fail
    elif 7.0 <= mean <= 9.9:
        return "Third (III)"
    else:
        return "Not of Honours standard"
    
def degree_summary(
    completed: List[Tuple[float, float]],
    target_class: int,
):
    """
    Minimal current-position summary.
    No 'requirements' logic.
    """

    class_labels = {
        1: "First (I)",
        2: "Upper Second (II.1)",
        3: "Lower Second (II.2)",
        4: "Third (III)",
    }

    if target_class not in class_labels:
        raise ValueError("target_class must be an integer from 1 to 4")

    target_label = class_labels[target_class]

    if completed:
        gc_completed = np.array(completed, dtype=float)
        current_mean, _ = weighted_mean(gc_completed)
        current_median = weighted_median(gc_completed)
        current_class = classify_degree(current_mean, current_median)
    else:
        current_mean = np.nan
        current_median = np.nan
        current_class = "Not of Honours standard"

    return {
        "current_mean": current_mean,
        "current_median": current_median,
        "current_class": current_class,
        "target_class_label": target_label,
    }


def check_suggestion_meets_requirements(
    completed: List[Tuple[float, float]],
    outstanding: List[Tuple[float, float]],
    target_class: int,
    suggested_remaining_grades: List[float],
):

    class_labels = {
        1: "First (I)",
        2: "Upper Second (II.1)",
        3: "Lower Second (II.2)",
        4: "Third (III)",
    }
    mean_thresholds = {
        1: 16.5,  # First
        2: 13.5,  # Upper Second
        3: 10.5,  # Lower Second
        4: 7.0,   # Third
    }
    median_thresholds = mean_thresholds  # same boundaries

    if target_class not in class_labels:
        raise ValueError("target_class must be an integer from 1 to 4")

    if len(outstanding) != len(suggested_remaining_grades):
        raise ValueError("Length of outstanding and suggested_remaining_grades must match")

    target_label = class_labels[target_class]
    target_mean_band = mean_thresholds[target_class]
    target_median_band = median_thresholds[target_class]

    # --- build arrays for completed and suggested remaining modules ---
    if len(completed) > 0:
        gc_completed = np.array(completed, dtype=float)
    else:
        gc_completed = np.zeros((0, 2), dtype=float)

    if len(outstanding) > 0:
        remaining_credits = np.array([w for (_, w) in outstanding], dtype=float)
        gc_remaining = np.column_stack(
            [np.array(suggested_remaining_grades, dtype=float), remaining_credits]
        )
    else:
        gc_remaining = np.zeros((0, 2), dtype=float)

    # all modules together
    all_gc = np.vstack([gc_completed, gc_remaining])

    # --- compute final mean, median, and classification ---
    final_mean, _ = weighted_mean(all_gc)
    final_median = weighted_median(all_gc)
    final_class = classify_degree(final_mean, final_median)

    # --- check against band thresholds (mean/median) ---
    meets_mean_band = final_mean >= target_mean_band
    meets_median_band = final_median >= target_median_band

    final_order = CLASS_ORDER.get(final_class, 999)
    target_order = CLASS_ORDER[target_label]
    meets_target_class = final_order <= target_order

    result = {
        "final_mean": final_mean,
        "final_median": final_median,
        "final_class": final_class,
        "target_class_label": target_label,
        "meets_mean_band": meets_mean_band,
        "meets_median_band": meets_median_band,
        "meets_target_class": meets_target_class,
        "delta_mean_to_band": final_mean - target_mean_band,
        "delta_median_to_band": final_median - target_median_band,
    }

    return result