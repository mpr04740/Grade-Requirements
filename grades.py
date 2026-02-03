import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------
# Core logic 
# ------------------------

def weighted_mean(gc: np.ndarray) -> float:
    """
    gc: Nx2 numpy array -> [grade, credit]
    """
    grades = gc[:, 0].astype(float)
    credits = gc[:, 1].astype(float)
    total_credits = credits.sum()
    if total_credits == 0:
        return np.nan
    return float(np.dot(grades, credits) / total_credits), total_credits


def weighted_median(gc: np.ndarray) -> float:
    """
    Credit-weighted median of grades.
    """
    grades = gc[:, 0].astype(float)
    credits = gc[:, 1].astype(float)

    # sort by grade
    order = np.argsort(grades)
    grades = grades[order]
    credits = credits[order]

    cum_credits = np.cumsum(credits)
    total = cum_credits[-1]
    if total == 0:
        return np.nan

    cutoff = total / 2.0
    idx = np.searchsorted(cum_credits, cutoff)
    return float(grades[idx])


CLASS_ORDER = {
    "Not of Honours standard": 5,
    "Third (III)": 4,
    "Lower Second (II.2)": 3,
    "Upper Second (II.1)": 2,
    "First (I)": 1,
}

def classify_degree(mean: float, median: float) -> str:
    """
    Implements the rules from the table.
    First decide band by mean, then apply 'uplift' rules using median.
    """
    # Base class from mean alone
    if mean >= 16.5:
        base = "First (I)"
    elif mean >= 13.5:
        base = "Upper Second (II.1)"
    elif mean >= 10.5:
        base = "Lower Second (II.2)"
    elif mean >= 7.0:
        base = "Third (III)"
    else:
        return "Not of Honours standard"

    # Borderline uplifts
    if 16.0 <= mean <= 16.4 and median >= 16.5:
        return "First (I)"
    if 13.0 <= mean <= 13.4 and median >= 13.5:
        return "Upper Second (II.1)"
    if 10.0 <= mean <= 10.4 and median >= 10.5:
        return "Lower Second (II.2)"

    return base


def minimal_forward_average_for_target_mean(target_mean,
                                            credits_outstanding,
                                            current_mean,
                                            credits_completed):
    Ca = credits_completed
    Cr = credits_outstanding
    Ma = current_mean
    
    if Cr == 0:
        return float('nan')

    x = (target_mean * (Ca + Cr) - Ma * Ca) / Cr
    return x


def required_grade_for_target_median(existing_gc: np.ndarray,
                                     remaining_credits: np.ndarray,
                                     target: float,
                                     grade_min: float = 0.0,
                                     grade_max: float = 20.0,
                                     tol: float = 1e-3):
    """
    Find the minimum constant grade on all remaining classes needed to reach
    a target weighted median.
    """

    remaining_credits = np.asarray(remaining_credits, dtype=float)

    # helper: compute median if all remaining classes have grade g
    def median_if_grade(g: float) -> float:
        remaining_gc = np.column_stack([
            np.full_like(remaining_credits, g, dtype=float),
            remaining_credits
        ])
        all_gc = np.vstack([existing_gc, remaining_gc])
        return weighted_median(all_gc)

    # Check if it is even possible
    if median_if_grade(grade_max) < target:
        return None  # impossible to reach target even with perfect grades

    lo, hi = grade_min, grade_max
    # Binary search for minimal grade g such that median >= target
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        med = median_if_grade(mid)
        if med >= target:
            hi = mid
        else:
            lo = mid

    return hi


def degree_summary_and_requirements(
    completed: List[Tuple[float, float]],
    outstanding: List[Tuple[float, float]],
    target_class: int,
):
    """
    completed:   list of (grade, credit) for modules already done
    outstanding: list of (placeholder_grade, credit) for future modules;
                 only the credits are used.
    target_class: 1 = First, 2 = Upper Second, 3 = Lower Second, 4 = Third
    """

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
    median_thresholds = mean_thresholds  # same band boundaries for median

    if target_class not in class_labels:
        raise ValueError("target_class must be an integer from 1 to 4")

    target_mean = mean_thresholds[target_class]
    target_median = median_thresholds[target_class]
    target_label = class_labels[target_class]

    # ---- Completed modules ----
    if len(completed) > 0:
        gc_completed = np.array(completed, dtype=float)  # shape (n, 2)
        current_mean, credits_completed = weighted_mean(gc_completed)
        current_median = weighted_median(gc_completed)
        current_class = classify_degree(current_mean, current_median)
    else:
        gc_completed = np.zeros((0, 2), dtype=float)
        current_mean = np.nan
        current_median = np.nan
        current_class = "Not of Honours standard"
        credits_completed = 0.0

    # ---- Outstanding modules (only credits matter here) ----
    if len(outstanding) > 0:
        remaining_credits_arr = np.array([w for (_, w) in outstanding], dtype=float)
        credits_outstanding = float(remaining_credits_arr.sum())
    else:
        remaining_credits_arr = np.zeros(0, dtype=float)
        credits_outstanding = 0.0

    # ---- Needed forward mean (average over remaining modules) ----
    if credits_outstanding > 0:
        needed_forward_mean = minimal_forward_average_for_target_mean(
            target_mean=target_mean,
            credits_outstanding=credits_outstanding,
            current_mean=current_mean if not np.isnan(current_mean) else 0.0,
            credits_completed=credits_completed,
        )
    else:
        needed_forward_mean = np.nan

    # ---- Needed uniform grade on remaining modules for target median ----
    if credits_outstanding > 0:
        needed_uniform_for_median = required_grade_for_target_median(
            existing_gc=gc_completed,
            remaining_credits=remaining_credits_arr,
            target=target_median,
            grade_min=0.0,
            grade_max=20.0,
            tol=1e-3,
        )
    else:
        needed_uniform_for_median = None

    return {
        "current_mean": current_mean,
        "current_median": current_median,
        "current_class": current_class,
        "target_class_label": target_label,
        "needed_forward_mean": needed_forward_mean,
        "needed_uniform_for_median": needed_uniform_for_median,
    }


def check_suggestion_meets_requirements(
    completed: List[Tuple[float, float]],
    outstanding: List[Tuple[float, float]],
    target_class: int,
    suggested_remaining_grades: List[float],
):
    """
    See your original docstring – unchanged behaviour.
    """

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


# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(
    page_title="Degree Classification Planner",
    layout="wide",
)

st.title("🎓 Degree Classification Planner")

st.markdown(
    "Enter your **completed modules** and **remaining modules**, choose a "
    "target classification, and then experiment with different future grades."
)

TARGET_CLASS_OPTIONS = {
    "First (I)": 1,
    "Upper Second (II.1)": 2,
    "Lower Second (II.2)": 3,
    "Third (III)": 4,
}


def parse_completed(df: pd.DataFrame) -> List[Tuple[float, float]]:
    rows = []
    for _, row in df.iterrows():
        grade = row.get("Grade")
        credit = row.get("Credits")
        if pd.isna(grade) or pd.isna(credit):
            continue
        if float(credit) <= 0:
            continue
        rows.append((float(grade), float(credit)))
    return rows


def parse_outstanding(df: pd.DataFrame) -> List[Tuple[float, float]]:
    """
    Only credits matter for requirements; we use 0.0 as a placeholder grade.
    """
    rows = []
    for _, row in df.iterrows():
        credit = row.get("Credits")
        if pd.isna(credit):
            continue
        if float(credit) <= 0:
            continue
        rows.append((0.0, float(credit)))
    return rows


# ------------------------
# Input form
# ------------------------

with st.form("degree_input_form"):
    st.subheader("1. Enter your modules")

    col_completed, col_outstanding = st.columns(2)

    with col_completed:
        st.markdown("**Completed modules** (grade & credits)")
        default_completed = pd.DataFrame(
            [
                {"Grade": 14.0, "Credits": 20.0},
                {"Grade": 16.0, "Credits": 20.0},
            ]
        )
        completed_df = st.data_editor(
            default_completed,
            key="completed_df",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Grade": st.column_config.NumberColumn("Grade", step=0.1),
                "Credits": st.column_config.NumberColumn("Credits", step=5),
            },
        )

    with col_outstanding:
        st.markdown("**Remaining modules** (credits only)")
        default_outstanding = pd.DataFrame(
            [
                {"Credits": 20.0},
                {"Credits": 20.0},
            ]
        )
        outstanding_df = st.data_editor(
            default_outstanding,
            key="outstanding_df",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Credits": st.column_config.NumberColumn("Credits", step=5),
            },
        )

    st.subheader("2. Choose your target classification")
    target_label = st.selectbox(
        "Target classification",
        list(TARGET_CLASS_OPTIONS.keys()),
        index=0,
    )
    target_class = TARGET_CLASS_OPTIONS[target_label]

    submitted = st.form_submit_button("Run analysis", type="primary")

if submitted:
    completed_list = parse_completed(completed_df)
    outstanding_list = parse_outstanding(outstanding_df)

    if len(completed_list) == 0:
        st.warning("Please enter at least one completed module (grade + credits).")
    else:
        summary = degree_summary_and_requirements(
            completed=completed_list,
            outstanding=outstanding_list,
            target_class=target_class,
        )

        # Persist in session_state for the planner
        st.session_state["completed_list"] = completed_list
        st.session_state["outstanding_list"] = outstanding_list
        st.session_state["target_class"] = target_class
        st.session_state["summary"] = summary

# ------------------------
# Show summary if we have it
# ------------------------

if "summary" in st.session_state:
    summary = st.session_state["summary"]
    completed_list = st.session_state["completed_list"]
    outstanding_list = st.session_state["outstanding_list"]
    target_class = st.session_state["target_class"]

    st.markdown("---")
    st.subheader("Current position & requirements")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current weighted mean",
            f"{summary['current_mean']:.2f}"
            if not np.isnan(summary["current_mean"])
            else "N/A",
        )
    with col2:
        st.metric(
            "Current weighted median",
            f"{summary['current_median']:.2f}"
            if not np.isnan(summary["current_median"])
            else "N/A",
        )
    with col3:
        st.metric("Current classification", summary["current_class"])
    with col4:
        st.metric("Target classification", summary["target_class_label"])

    col5, col6 = st.columns(2)

    with col5:
        needed_forward_mean = summary["needed_forward_mean"]
        if np.isnan(needed_forward_mean):
            st.info("No remaining credits – no forward average required.")
        else:
            st.metric(
                "Required average on remaining modules (mean)",
                f"{needed_forward_mean:.2f}",
            )

    with col6:
        needed_uniform = summary["needed_uniform_for_median"]
        if needed_uniform is None:
            st.info(
                "Target median band may be impossible to reach even with perfect "
                "grades on remaining modules."
            )
        elif np.isnan(needed_uniform):
            st.info("No remaining modules, so median can't be improved.")
        else:
            st.metric(
                "Uniform grade on remaining modules for target median",
                f"{needed_uniform:.2f}",
            )

    # ------------------------
    # Scenario planner with sliders
    # ------------------------
    if len(outstanding_list) > 0:
        st.markdown("---")
        st.subheader("Scenario planner: adjust your future grades")

        # Initialise or update suggested grades
        n_outstanding = len(outstanding_list)
        if "suggested_grades" not in st.session_state or len(
            st.session_state["suggested_grades"]
        ) != n_outstanding:
            # sensible default: use required forward mean if available, else 10
            default_grade = summary["needed_forward_mean"]
            if np.isnan(default_grade):
                default_grade = 10.0
            default_grade = float(max(0.0, min(20.0, default_grade)))
            st.session_state["suggested_grades"] = [
                default_grade for _ in range(n_outstanding)
            ]

        suggested_grades = []

        st.markdown(
            "Use the sliders to set the **grades you think you can achieve** "
            "on each remaining module. The app will compute the final classification."
        )

        for idx, (_, credit) in enumerate(outstanding_list):
            key = f"remaining_grade_{idx}"
            current_default = st.session_state["suggested_grades"][idx]
            grade_val = st.slider(
                f"Remaining module {idx + 1} (credits: {credit:g})",
                min_value=0.0,
                max_value=20.0,
                step=0.1,
                value=float(current_default),
                key=key,
            )
            suggested_grades.append(float(grade_val))

        # store back
        st.session_state["suggested_grades"] = suggested_grades

        # Immediately check (in Streamlit this is the most practical UX;
        # if you really want a 1s debounce, you can add custom JS or a timed rerun.)
        result = check_suggestion_meets_requirements(
            completed=completed_list,
            outstanding=outstanding_list,
            target_class=target_class,
            suggested_remaining_grades=suggested_grades,
        )

        st.markdown("### Result for this plan")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Final weighted mean", f"{result['final_mean']:.2f}")
        with c2:
            st.metric("Final weighted median", f"{result['final_median']:.2f}")
        with c3:
            st.metric("Final classification", result["final_class"])

        meets = result["meets_target_class"]
        delta_mean = result["delta_mean_to_band"]
        delta_med = result["delta_median_to_band"]

        if meets:
            st.success(
                f"✅ This plan **meets or exceeds** your target classification "
                f"({result['target_class_label']})."
            )
        else:
            st.error(
                f"❌ This plan **does not yet meet** your target classification "
                f"({result['target_class_label']})."
            )

        st.markdown(
            f"- Distance to target **mean band**: "
            f"{delta_mean:+.2f} grade points\n"
            f"- Distance to target **median band**: "
            f"{delta_med:+.2f} grade points"
        )

    else:
        st.info(
            "No outstanding modules were entered, so there is nothing to plan forward."
        )
else:
    st.info("Fill in your modules and click **Run analysis** to get started.")


'''
To open the sit run the following comman in terminal 
streamlit run grades.py
'''