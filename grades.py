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
    return round_1dp_half_up(mean,1), total_credits

def cw_median_js_equivalent(grades_and_credits):
    """
    Exact Python equivalent of the JavaScript CWMedian().

    - Sorts input IN PLACE
    - Uses (total_credits + 1) / 2
    - Averages grades if midpoint falls exactly between modules
    - No extra safety guards beyond the JS logic
    """

    # JS sorts in place
    grades_and_credits.sort(key=lambda x: x["grade"])

    # Sum credits
    credits = 0
    for item in grades_and_credits:
        credits += item["credits"]

    # Middle credit number
    credit_total = (credits + 1) / 2

    def is_int(x):
        return x == int(x)

    if is_int(credit_total):
        # Integer midpoint case
        credits = 0
        for item in grades_and_credits:
            credits += item["credits"]
            if credits >= credit_total:
                return item["grade"]
    else:
        # Non-integer midpoint
        credits = 0
        for i in range(len(grades_and_credits)):
            credits += grades_and_credits[i]["credits"]

            # EXACT translation of JS condition
            if ((credit_total - 0.5) == credits) and ((credit_total + 0.5) > credits):
                return (
                    grades_and_credits[i]["grade"]
                    + grades_and_credits[i + 1]["grade"]
                ) / 2

            elif credits > credit_total:
                return grades_and_credits[i]["grade"]

def weighted_median_by_expansion_5credits(
    gc: List[Tuple[float, int]],
    *,
    average_middle_two: bool = True,
) -> float:
    """
    Credit-weighted median by *explicit expansion* into a list of grades,
    using 5-credit units (St Andrews style).

    Concept:
      - All module credits are multiples of 5.
      - Treat each 5 credits as one 'unit' or 'vote'.
      - Repeat each grade (credits / 5) times.
      - Sort the expanded list.
      - Take the middle value(s).

    Parameters
    ----------
    gc : list of (grade, credits)
        Example: [(16.4, 15), (12.0, 20), (18.0, 30)]

    average_middle_two : bool, default False
        False  -> policy/admin style:
                  if expanded length is even, return the *lower* middle value
        True   -> textbook median:
                  if expanded length is even, return the average of the two middle values

    Returns
    -------
    float
        The weighted median grade.
        Returns NaN if no valid credits.
    """
    expanded: List[float] = []

    for grade, credits in gc:
        # Skip invalid or zero-credit entries
        if credits is None or credits <= 0:
            continue

        # Enforce St Andrews credit structure
        if credits % 5 != 0:
            raise ValueError(
                f"Credits must be multiples of 5 for this method (got {credits})."
            )

        # Number of repetitions = credits / 5
        repetitions = credits // 5

        # Repeat the grade explicitly
        expanded.extend([float(grade)] * repetitions)

    # If nothing survived validation
    if not expanded:
        return float("nan")
    
    expanded.sort()

    n = len(expanded)

    # Odd length: single middle element
    if n % 2 == 1:
        mid_index = n // 2
        return expanded[mid_index]

    # Even length
    lower_mid_index = (n // 2) - 1
    upper_mid_index = n // 2

    if average_middle_two:
        # Textbook median
        return (expanded[lower_mid_index] + expanded[upper_mid_index]) / 2.0
    else:
        # Policy/admin style: choose lower middle
        return expanded[lower_mid_index]


def weighted_median(gc, double_check: bool = True) -> float:

    median1 = weighted_median_by_expansion_5credits(gc)
    if double_check:
        median2 = cw_median_js_equivalent(gc)
        if median1 != median2:
            raise ValueError(
                f"Weighted median mismatch: {median1} (expansion) != {median2} (JS equivalent)"
            )
    return round_1dp_half_up(median1)


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
# Streamlit UI (with optional CSV upload)
# ------------------------

st.set_page_config(
    page_title="St Andrews Degree Classification Calculator | Weighted Mean & Median",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .coffee-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;

        padding: 10px 18px;
        background: linear-gradient(135deg, #FFDD00, #FFC107);
        color: #111111;

        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.2px;

        border-radius: 14px;
        text-decoration: none;

        box-shadow: 0 6px 16px rgba(255, 221, 0, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }

    .coffee-btn:hover {
        background: linear-gradient(135deg, #FFE066, #FFB300);
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(255, 221, 0, 0.35);
        color: #000000;
    }

    .coffee-btn:active {
        transform: translateY(0);
        box-shadow: 0 4px 10px rgba(255, 221, 0, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    st.markdown(
        '<a class="coffee-btn" href="https://buymeacoffee.com/michaelangelospr?status=1" target="_blank">☕ Buy me a coffee</a>',
        unsafe_allow_html=True
    )

st.title("🎓 St Andrews Degree Classification Calculator")
st.write(
    "A University of St Andrews honours classification tool using the credit-weighted "
    "mean and credit-weighted median (20-point scale). Upload grades + credits, set a target, "
    "and plan remaining modules."
)

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

# ------------------------
# CSV helpers (UI-side)
# ------------------------

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # allow singular "credit"
    if "credit" in df.columns and "credits" not in df.columns:
        df = df.rename(columns={"credit": "credits"})
    return df

def read_csv_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return _normalise_cols(df)

def validate_completed_csv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"grade", "credits"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Expected: Grade, Credits.")
    out = df[["grade", "credits"]].copy()
    out = out.rename(columns={"grade": "Grade", "credits": "Credits"})
    return out

def validate_remaining_csv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"credits"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Expected: Credits.")
    out = df[["credits"]].copy()
    out = out.rename(columns={"credits": "Credits"})
    return out

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

    st.markdown(
        "Use the tables below to enter your completed and remaining modules.\n\n"
    )

    up1, up2 = st.columns(2)
    with up1:
        completed_csv = st.file_uploader(
            "Optionally upload completed modules CSV (Grade, Credits)",
            type=["csv"],
            key="completed_csv",
        )
    with up2:
        remaining_csv = st.file_uploader(
            "Upload remaining modules CSV (Credits)",
            type=["csv"],
            key="remaining_csv",
        )

    col_completed, col_outstanding = st.columns(2)

    # ---- Defaults (used if no upload) ----
    default_completed = pd.DataFrame(
        [
            {"Grade": 14.0, "Credits": 15.0},
            {"Grade": 16.0, "Credits": 15.0},
        ]
    )
    default_outstanding = pd.DataFrame(
        [
            {"Credits": 15.0},
            {"Credits": 15.0},
        ]
    )

    # ---- If uploaded, use uploaded data; otherwise use defaults ----
    completed_seed = default_completed
    completed_upload_error = None
    if completed_csv is not None:
        try:
            completed_seed = validate_completed_csv(read_csv_upload(completed_csv))
        except Exception as e:
            completed_upload_error = str(e)

    outstanding_seed = default_outstanding
    remaining_upload_error = None
    if remaining_csv is not None:
        try:
            outstanding_seed = validate_remaining_csv(read_csv_upload(remaining_csv))
        except Exception as e:
            remaining_upload_error = str(e)

    with col_completed:
        st.markdown("**Completed modules** (grade & credits)")
        if completed_upload_error:
            st.error(f"Completed CSV error: {completed_upload_error}")

        completed_df = st.data_editor(
            completed_seed,
            key="completed_df",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Grade": st.column_config.NumberColumn("Grade", step=0.1, format="%.1f"),
                "Credits": st.column_config.NumberColumn("Credits", step=5, format="%.0f"),
            },
        )

    with col_outstanding:
        st.markdown("**Remaining modules** (credits only)")
        if remaining_upload_error:
            st.error(f"Remaining CSV error: {remaining_upload_error}")

        outstanding_df = st.data_editor(
            outstanding_seed,
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
    # If uploads were invalid, stop early so users don't get confusing results
    if (completed_csv is not None and completed_upload_error) or (
        remaining_csv is not None and remaining_upload_error
    ):
        st.warning("Please fix the CSV upload errors above (or remove the upload) and try again.")
    else:
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
            f"{summary['current_mean']:.1f}"
            if not np.isnan(summary["current_mean"])
            else "N/A",
        )
    with col2:
        st.metric(
            "Current weighted median",
            f"{summary['current_median']:.1f}"
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
            st.info("No remaining credits - no forward average required.")
        else:
            st.metric(
                "Required average on remaining modules (mean)",
                f"{needed_forward_mean:.1f}",
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
                f"{needed_uniform:.1f}",
            )

    # ------------------------------
    # Scenario planner with sliders
    # ------------------------------
    if len(outstanding_list) > 0:
        st.markdown("---")
        st.subheader("Scenario planner: adjust your future grades")
        
        n_outstanding = len(outstanding_list)
        if "suggested_grades" not in st.session_state or len(st.session_state["suggested_grades"]) != n_outstanding:
            default_grade = summary["needed_forward_mean"]
            if np.isnan(default_grade):
                default_grade = 10.0
            default_grade = float(max(0.0, min(20.0, default_grade)))
            st.session_state["suggested_grades"] = [default_grade for _ in range(n_outstanding)]

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

        st.session_state["suggested_grades"] = suggested_grades

        result = check_suggestion_meets_requirements(
            completed=completed_list,
            outstanding=outstanding_list,
            target_class=target_class,
            suggested_remaining_grades=suggested_grades,
        )

        st.markdown("### Result for this plan")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Final weighted mean", f"{result['final_mean']:.1f}")
        with c2:
            st.metric("Final weighted median", f"{result['final_median']:.1f}")
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
            f"- Distance to target **mean band**: {delta_mean:+.1f} grade points\n"
            f"- Distance to target **median band**: {delta_med:+.1f} grade points"
        )

    else:
        st.info("No outstanding modules were entered, so there is nothing to plan forward.")
else:
    st.info("Fill in your modules and click **Run analysis** to get started.")


st.header("FAQ")

st.subheader("How does St Andrews calculate degree classification?")
st.write(
    "The University of St Andrews determines Honours degree classification uses both mean and median grades" \
    ", weighted by module credits, on a 20-point scale. This tool replicates the calculation method as per the official guidance." 
)

st.subheader("Where can I find the official St Andrews classification guidance?")
st.write(
    "You can view the University of St Andrews’ official degree classification guidance here:"
)
st.markdown(
    "- **[https://www.st-andrews.ac.uk/policy/academic-policies-assessment-examination-and-award-classification/classification-policy.pdf]**"
)

st.subheader("What data do you collect or store?")
st.write(
    "This tool does **not** store, save, or transmit your data. "
    "All grades and module credits you enter are processed **locally in your browser session** "
    "and are cleared when you refresh or close the page."
)

st.subheader("Are my grades uploaded or shared with anyone?")
st.write(
    "No. Uploaded CSV files and manually entered grades are used only for on-screen calculations. "
    "They are not written to a database, logged, or shared with third parties."
)

st.subheader("Can the University see my results?")
st.write(
    "No. This is an independent, unofficial tool created for planning and exploration. "
    "It is **not connected to the University of St Andrews’ systems** and cannot access student records."
)

st.subheader("Does this tool guarantee my final degree classification?")
st.write(
    "No. This calculator is for guidance and planning only. "
    "Final degree classifications are determined by the University of St Andrews in accordance "
    "with its academic regulations."
)

st.subheader("Who is this tool for?")
st.write(
    "This calculator is intended for St Andrews students who want to understand how their "
    "current grades contribute to their overall classification and to explore possible outcomes "
    "based on remaining modules."
)

# To run:
# streamlit run grades.py
