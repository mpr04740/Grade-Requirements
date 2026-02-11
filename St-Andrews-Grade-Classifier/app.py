import streamlit as st
import numpy as np  
from src.backend_logic import *
from src.io_csv import *

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
# Input form
# ------------------------

with st.form("degree_input_form"):
    st.subheader("1. Enter your modules")

    st.markdown(
        "**Use the tables below to enter your completed and remaining modules.**\n\n"
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
            summary = degree_summary(
                completed=completed_list,
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
    st.subheader("Current position")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current weighted mean",
            f"{summary['current_mean_rounded']:.1f}"
            if not np.isnan(summary["current_mean_rounded"])
            else "N/A",
        )
    with col2:
        st.metric(
            "Current weighted median",
            f"{summary['current_median_rounded']:.1f}"
            if not np.isnan(summary["current_median_rounded"])
            else "N/A",
        )
    with col3:
        st.metric("Current classification", summary["current_class"])
    with col4:
        st.metric("Target classification", summary["target_class_label"])


    # ------------------------------
    # Scenario planner with sliders
    # ------------------------------
    if len(outstanding_list) > 0:
        st.markdown("---")
        st.subheader("Scenario planner: adjust your future grades")
        
        n_outstanding = len(outstanding_list)
        if "suggested_grades" not in st.session_state or len(st.session_state["suggested_grades"]) != n_outstanding:
            default_grade = summary["current_mean"]
            if np.isnan(default_grade):
                default_grade = 10.0
            default_grade = float(max(0.0, min(20.0, default_grade)))
            st.session_state["suggested_grades"] = [default_grade for _ in range(n_outstanding)]

        suggested_grades = []

        st.markdown(
            "Use the sliders to set the **grades you think you can achieve** "
            "on each remaining module. The app will compute the final classification. Use this to help set goals and meet your target classification."
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
                format="%.1f",
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
            st.metric("Final weighted mean", f"{result['final_mean_rounded']:.1f}")
        with c2:
            st.metric("Final weighted median", f"{result['final_median_rounded']:.1f}")
        with c3:
            st.metric("Final classification", result["final_class"])

        meets = result["meets_target_class"]

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
