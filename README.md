# University Grade Classification Calculator (UK)

This project is a **Python + Streamlit application** that allows students to calculate their current and projected degree classification based on **UK undergraduate grading rules**, specifically following **University of St Andrews classification regulations**.

It is designed to help students understand:
- Their current academic standing
- What degree classification they are on track for
- What grades they would need in remaining modules to achieve a desired classification

---

## Features

### 📊 Current Performance Analysis
- Input completed university modules with:
  - Grade
  - Credit value
- Automatically calculates:
  - **Weighted average**
  - **Weighted median (average of middle two)**
  - Other summary statistics
- Applies **St Andrews classification rules** to determine:
  - First
  - Upper Second (2:1)
  - Lower Second (2:2)
  - Third

---

### 🎯 Target Classification Tool
- Choose a **desired UK degree classification**
- The app evaluates whether the target is currently achievable based on existing grades

---

### 🔮 Grade Projection for Remaining Modules
- Add hypothetical grades for **remaining subjects**
- Instantly see:
  - Updated averages
  - Updated classification
- Useful for **“what-if” planning** and goal setting

---

## Classification Rules

This calculator follows the **University of St Andrews undergraduate degree classification methodology**, which includes:
- Credit-weighted averages
- Median-based considerations
- Borderline handling in line with institutional rules

> ⚠️ This tool is intended for guidance only and does not replace official university calculations.

---

## Tech Stack

- **Python 3**
- **Streamlit** – interactive web UI
- **NumPy** – numerical calculations
- **Pandas** – data handling

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/grade-requirements.git
cd grade-requirements
