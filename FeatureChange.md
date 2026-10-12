
# Community Complaint Categorization: Schema & Feature Mapping Guide

## 1. Revised Dataset Schema

This schema comprises one reference identifier, 5 input features, and 3 target output features. The `date` feature serves as an operational timestamp and an arithmetic anchor for dynamic SLA resolution calculations.

| Column Name | Feature Type | Role | Example Value |
| :--- | :--- | :--- | :--- |
| **`complaint_id`** | Reference | Unique record identifier | `CC-101` |
| **`Complaint text`** | Input | Analytical (Primary ML Feature) | *"Pothole on Main St causing traffic chaos"* |
| **`urgency level`** | Input | Analytical (Priority Modifier) | `High` |
| **`location`** | Input | Analytical / Routing Filter | `A` |
| **`submission channel`** | Input | Operational Metadata | `Mobile App` |
| **`date`** | Input | Operational Metadata & Date Anchor | `2026-10-09` |
| **`Complaint category`** | Target Output | Machine Learning Model Target | `Road and sidewalk repair` |
| **`routing suggestion`** | Target Output | ML / Hybrid Logic Target | `Branch 1 Public Works Office - General Queue` |
| **`Estimated Resolution (date)`** | Target Output | Derived Target Output (Date) | `2026-10-10` |

---

### Possible values for each column

| Column | Possible values or format |
| :--- | :--- |
| `complaint_id` | Unique identifier in the format `CC-###`; the 100 records in `test_records2.csv` use `CC-101` through `CC-200`. |
| `Complaint text` | Nonempty, free-form description of a community complaint. |
| `urgency level` | `High`, `Medium`, `Low`. |
| `location` | `A`, `B`, `C`, `D`, `E`, `V`, `W`, `X`, `Y`, `Z`. |
| `submission channel` | `Mobile App`, `Web Portal`, `Hotline`, `Walk-in`. |
| `date` | Valid calendar date in `YYYY-MM-DD` format. |
| `Complaint category` | `Waste and sanitation`, `Road and sidewalk repair`, `Drainage and flooding`, `Water supply and leaks`, `Street lighting`. |
| `routing suggestion` | Template: `{branch} {office} - {queue}`. `{branch}` is `Branch 1` for locations `A`–`E` or `Branch 2` for `V`–`Z`. `{office}` is `Sanitation Office`, `Public Works Office`, `Drainage Services`, `Water Utility`, or `Electrical Maintenance`, according to complaint category. `{queue}` is `Rapid Response` for `High`, or `General Queue` for `Medium` and `Low`. |
| `Estimated Resolution (date)` | Valid calendar date in `YYYY-MM-DD` format, calculated as `date` plus 1 day for `High`, 3 days for `Medium`, or 7 days for `Low`. |

---

## 2. Feature Mapping & Logic Workflow

### Output Dependencies & Mechanisms

#### 1. `Complaint category`
* **Inputs Used:** `Complaint text` (Primary), `urgency level` (Secondary)[cite: 1]
* **Mechanism:**
  * `Complaint text` is transformed into TF-IDF vector representations and fed into Logistic Regression or Naive Bayes classifiers[cite: 1].
  * `urgency level` can be one-hot encoded alongside TF-IDF vectors to provide secondary predictive signal for high-severity categories[cite: 1].

#### 2. `routing suggestion`
* **Inputs Used:** `Complaint category` (derived from text), `location`, `urgency level`[cite: 1]
* **Mechanism:**
  * **Department Level:** `Complaint category` dictates the primary functional office (e.g., Public Works, Health, Sanitation)[cite: 1].
  * **Branch Level:** Locations `A`–`E` route to `Branch 1`; locations `V`–`Z` route to `Branch 2`.
  * **Dispatch Type:** `urgency level` routes standard requests to general queues and critical issues to rapid-response units.

#### 3. `Estimated Resolution (date)`
* **Inputs Used:** `date`, `urgency level`, `Complaint category`[cite: 1]
* **Mechanism:** Calculated via deterministic date arithmetic:

$$\text{Estimated Resolution Date} = \text{Submission Date} + \text{SLA Days}(\text{Urgency Level}, \text{Complaint Category})$$

1. **`date`** sets the baseline submission anchor date ($T_0$)[cite: 1].
2. **`urgency level`** and **`Complaint category`** determine the SLA resolution offset ($\Delta T$ in days):
   * **High Urgency / Emergency:** $+1\text{ day}$
   * **Medium Urgency (Infrastructure / Utilities):** $+3\text{ days}$
   * **Low Urgency (Administrative / General Enquiry):** $+7\text{ days}$
3. **Calculation Example:**
   * Submission `date` = `2026-10-09`
   * `urgency level` = `High` ($\text{SLA} = 1\text{ day}$)
   * `Estimated Resolution (date)` = `2026-10-10`

---

## 3. Operational Metadata Specification

Operational metadata fields do not directly feed into the core text classification algorithm (TF-IDF NLP), but fulfill essential administrative and user interface requirements[cite: 1]:

1. **`submission channel`**
   * **Type:** Pure Operational Metadata[cite: 1]
   * **Function:** Tracks intake origin (`Mobile App`, `Web Portal`, `Hotline`, `Walk-in`)[cite: 1]. Used in Streamlit analytics dashboards for channel utilization tracking and accessibility analysis without introducing bias to model predictions[cite: 1].

2. **`date`**
   * **Type:** Operational Metadata & Arithmetic Parameter[cite: 1]
   * **Function:** Serves as the system submission timestamp for record indexing, audit tracking, and as the numerical anchor for target SLA date math[cite: 1].
