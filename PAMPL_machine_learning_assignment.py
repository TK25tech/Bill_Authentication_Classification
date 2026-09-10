"""
PAMPL Set Exercise: Implementation of Machine Learning Models Using Python

The script answers the five tasks in the assignment brief:
1. Load and clean bill_authentication.csv.
2. Build a decision-tree classification model.
3. Perform k-means clustering.
4. Evaluate the classification model.
5. Build and evaluate a linear-regression model.

Run in PyCharm after placing bill_authentication.csv in the same folder:
    python PAMPL_machine_learning_assignment.py

For a differently named or located file, pass its path:
    python PAMPL_machine_learning_assignment.py "bill_authentication (1).csv"
"""

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "bill_authentication.csv"
OUTPUT_DIR = BASE_DIR / "assignment_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FEATURES = ["Variance", "Skewness", "Curtosis", "Entropy"]
CLASS_TARGET = "Class"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
warnings.filterwarnings("ignore", category=FutureWarning)


def section(title):
    """Print a clear heading in the console."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def save_figure(filename):
    """Save the current Matplotlib figure at report quality."""
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


# =============================================================================
# TASK 1 (20 MARKS): LOAD AND CLEAN THE DATASET
# =============================================================================

section("TASK 1: DATA LOADING, INSPECTION AND CLEANING")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place bill_authentication.csv in the same folder as this script, "
        "or pass the file path as a command-line argument."
    )

df_raw = pd.read_csv(DATA_FILE)

# Standardise headings without changing their meaning.
df_raw.columns = df_raw.columns.str.strip().str.title()

required_columns = FEATURES + [CLASS_TARGET]
missing_columns = sorted(set(required_columns) - set(df_raw.columns))
unexpected_columns = sorted(set(df_raw.columns) - set(required_columns))

if missing_columns:
    raise ValueError(f"Required columns are missing: {missing_columns}")
if unexpected_columns:
    print(f"Warning - unexpected columns found and retained: {unexpected_columns}")

print(f"Data file: {DATA_FILE}")
print(f"Original dimensions: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
print("\nFirst five rows:")
print(df_raw.head())
print("\nData types:")
print(df_raw.dtypes)

# Coerce required variables to numeric. Invalid text becomes NaN and is counted.
df = df_raw.copy()
for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

missing_before = int(df[required_columns].isna().sum().sum())
duplicates_before = int(df.duplicated().sum())
invalid_class_before = int((~df[CLASS_TARGET].isin([0, 1]) & df[CLASS_TARGET].notna()).sum())

# Remove unusable observations. The supplied file has no missing or invalid values,
# but the checks make the workflow transparent and reproducible.
df = df.dropna(subset=required_columns).copy()
df = df[df[CLASS_TARGET].isin([0, 1])].copy()
df = df.drop_duplicates().reset_index(drop=True)
df[CLASS_TARGET] = df[CLASS_TARGET].astype(int)

cleaning_summary = pd.DataFrame(
    {
        "Check": [
            "Missing required values",
            "Exact duplicate rows",
            "Invalid class values",
        ],
        "Number_found": [missing_before, duplicates_before, invalid_class_before],
        "Treatment": [
            "Remove rows only if required model values are missing",
            "Remove exact duplicates before model splitting",
            "Keep only valid binary labels 0 and 1",
        ],
        "Reason": [
            "Models require complete numeric inputs",
            "Prevents repeated records and possible train-test leakage",
            "The classification target must be binary",
        ],
    }
)

data_audit = pd.DataFrame(
    {
        "Measure": ["Rows", "Columns", "Missing values", "Duplicate rows"],
        "Before_cleaning": [
            len(df_raw),
            df_raw.shape[1],
            missing_before,
            duplicates_before,
        ],
        "After_cleaning": [
            len(df),
            df.shape[1],
            int(df[required_columns].isna().sum().sum()),
            int(df.duplicated().sum()),
        ],
    }
)

descriptive_statistics = df[required_columns].describe().T
class_distribution = (
    df[CLASS_TARGET]
    .value_counts()
    .sort_index()
    .rename_axis("Class")
    .reset_index(name="Count")
)
class_distribution["Percentage"] = 100 * class_distribution["Count"] / len(df)

cleaning_summary.to_csv(TABLE_DIR / "task1_cleaning_summary.csv", index=False)
data_audit.to_csv(TABLE_DIR / "task1_data_audit.csv", index=False)
descriptive_statistics.to_csv(TABLE_DIR / "task1_descriptive_statistics.csv")
class_distribution.to_csv(TABLE_DIR / "task1_class_distribution.csv", index=False)
df.to_csv(TABLE_DIR / "cleaned_banknote_data.csv", index=False)

print("\nCleaning summary:")
print(cleaning_summary.to_string(index=False))
print("\nBefore-and-after audit:")
print(data_audit.to_string(index=False))
print("\nClass distribution:")
print(class_distribution.to_string(index=False, formatters={"Percentage": "{:.2f}".format}))
print("\nDescriptive statistics:")
print(descriptive_statistics.round(3))

# Figure 1: distributions of all four features by known class.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for feature, ax in zip(FEATURES, axes.flatten()):
    sns.histplot(
        data=df,
        x=feature,
        hue=CLASS_TARGET,
        kde=True,
        stat="density",
        common_norm=False,
        element="step",
        ax=ax,
        palette="Set1",
    )
    ax.set_title(f"Distribution of {feature} by Class")
fig.suptitle("Banknote Feature Distributions", fontsize=16, y=1.02)
save_figure("task1_feature_distributions.png")

# Figure 2: correlation heatmap for initial relationships and multicollinearity.
plt.figure(figsize=(8, 6))
sns.heatmap(df[required_columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Banknote Variables")
save_figure("task1_correlation_heatmap.png")


# =============================================================================
# TASK 2 (20 MARKS): DECISION-TREE CLASSIFICATION
# =============================================================================

section("TASK 2: DECISION-TREE CLASSIFICATION")

X = df[FEATURES]
y = df[CLASS_TARGET]

# Stratification preserves the class proportions in both partitions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training observations: {len(X_train)}")
print(f"Testing observations:  {len(X_test)}")
print("Training class proportions:")
print(y_train.value_counts(normalize=True).sort_index().round(3))
print("Testing class proportions:")
print(y_test.value_counts(normalize=True).sort_index().round(3))

# A dummy model provides a minimum benchmark for meaningful performance.
dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit(X_train, y_train)

# Tune only on training data. The test set remains untouched until Task 4.
decision_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
parameter_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5, 10],
}

classification_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
tree_search = GridSearchCV(
    estimator=decision_tree,
    param_grid=parameter_grid,
    scoring="f1",
    cv=classification_cv,
    n_jobs=-1,
    return_train_score=True,
)
tree_search.fit(X_train, y_train)

best_tree = tree_search.best_estimator_
print(f"Best cross-validated F1 score: {tree_search.best_score_:.4f}")
print(f"Best decision-tree parameters: {tree_search.best_params_}")

tree_cv_results = pd.DataFrame(tree_search.cv_results_).sort_values("rank_test_score")
tree_cv_results[
    [
        "params",
        "mean_train_score",
        "std_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
].head(15).to_csv(TABLE_DIR / "task2_top_tuning_results.csv", index=False)

feature_importance = pd.DataFrame(
    {"Feature": FEATURES, "Importance": best_tree.feature_importances_}
).sort_values("Importance", ascending=False)
feature_importance.to_csv(TABLE_DIR / "task2_feature_importance.csv", index=False)

print("\nFeature importance:")
print(feature_importance.to_string(index=False, formatters={"Importance": "{:.4f}".format}))

# Figure 3: the top levels remain readable even if the selected model is deeper.
plt.figure(figsize=(18, 9))
plot_tree(
    best_tree,
    feature_names=FEATURES,
    class_names=["Class 0", "Class 1"],
    filled=True,
    rounded=True,
    proportion=True,
    precision=2,
    max_depth=3,
    fontsize=8,
)
plt.title("Decision Tree (First Four Levels Shown)")
save_figure("task2_decision_tree.png")

# Figure 4: feature importance.
plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x="Importance", y="Feature", color="#2A6FBB")
plt.title("Decision-Tree Feature Importance")
plt.xlim(0, max(feature_importance["Importance"].max() * 1.15, 0.05))
save_figure("task2_feature_importance.png")


# =============================================================================
# TASK 3 (20 MARKS): K-MEANS CLUSTERING
# =============================================================================

section("TASK 3: K-MEANS CLUSTERING")

# The true Class is deliberately excluded because k-means is unsupervised.
cluster_features = df[FEATURES]
cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(cluster_features)

k_values = range(2, 9)
cluster_selection_rows = []

for k in k_values:
    candidate_model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        random_state=RANDOM_STATE,
    )
    candidate_labels = candidate_model.fit_predict(X_cluster_scaled)
    cluster_selection_rows.append(
        {
            "k": k,
            "Inertia": candidate_model.inertia_,
            "Silhouette_score": silhouette_score(X_cluster_scaled, candidate_labels),
        }
    )

cluster_selection = pd.DataFrame(cluster_selection_rows)
selected_k = int(cluster_selection.loc[cluster_selection["Silhouette_score"].idxmax(), "k"])

print("Cluster-selection results:")
print(cluster_selection.round(4).to_string(index=False))
print(f"Selected k using the highest silhouette score: {selected_k}")

cluster_selection.to_csv(TABLE_DIR / "task3_cluster_selection_metrics.csv", index=False)

final_kmeans = KMeans(
    n_clusters=selected_k,
    init="k-means++",
    n_init=20,
    random_state=RANDOM_STATE,
)
cluster_labels = final_kmeans.fit_predict(X_cluster_scaled)

df_clustered = df.copy()
df_clustered["Cluster"] = cluster_labels

# Convert centres back to the original measurement scale for interpretation.
cluster_centres = pd.DataFrame(
    cluster_scaler.inverse_transform(final_kmeans.cluster_centers_),
    columns=FEATURES,
)
cluster_centres.insert(0, "Cluster", range(selected_k))

cluster_sizes = (
    df_clustered["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Count")
)
cluster_sizes["Percentage"] = 100 * cluster_sizes["Count"] / len(df_clustered)

# The target is used only after clustering to assess agreement, not to create clusters.
cluster_class_table = pd.crosstab(
    df_clustered["Cluster"],
    df_clustered[CLASS_TARGET],
    rownames=["Cluster"],
    colnames=["Known_class"],
)
ari = adjusted_rand_score(df_clustered[CLASS_TARGET], df_clustered["Cluster"])

cluster_centres.to_csv(TABLE_DIR / "task3_cluster_centres_original_scale.csv", index=False)
cluster_sizes.to_csv(TABLE_DIR / "task3_cluster_sizes.csv", index=False)
cluster_class_table.to_csv(TABLE_DIR / "task3_cluster_class_crosstab.csv")
df_clustered.to_csv(TABLE_DIR / "task3_clustered_data.csv", index=False)

print("\nCluster centres on the original scale:")
print(cluster_centres.round(3).to_string(index=False))
print("\nCluster sizes:")
print(cluster_sizes.to_string(index=False, formatters={"Percentage": "{:.2f}".format}))
print("\nPost-hoc cluster versus known-class comparison:")
print(cluster_class_table)
print(f"Adjusted Rand Index: {ari:.4f}")

# Figure 5: elbow and silhouette evidence together.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.lineplot(data=cluster_selection, x="k", y="Inertia", marker="o", ax=axes[0])
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (k)")
sns.lineplot(
    data=cluster_selection,
    x="k",
    y="Silhouette_score",
    marker="o",
    color="#C43C39",
    ax=axes[1],
)
axes[1].axvline(selected_k, color="black", linestyle="--", label=f"Selected k = {selected_k}")
axes[1].set_title("Silhouette Analysis")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].legend()
save_figure("task3_cluster_selection.png")

# Figure 6: PCA is used only to display the four-dimensional clusters in 2D.
pca = PCA(n_components=2)
cluster_coordinates = pca.fit_transform(X_cluster_scaled)
pca_centres = pca.transform(final_kmeans.cluster_centers_)

pca_plot_data = pd.DataFrame(
    {
        "PC1": cluster_coordinates[:, 0],
        "PC2": cluster_coordinates[:, 1],
        "Cluster": cluster_labels.astype(str),
    }
)

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_plot_data,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    alpha=0.65,
    s=45,
)
plt.scatter(
    pca_centres[:, 0],
    pca_centres[:, 1],
    marker="X",
    s=220,
    c="black",
    label="Centres",
)
explained = 100 * pca.explained_variance_ratio_
plt.xlabel(f"PC1 ({explained[0]:.1f}% variance explained)")
plt.ylabel(f"PC2 ({explained[1]:.1f}% variance explained)")
plt.title("K-Means Clusters Displayed Using PCA")
plt.legend(title="Cluster")
save_figure("task3_pca_clusters.png")


# =============================================================================
# TASK 4 (20 MARKS): EVALUATE THE CLASSIFICATION ALGORITHM
# =============================================================================

section("TASK 4: CLASSIFICATION EVALUATION")

tree_train_predictions = best_tree.predict(X_train)
tree_test_predictions = best_tree.predict(X_test)
tree_test_probabilities = best_tree.predict_proba(X_test)[:, 1]
dummy_test_predictions = dummy_classifier.predict(X_test)

classification_metrics = pd.DataFrame(
    [
        {
            "Model": "Majority-class baseline",
            "Training_accuracy": accuracy_score(y_train, dummy_classifier.predict(X_train)),
            "Test_accuracy": accuracy_score(y_test, dummy_test_predictions),
            "Test_precision": precision_score(y_test, dummy_test_predictions, zero_division=0),
            "Test_recall": recall_score(y_test, dummy_test_predictions, zero_division=0),
            "Test_F1": f1_score(y_test, dummy_test_predictions, zero_division=0),
            "Test_ROC_AUC": 0.5,
        },
        {
            "Model": "Tuned decision tree",
            "Training_accuracy": accuracy_score(y_train, tree_train_predictions),
            "Test_accuracy": accuracy_score(y_test, tree_test_predictions),
            "Test_precision": precision_score(y_test, tree_test_predictions, zero_division=0),
            "Test_recall": recall_score(y_test, tree_test_predictions, zero_division=0),
            "Test_F1": f1_score(y_test, tree_test_predictions, zero_division=0),
            "Test_ROC_AUC": roc_auc_score(y_test, tree_test_probabilities),
        },
    ]
)

classification_report_table = pd.DataFrame(
    classification_report(
        y_test,
        tree_test_predictions,
        target_names=["Class 0", "Class 1"],
        output_dict=True,
        zero_division=0,
    )
).T

cv_f1_scores = cross_val_score(
    best_tree,
    X_train,
    y_train,
    cv=classification_cv,
    scoring="f1",
    n_jobs=-1,
)

classification_metrics.to_csv(TABLE_DIR / "task4_classification_metrics.csv", index=False)
classification_report_table.to_csv(TABLE_DIR / "task4_classification_report.csv")

print("Classification metrics:")
print(classification_metrics.round(4).to_string(index=False))
print("\nDetailed classification report:")
print(classification_report_table.round(4))
print(
    f"\nFive-fold training CV F1: {cv_f1_scores.mean():.4f} "
    f"(+/- {cv_f1_scores.std():.4f})"
)
print(
    "Training-test accuracy gap: "
    f"{accuracy_score(y_train, tree_train_predictions) - accuracy_score(y_test, tree_test_predictions):.4f}"
)

# Figure 7: confusion matrix.
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    tree_test_predictions,
    display_labels=["Class 0", "Class 1"],
    cmap="Blues",
    colorbar=False,
    ax=ax,
)
ax.set_title("Decision-Tree Confusion Matrix")
save_figure("task4_confusion_matrix.png")

# Figure 8: ROC curve.
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, tree_test_probabilities)
test_auc = roc_auc_score(y_test, tree_test_probabilities)

plt.figure(figsize=(7, 6))
plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"Decision tree (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Curve")
plt.legend(loc="lower right")
save_figure("task4_roc_curve.png")


# =============================================================================
# TASK 5 (20 MARKS): EVALUATE A LINEAR-REGRESSION ALGORITHM
# =============================================================================

section("TASK 5: LINEAR-REGRESSION EVALUATION")

# The dataset's intended outcome is categorical. For this required regression
# exercise, Variance is treated as a continuous response and predicted from the
# other image measurements. Class is excluded to avoid using the known label.
REGRESSION_TARGET = "Variance"
REGRESSION_FEATURES = ["Skewness", "Curtosis", "Entropy"]

X_regression = df[REGRESSION_FEATURES]
y_regression = df[REGRESSION_TARGET]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_regression,
    y_regression,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

regression_model = LinearRegression()
regression_model.fit(X_reg_train, y_reg_train)

regression_baseline = DummyRegressor(strategy="mean")
regression_baseline.fit(X_reg_train, y_reg_train)

regression_predictions = regression_model.predict(X_reg_test)
regression_train_predictions = regression_model.predict(X_reg_train)
baseline_regression_predictions = regression_baseline.predict(X_reg_test)
residuals = y_reg_test - regression_predictions


def regression_scores(actual, predicted):
    """Return commonly reported regression evaluation measures."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R_squared": r2_score(actual, predicted),
    }


linear_scores = regression_scores(y_reg_test, regression_predictions)
baseline_scores = regression_scores(y_reg_test, baseline_regression_predictions)

regression_metrics = pd.DataFrame(
    [
        {"Model": "Mean baseline", **baseline_scores},
        {"Model": "Multiple linear regression", **linear_scores},
    ]
)

coefficient_table = pd.DataFrame(
    {
        "Predictor": REGRESSION_FEATURES,
        "Coefficient": regression_model.coef_,
    }
)
coefficient_table.loc[len(coefficient_table)] = ["Intercept", regression_model.intercept_]

# Cross-validation is performed on training data, leaving the final test result independent.
regression_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_negative_mse = cross_val_score(
    regression_model,
    X_reg_train,
    y_reg_train,
    cv=regression_cv,
    scoring="neg_mean_squared_error",
)
cv_rmse_scores = np.sqrt(-cv_negative_mse)

# Shapiro-Wilk is reported as a residual diagnostic, not as proof of model validity.
shapiro_statistic, shapiro_p_value = stats.shapiro(residuals)
train_r_squared = r2_score(y_reg_train, regression_train_predictions)

regression_diagnostics = pd.DataFrame(
    {
        "Diagnostic": [
            "Training R-squared",
            "Test R-squared",
            "Mean training CV RMSE",
            "Training CV RMSE standard deviation",
            "Mean test residual",
            "Shapiro-Wilk statistic",
            "Shapiro-Wilk p-value",
        ],
        "Value": [
            train_r_squared,
            linear_scores["R_squared"],
            cv_rmse_scores.mean(),
            cv_rmse_scores.std(),
            residuals.mean(),
            shapiro_statistic,
            shapiro_p_value,
        ],
    }
)

regression_metrics.to_csv(TABLE_DIR / "task5_regression_metrics.csv", index=False)
coefficient_table.to_csv(TABLE_DIR / "task5_regression_coefficients.csv", index=False)
regression_diagnostics.to_csv(TABLE_DIR / "task5_regression_diagnostics.csv", index=False)

print("Regression metrics:")
print(regression_metrics.round(4).to_string(index=False))
print("\nRegression coefficients:")
print(coefficient_table.round(4).to_string(index=False))
print("\nRegression diagnostics:")
print(regression_diagnostics.round(4).to_string(index=False))

# Figure 9: actual versus predicted values.
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_reg_test, y=regression_predictions, alpha=0.7, color="#2A6FBB")
minimum = min(y_reg_test.min(), regression_predictions.min())
maximum = max(y_reg_test.max(), regression_predictions.max())
plt.plot([minimum, maximum], [minimum, maximum], linestyle="--", color="black", label="Perfect prediction")
plt.xlabel("Actual Variance")
plt.ylabel("Predicted Variance")
plt.title("Linear Regression: Actual vs Predicted Variance")
plt.legend()
save_figure("task5_actual_vs_predicted.png")

# Figure 10: residuals against predictions.
plt.figure(figsize=(7, 6))
sns.scatterplot(x=regression_predictions, y=residuals, alpha=0.7, color="#C43C39")
plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Predicted Variance")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Values")
save_figure("task5_residual_plot.png")

# Figure 11: residual histogram and Q-Q plot.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=axes[0], color="#2A6FBB")
axes[0].set_title("Distribution of Test Residuals")
axes[0].set_xlabel("Residual")
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Normal Q-Q Plot of Test Residuals")
save_figure("task5_residual_diagnostics.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section("ANALYSIS COMPLETE")
print(f"Cleaned observations used: {len(df)}")
print(f"Selected decision-tree parameters: {tree_search.best_params_}")
print(f"Decision-tree test F1 score: {f1_score(y_test, tree_test_predictions):.4f}")
print(f"Selected number of k-means clusters: {selected_k}")
print(f"K-means Adjusted Rand Index: {ari:.4f}")
print(f"Linear-regression test RMSE: {linear_scores['RMSE']:.4f}")
print(f"Linear-regression test R-squared: {linear_scores['R_squared']:.4f}")
print(f"\nAll figures and tables are saved in: {OUTPUT_DIR}")

