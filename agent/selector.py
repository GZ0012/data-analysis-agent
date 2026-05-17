import anthropic
from pydantic import BaseModel
from agent.profiler import DataProfile


SYSTEM_PROMPT = """You are an expert data scientist. Given a dataset profile and a user's analysis goal,
decide the best analysis method and which columns to use.

Supervised learning tasks:
- "linear_regression"   : predict a continuous numeric target
- "logistic_regression" : classify a binary or multi-class target (interpretable coefficients)
- "random_forest"       : classification or regression, handles non-linear patterns well
- "svm"                 : classification or regression with kernel trick
- "knn"                 : simple distance-based classification or regression

Unsupervised learning tasks:
- "kmeans"  : find natural clusters in data (auto-selects K if not specified)
- "pca"     : reduce dimensionality, understand variance structure
- "dbscan"  : density-based clustering, good for irregular shapes and noise detection

Rules:
- Choose supervised if the user mentions a target variable or wants to predict / classify something.
- Choose unsupervised if the user wants to find patterns, groups, or explore without a named target.
- For regression: target must be numeric. For classification: target must be categorical or binary.
- Only include columns that make analytical sense as features — exclude free-text IDs.
- __was_missing indicator columns are valid features; include them if they exist."""


class AnalysisPlan(BaseModel):
    method_type: str          # "supervised" | "unsupervised"
    task: str                 # one of the task strings above
    target_column: str | None
    feature_columns: list[str]
    reasoning: str
    suggested_algorithms: list[str]

    @property
    def module_path(self) -> str:
        supervised = {"linear_regression", "logistic_regression", "random_forest", "svm", "knn"}
        unsupervised = {"kmeans", "pca", "dbscan"}
        if self.task in supervised:
            return f"methods.supervised.{self.task}"
        if self.task in unsupervised:
            return f"methods.unsupervised.{self.task}"
        raise ValueError(f"Unknown task: {self.task}")


def select_analysis(
    profile: DataProfile,
    user_goal: str,
    client: anthropic.Anthropic,
) -> AnalysisPlan:
    user_message = f"""Dataset Profile:
{profile.to_llm_summary()}

User's Analysis Goal:
{user_goal}

Choose the best analysis method and return your decision."""

    response = client.messages.parse(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
        output_format=AnalysisPlan,
    )

    return response.parsed_output
