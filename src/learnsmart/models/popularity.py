from pathlib import Path

import pandas as pd

from src.learnsmart.exception.exception import LearnSmartAIException
from src.learnsmart.logger.logger import logging


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "interactions.csv"
)


def load_interactions():
    try:
        logging.info("Loading interaction data.")

        interactions_df = pd.read_csv(DATA_PATH)

        logging.info("Interaction data loaded successfully.")

        return interactions_df

    except Exception as e:
        logging.error("Failed to load interaction data.")
        raise LearnSmartAIException(e, __import__("sys"))


def get_popular_courses(interactions_df, top_n=5):
    try:
        logging.info("Calculating popular courses.")

        popular_courses = (
            interactions_df
            .groupby("course_id")
            .agg(
                interaction_count=("course_id", "size"),
                average_score=("interaction_score", "mean")
            )
            .sort_values(
                ["interaction_count", "average_score"],
                ascending=False
            )
            .head(top_n)
            .reset_index()
        )

        logging.info(
            f"Top {top_n} popular courses calculated."
        )

        return popular_courses

    except Exception as e:
        logging.error("Failed to calculate popular courses.")
        raise LearnSmartAIException(e, __import__("sys"))


if __name__ == "__main__":
    interactions_df = load_interactions()

    popular_courses = get_popular_courses(
        interactions_df,
        top_n=5
    )

    print("\nTop 5 Popular Courses:")
    print(popular_courses)