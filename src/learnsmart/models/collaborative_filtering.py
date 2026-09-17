from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.learnsmart.logger.logger import logging
from src.learnsmart.exception.exception import LearnSmartAIException


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = (
    PROJECT_ROOT
    / "data"
    / "transformed"
    / "interaction_matrix.csv"
)


def load_matrix():
    try:
        logging.info("Loading interaction matrix.")

        matrix = pd.read_csv(
            MATRIX_PATH,
            index_col=0
        )

        logging.info(
            f"Interaction matrix loaded: {matrix.shape}"
        )

        return matrix

    except Exception as e:
        logging.error("Failed to load interaction matrix.")
        raise LearnSmartAIException(e, sys)


def calculate_item_similarity(interaction_matrix):
    try:
        logging.info("Calculating item-item similarity.")

        item_matrix = interaction_matrix.T

        similarity = cosine_similarity(item_matrix)

        similarity_df = pd.DataFrame(
            similarity,
            index=item_matrix.index,
            columns=item_matrix.index
        )

        logging.info("Item similarity calculated.")

        return similarity_df

    except Exception as e:
        logging.error("Item similarity calculation failed.")
        raise LearnSmartAIException(e, sys)

def recommend_courses(
    student_id,
    interaction_matrix,
    similarity_df,
    top_n=5
):
    try:
        logging.info(
            f"Generating CF recommendations for {student_id}."
        )

        student_scores = interaction_matrix.loc[student_id]

        interacted_courses = student_scores[
            student_scores > 0
        ].index

        scores = similarity_df[interacted_courses].dot(
            student_scores[interacted_courses]
        )

        scores = scores.drop(
            labels=interacted_courses,
            errors="ignore"
        )

        recommendations = (
            scores
            .sort_values(ascending=False)
            .head(top_n)
        )

        logging.info(
            f"Generated {len(recommendations)} recommendations."
        )

        return recommendations

    except Exception as e:
        logging.error(
            f"Failed to generate recommendations for {student_id}."
        )
        raise LearnSmartAIException(e, sys)


if __name__ == "__main__":
    matrix = load_matrix()
    similarity = calculate_item_similarity(matrix)

    student_id = matrix.index[0]

    recommendations = recommend_courses(
        student_id,
        matrix,
        similarity,
        top_n=5
    )

    print(f"\nRecommendations for {student_id}:")
    print(recommendations)

