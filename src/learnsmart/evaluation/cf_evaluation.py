from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.learnsmart.logger.logger import logging
from src.learnsmart.exception.exception import LearnSmartAIException


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INTERACTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "interactions.csv"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_interactions():
    """
    Load interaction data from the raw dataset.
    """
    try:
        logging.info("Loading interaction data.")

        interactions = pd.read_csv(
            INTERACTIONS_PATH
        )

        logging.info(
            f"Interactions loaded: {interactions.shape}"
        )

        return interactions

    except Exception as e:
        logging.error(
            "Failed to load interaction data."
        )

        raise LearnSmartAIException(
            e,
            sys
        )


# ---------------------------------------------------------
# Train-Test Split
# ---------------------------------------------------------

def create_train_test(interactions):
    """
    Perform leave-one-out evaluation.

    For every student:
    - Select one completed course as test data.
    - Use all remaining interactions as training data.
    """

    try:
        logging.info(
            "Creating leave-one-out train-test split."
        )

        train_data = []
        test_data = []

        for student_id, group in interactions.groupby(
            "student_id"
        ):

            completed = group[
                group["completion_status"]
                == "Completed"
            ]

            # Skip students with no completed course
            if completed.empty:
                logging.warning(
                    f"No completed course for "
                    f"{student_id}. Skipping."
                )
                continue

            # Select one completed course
            test_row = completed.sample(
                n=1,
                random_state=42
            )

            # Remaining interactions become training data
            train_group = group.drop(
                test_row.index
            )

            train_data.append(
                train_group
            )

            test_data.append(
                test_row
            )

        train_df = pd.concat(
            train_data,
            ignore_index=True
        )

        test_df = pd.concat(
            test_data,
            ignore_index=True
        )

        logging.info(
            f"Train data: {train_df.shape}"
        )

        logging.info(
            f"Test data: {test_df.shape}"
        )

        return train_df, test_df

    except Exception as e:
        logging.error(
            "Failed to create train-test split."
        )

        raise LearnSmartAIException(
            e,
            sys
        )


# ---------------------------------------------------------
# Build Interaction Matrix
# ---------------------------------------------------------

def create_interaction_matrix(train_df):
    """
    Create student-course interaction matrix.
    """

    try:
        logging.info(
            "Creating student-course interaction matrix."
        )

        matrix = train_df.pivot(
            index="student_id",
            columns="course_id",
            values="interaction_score"
        ).fillna(0)

        logging.info(
            f"Interaction matrix shape: {matrix.shape}"
        )

        return matrix

    except Exception as e:
        logging.error(
            "Failed to create interaction matrix."
        )

        raise LearnSmartAIException(
            e,
            sys
        )


# ---------------------------------------------------------
# Calculate Item Similarity
# ---------------------------------------------------------

def calculate_item_similarity(
    interaction_matrix
):
    """
    Calculate item-item cosine similarity.
    """

    try:
        logging.info(
            "Calculating item-item cosine similarity."
        )

        # Student-course matrix
        # Shape:
        # students x courses
        #
        # Transpose:
        # courses x students

        item_matrix = (
            interaction_matrix.T
        )

        similarity = cosine_similarity(
            item_matrix
        )

        similarity_df = pd.DataFrame(
            similarity,
            index=item_matrix.index,
            columns=item_matrix.index
        )

        logging.info(
            "Item similarity calculated."
        )

        return similarity_df

    except Exception as e:
        logging.error(
            "Failed to calculate item similarity."
        )

        raise LearnSmartAIException(
            e,
            sys
        )


# ---------------------------------------------------------
# Generate Recommendations
# ---------------------------------------------------------

def recommend_courses(
    student_id,
    interaction_matrix,
    similarity_df,
    top_n=5
):
    """
    Generate item-based collaborative filtering
    recommendations for a student.
    """

    try:
        student_scores = (
            interaction_matrix.loc[student_id]
        )

        # Courses already interacted with
        interacted_courses = (
            student_scores[
                student_scores > 0
            ].index
        )

        if len(interacted_courses) == 0:
            return pd.Index([])

        # Weighted similarity score
        scores = (
            similarity_df[
                interacted_courses
            ]
            .dot(
                student_scores[
                    interacted_courses
                ]
            )
        )

        # Remove courses already interacted with
        scores = scores.drop(
            labels=interacted_courses,
            errors="ignore"
        )

        # Sort and select Top N
        recommendations = (
            scores
            .sort_values(
                ascending=False
            )
            .head(top_n)
        )

        return recommendations.index

    except Exception as e:
        logging.error(
            f"Failed to generate recommendations "
            f"for {student_id}."
        )

        raise LearnSmartAIException(
            e,
            sys
        )


# ---------------------------------------------------------
# Precision@K + Hit Rate@K
# ---------------------------------------------------------

def precision_at_k(
    train_df,
    test_df,
    k=5
):
    """
    Evaluate collaborative filtering using:
    - Precision@K
    - Hit Rate@K
    """

    try:
        logging.info(
            f"Starting Precision@{k} evaluation."
        )

        # Create training matrix
        matrix = create_interaction_matrix(
            train_df
        )

        # Calculate item similarity
        similarity_df = (
            calculate_item_similarity(
                matrix
            )
        )

        precisions = []

        hits_at_k = 0

        evaluated_students = 0

        # Evaluate every student
        for student_id in matrix.index:

            recommendations = recommend_courses(
                student_id=student_id,
                interaction_matrix=matrix,
                similarity_df=similarity_df,
                top_n=k
            )

            # Actual held-out course
            actual_courses = set(
                test_df[
                    test_df["student_id"]
                    == student_id
                ]["course_id"]
            )

            # Skip if no test course
            if not actual_courses:
                continue

            evaluated_students += 1

            # Calculate hits
            hits = len(
                set(recommendations)
                & actual_courses
            )

            # Hit Rate
            if hits > 0:
                hits_at_k += 1

            # Precision@K
            precision = hits / k

            precisions.append(
                precision
            )

            # Debug information for first student
            if (
                evaluated_students == 1
            ):
                print(
                    f"\nStudent: {student_id}"
                )

                print(
                    "Actual test course:",
                    actual_courses
                )

                print(
                    "Recommended:",
                    list(recommendations)
                )

                print(
                    "\nInteracted courses:"
                )

                interacted = matrix.loc[
                    student_id
                ]

                interacted = interacted[
                    interacted > 0
                ].index

                print(
                    list(interacted)
                )

        # Final metrics
        if not precisions:
            raise ValueError(
                "No students were available "
                "for evaluation."
            )

        precision_at_k_value = (
            sum(precisions)
            / len(precisions)
        )

        hit_rate_at_k = (
            hits_at_k
            / evaluated_students
        )

        print(
            f"\nStudents evaluated: "
            f"{evaluated_students}"
        )

        print(
            f"Hits at {k}: "
            f"{hits_at_k}"
        )

        print(
            f"Hit Rate@{k}: "
            f"{hit_rate_at_k:.4f}"
        )

        print(
            f"Precision@{k}: "
            f"{precision_at_k_value:.4f}"
        )

        return precision_at_k_value

    except Exception as e:
        logging.error(
            "Collaborative filtering evaluation failed."
        )

        raise LearnSmartAIException(
            e,
            sys
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    interactions = load_interactions()

    train_df, test_df = (
        create_train_test(
            interactions
        )
    )

    precision = precision_at_k(
        train_df=train_df,
        test_df=test_df,
        k=5
    )

    print(
        "\n========== EVALUATION SUMMARY =========="
    )

    print(
        f"Precision@5: {precision:.4f}"
    )

    print(
        f"Train interactions: "
        f"{len(train_df)}"
    )

    print(
        f"Test interactions: "
        f"{len(test_df)}"
    )

    print(
        f"Unique test students: "
        f"{test_df['student_id'].nunique()}"
    )