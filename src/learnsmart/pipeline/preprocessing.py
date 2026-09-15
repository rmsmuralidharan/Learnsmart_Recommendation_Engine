from pathlib import Path

import pandas as pd
import sys

from src.learnsmart.logger.logger import logging
from src.learnsmart.exception.exception import LearnSmartAIException

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" 


def load_data():
    """
    Load the raw synthetic datasets.
    """
    try:
        logging.info("Starting raw data loading.")

        students_path = RAW_DATA_DIR / "students.csv"
        courses_path = RAW_DATA_DIR / "courses.csv"
        interactions_path = RAW_DATA_DIR / "interactions.csv"

        logging.info("Loading students.csv.")
        students_df = pd.read_csv(students_path)

        logging.info("Loading courses.csv.")
        courses_df = pd.read_csv(courses_path)

        logging.info("Loading interactions.csv.")
        interactions_df = pd.read_csv(interactions_path)

        logging.info("All raw datasets loaded successfully.")

        return students_df, courses_df, interactions_df

    except Exception as e:
        logging.error("Error occurred while loading raw datasets.")
        raise LearnSmartAIException(e, sys)


### validating the data

def validate_data(students_df, courses_df, interactions_df):
    try:
        logging.info("Starting data validation.")

        assert students_df["student_id"].is_unique
        assert courses_df["course_id"].is_unique

        assert interactions_df["student_id"].isin(
            students_df["student_id"]
        ).all()

        assert interactions_df["course_id"].isin(
            courses_df["course_id"]
        ).all()

        assert interactions_df["quiz_score"].between(0, 100).all()
        assert interactions_df["interaction_score"].between(0, 1).all()

        assert not interactions_df.duplicated(
            ["student_id", "course_id"]
        ).any()

        logging.info("Data validation completed successfully.")
        return True

    except Exception as e:
        logging.error("Data validation failed.")
        raise LearnSmartAIException(e, sys)


def handle_missing_values(
    students_df,
    courses_df,
    interactions_df
):
    try:
        logging.info("Handling missing values.")

        # Beginner courses have no prerequisite
        courses_df["prerequisites"] = (
            courses_df["prerequisites"].fillna("None")
        )

        logging.info("Missing values handled successfully.")

        return students_df, courses_df, interactions_df

    except Exception as e:
        logging.error("Missing value handling failed.")
        raise LearnSmartAIException(e, sys)


def prepare_features(
    students_df,
    courses_df,
    interactions_df
):
    try:
        logging.info("Preparing model features.")

        students_df = students_df.copy()
        courses_df = courses_df.copy()
        interactions_df = interactions_df.copy()

        # Convert categorical features to numerical values
        experience_mapping = {
            "Beginner": 0,
            "Intermediate": 1,
            "Advanced": 2
        }

        students_df["experience_level_encoded"] = (
            students_df["experience_level"]
            .map(experience_mapping)
        )

        courses_df["difficulty_level_encoded"] = (
            courses_df["difficulty_level"]
            .map(experience_mapping)
        )

        # Convert completion status to numerical value
        interactions_df["completion_encoded"] = (
            interactions_df["completion_status"]
            .map({
                "Not completed": 0,
                "Completed": 1
            })
        )

        logging.info("Model features prepared successfully.")

        return students_df, courses_df, interactions_df

    except Exception as e:
        logging.error("Feature preparation failed.")
        raise LearnSmartAIException(e, sys)


def create_interaction_matrix(interactions_df):
    try:
        logging.info("Creating student-course interaction matrix.")

        interaction_matrix = interactions_df.pivot(
            index="student_id",
            columns="course_id",
            values="interaction_score"
        ).fillna(0)

        logging.info(
            f"Interaction matrix created: "
            f"{interaction_matrix.shape}"
        )

        return interaction_matrix

    except Exception as e:
        logging.error("Interaction matrix creation failed.")
        raise LearnSmartAIException(e, sys)

def save_processed_data(
    students_df,
    courses_df,
    interactions_df,
    interaction_matrix
):
    try:
        logging.info("Saving processed data.")

        transformed_dir = (
            PROJECT_ROOT / "data" / "transformed"
        )
        transformed_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        students_df.to_csv(
            transformed_dir / "students_processed.csv",
            index=False
        )
        courses_df.to_csv(
            transformed_dir / "courses_processed.csv",
            index=False
        )
        interactions_df.to_csv(
            transformed_dir / "interactions_processed.csv",
            index=False
        )
        interaction_matrix.to_csv(
            transformed_dir / "interaction_matrix.csv"
        )

        logging.info("Processed data saved successfully.")

    except Exception as e:
        logging.error("Failed to save processed data.")
        raise LearnSmartAIException(e, sys)


def encode_and_scale_features(
        students_df,
        courses_df,
        interactions_df
):
    try:
        logging.info("Encoding and scaling features.")

        # One-hot encode student categorical features
        student_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )

        student_encoded = student_encoder.fit_transform(
            students_df[["experience_level", "learning_mode"]]
        )

        student_encoded_df = pd.DataFrame(
            student_encoded,
            columns=student_encoder.get_feature_names_out(
                ["experience_level", "learning_mode"]
            )
        )

        # One-hot encode course categorical features
        course_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )

        course_encoded = course_encoder.fit_transform(
            courses_df[["category"]]
        )

        course_encoded_df = pd.DataFrame(
            course_encoded,
            columns=course_encoder.get_feature_names_out(
                ["category"]
            )
        )

        # Scale numerical interaction features
        scaler = StandardScaler()

        numerical_features = interactions_df[
            ["quiz_score", "time_spent", "engagement_score"]
        ]

        scaled_features = scaler.fit_transform(
            numerical_features
        )

        scaled_features_df = pd.DataFrame(
            scaled_features,
            columns=[
                "quiz_score_scaled",
                "time_spent_scaled",
                "engagement_score_scaled"
            ]
        )

        logging.info("Encoding and scaling completed.")

        return (
            student_encoded_df,
            course_encoded_df,
            scaled_features_df
        )

    except Exception as e:
        logging.error("Encoding and scaling failed.")
        raise LearnSmartAIException(e, sys)


def prepare_content_features(students_df, courses_df):
    try:
        logging.info("Preparing content-based features.")

        # Multi-label encode student interests
        students_copy = students_df.copy()
        students_copy["interests_list"] = (
            students_copy["interests"].str.split(", ")
        )

        mlb = MultiLabelBinarizer()

        interests_encoded = mlb.fit_transform(
            students_copy["interests_list"]
        )

        interests_df = pd.DataFrame(
            interests_encoded,
            columns=[
                f"interest_{name}"
                for name in mlb.classes_
            ]
        )

        # TF-IDF for course descriptions
        tfidf = TfidfVectorizer(
            stop_words="english"
        )

        description_tfidf = tfidf.fit_transform(
            courses_df["description"]
        )

        description_df = pd.DataFrame(
            description_tfidf.toarray(),
            columns=[
                f"tfidf_{word}"
                for word in tfidf.get_feature_names_out()
            ]
        )

        logging.info(
            "Content-based features prepared successfully."
        )

        return interests_df, description_df

    except Exception as e:
        logging.error("Content feature preparation failed.")
        raise LearnSmartAIException(e, sys)


if __name__ == "__main__":
    try:
        students_df, courses_df, interactions_df = load_data()

        print("Students:", students_df.shape)
        print("Courses:", courses_df.shape)
        print("Interactions:", interactions_df.shape)

        validate_data(
            students_df,
            courses_df,
            interactions_df
        )

        students_df, courses_df, interactions_df = (
            handle_missing_values(
                students_df,
                courses_df,
                interactions_df
            )
        )

        students_df, courses_df, interactions_df = (
            prepare_features(
                students_df,
                courses_df,
                interactions_df
            )
        )

        interaction_matrix = create_interaction_matrix(
            interactions_df
        )

        save_processed_data(
            students_df,
            courses_df,
            interactions_df,
            interaction_matrix
        )

        student_encoded_df, course_encoded_df, scaled_features_df = (
            encode_and_scale_features(
                students_df,
                courses_df,
                interactions_df
            )
        )

        interests_df, description_df = prepare_content_features(
            students_df,
            courses_df
        )

        print("Interest features:", interests_df.shape)
        print("TF-IDF features:", description_df.shape)

    except Exception as e:
        logging.error("Preprocessing pipeline failed.")
        raise LearnSmartAIException(e, sys)

