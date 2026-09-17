from pathlib import Path
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.learnsmart.logger.logger import logging
from src.learnsmart.exception.exception import LearnSmartAIException


PROJECT_ROOT = Path(__file__).resolve().parents[3]

STUDENTS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "students.csv"
)

COURSES_PATH = (
    PROJECT_ROOT / "data" / "raw" / "courses.csv"
)

INTERACTIONS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "interactions.csv"
)


def load_data():
    """Load student, course, and interaction data."""
    try:
        logging.info("Loading data for content-based filtering.")

        students = pd.read_csv(STUDENTS_PATH)
        courses = pd.read_csv(COURSES_PATH)
        interactions = pd.read_csv(INTERACTIONS_PATH)

        logging.info(
            f"Students: {students.shape}, "
            f"Courses: {courses.shape}, "
            f"Interactions: {interactions.shape}"
        )

        return students, courses, interactions

    except Exception as e:
        logging.error("Failed to load content-based data.")
        raise LearnSmartAIException(e, sys)


def create_course_features(courses):
    """Create TF-IDF features from course content."""
    try:
        logging.info("Creating course TF-IDF features.")

        courses = courses.copy()

        courses["course_text"] = (
            courses["category"].fillna("")
            + " "
            + courses["difficulty_level"].fillna("")
            + " "
            + courses["description"].fillna("")
        )

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        course_features = vectorizer.fit_transform(
            courses["course_text"]
        )

        logging.info(
            f"Course feature matrix: "
            f"{course_features.shape}"
        )

        return course_features

    except Exception as e:
        logging.error(
            "Failed to create course features."
        )
        raise LearnSmartAIException(e, sys)


def calculate_course_similarity(course_features):
    """Calculate cosine similarity between courses."""
    try:
        logging.info(
            "Calculating course content similarity."
        )

        similarity = cosine_similarity(
            course_features
        )

        logging.info(
            "Course content similarity calculated."
        )

        return similarity

    except Exception as e:
        logging.error(
            "Failed to calculate course similarity."
        )
        raise LearnSmartAIException(e, sys)


def recommend_courses(
    student_id,
    students,
    courses,
    interactions,
    similarity,
    top_n=5
):
    """Generate content-based recommendations."""
    try:
        logging.info(
            f"Generating content recommendations "
            f"for {student_id}."
        )

        student = students[
            students["student_id"] == student_id
        ]

        if student.empty:
            raise ValueError(
                f"Student {student_id} not found."
            )

        student_interactions = interactions[
            interactions["student_id"] == student_id
        ]

        interacted_courses = set(
            student_interactions["course_id"]
        )

        if not interacted_courses:
            return courses.head(top_n)[
                "course_id"
            ].tolist()

        course_index = {
            course_id: index
            for index, course_id
            in enumerate(courses["course_id"])
        }

        scores = {}

        for course_id in courses["course_id"]:

            if course_id in interacted_courses:
                continue

            course_idx = course_index[course_id]

            similarities = []

            for interacted_id in interacted_courses:

                if interacted_id not in course_index:
                    continue

                interacted_idx = (
                    course_index[interacted_id]
                )

                similarities.append(
                    similarity[
                        course_idx,
                        interacted_idx
                    ]
                )

            if similarities:
                scores[course_id] = (
                    sum(similarities)
                    / len(similarities)
                )

        recommendations = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return pd.Series(
            dict(recommendations)
        )

    except Exception as e:
        logging.error(
            f"Failed to generate recommendations "
            f"for {student_id}."
        )
        raise LearnSmartAIException(e, sys)


if __name__ == "__main__":

    students, courses, interactions = (
        load_data()
    )

    course_features = create_course_features(
        courses
    )

    similarity = calculate_course_similarity(
        course_features
    )

    student_id = students.iloc[0]["student_id"]

    recommendations = recommend_courses(
        student_id=student_id,
        students=students,
        courses=courses,
        interactions=interactions,
        similarity=similarity,
        top_n=5
    )

    print(
        f"\nContent-Based Recommendations "
        f"for {student_id}:"
    )

    print(recommendations)