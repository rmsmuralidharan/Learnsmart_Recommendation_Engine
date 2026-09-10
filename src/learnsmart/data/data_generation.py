import pandas as pd
import numpy as np
import faker
from pathlib import Path


NUM_STUDENTS = 5000
NUM_COURSES = 500
NUM_INTERACTIONS = 100000
RANDOM_SEED = 42

### setting the path for the generated data to store in
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'raw'



INTERESTS = [
    "Python",
    "Data Science",
    "Machine Learning",
    "Artificial Intelligence",
    "SQL",
    "Web Development",
    "Cloud Computing",
    "DevOps",
    "Cybersecurity",
    "Data Analytics",
    "Deep Learning",
    "Software Development",
]

EXPERIENCE_LEVEL = ['Beginner', 'Intermediate', 'Advanced']

LEARNING_STYLE = ['Visual', 'practical', 'Reading']

def generate_students():

    students = []

    ## generate student ids
    NUM_STUDENTS = 5000

    for i in range(1, NUM_STUDENTS + 1):
        student_id = f"STU{i:05d}"

        ### generate minimum 1 to 3 interests for every students

        num_interests = np.random.randint(1,4)

        select_interests = np.random.choice(
            INTERESTS,
            size=num_interests,
            replace=False
        )

        interests = ", ".join(select_interests)

        ### generate experience levels for each students

        experience_level = np.random.choice(EXPERIENCE_LEVEL)

        ## generate learning mode
        learning_style = np.random.choice(LEARNING_STYLE)


        ### create the data structure
        student = {
            "student_id": student_id,
            "interests": interests,
            "experience_level": experience_level,
            "learning_mode": learning_style,
        }

        students.append(student)


    ### convert the lists into dataframe
    students_df = pd.DataFrame(students)

    ### saving it as csv file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    students_df.to_csv(OUTPUT_DIR / "students.csv", index=False)

    return students_df


students_df = generate_students()



### course generation 

COURSE_CATEGORIES = [
    "Python",
    "Data Science",
    "Machine Learning",
    "Artificial Intelligence",
    "SQL",
    "Web Development",
    "Cloud Computing",
    "DevOps",
    "Cybersecurity",
    "Data Analytics",
    "Deep Learning",
    "Software Development",
]

COURSE_DIFFICULTIES = [
    'Beginner',
    'Intermediate',
    'Advanced'
]


COURSE_TITLE_TEMPLATES = {
    "Beginner": [
        "Introduction to {category}",
        "{category} Fundamentals",
        "Getting Started with {category}",
    ],
    "Intermediate": [
        "Applied {category}",
        "Intermediate {category}",
        "{category} in Practice",
    ],
    "Advanced": [
        "Advanced {category}",
        "{category} Techniques",
        "Mastering {category}",
    ]
}

COURSE_DESCRIPTION_TEMPLATE = (
    "This {difficulty_level} course teaches {category} "
    "concepts and practical skills."
)

COURSES_PER_LEVEL = {
    "Beginner": 170,
    "Intermediate": 165,
    "Advanced": 165,
}


def generate_courses():

    courses = []

    category_courses = {}

    course_number = 1

    for difficulty_level, count in COURSES_PER_LEVEL.items():
        for _ in range(count):

            course_id = f"CRS{course_number:05d}"


            ### course categories
            category = np.random.choice(COURSE_CATEGORIES)
            if category not in category_courses:
                category_courses[category] = {}


            ### course difficullties

            if difficulty_level == "Beginner":
                prerequisite = None
            elif difficulty_level == "Intermediate":
                prerequisite = category_courses[category].get('Beginner')
            else:
                prerequisite = category_courses[category].get('Intermediate')


            ### COURSE NAME
            title_template = np.random.choice(
                COURSE_TITLE_TEMPLATES[difficulty_level]
            )

            course_name = title_template.format(
                category = category
            )


            ### description for each courses
            description = COURSE_DESCRIPTION_TEMPLATE.format(
                difficulty_level = difficulty_level,
                category = category
            )


            course = {
                "course_id": course_id,
                "course_name": course_name,
                "category": category,
                "difficulty_level": difficulty_level,
                "description": description,
                "prerequisites": prerequisite,
            }

            courses.append(course)
            category_courses[category][difficulty_level] = course_id

            course_number += 1

    courses_df = pd.DataFrame(courses)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    courses_df.to_csv(OUTPUT_DIR/'courses.csv', index=False)

    return courses_df

courses_df = generate_courses()



### interactions generation


def generate_interactions():
    interactions = []

    for _, student in students_df.iterrows():

        student_id = student["student_id"]
        interests = student["interests"].split(", ")
        experience_level = student["experience_level"]

        preferred_courses = courses_df[
            courses_df["category"].isin(interests)
        ]

        level_matched_courses = preferred_courses[
            preferred_courses["difficulty_level"] == experience_level
        ]

        other_courses = courses_df[
            ~courses_df["category"].isin(interests)
        ]

        # If enough courses match both interest and experience,
        # select most of the courses from this group.
        if len(level_matched_courses) >= 16:

            selected_level_courses = np.random.choice(
                level_matched_courses["course_id"],
                size=12,
                replace=False
            )

            remaining_preferred = preferred_courses[
                ~preferred_courses["course_id"].isin(
                    selected_level_courses
                )
            ]

            selected_preferred = np.random.choice(
                remaining_preferred["course_id"],
                size=4,
                replace=False
            )

        else:
            selected_level_courses = np.random.choice(
                preferred_courses["course_id"],
                size=min(12, len(preferred_courses)),
                replace=False
            )

            remaining_count = 16 - len(selected_level_courses)

            remaining_preferred = preferred_courses[
                ~preferred_courses["course_id"].isin(
                    selected_level_courses
                )
            ]

            selected_preferred = np.random.choice(
                remaining_preferred["course_id"],
                size=remaining_count,
                replace=False
            )

        selected_other = np.random.choice(
            other_courses["course_id"],
            size=4,
            replace=False
        )

        selected_course_ids = np.concatenate(
            [
                selected_level_courses,
                selected_preferred,
                selected_other
            ]
        )

        for course_id in selected_course_ids:

            course = courses_df[
                courses_df["course_id"] == course_id
            ].iloc[0]

            course_difficulty = course["difficulty_level"]

            # Experience level influences quiz performance.
            if course_difficulty == experience_level:
                quiz_score = np.random.randint(60, 101)

            elif (
                EXPERIENCE_LEVEL.index(course_difficulty)
                < EXPERIENCE_LEVEL.index(experience_level)
            ):
                quiz_score = np.random.randint(70, 101)

            else:
                quiz_score = np.random.randint(30, 81)

            completion_status = np.random.choice(
                ["Completed", "Not completed"]
            )

            time_spent = np.random.randint(15, 301)

            engagement_score = np.random.uniform(0, 1)

            if completion_status == "Completed":
                completion_value = 1
            else:
                completion_value = 0

            interaction_score = (
                0.4 * completion_value
                + 0.4 * (quiz_score / 100)
                + 0.2 * engagement_score
            )

            interaction = {
                "student_id": student_id,
                "course_id": course_id,
                "completion_status": completion_status,
                "quiz_score": quiz_score,
                "time_spent": time_spent,
                "engagement_score": engagement_score,
                "interaction_score": interaction_score,
            }

            interactions.append(interaction)

    interactions_df = pd.DataFrame(interactions)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    interactions_df.to_csv(
        OUTPUT_DIR / "interactions.csv",
        index=False
    )

    return interactions_df


interaction = generate_interactions()

print("\n========== FINAL DATA VALIDATION ==========")

# 1. Row counts
print("\n--- Row Counts ---")
print(f"Students: {len(students_df)}")
print(f"Courses: {len(courses_df)}")
print(f"Interactions: {len(interaction)}")


# 2. Unique IDs
print("\n--- Unique IDs ---")
print(f"Unique students: {students_df['student_id'].nunique()}")
print(f"Unique courses: {courses_df['course_id'].nunique()}")


# 3. Interactions per student
print("\n--- Interactions Per Student ---")
print(
    interaction
    .groupby("student_id")
    .size()
    .value_counts()
)


# 4. Duplicate student-course pairs
print("\n--- Duplicate Student-Course Pairs ---")
print(
    interaction
    .duplicated(
        subset=["student_id", "course_id"]
    )
    .sum()
)


# 5. Foreign-key validation
print("\n--- Foreign Key Validation ---")

print(
    "All student IDs valid:",
    interaction["student_id"]
    .isin(students_df["student_id"])
    .all()
)

print(
    "All course IDs valid:",
    interaction["course_id"]
    .isin(courses_df["course_id"])
    .all()
)


# 6. Missing values
print("\n--- Missing Values ---")
print(interaction.isnull().sum())


# 7. Quiz score range
print("\n--- Quiz Score Range ---")
print(
    f"Min: {interaction['quiz_score'].min()}"
)
print(
    f"Max: {interaction['quiz_score'].max()}"
)


# 8. Interaction score range
print("\n--- Interaction Score Range ---")
print(
    f"Min: {interaction['interaction_score'].min()}"
)
print(
    f"Max: {interaction['interaction_score'].max()}"
)


# 9. Completion distribution
print("\n--- Completion Distribution ---")
print(
    interaction["completion_status"]
    .value_counts()
)


# 10. Course difficulty distribution
print("\n--- Course Difficulty Distribution ---")
print(
    courses_df["difficulty_level"]
    .value_counts()
)


# 11. Course category distribution
print("\n--- Course Category Distribution ---")
print(
    courses_df["category"]
    .value_counts()
)


print("\n========== VALIDATION COMPLETE ==========")

