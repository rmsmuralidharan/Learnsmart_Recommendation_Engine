# LearnSmart EdTech — Architecture & Evaluation Design

## 1. Project Objective

LearnSmart EdTech is an adaptive learning recommendation engine that recommends the next best courses using:

- Collaborative Filtering
- Content-Based Filtering
- Completed-course history
- Quiz scores
- REST API
- A/B testing on 100 sample users

The project acceptance criteria are:

- Cold-start working
- Warm-start working
- Precision@5 >= 0.6
- API documented
- A/B test results delivered on GitHub

> Research findings and paper/repository analysis are documented separately in `RESEARCH.md`.

---

## 2. Dataset Decision

The primary modeling dataset will be a **synthetically generated LearnSmart dataset**.

The real datasets investigated during project setup did not align sufficiently with all project requirements, particularly the combination of student-course interactions, course content, completion information, and quiz scores.

Synthetic data allows the project to explicitly represent the required learning signals while remaining reproducible and transparent.

The dataset will be clearly identified as **synthetic** and will not be presented as real student data.

---

## 3. Initial Dataset Size

| Entity | Target |
|---|---:|
| Students | 5,000 |
| Courses | 500 |
| Student-course interactions | 100,000 |

This gives approximately:

- 20 interactions per student
- 200 interactions per course

100,000 interactions is the initial target. The dataset can be expanded later if evaluation demonstrates that additional interactions are necessary.

---

## 4. Dataset Schema

### 4.1 Students

| Feature | Description |
|---|---|
| `student_id` | Unique student identifier |
| `interests` | Student learning interests |
| `experience_level` | Beginner / Intermediate / Advanced |
| `learning_style` | Student learning preference/style |

### 4.2 Courses

| Feature | Description |
|---|---|
| `course_id` | Unique course identifier |
| `course_name` | Course title |
| `category` | Course subject/category |
| `difficulty_level` | Beginner / Intermediate / Advanced |
| `description` | Course description |
| `prerequisites` | Required prior courses/knowledge |

### 4.3 Student-Course Interactions

| Feature | Description |
|---|---|
| `student_id` | Student identifier |
| `course_id` | Course identifier |
| `completion_status` | Whether the course was completed |
| `quiz_score` | Student quiz/performance score |
| `time_spent` | Time spent learning |
| `engagement_score` | Learning engagement signal |
| `interaction_score` | Derived strength of the student-course interaction |

`completion_status` and `quiz_score` are mandatory project signals because the project brief explicitly requires recommendations based on completed courses and quiz scores.

---

## 5. Final Model Architecture

```text
                 SYNTHETIC DATA GENERATOR
                         |
                         v
                    DATA / ETL
                         |
              +----------+----------+
              |          |          |
              v          v          v
           Students   Courses   Interactions
              |          |          |
              +----------+----------+
                         |
                         v
                 DATA PREPROCESSING
                         |
                         v
              Student-Course Matrix
                         |
                +--------+--------+
                |                 |
                v                 v
       Collaborative       Content-Based
         Filtering           Filtering
                |                 |
                +--------+--------+
                         |
                         v
                  HYBRID SCORING
                         |
                         v
                   RANKING ENGINE
                         |
                         v
                    TOP 5 COURSES
                         |
                         v
                    FASTAPI REST
                         |
                         v
                  Student / Client