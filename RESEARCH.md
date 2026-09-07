# LearnSmart EdTech

## Literature Review — Key Findings

### Purpose

This literature review summarizes three relevant recommendation-system papers studied during Week 1. The findings are used to guide the architecture and evaluation approach for the LearnSmart EdTech recommendation engine.

---

## 1. Neural Collaborative Filtering (NCF)

**Authors:** Xiangnan He et al.

### Key Finding

The paper presents Neural Collaborative Filtering as an approach for learning user–item interactions with neural networks. Instead of relying only on a predefined interaction function, the approach uses neural network models to learn more complex, non-linear relationships between users and items. This provides useful background for understanding more advanced collaborative filtering methods.

**Paper:** https://arxiv.org/abs/1708.05031

---

## 2. Wide & Deep Learning for Recommender Systems

**Authors:** Heng-Tze Cheng et al.

### Key Finding

The paper combines a wide component and a deep component for recommendation. The wide component is designed to capture and memorize important existing patterns, while the deep component learns general patterns and feature combinations. The key takeaway is that complementary learning approaches can be combined to improve recommendation capability.

**Paper:** https://arxiv.org/abs/1606.07792

---

## 3. Content-Boosted Collaborative Filtering for Improved Recommendations

**Authors:** Prem Melville, Raymond J. Mooney, and Ramadass Nagarajan

### Key Finding

This paper is the most directly relevant to LearnSmart because it combines content-based and collaborative filtering. The content-based predictor is used to generate predictions for missing entries in a sparse user–item matrix, after which collaborative filtering uses the resulting information to produce personalized recommendations.

The paper specifically addresses data sparsity and the first-rater problem. In its experiments, the content-boosted approach outperformed pure content-based filtering, pure collaborative filtering, and a naive hybrid approach.

**Paper:** https://aaai.org/papers/00187-aaai02-029-content-boosted-collaborative-filtering-for-improved-recommendations/

---

## Overall Findings for LearnSmart

The literature review leads to the following findings:

- **Collaborative filtering** can learn learner preferences from interactions with courses and from similarities between learners.
- **Content-based filtering** can use course characteristics and provide useful recommendations when interaction history is limited.
- A **hybrid approach** can combine the complementary strengths of collaborative and content-based filtering.
- The **Content-Boosted Collaborative Filtering** paper provides the strongest direct support for LearnSmart's required hybrid recommendation approach.
- **NCF** and **Wide & Deep** are used as research references and are not being claimed as implementation architectures for LearnSmart.

---

## Architecture Direction

Based on the literature review and the project brief, LearnSmart will focus on a **practical hybrid recommendation engine** using:

- Collaborative filtering
- Content-based filtering
- Cold-start evaluation
- Warm-start evaluation

The literature review will therefore serve as the research foundation for the subsequent architecture and evaluation design.