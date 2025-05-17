# Complementary Representations of Invariant in Domain Generalization for Industrial Data Drift

This repository contains the official implementation of **Complementary Representations of Invariant in Domain Generalization for Industrial Data Drift**, which is prepared for publication in IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS: SYSTEMS.

As the paper is still in the under review stage, I have removed the data processing. The full code will be filled in as soon as the paper is accepted.

## 🚀 Highlights

1. **Novel perspective** is proposed by ICDG to enhance domain generalization from the perspective of invariant-complement representations, which is more effective than existing domain generalization methods that focus on invariant-only representations.
2. **Theoretical insights** indicate that optimizing the objective function of ICDG is equivalent to inferring the minimal upper bound of the empirical risk for unseen domains. This is achieved by complementing invariant representations with covariant representations.
3. **Competitive and insightful results:** Extensive quantitative and qualitative experiments conducted on the gas turbine and polyester esterification processes demonstrate the superiority of ICDG.

## 📊 Performance Comparison
As shown in the table, ICDG outperforms the other methods in both MAE, RMSE and R²:
### Table: Performance Metrics for Six Methods on CO and NOₓ Prediction

| Method   | MAE (CO) | RMSE (CO) | R² (CO) | MAE (NOₓ) | RMSE (NOₓ) | R² (NOₓ) |
|----------|----------|-----------|---------|------------|-------------|----------|
| AdaRNN   | 1.6816   | 2.0685    | 0.4034  | 5.7769     | 6.9776      | 0.5898   |
| CIDA     | 0.8055   | 1.1027    | 0.3418  | 2.9546     | 3.7277      | 0.6503   |
| VDI      | 0.6678   | 0.8960    | 0.4651  | _2.1998_   | _2.7770_    | _0.7160_ |
| SAD      | _0.6410_ | _0.8488_  | _0.5224_| 2.2573     | 2.7953      | 0.6945   |
| NU       | 0.6686   | 0.8862    | 0.3028  | 2.2970     | 2.8905      | 0.6487   |
| **ICDG (Ours)** | **0.6213** | **0.8090** | **0.6508** | **2.0457** | **2.5462** | **0.7824** |

> **Note**: Bold indicates the best performance, and *italic* indicates the second-best performance.
