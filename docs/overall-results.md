| Method                  | Rank | Mean Win Rate (%) | Average AUC |
|-------------------------|------|-------------------|-------------|
| **Llama-3-8B (FT) (Ours)** | **1** | **78.11** | **78.037** |
| **Flan-T5-L (FT) (Ours)**  | **2** | **76.43** | **78.663** |
| MiniCheck-T5-L          | 3    | 72.39              | 76.674      |
| gpt-3.5-turbo           | 4    | 69.36              | 77.007      |
| Flan-T5-B (FT) (Ours)   | 5    | 66.00              | 76.126      |
| AlignScore-L            | 6    | 53.19              | 73.074      |
| Llama-3-8B              | 7    | 53.20              | 75.085      |
| AlignScore-B            | 8    | 39.39              | 71.319      |
| QuestEval               | 9    | 37.37              | 66.089      |
| BARTScore               | 10   | 26.94              | 62.637      |
| BERTScore               | 11   | 20.88              | 61.263      |
| ROUGE-L                 | 12   | 6.73               | 54.678      |

*Comparison of different factuality evaluation methods across all test datasets. The methods are ranked based on the Mean Win Rate, which measures overall performance on factuality tasks. The Average AUC column represents the average of all individual AUC-ROC scores.*
