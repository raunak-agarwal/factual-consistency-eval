# Zero-shot Factual Consistency Evaluation Across Domains

Code, Data, and Models for the paper **Zero-shot Factual Consistency Evaluation Across Domains** [(arxiv)](https://arxiv.org/abs/2408.04114)

**Abstract**: This work addresses the challenge of factual consistency in text generation systems. We unify the tasks of Natural Language Inference, Summarization Evaluation, Factuality Verification and Factual Consistency Evaluation to train models capable of evaluating the factual consistency of source-target pairs across diverse domains. We rigorously evaluate these against eight baselines on a comprehensive benchmark suite comprising 22 datasets that span various tasks, domains, and document lengths. Results demonstrate that our method achieves state-of-the-art performance on this heterogeneous benchmark while addressing efficiency concerns and attaining cross-domain generalization. 

Models:
- [factual-consistency-t5-base](https://huggingface.co/ragarwal/factual-consistency-t5-base)
- [factual-consistency-t5-large](https://huggingface.co/ragarwal/factual-consistency-t5-large)
- [factual-consistency-llama3-8b](https://huggingface.co/ragarwal/factual-consistency-llama3-8b)

Data:
- [Training Data](https://huggingface.co/datasets/ragarwal/factual-consistency-training-mix)
- [Evaluation Benchmark](https://huggingface.co/datasets/ragarwal/factual-consistency-evaluation-benchmark)

Results:
- Overall Results available [here](docs/overall-results.md)
- Dataset-specific Results available [here](docs/dataset-results.md)



Cite this work as follows:
```
@misc{agarwal2024zeroshotfactualconsistencyevaluation,
      title={Zero-shot Factual Consistency Evaluation Across Domains}, 
      author={Raunak Agarwal},
      year={2024},
      eprint={2408.04114},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.04114}, 
}
```
