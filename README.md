# VisualPuzzles: Decoupling Multimodal Reasoning Evaluation from Domain Knowledge

## Overview
**VisualPuzzles** is a multimodal benchmark specifically designed to evaluate **reasoning abilities** in large models while deliberately minimizing reliance on domain-specific knowledge.

Key features:
- 1168 diverse puzzles
- 5 reasoning categories: Algorithmic, Analogical, Deductive, Inductive, Spatial
- Difficulty labels: Easy, Medium, Hard
- Less knowledge-intensive than existing benchmarks (e.g., MMMU)
- More reasoning-complex than existing benchmarks (e.g., MMMU)

## Key Findings
- All models perform worse than humans; most can't surpass even 5th-percentile human performance.
- Strong performance on knowledge-heavy benchmarks does not transfer well.
- Larger models and structured "thinking modes" don't guarantee better results.
- Scaling model size does not ensure stronger reasoning

## Dataset
The dataset is available on HuggingFace 🤗.

## Model Outputs
Outputs of all models we evaluated are available on [Zeno](https://hub.zenoml.com/project/2e727b03-a677-451a-b714-f2c07ad2b49f/VisualPuzzles).

## Experiments

We gratefully use the [lmms-eval package](https://github.com/EvolvingLMMs-Lab/lmms-eval) to evaluate VisualPuzzles.

To reproduce experimental results on VisualPuzzles, run the following commands:

Installation:
```bash
cd evaluation/lmms-eval
pip install -e .
```

Experiments:
```bash
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model model_type \ # for example, llava
    --model_args pretrained=model_name \ # for example, "liuhaotian/llava-v1.5-7b"
    --tasks VisualPuzzles_cot \ # use VisualPuzzles_cot if you are evaluating CoT performance, or use VisualPuzzles_direct if not.
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix VisualPuzzles \
    --output_path ./logs/
```

Alternatively, you could also run model_evaluation.py with a custom model.

## Knowledge Intensity Evaluation of MMMU v.s. VisualPuzzles

This experiment investigates 
- the extent to which solving problems in the VisualPuzzles benchmark relies on domain-specific knowledge, compared to the widely-used MMMU dataset; and
- whether models already possess the knowledge required to solve VisualPuzzles, as compared to MMMU.

### Knowledge Checklist Generation

We prompted GPT-4o to generate "knowledge concept checklists" for 50 randomly selected questions from each of MMMU and VisualPuzzles.

The knowledge concept checklists we generated for MMMU and VisualPuzzles could be found in [knowledge/mmmu_questions.json](knowledge/mmmu_questions.json) and [knowledge/puzzle_questions.json](knowledge/puzzle_questions.json) respectively.

Run the following command to reproduce this experiment.
```bash
python get_knowledge_checklists.py
```
Note that we went through manual validation as dicussed in the [paper](https://arxiv.org/pdf/2504.10342).

### Knowledge Accuracy

We measured models' knowledge accuracy - their ability to answer the knowledge checklist questions correctly - on both benchmarks. We used llm-as-a-judge with GPT-4o to evaluate whether models answered the knowledge checklist questions correctly. Model outputs and judge outputs could be found in [knowledge/knowledge_eval_output](knowledge/knowledge_eval_output).

After generating model responses for the knowledge checklist questions [knowledge/mmmu_questions.json](knowledge/mmmu_questions.json) and [knowledge/puzzle_questions.json](puzzle_questions.json), run the following command to reproduce this experiment on models' knowledge accuracy.
```bash
cd knowledge
python get_knowledge_scores.py
```