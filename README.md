# LongHisDoc: A Benchmark for Long-context Historical Document Understanding

![LongHisDoc](LongHisDoc_overview.png)

## 📖 Introduction
we present **LongHisDoc**, a pioneering benchmark specifically designed to evaluate the capabilities of LLMs and LVLMs in long-context historical document understanding tasks. This benchmark includes 101 historical documents across 10 categories, with 1,012 expert-annotated question-answer pairs covering four types, and the evidence for the questions is drawn from three modalities. 

## 🔎 Evaluate
Our evaluation pipeline consists of three stages: *response generation*, *answer extraction*, and *score calculation*. The corresponding code for each stage can be found in their respective directories. To begin, please download the dataset from [LongHisDoc Data](https://huggingface.co/datasets/qweq12433454/LongHisDoc), and then use the evaluation scripts to score LLMs/LVLMs accordingly.

## 📜 License
The data should be used and distributed under [ (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.zh-hans) for non-commercial research purposes.
