# LLM Preference Prediction with KerasNLP(Kaggle Competition)

This repository contains an end-to-end solution for **predicting human preferences between two LLM (Large Language Model) responses** to a given prompt.  
The project fine-tunes the **DeBERTaV3** model using **KerasNLP** and **Keras 3**, following the *Shared Weight (Siamese)* approach to compare two responses contextually.

---

## Competition Context

This work is based on the **LMSYS LLM Arena Preference Prediction** challenge, where the goal is to:
> Predict which chatbot response users will prefer in head-to-head battles between LLMs.

Each data sample contains:
- A **prompt** (question or instruction)
- Two **responses** (from different models)
- A **label** indicating which response users preferred

Your model must learn to judge which response is *better*, similar to how human evaluators do.

---

## Project Highlights

- **Model:** Fine-tuned [DeBERTaV3 Base](https://huggingface.co/microsoft/deberta-v3-base)
- **Framework:** [KerasNLP](https://keras.io/keras_nlp/) with Keras 3 backend (supports TensorFlow, JAX, or PyTorch)
- **Optimization:** Mixed Precision (`float16`) for faster training
- **Architecture:** Shared-weight (Siamese) dual encoder for pairwise comparison
- **Goal:** Binary classification — predict whether Response A or B is preferred
- **Data Pipeline:** Built using `tf.data.Dataset` with caching, shuffling, batching, and prefetching

---
Developed as part of the LMSYS LLM Preference Prediction competition.
Built with using KerasNLP, DeBERTaV3, and TensorFlow.

## This Project Made By Mohammad Parsa Yahyazadeh As Member Of Kaggle Competition.
