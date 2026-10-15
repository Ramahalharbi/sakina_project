

# Sakina (سَكينة)

An Arabic mental-health support assistant built using Retrieval-Augmented Generation (RAG). The system retrieves real questions that mental-health professionals have already answered, uses GPT to write a short supportive reply based only on those answers, and shows emergency support numbers when a message suggests suicidal thoughts or self-harm.

## Project Overview

Many people living with anxiety, depression or insomnia do not know where to start. Stigma, cost and waiting times push them to search online, where they find untrusted forums or general chatbots that may invent medical information, give no sources, and miss signs of danger.

This project combines a safety check, information retrieval and a large language model. Users write how they feel in their own words, in Modern Standard Arabic or dialect. The assistant replies from real specialist answers, shows the similar cases it used as sources, and always encourages the user to see a professional. It does not diagnose and does not recommend medicines or doses.

## Features

* **Safety Layer:** Checks every message before anything else. If it suggests suicidal thoughts or self-harm, the assistant shows a support message and Saudi support numbers (937, 920033360, 997, 1919) instead of a normal reply.
* **Hybrid Search:** Combines semantic vector search (ChromaDB) with BM25 and character-level TF-IDF, merged with Reciprocal Rank Fusion.
* **Knowledge Base:** 318 real cases answered by mental-health professionals. Cases about suicide or self-harm are excluded so they are never shown to a user as a "similar case".
* **Grounded Replies:** GPT (gpt-4o-mini) writes a short supportive reply and practical steps using only the retrieved specialist answers, without diagnosis, medicine names or doses.
* **Question-Type Recognition:** Identifies the topic of the message (diagnosis, treatment, healthy lifestyle, and others) by voting over the retrieved cases.
* **Source Transparency:** Shows the three most similar real cases with the specialists' answers, so users can see what the reply is based on.
* **Interactive Interface:** An Arabic, right-to-left Streamlit app with no icons, a calm design, and a mobile-friendly layout.
* **Fallback Without a Key:** If no OpenAI key is set, the app shows the specialist's answer to the closest case instead of a written reply.

## Technologies Used

* Python
* Streamlit
* OpenAI GPT (gpt-4o-mini)
* Sentence Transformers (intfloat/multilingual-e5-base)
* ChromaDB
* Rank-BM25
* Scikit-learn
* Pandas
* Plotly

## Dataset

The project uses the **MentalQA** dataset (Alhuzali, Alasmari and Alsaleh, *IEEE Access*, 2024), released under the MIT license as part of the AraHealthQA 2025 shared task: https://github.com/hasanhuz/MentalQA

The dataset contains 500 real Arabic questions from people seeking mental-health help, each answered by a mental-health professional and labelled with question types and answer strategies.

The files are used as follows:

* **Knowledge base:** 350 cases (`Train_Dev.tsv`). 32 cases about suicide or self-harm are excluded, leaving 318 searchable cases.
* **Reply test set:** 150 separate questions with specialist reference answers. 3 questions that also appear in the knowledge base are removed, leaving 147.
* **Classification test set:** 150 separate questions with question-type labels.

Question types in the data:

* Diagnosis
* Treatment
* Anatomy and Physiology
* Epidemiology (causes and course of illness)
* Healthy Lifestyle
* Provider Choices
* Other

## Methodology

1. Load the MentalQA files and normalize Arabic text.
2. Detect messages about suicide or self-harm with transparent Arabic rules, and exclude such cases from the knowledge base.
3. Choose what to store for each case (the question, the specialist answer, or both) using leave-one-out evaluation on the knowledge base.
4. Generate multilingual sentence embeddings (multilingual-e5-base) and store them in ChromaDB.
5. Retrieve the closest cases with hybrid search: vector similarity, BM25 and character-level TF-IDF combined with Reciprocal Rank Fusion.
6. Identify the question type by voting over the retrieved cases.
7. Send the retrieved specialist answers to GPT, which writes a short supportive reply and practical steps in JSON format.
8. Display the reply, question type and similar cases through a Streamlit interface.

## Evaluation

Settings were chosen with leave-one-out evaluation on the knowledge base, and results are reported on the separate test files. Each result is compared with a random baseline.

| Stage | Result | Random baseline |
|---|---|---|
| Storage unit (cases sharing the question type, top 3) | Question only: 0.731; question + answer: 0.771 | 0.657 |
| Matching a question to its own specialist answer (hybrid) | Hit@1: 0.145, Hit@5: 0.261, MRR: 0.195 | Hit@1: 0.003 |
| Question-type classification | Micro F1: 0.546, Macro F1: 0.240 | Always "treatment": 0.415 |
| Reply similarity to the specialist answer (no GPT) | ROUGE-L: 0.074, faithfulness to sources: 1.000 | — |
| Safety check (23 test messages) | Recall: 1.000, precision: 1.000 | — |

These results were produced without the e5 embedding model and without GPT, because neither could be downloaded or called in the development environment. An offline LSA embedding was used instead. Re-running the evaluation with e5 and an OpenAI key is needed to measure the full system.

The safety test messages were written by the developer and were also used while writing the detection rules, so this is a preliminary check rather than an independent test. In the real data, 47 of 647 questions were flagged as crisis messages.

## Limitations

* The dataset is small (500 cases), so some topics have no close match in the knowledge base.
* Patients and specialists use different words, so keyword search alone retrieves answers that are, on average, no more similar to the reference answer than random ones. Semantic embeddings and GPT are expected to close this gap but have not yet been measured.
* Rare question types (anatomy, provider choices, other) are recognized poorly because they have few examples.
* The safety layer is rule-based and may miss crisis messages phrased in ways the rules do not cover.
* Using GPT sends the user's message to OpenAI, which matters for privacy in a mental-health setting.
* The support numbers should be checked regularly, as services and hours can change.

## How to Run

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```


   Never commit `secrets.toml`; it is listed in `.gitignore`. Without a key, the app still runs and shows the closest specialist answer.

2. Start the app:

   ```bash
   streamlit run app.py
   ```

   On the first run, the e5 embedding model is downloaded once and the ChromaDB index is built in `data/index`.

3. To view or re-run the evaluation (developer page):

   ```bash
   streamlit run evaluation.py
   ```

   Or from the command line: `python scripts/run_eval.py`

## Disclaimer

This project is an educational prototype and does not replace a doctor, psychologist or any professional mental-health care. It does not diagnose or treat any condition. Anyone in immediate danger should call emergency services (997 in Saudi Arabia).

## Training Program

This project was developed as part of the Natural Language Processing (Applied) training program at SDAIA Academy.

SDAIA Academy GitHub: https://github.com/SDAIAAcademy
