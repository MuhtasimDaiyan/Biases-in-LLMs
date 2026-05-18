import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import pandas as pd

BBQ_BASE_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/{category}.jsonl"

def load_bbq_category(category):
    url = BBQ_BASE_URL.format(category=category)
    df = pd.read_json(url, lines=True)
    return df.to_dict("records")

os.makedirs("results", exist_ok=True)

MODELS = [
    "bert-base-uncased",
    "distilbert-base-uncased",
]

PROMPTS = {
    "baseline": "{context} Question: {question}",
    "mitigation": "Answer fairly and avoid stereotypes. {context} Question: {question}"
}


def multiple_choice_predict(context_question, choices, tokenizer, model, device):
    """
    Scores each answer choice using a multiple-choice model.
    Returns predicted index.
    """

    first_sentences = [context_question] * len(choices)
    second_sentences = choices

    inputs = tokenizer(
        first_sentences,
        second_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    pred = torch.argmax(logits, dim=1).item()
    return pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    categories = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Race_x_gender",
        "Race_x_SES",
        "Religion",
        "SES",
        "Sexual_orientation"
    ]

    all_results = []

    for model_name in MODELS:
        print(f"\nLoading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)
        model.eval()

        for category in categories:
            print(f"\nLoading BBQ category: {category}")
            dataset = load_bbq_category(category)

            for prompt_type, prompt_template in PROMPTS.items():
                print(f"Running BBQ: {model_name} | {category} | {prompt_type}")

                correct = 0
                total = 0
                ambiguous_total = 0
                ambiguous_correct = 0
                disambig_total = 0
                disambig_correct = 0

                for example in tqdm(dataset):
                    context = example["context"]
                    question = example["question"]

                    choices = [
                        example["ans0"],
                        example["ans1"],
                        example["ans2"]
                    ]

                    label = example["label"]
                    context_condition = example["context_condition"]

                    context_question = prompt_template.format(
                        context=context,
                        question=question
                    )

                    pred = multiple_choice_predict(
                        context_question,
                        choices,
                        tokenizer,
                        model,
                        device
                    )

                    is_correct = pred == label

                    correct += int(is_correct)
                    total += 1

                    if context_condition == "ambig":
                        ambiguous_total += 1
                        ambiguous_correct += int(is_correct)
                    else:
                        disambig_total += 1
                        disambig_correct += int(is_correct)

                    all_results.append({
                        "benchmark": "BBQ",
                        "model": model_name,
                        "prompt_type": prompt_type,
                        "category": category,
                        "context_condition": context_condition,
                        "context": context,
                        "question": question,
                        "answer_0": choices[0],
                        "answer_1": choices[1],
                        "answer_2": choices[2],
                        "gold_label": label,
                        "prediction": pred,
                        "correct": is_correct
                    })

                accuracy = correct / total * 100
                ambig_acc = ambiguous_correct / ambiguous_total * 100 if ambiguous_total else 0
                disambig_acc = disambig_correct / disambig_total * 100 if disambig_total else 0

                print(f"Accuracy: {accuracy:.2f}%")
                print(f"Ambiguous accuracy: {ambig_acc:.2f}%")
                print(f"Disambiguated accuracy: {disambig_acc:.2f}%")

    df = pd.DataFrame(all_results)
    df.to_csv("results/bbq_results.csv", index=False)

    summary = (
        df.groupby(["model", "prompt_type", "category", "context_condition"])["correct"]
        .mean()
        .reset_index()
    )
    summary["accuracy_percent"] = summary["correct"] * 100
    summary.to_csv("results/bbq_summary.csv", index=False)

    print("\nSaved:")
    print("results/bbq_results.csv")
    print("results/bbq_summary.csv")


if __name__ == "__main__":
    main()