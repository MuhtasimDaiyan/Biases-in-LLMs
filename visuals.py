import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("visuals", exist_ok=True)


def plot_crows_pairs():
    df = pd.read_csv("results/crows_pairs_summary.csv")

    labels = df["model"] + " - " + df["prompt_type"]
    values = df["stereotype_preference_percent"]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.ylabel("Stereotype Preference Rate (%)")
    plt.xlabel("Model and Prompt Type")
    plt.title("CrowS-Pairs: Stereotype Preference by Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("visuals/crows_pairs_stereotype_preference.png", dpi=300)
    plt.close()


def plot_bbq_overall_accuracy():
    df = pd.read_csv("results/bbq_results.csv")

    summary = (
        df.groupby(["model", "prompt_type"])["correct"]
        .mean()
        .reset_index()
    )

    summary["accuracy_percent"] = summary["correct"] * 100
    labels = summary["model"] + " - " + summary["prompt_type"]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, summary["accuracy_percent"])
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Model and Prompt Type")
    plt.title("BBQ: Overall Accuracy by Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("visuals/bbq_overall_accuracy.png", dpi=300)
    plt.close()


def plot_bbq_ambiguous_vs_disambiguated():
    df = pd.read_csv("results/bbq_summary.csv")

    for model in df["model"].unique():
        model_df = df[df["model"] == model]

        labels = (
            model_df["prompt_type"]
            + " - "
            + model_df["context_condition"]
            + " - "
            + model_df["category"]
        )

        plt.figure(figsize=(14, 7))
        plt.bar(labels, model_df["accuracy_percent"])
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Prompt / Context / Category")
        plt.title(f"BBQ: Ambiguous vs Disambiguated Accuracy ({model})")
        plt.xticks(rotation=75, ha="right")
        plt.tight_layout()
        filename = f"visuals/bbq_ambiguous_vs_disambiguated_{model.replace('/', '_')}.png"
        plt.savefig(filename, dpi=300)
        plt.close()


def make_result_tables():
    crows = pd.read_csv("results/crows_pairs_summary.csv")
    bbq = pd.read_csv("results/bbq_summary.csv")

    crows.to_csv("visuals/table_crows_pairs_summary.csv", index=False)
    bbq.to_csv("visuals/table_bbq_summary.csv", index=False)


def main():
    plot_crows_pairs()
    plot_bbq_overall_accuracy()
    plot_bbq_ambiguous_vs_disambiguated()
    make_result_tables()

    print("Visuals saved in visuals/ folder:")
    print("- crows_pairs_stereotype_preference.png")
    print("- bbq_overall_accuracy.png")
    print("- bbq_ambiguous_vs_disambiguated_MODEL.png")
    print("- table_crows_pairs_summary.csv")
    print("- table_bbq_summary.csv")


if __name__ == "__main__":
    main()