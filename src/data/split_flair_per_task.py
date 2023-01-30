output_path = "data/interim/flair/"
output_files = ["st1_train_flair.txt", "st1_val_flair.txt"]


tasks = {
    "question": ["B-question", "I-question"],
    "per_exp": ["B-per_exp", "I-per_exp", "B-claim_per_exp", "I-claim_per_exp"],
    "claim": ["B-claim", "I-claim", "B-claim_per_exp", "I-claim_per_exp"],
}

for file in output_files:
    with open(f"{output_path}/{file}", "r") as f:
        lines = f.readlines()
    for task_name, task_tags in tasks.items():
        output_file = f"{output_path}/{task_name}_{file}"
        with open(output_file, "w") as f:
            for line in lines:
                if line != "\n":
                    word, label = line.split()
                    if label in task_tags:
                        if label == "B-claim_per_exp":
                            label = task_tags[0]
                        elif label == "I-claim_per_exp":
                            label = task_tags[1]
                        f.write(f"{word} {label}\n")
                    else:
                        f.write(f"{word} O\n")
                else:
                    f.write("\n")
