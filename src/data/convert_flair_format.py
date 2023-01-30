import os.path

from sklearn.model_selection import train_test_split

texts_file =  "data/interim/st1_train_tokens.txt"
bio_file = "data/interim/st1_train_bio_tokens.txt"

with open(texts_file, "r") as f:
    texts = f.readlines()

with open(bio_file, "r") as f:
    tags = f.readlines()

if not os.path.exists('data/interim/flair/'):
    os.makedirs('data/interim/flair/')

output_files = ["data/interim/flair/st1_train_flair.txt",
                "data/interim/flair/st1_val_flair.txt"]

# split into train and val
train_texts, val_texts, train_tags, val_tags = train_test_split(
    texts, tags, test_size=0.2, random_state=42
)

# write to file
for i, (texts, tags) in enumerate(zip([train_texts, val_texts], [train_tags, val_tags])):
    with open(output_files[i], "w") as f:
        for text, tag in zip(texts, tags):
            for word, label in zip(text.split(), tag.split()):
                f.write(f"{word} {label}\n")
            f.write("\n")
