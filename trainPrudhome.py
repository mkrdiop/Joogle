import os
import json
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from rouge import Rouge

# Configure model and tokenizer
MODEL_NAME = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Define data path
data_path = 'C:\\Users\\mkrdi\\Documents\\JOOGLE\\prudhomme\\data\\'

# Function to read PDF and extract text
def read_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    page_content = []
    for page in pdf_reader.pages:
        page_content.append(page.extract_text())
    return " ".join(page_content)

# Function to preprocess data
def preprocess_function(file_name):
    summary_file = os.path.join(data_path, file_name + ".txt")
    text = read_pdf(os.path.join(data_path, file_name + ".pdf"))
    
    with open(summary_file, encoding="utf-8") as f:
        summary = f.read()
    
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(summary, truncation=True, padding="max_length", max_length=128)

    data_point = {"text": text, "summary": summary, "input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels["input_ids"]}
    
    return data_point

# Function to evaluate summary
def evaluate_summary(summary, gold_standard):
    rouge = Rouge(metrics=['rouge-n', 'rouge-l'], max_n=4)
    score = rouge.get_scores(summary, gold_standard)
    return score

# Load and prepare data
dataset = []
print('Preparation of the dataset')
for filename in os.listdir(data_path):
    if filename.endswith(".pdf"):
        text = read_pdf(os.path.join(data_path, filename))
        file_name = os.path.splitext(filename)[0]
        data_point = preprocess_function(file_name)
        dataset.append(data_point)

# Split the dataset into training and evaluation sets
train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Save the training dataset in JSON format
print('Saving the training dataset')
with open("train_dataset.json", "w", encoding="utf-8") as f:
    json.dump(train_dataset, f, indent=4)

# Save the evaluation dataset in JSON format
print('Saving the evaluation dataset')
with open("eval_dataset.json", "w", encoding="utf-8") as f:
    json.dump(eval_dataset, f, indent=4)

# Train the model
print('training the model')
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
)

# Train the model
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset = eval_dataset,
)

trainer.train()

# Save the model
print('saving the model')
model.save_pretrained("./summarization_model")

# Define function to summarize text
def summarize(text):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    outputs = model.generate(**inputs)
    summary = tokenizer.decode(outputs[0])
    return summary
