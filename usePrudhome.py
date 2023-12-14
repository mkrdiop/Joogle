import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import PyPDF2

def read_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    page_content = []
    for page in pdf_reader.pages:
        page_content.append(page.extract_text())
    return " ".join(page_content)
pdffile = "cour-d'appel_n°1910752_22_11_2023.pdf"
# Load the trained model and tokenizer
model_path = "./summarization_model"  # Update with the path where you saved your model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Example text to be summarized
#input_text = read_pdf(pdffile)
input_text = " Cour d'appel de Douai : le licenciement pour faute grave d'un outilleur régleur requalifié en licenciement sans cause réelle et sérieuse : le licenciement pour faute grave d'un outilleur régleur requalifié en licenciement sans cause réelle et sérieuse."
# Tokenize and generate summary
inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"], max_length=1000, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decode the generated summary
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the generated summary
print("Generated Summary:", generated_summary)

"""

input_text = read_pdf("cour-d'appel_n°1910752_22_11_2023.pdf")
"./summarization_model/pytorch_model.bin"
"./summarization_model/config.json"

"""