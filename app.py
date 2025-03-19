from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load the tokenizer and model
model_name = "google/long-t5-tglobal-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    if not text.strip():
        return "Please enter text to summarize.", 400  # Return error if text is empty
    
    # Tokenize and summarize text
    inputs = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return render_template('summarize.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
