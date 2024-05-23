from flask import Flask, request, render_template, redirect, url_for, send_file
import os
import gc
import torch
import matplotlib.pyplot as plt
import io
import base64
from Mistral import run_model
from main import process_audio_files# Ensure these functions are correctly imported
def sanitize_filename(filename):
    # Replace invalid characters with underscores
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)

# Set matplotlib to use the 'Agg' backend
plt.switch_backend('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['OUTPUT_FOLDER'] = 'output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    processed_files = os.listdir(app.config['OUTPUT_FOLDER'])
    return render_template('index.html', files=processed_files)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('files[]')
    for file in files:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
def process_files():
    data_dir = app.config['UPLOAD_FOLDER']
    output_dir = app.config['PROCESSED_FOLDER']
    batch_size = 32  # Adjust batch size as needed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    process_audio_files(data_dir, output_dir, batch_size, device)
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    run_model(app.config['PROCESSED_FOLDER'], app.config['OUTPUT_FOLDER'], question, batch_size=1)
    return redirect(url_for('index'))

@app.route('/clear', methods=['POST'])
def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    return redirect(url_for('index'))

@app.route('/plot/<filename>')
def plot_graph(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        binary_string = file.read().strip()
        counts = [int(bit) for bit in binary_string]

    # Create the bar chart (histogram)
    plt.figure(figsize=(10, 5))
    colors = ['#FF5733', '#33C1FF']  # Colors for the bars
    bars = plt.bar([0, 1], [counts.count(0), counts.count(1)], color=colors, edgecolor='black')
    plt.title(f'Binary String Histogram for {filename}')
    plt.xlabel('Bit Value')
    plt.ylabel('Frequency')

    # Add labels to the bars
    plt.text(bars[0].get_x() + bars[0].get_width() / 2, 0.01, 'No', ha='center')
    plt.text(bars[1].get_x() + bars[1].get_width() / 2, 0.01, 'Yes', ha='center')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    return render_template('plot.html', img_data=img_base64, filename=filename)
if __name__ == '__main__':
    app.run(debug=True)