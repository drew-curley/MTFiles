
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import zipfile
import io
from TranslatorFactory import TranslatorFactory
from pathlib import Path
import json
import tempfile

app = Flask(__name__)

with open("./constants/mimeTypeToFileTypeMap.json", 'r') as json_file:
    mimetypeToFiletypeMap = json.load(json_file)

translator_factory = TranslatorFactory()

@app.route('/')
def index():
    with open('./constants/languages.json', 'r') as file:
        languages = json.load(file)
    with open('./constants/model_checkpoints.json', 'r') as file:
        models = json.load(file)
    with open('./constants/wa-language-metadata.json', 'r') as file:
        wa_language_metadata = json.load(file)['wa_language_metadata']
    
    target_languages_list = [{"value": key, "label": value} for key, value in languages.items()]
    models_list = [{"value": key, "label": key} for key in models]
    wa_language_metadata_list = [
        {
            "value": item['language']['supported_nllb_variants'][0] if item['language']['supported_nllb_variants'] else item['language']['iso6393'],
            "label": item['language']['english_name']
        }
        for item in wa_language_metadata
    ]
    
    return render_template('index.html', target_languages=target_languages_list, wa_language_metadata=wa_language_metadata_list, models=models_list)


@app.route('/submit', methods=['POST'])
def submit():
    selected_target_languages = request.form.getlist('target_languages')
    selected_models = request.form.getlist('models')
    uploaded_files = request.files.getlist('files')
    selected_input_languages = request.form.getlist('input_languages')
    
    output_paths = []
    for input_language in selected_input_languages:
        for target_language in selected_target_languages:
            for model in selected_models:
                for file in uploaded_files:
                    file_type = file.mimetype
                    file_extension = mimetypeToFiletypeMap[file_type]

                    original_filename = file.filename
                    temp_dir = tempfile.gettempdir()
                    temp_file_path = os.path.join(temp_dir, original_filename)

                    with open(temp_file_path, 'wb') as temp_file:
                        temp_file.write(file.read())
                        file.seek(0)

                    translator = translator_factory.get_translator(file_extension)
                    output_path = translator.translate(Path(temp_file_path), input_language, target_language, model)

                    if output_path:
                        output_paths.append(str(output_path))

                    os.remove(temp_file_path)
    
    # Redirect to the download route with the list of output files
    return redirect(url_for('download', files=','.join(output_paths)))  # Ensure all paths are strings


@app.route('/download', methods=['GET'])
def download():
    file_paths = request.args.get('files', '').split(',')

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for file_path in file_paths:
            if os.path.exists(file_path):
                zf.write(file_path, os.path.basename(file_path))

                # Remove the file after adding it to the zip
                os.remove(file_path)
            else:
                return f"<h1>Error: File {file_path} not found!</h1>"
    
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='translated_files.zip'
    )

if __name__ == '__main__':
    app.run(debug=True)
