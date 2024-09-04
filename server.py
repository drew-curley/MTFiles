
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
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    selected_target_languages = request.form.getlist('languages')
    selected_models = request.form.getlist('models')
    uploaded_files = request.files.getlist('files')
    input_language = "eng_Latn"
    
    output_paths = []
    
    for target_language in selected_target_languages:
        for model in selected_models:
            print(f"{target_language} {model}")

            for file in uploaded_files:
                # TODO: clean this up. Be able to pass in file type to get_translator
                file_type = file.mimetype  # Get the file type (MIME type)
                file_extension = mimetypeToFiletypeMap[file_type]

                # TODO: see if I can do this without making a temp file. Just use the file that is given
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                    temp_file.write(file.read())
                    file.seek(0)
                    temp_file_path = temp_file.name

                    print(temp_file_path)
                    # Create translator and translate the file
                    translator = translator_factory.get_translator(file_extension)
                    output_path = translator.translate(Path(temp_file_path), input_language, target_language, model)
                    
                # Store the output path
                if output_path:
                    output_paths.append(str(output_path))  # Convert Path to string

                # Optionally, delete the temporary file
                os.remove(temp_file_path)
    
    # Redirect to the download route with the list of output files
    return redirect(url_for('download', files=','.join(output_paths)))  # Ensure all paths are strings

@app.route('/download', methods=['GET'])
def download():
    # Get file paths from query parameters
    file_paths = request.args.get('files', '').split(',')

    # Create a Zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for file_path in file_paths:
            if os.path.exists(file_path):
                zf.write(file_path, os.path.basename(file_path))
            else:
                return f"<h1>Error: File {file_path} not found!</h1>"
    zip_buffer.seek(0)

    # Send the zip file as a downloadable attachment
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='translated_files.zip'
    )

if __name__ == '__main__':
    app.run(debug=True)
