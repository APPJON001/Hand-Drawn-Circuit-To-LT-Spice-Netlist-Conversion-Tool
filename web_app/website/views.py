from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, session, send_file
from flask_login import login_required, current_user
from .models import Note
from PIL import Image
from . import db
import io
import os
import sys
import shutil
import time
import csv

sys.path.insert(1, '/Users/jonathanapps/UCT/2023 (4TH YEAR)/Thesis_Work/code')
from code_main import mainProcessing

views  = Blueprint('views', __name__)
mainProcessor = None
process_time = 0

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    global process_time
    global mainProcessor
    process_time = 0
    if request.method == 'POST':
        try:
            print("Posting to home")
            # Check if an image was uploaded in the form
            if 'image' in request.files:
                
                print("Image is here!")
                uploaded_image = request.files['image']
                
                if uploaded_image:
                    start_time = time.time()
                    # Process the uploaded image (e.g., resize)
                    image = Image.open(io.BytesIO(uploaded_image.read()))
                    mainProcessor = mainProcessing(image)
                    new_image, new_labels = mainProcessor.insert_to_pipeline() # Feed the image to the pipeline to render image to be displayed next
                    
                    print("new_labels = ", new_labels)
                    process_time = time.time() - start_time
                    return redirect(url_for("views.valuation", new_labels=",".join(new_labels)))
            else:
                print("no image found")

        except Exception as e:
            flash(f'Error processing the image: {str(e)}', 'danger')
            
    return render_template("home.html", user=current_user)


@views.route('/valuation', methods=['GET', 'POST'])
@login_required
def valuation():
    global mainProcessor
    global process_time
    if mainProcessor is None:
        return redirect(url_for("views.home"))
    start_time = time.time()
    new_labels = request.args.get('new_labels')  # Retrieve new_labels as a comma-separated string
    if new_labels: new_labels = new_labels.split(',')  # Split the string to get the list of labels
    # Define a dictionary to map label prefixes to symbols
    symbol_mapping = {'DC':'V', 'AC':'V', 'L':'mH', 'R':'Ω', 'D':'None', 'C':'µF'}

    # Create indexed labels with symbols
    indexed_labels = []
    for index, label in enumerate(new_labels):
        # Extract the prefix (first two characters) of the label
        prefix = label[:2]
        if prefix in ['DC', 'AC']:
            symbol = symbol_mapping.get(prefix, None)
        else:
            symbol = symbol_mapping.get(prefix[0], None)  # Get the first character as symbol

        indexed_labels.append((index, label, symbol))
    session['indexed_labels'] = indexed_labels
    print("indexed labels = ", indexed_labels)
    process_time += (time.time() - start_time)
    return render_template("valuation.html", user=current_user, indexed_labels=indexed_labels)


@views.route('/results', methods=['GET', 'POST'])
@login_required
def results():
    global process_time
    global mainProcessor  # Access the global variable
    if mainProcessor is None:
        # Handle the case where mainProcessor is not yet initialized
        return redirect(url_for("views.home"))
    indexed_labels = session.get('indexed_labels', [])
    start_time = time.time()
    component_values = []
    for index, label, symbol in indexed_labels:
        input_name = f'label_{index}'
        component_values.append(request.form.get(input_name))
        
    print('Generating Netlist!')
    netlist_file = mainProcessor.process_image(component_values, indexed_labels) # Process the image
    print('Data processed successfully!')
    
    img_source_dir = '../web_app/'
    img_dest_dir = './website/static/images/'
    
    # Create the destination directory if it doesn't exist
    os.makedirs(img_dest_dir, exist_ok=True)
    
    # Iterate through files in the source directory
    for filename in os.listdir(img_source_dir):
        print("filename = ", filename)
        if filename.endswith('.png'):
            source_file = os.path.join(img_source_dir, filename)
            destination_file = os.path.join(img_dest_dir, filename)
            shutil.copy(source_file, destination_file)
    process_time += (time.time() - start_time) 
    
    # Specify the CSV file path
    csv_file = 'process_times.csv'
    # Open the CSV file in append mode and write the process_time value
    with open(csv_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([process_time])
    print(f'Process time {process_time} saved to {csv_file}')
        
    return render_template("results.html", user=current_user, netlist_file = netlist_file)


@views.route('/download_file', methods=['GET', 'POST'])
@login_required
def download_file():
    # Define the path to the file in your local directory
    file_path = '../netlist.cir'  # Change this to the actual file path
    
    # Use Flask's send_file function to send the file for download
    return send_file(file_path, as_attachment=True)