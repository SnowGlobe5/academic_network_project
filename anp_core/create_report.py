import os
import json
import re
from collections import defaultdict
from pdf2image import convert_from_path
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# Folder containing the experiment folders
base_folder = "/home/sguidotti/academic_network_project/anp_models"

date_lower = "2024_02_01"
date_upper = "2024_04_01"

# Convert date strings to integers for comparison
date_lower_int = int(date_lower.replace("_", ""))
date_upper_int = int(date_upper.replace("_", ""))

# Dictionary to organize the results
results = [defaultdict(list), defaultdict(list)]

# Function to read the info.json file
def read_info_json(folder):
    info_file = os.path.join(folder, "info.json")
    if os.path.exists(info_file):
        with open(info_file) as f:
            info_data = json.load(f)
            del info_data['data']
            if not info_data['use_infosphere']:
                del info_data['infosphere_expansion']
            return info_data
    return None

# Function to read the anp_link_prediction_co_author_log.json file
def read_log_json(folder):
    log_file = os.path.join(folder, "anp_link_prediction_co_author_log.json")
    if os.path.exists(log_file):
        with open(log_file) as f:
            log_data = json.load(f)
            return log_data
    return None

# Regex for the folder format
folder_pattern = re.compile(r'anp_link_prediction_co_author_(\d{4}_\d{2}_\d{2})_\d{2}_\d{2}_\d{2}')

# Scan all the experiment folders
for experiment_folder in os.listdir(base_folder):
    experiment_path = os.path.join(base_folder, experiment_folder)
    if not os.path.isdir(experiment_path):
        continue

    # Check if the folder name matches the desired format and falls within the date range
    match = folder_pattern.match(experiment_folder)
    if not match:
        continue

    folder_date = match.group(1)
    if not (date_lower_int <= int(folder_date.replace("_", "")) <= date_upper_int):
        continue

    # Read info.json
    info = read_info_json(experiment_path)
    if info is None:
        continue

    # Read anp_link_prediction_co_author_log.json
    log_data = read_log_json(experiment_path)
    if log_data is None:
        continue

    # Extract the values of interest
    prediction_type = info.get("only_new", "N/A")
    lr = info.get("lr", "N/A")
    use_infosphere = info.get("use_infosphere", "N/A")
    validation_accuracy = log_data["validation_accuracy_list"][-1] if "validation_accuracy_list" in log_data else "N/A"

    # Create a title for the report
    title = f"Folder: {experiment_folder}"
    
    # Create the body of the report
    body = f"Learning Rate: {lr}\n"
    body += f"Use Infosphere: {use_infosphere}\n"
    body += f"Validation Accuracy: {validation_accuracy}\n"
    body += "Info JSON:\n"
    body += json.dumps(info, indent=4)
    
    # Save the images if they exist
    images = []
    for image_name in ["anp_link_prediction_co_author_accuracy.pdf", "anp_link_prediction_co_author_CM.pdf", "anp_link_prediction_co_author_loss.pdf"]:
        image_path = os.path.join(experiment_path, image_name)
        if os.path.exists(image_path):
            # Convert the PDF to a JPEG image
            pdf_images = convert_from_path(image_path, dpi=300, fmt='jpeg')
            for idx, pdf_image in enumerate(pdf_images):
                # Save the temporary image
                temp_image_path = f"temp_{image_name}_{experiment_folder}.jpeg"
                pdf_image.save(temp_image_path, "JPEG")
                images.append(temp_image_path)

    # Add the data to the results dictionary
    results[prediction_type][(float(lr), use_infosphere)].append({
        "title": title,
        "body": body,
        "images": images
    })

# Create the PDF
doc = SimpleDocTemplate("report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
style_title = styles["Title"]
style_body = styles["Normal"]
style_body.alignment = TA_CENTER

# List of elements to insert into the PDF
elements = []

# Header
elements.append(Paragraph("Experiment Report", style_title))
elements.append(Spacer(1, 12))

# Sort the results by Learning Rate and Use Infosphere
sorted_results = [[], []]
for only_new in range(2):
    sorted_results[only_new] = sorted(results[only_new].items(), key=lambda x: (x[0][0], x[0][1]))

# Create the report for each learning rate and infosphere
for only_new in range(2):
    elements.append(Paragraph("All co-authors" if only_new else "Only new", style_title))
    for key, entries in sorted_results[only_new]:
        elements.append(Paragraph(f"Learning Rate: {key[0]}, Use Infosphere: {key[1]}", style_title))
        elements.append(Spacer(1, 6))

        for entry in entries:
            elements.append(Paragraph(entry["title"], style_body))
            elements.append(Paragraph(entry["body"], style_body))

            # List of images to insert in this section
            images_section = []
            
            # Add the images to the current section list
            for image_path in entry["images"]:
                image = PILImage.open(image_path)
                width, height = image.size
                max_width = 200  # Width in points
                aspect = height / width
                adjusted_width = max_width
                adjusted_height = adjusted_width * aspect
                image_obj = Image(image_path, width=adjusted_width, height=adjusted_height)
                images_section.append(image_obj)

            # Create a table to display the images horizontally
            if images_section:
                table_data = [images_section]
                table = Table(table_data, colWidths=len(images_section) * [200])
                elements.append(table)

            elements.append(Spacer(1, 12))

doc.build(elements)

# Remove the temporary images
for only_new in range(2):
    for key, entries in results[only_new].items():
        for entry in entries:
            for image_path in entry["images"]:
                os.remove(image_path)

print("The report was successfully created.")
