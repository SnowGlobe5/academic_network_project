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

date_lower = "2024_04_24"
date_upper = "2024_05_16"
lr_lower = 0.00001
lr_upper = 0.00000005

infosphere_string = ["Baseline (No infosphere)", "Future infosphere", "Infosphere TOP PAPER", "Infosphere TOP PAPER PER TOPIC"]
# Convert date strings to integers for comparison
date_lower_int = int(date_lower.replace("_", ""))
date_upper_int = int(date_upper.replace("_", ""))

# Dictionary to organize the results
results = [[{}, {}, {}, {}], [{}, {}, {}, {}]]

# Function to read the info.json file
def read_info_json(folder):
    info_file = os.path.join(folder, "info.json")
    if os.path.exists(info_file):
        with open(info_file) as f:
            info_data = json.load(f)
            del info_data['data']
            if info_data['infosphere_type'] == 0:
                del info_data['infosphere_parameters']
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

    if not (lr_upper <= float(lr) <= lr_lower):
        continue

    validation_accuracy = log_data["validation_accuracy_list"][-1] if "validation_accuracy_list" in log_data else "N/A"

    # Calculate highest accuracy and average of last 10 epochs
    highest_accuracy = max(log_data["validation_accuracy_list"]) if "validation_accuracy_list" in log_data else "N/A"
    avg_last_10_epochs = sum(log_data["validation_accuracy_list"][-10:]) / 10 if "validation_accuracy_list" in log_data and len(log_data["validation_accuracy_list"]) >= 10 else "N/A"

    # Create a title for the report
    folder = f"Folder: {experiment_folder}"

    # Create the body of the report
    body = f"Learning Rate: {lr},\n"
    body += f"Last Validation Accuracy: {validation_accuracy},\n"
    body += f"Highest Accuracy: {highest_accuracy},\n"
    body += f"Average of Last 10 Epochs: {avg_last_10_epochs}\n"

    infosphere_parameters = info.get("infosphere_parameters", None)


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
    infotype = info["infosphere_type"] if info.get("infosphere_type") else 0
    if not results[prediction_type][infotype].get(infosphere_parameters):
        results[prediction_type][infotype][infosphere_parameters] = []
    results[prediction_type][infotype][infosphere_parameters].append({
        "folder": folder,
        "body": body,
        "images": images,
        "learning_rate": lr
    })

# Create the PDF
doc = SimpleDocTemplate(f"report_{date_lower}_{date_upper}_{lr_lower}_{lr_upper}.pdf", pagesize=letter)
styles = getSampleStyleSheet()
# style_body.alignment = TA_CENTER

# List of elements to insert into the PDF
elements = []

# Header
elements.append(Paragraph("Experiment Report", styles["Title"]))
elements.append(Spacer(1, 12))

# Sort the results by Learning Rate and Use Infosphere
# sorted_results = [[{}, {}, {}, {}], [{}, {}, {}, {}]]
# for only_new in range(2):
#     for infotype in range(4):
#         try:
#             sorted_results[only_new][infotype] = sorted(results[only_new][infotype].items(), key=lambda x: (x[0][0], x[0][1]))
#         except:
#             print(f"skip {only_new}, {infotype}")

# Create the report for each learning rate and infosphere
for only_new in range(2):
    elements.append(Paragraph("Only new" if only_new else "All co-authors",  styles["Heading1"]))
    for i, entries2 in enumerate(results[only_new]):
        elements.append(Paragraph(f"{infosphere_string[i]}", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        sorted_entries2 = dict(sorted(entries2.items(), key=lambda x: x[0]))

        for infosphere_parameter, entries in sorted_entries2.items():
            if infosphere_parameter:
                    elements.append(Paragraph(f"parameters: {infosphere_parameter}", styles["Heading3"]))
            entries.sort(key=lambda x: x.get("learning_rate", 0)) 
            for entry in entries:
                elements.append(Paragraph(entry["body"], styles["Normal"]))

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

                elements.append(Paragraph(entry["folder"], styles["Heading6"]))

                elements.append(Spacer(1, 12))

doc.build(elements)

# Remove the temporary images
for only_new in range(2):
      for infosphere_parameter, entries in entries2.items():
            for entry in entries:
                for image_path in entry["images"]:
                    os.remove(image_path)

print("The report was successfully created.")