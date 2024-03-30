from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os
import json
import re
from collections import defaultdict
from pdf2image import convert_from_path
from PIL import Image as PILImage

# Cartella contenente le cartelle degli esperimenti
base_folder = "/home/sguidotti/academic_network_project/anp_models"

# Dizionario per organizzare i risultati
results = [defaultdict(list), defaultdict(list)]

# Funzione per leggere il file info.json
def read_info_json(folder):
    info_file = os.path.join(folder, "info.json")
    if os.path.exists(info_file):
        with open(info_file) as f:
            info_data = json.load(f)
            # Rimuovi il campo "data" se presente
            del info_data['data']
            return info_data
    return None

# Funzione per leggere il file anp_link_prediction_co_author_log.json
def read_log_json(folder):
    log_file = os.path.join(folder, "anp_link_prediction_co_author_log.json")
    if os.path.exists(log_file):
        with open(log_file) as f:
            log_data = json.load(f)
            return log_data
    return None

# Regex per il formato della cartella
folder_pattern = re.compile(r'anp_link_prediction_co_author_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}')

# Scandire tutte le cartelle degli esperimenti
for experiment_folder in os.listdir(base_folder):
    experiment_path = os.path.join(base_folder, experiment_folder)
    if not os.path.isdir(experiment_path):
        continue

    # Verifica se il nome della cartella corrisponde al formato desiderato
    if not folder_pattern.match(experiment_folder):
        continue

    # Leggere info.json
    info = read_info_json(experiment_path)
    if info is None:
        continue

    # Leggere anp_link_prediction_co_author_log.json
    log_data = read_log_json(experiment_path)
    if log_data is None:
        continue

    # Estraiamo i valori di interesse
    prediction_type = info.get("only_new", "N/A")
    lr = info.get("lr", "N/A")
    use_infosphere = info.get("use_infosphere", "N/A")
    validation_accuracy = log_data["validation_accuracy_list"][-1] if "validation_accuracy_list" in log_data else "N/A"

    # Creiamo un titolo per il report
    title = f"Cartella: {experiment_folder}"
    
    # Creiamo il corpo del report
    body = f"Learning Rate: {lr}\n"
    body += f"Use Infosphere: {use_infosphere}\n"
    body += f"Validation Accuracy: {validation_accuracy}\n"
    body += "Info JSON:\n"
    body += json.dumps(info, indent=4)
    
    # Salviamo le immagini se esistono
    images = []
    for image_name in ["anp_link_prediction_co_author_accuracy.pdf", "anp_link_prediction_co_author_CM.pdf", "anp_link_prediction_co_author_loss.pdf"]:
        image_path = os.path.join(experiment_path, image_name)
        if os.path.exists(image_path):
            # Convertiamo il PDF in un'immagine JPEG
            pdf_images = convert_from_path(image_path, dpi=300, fmt='jpeg')
            for idx, pdf_image in enumerate(pdf_images):
                # Salviamo l'immagine temporanea
                temp_image_path = f"temp_{image_name}_{experiment_folder}.jpeg"
                pdf_image.save(temp_image_path, "JPEG")
                images.append(temp_image_path)

    # Aggiungiamo i dati al dizionario dei risultati
    results[prediction_type][(float(lr), use_infosphere)].append({
        "title": title,
        "body": body,
        "images": images
    })

# Creiamo il PDF
doc = SimpleDocTemplate("report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
style_title = styles["Title"]
style_body = styles["Normal"]
style_body.alignment = TA_CENTER

# Lista di elementi da inserire nel PDF
elements = []

# Intestazione
elements.append(Paragraph("Report degli Esperimenti", style_title))
elements.append(Spacer(1, 12))

# Ordiniamo i risultati per Learning Rate
sorted_results = [[], []]
for only_new in range(2):
    sorted_results[only_new] = sorted(results[only_new].items(), key=lambda x: x[0][0])

# Creazione del report per ogni learning rate e infosphere
for only_new in range(2):
    elements.append(Paragraph("All co-authors" if only_new else "Only new", style_title))
    for key, entries in sorted_results[only_new]:
        elements.append(Paragraph(f"Learning Rate: {key[0]}, Use Infosphere: {key[1]}", style_title))
        elements.append(Spacer(1, 6))

        for entry in entries:
            elements.append(Paragraph(entry["title"], style_body))
            elements.append(Paragraph(entry["body"], style_body))

            # Lista delle immagini da inserire in questa sezione
            images_section = []
            
            # Aggiungiamo le immagini alla lista della sezione corrente
            for image_path in entry["images"]:
                image = PILImage.open(image_path)
                width, height = image.size
                max_width = 200  # Width in points
                aspect = height / width
                adjusted_width = max_width
                adjusted_height = adjusted_width * aspect
                image_obj = Image(image_path, width=adjusted_width, height=adjusted_height)
                images_section.append(image_obj)

            # Creiamo una tabella per visualizzare le immagini orizzontalmente
            if images_section:
                table_data = [images_section]
                table = Table(table_data, colWidths=len(images_section) * [200])
                elements.append(table)

            elements.append(Spacer(1, 12))

# Costruiamo il PDF
doc.build(elements)

# Rimuoviamo le immagini temporanee
for only_new in range(2):
    for key, entries in results[only_new].items():
        for entry in entries:
            for image_path in entry["images"]:
                os.remove(image_path)

print("Il report è stato creato con successo.")
