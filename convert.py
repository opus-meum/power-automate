import comtypes.client
import os

def ppt_to_pdf(input_file_path, output_file_path, formatType=32):
    powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
    powerpoint.Visible = 1

    if os.path.isabs(input_file_path):
        input_file_path = os.path.abspath(input_file_path)
    if not os.path.isabs(output_file_path):
        output_file_path = os.path.abspath(output_file_path)

    deck = powerpoint.Presentations.Open(input_file_path)
    deck.SaveAs(output_file_path, formatType)  # formatType = 32 for PDF
    deck.Close()
    powerpoint.Quit()

ppt_file = "Capability Assessment Report.pptx"
pdf_file = "Capability Assessment Report.pdf"
ppt_to_pdf(ppt_file, pdf_file)
