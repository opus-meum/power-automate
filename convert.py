import subprocess
import os

# Define paths
pptx_path = 'Capability Assessment Report.pptx'
output_dir = os.getcwd()  # Current directory, adjust as needed

# Convert PPTX to PDF
subprocess.run(['libreoffice', '--convert-to', 'pdf', '--outdir', output_dir, pptx_path], check=True)

# Assuming the PDF has the same base name as the PPTX file
pdf_path = os.path.splitext(pptx_path)[0] + '.pdf'