import subprocess
import os

def convert_pptx_to_pdf(input_path, output_path):
    try:
        # Construct the unoconv command
        command = ['unoconv', '-f', 'pdf', '-o', output_path, input_path]
        
        # Execute the command
        subprocess.run(command, check=True)
        print(f"Conversion successful: '{input_path}' to '{output_path}'")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

# Example usage
input_pptx = os.path.realpath("Capability Assessment Report.pptx")
output_pdf = "Capability Assessment Report.pdf"
convert_pptx_to_pdf(input_pptx, output_pdf)
