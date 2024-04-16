import win32com.client
import os

x = os.path.realpath("Capability Assessment Report.pptx")
y = os.getcwd() + "Capability Assessment Reports.pdf"

powerpoint = win32com.client.gencache.EnsureDispatch('PowerPoint.Application')
powerpoint.Visible = True
pdf = powerpoint.Presentations.Open(x)
pdf.SaveAs(y, FileFormat = 32)
pdf.Close()
powerpoint.Quit()
