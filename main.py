from bs4 import BeautifulSoup
import os

ndc_html_dir = 'html/'
ndc_txt_dir = 'txt/'

for ndc_filename in os.listdir(ndc_html_dir):
    with open(ndc_html_dir + ndc_filename) as ndc_file:
        ndc_content = ndc_file.read()
        soup = BeautifulSoup(ndc_content, 'lxml')
        with open(ndc_txt_dir + ndc_filename.split('.')[0] + '.txt', 'w') as outfile:
            outfile.write(soup.text)
