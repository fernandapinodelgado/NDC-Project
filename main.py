from bs4 import BeautifulSoup
import os

ndc_html_dir = 'html/'
ndc_txt_dir = 'txt/'

for ndc_name in os.listdir('html/'):
    with open(ndc_html_dir + ndc_name) as ndc_file:
        ndc_content = ndc_file.read()
        soup = BeautifulSoup(ndc_content, 'lxml')
        with open(ndc_txt_dir + ndc_name.split('.')[0] + '.txt', 'w') as outfile:
            outfile.write(soup.text)
