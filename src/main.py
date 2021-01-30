from bs4 import BeautifulSoup
import os
import re

ndc_html_dir = '../html/'
ndc_txt_dir = '../txt/'


def html_to_txt(html_dir, txt_dir):
    for ndc_filename in os.listdir(html_dir):
        with open(html_dir + ndc_filename) as ndc_file:
            ndc_content = ndc_file.read()
            soup = BeautifulSoup(ndc_content, 'lxml')
            with open(txt_dir + ndc_file.split('.')[0] + '.txt', 'w') as outfile:
                outfile.write(soup.text)


def extract_a_ids(html_dir, a_id_csv, filter_EN):
    with open(a_id_csv, 'w') as a_id_file:
        for ndc_filename in os.listdir(html_dir):
            with open(html_dir + ndc_filename) as ndc_file:
                ndc_content = ndc_file.read()
                ndc_file_abrev = ndc_filename.split("-")[0]
                if filter_EN and ndc_filename.split("-")[2].split(".")[0] != "EN":
                    continue
                soup = BeautifulSoup(ndc_content, 'lxml')
                a_tags = soup.find_all("a")
                for tag in a_tags:
                    id = tag.get("id")
                    if id is not None and re.search("ref[0-9]", id) is None:
                        a_id_file.write(ndc_file_abrev + "," + id + "\n")
            

if __name__ == "__main__":
    extract_a_ids(ndc_html_dir, "../csv/A-IDs.csv", True)
    os.system("sort ../csv/A-IDs.csv > ../csv/sorted-A-IDs.csv")

    extract_a_ids(ndc_html_dir, "../csv/A-IDs-ALL-LANGS.csv", False)
    os.system("sort ../csv/A-IDs-ALL-LANGS.csv > ../csv/sorted-A-IDs-ALL-LANGS.csv")