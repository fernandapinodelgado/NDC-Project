from bs4 import BeautifulSoup
import os
import re

ndc_html_dir = '../html/'
ndc_txt_dir = '../txt/'
ndc_csv_dir = '../csv/'


def html_to_txt(html_dir, txt_dir):
    for ndc_filename in os.listdir(html_dir):
        with open(html_dir + ndc_filename) as ndc_file:
            ndc_content = ndc_file.read()
            soup = BeautifulSoup(ndc_content, 'lxml')
            with open(txt_dir + ndc_file.split('.')[0] + '.txt', 'w') as outfile:
                outfile.write(soup.text)


def extract_a_ids(html_dir, csv_dir):
    english_file = "TAGS-EN.csv"
    non_english_file = "TAGS-NON-EN.csv"
    with open(csv_dir + english_file, 'w') as en_id_file:
        with open(csv_dir + non_english_file, 'w') as non_en_id_file:

            for ndc_filename in os.listdir(html_dir):
                with open(html_dir + ndc_filename) as ndc_file:
                    ndc_content = ndc_file.read()
                    ndc_country = ndc_filename.split("-")[0]
                    ndc_language = ndc_filename.split("-")[2].split(".")[0]

                    soup = BeautifulSoup(ndc_content, 'lxml')
                    a_tags = soup.find_all("a")
                    for tag in a_tags:
                        id = tag.get("id")
                        if id is not None and re.search("ref[0-9]", id) is None:
                            if ndc_language == "EN":
                                en_id_file.write(ndc_country + "," + id + "\n")
                            else:
                                non_en_id_file.write(ndc_country + "," + id + "\n")

    uniq(csv_dir, sort(csv_dir, english_file))
    uniq(csv_dir, sort(csv_dir, non_english_file))


def sort(file_dir, file_name):
    sorted_file = "sorted-" + file_name
    os.system("sort " + file_dir + file_name + " > " + file_dir + sorted_file)
    return sorted_file


def uniq(file_dir, file_name):
    uniqed_file = "uniqed-" + file_name
    os.system("uniq " + file_dir + file_name + " > " + file_dir + uniqed_file)
    return uniqed_file
            

if __name__ == "__main__":
    extract_a_ids(ndc_html_dir, ndc_csv_dir)
