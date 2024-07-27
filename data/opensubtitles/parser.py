from bs4 import BeautifulSoup
import requests
import re
from pathlib import Path
import os
from zipfile import ZipFile
import time

CURRENT_PATH = Path(__file__).parent

THRESHOLD_DOWNLOADS = 1000

uploader_info = [
    {
        'name': "masousa",
        'language': "pt-PT"
    },
]


def get_html_page(uploader_name, offset=0):

    html_page = requests.get(
        f"https://www.opensubtitles.org/pb/search/sublanguageid-por/uploader-{uploader_name}/sort-7/asc-0/offset-{offset}")

    if not html_page.ok:
        return None

    return BeautifulSoup(html_page.content, 'html.parser')


def extract_number_downloads(html_page, id):
    table_row = html_page.find("tr", id=f"name{id}")
    fifth_column = table_row.find_all("td")[4]

    return re.search(r"\d+", fifth_column.text).group(0)


def get_info(uploader_info):

    results = {
        'id': [],
        'name': [],
        'downloads': [],
        'uploader': [],
        'link': [],
        'language': []
    }

    for uploader in uploader_info:
        for offset in range(0, 500, 40):
            html_page = get_html_page(uploader['name'], offset)

            if html_page is None:
                print(f"Error Getting HTML Page{uploader['name']}")
                break

            for link in html_page.find_all("a", class_="bnone"):
                #Extract id
                id = re.search(r"\d+", link.get("href")).group(0)

                #Extract number of downloads
                download_number = extract_number_downloads(html_page, id)

                # Only download subtitles with more than THRESHOLD_DOWNLOADS
                if int(download_number) < THRESHOLD_DOWNLOADS:
                    continue
                else:
                    results['id'].append(id)
                    results['downloads'].append(download_number)

                #Extract name
                results['name'].append(link.text.strip().replace("\n", ""))

                results['uploader'].append(uploader['name'])

                results['language'].append(uploader['language'])

        return results

# Parse srt file to txt


def srt_to_txt(srt):
    final_text = ""
    counter_line = 0

    for line in srt.split("\n"):

        counter_line += 1

        if line == "":
            counter_line = 0
        elif counter_line > 2:
            final_text += line.strip().replace('Watch any video online with Open-SUBTITLES Free Browser extension: osdb.link/ext', ' ').replace('api.OpenSubtitles.org is deprecated, please implement REST API from OpenSubtitles.com', ' ').replace('<i>', '').replace('</i>', '').replace(
                'Do you want subtitles for any video? -=[ ai.OpenSubtitles.com ]=- ', ' ').replace('Watch any video online with Open-SUBTITLES Free Browser extension: osdb.link/ext', ' ').replace('api.OpenSubtitles.org is deprecated, please implement REST API from OpenSubtitles.com', ' ').replace('Anuncie o seu produto ou marca aqui Contacte www.OpenSubtitles.org', ' ').replace("Ajude-nos e torne-se membro VIP para remover todos os anúncios do % url%", ' ') + " "

    return final_text[int(0.05*len(final_text)):int(0.95*len(final_text))]


def download_subtitles(results):
    data_path = os.path.join(CURRENT_PATH, "data")

    try:
        os.makedirs(os.path.join(data_path, 'pt-PT'), exist_ok=True)
    except:
        print(
            f"Warning: Folder {os.path.join(data_path, 'pt-PT')} already exists")

    try:
        os.makedirs(os.path.join(data_path, 'pt-BR'), exist_ok=True)
    except:
        print(
            f"Warning: Folder {os.path.join(data_path, 'pt-BR')} already exists")

    for id, name, downloads, uploader, language in zip(results['id'], results['name'], results['downloads'], results['uploader'], results['language']):
        print(f"Downloading {name}...")

        request = requests.get(
            f"https://dl.opensubtitles.org/pb/download/sub/{id}")

        if not request.ok:
            print(
                f"Error Downloading ZIP {name} | URL: https://dl.opensubtitles.org/pb/download/sub/{id}")
            print(f"Error Code: {request.status_code}")
            print('-------------------')
            time.sleep(5)
            continue
        """
        if os.name == "nt":
            temp_path = os.path.join(os.path.join(
                "C:\\Windows\\Temp", language), id)
        else:
            temp_path = os.path.join("/tmp", language, id)

        """
        temp_path = os.path.join("tmp", language, id)

        final_path = os.path.join(data_path, language, id)

        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)

        with open(os.path.join(temp_path, f"{id}.zip"), "wb") as f:
            f.write(request.content)

        with ZipFile(os.path.join(temp_path, f"{id}.zip"), 'r') as zipObj:
            zipObj.extractall(path=temp_path)

        # Read srt file
        for root, directories, files in os.walk(temp_path):
            for file in files:
                if file.endswith('.srt'):
                    #Read file
                    with open(os.path.join(root, file), "r", encoding="iso-8859-1") as f:
                        srt_file = f.read()

        if srt_file is None:
            print(f"Error Reading {name} SRT file")
            continue

        with open(os.path.join(final_path, f"{id}.txt"), "w", encoding="utf-8") as f:
            parsed_srt = srt_to_txt(srt_file)
            f.write(parsed_srt)

        time.sleep(5)


if __name__ == "__main__":
    results = get_info(uploader_info)
    download_subtitles(results)
