"""See https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url"""
import sys
import requests
import tarfile
import os.path
import sys
from pathlib import Path


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def fetch_lcd():
    file_id = "1BM3qJp1eNmrFsXQ-C4exlyJvH2buY45n"
    destination = "./datasets/LCD/lcd.tar.gz"
    folder = "./datasets/LCD/"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif len(os.listdir(folder)) != 0:
        print("You have downloaded LCD data set. ")
        return

    print(f"dowload {file_id} to {destination}")
    download_file_from_google_drive(file_id, destination)

    if destination.endswith("tar.gz"):
        tar = tarfile.open(destination, "r:gz")
        tar.extractall(path=folder)
        tar.close()
    elif destination.endswith("tar"):
        tar = tarfile.open(destination, "r:")
        tar.extractall(path=folder)
        tar.close()

    file_to_rm = Path(destination)
    file_to_rm.unlink()



def fetch_gmsc():  
    file_id = "1HdZbxgD06u7Xpu54OM4lie9QPnntwjcC"
    destination = "./datasets/GMSC/gmsc.tar.gz"
    folder = "./datasets/GMSC/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif len(os.listdir(folder)) != 0:
        print("You have downloaded LCD data set. ")
        return

    print(f"dowload {file_id} to {destination}")
    download_file_from_google_drive(file_id, destination)

    if destination.endswith("tar.gz"):
        tar = tarfile.open(destination, "r:gz")
        tar.extractall(path=folder)
        tar.close()
    elif destination.endswith("tar"):
        tar = tarfile.open(destination, "r:")
        tar.extractall(path=folder)
        tar.close()

    file_to_rm = Path(destination)
    file_to_rm.unlink()

def main():
    # testing purpose only
    if len(sys.argv) >= 3:
        file_id = sys.argv[1]
        destination = sys.argv[2]
    else:
        file_id = "1BM3qJp1eNmrFsXQ-C4exlyJvH2buY45n"
        destination = "./datasets/LCD/lcd.tar.gz"
    print(f"dowload {file_id} to {destination}")
    download_file_from_google_drive(file_id, destination)

    if destination.endswith("tar.gz"):
        tar = tarfile.open(destination, "r:gz")
        tar.extractall(path="./datasets/LCD/")
        tar.close()
    elif destination.endswith("tar"):
        tar = tarfile.open(destination, "r:")
        tar.extractall()
        tar.close()

DATASETS = {
    "LCD":fetch_lcd,
    "GMSC": fetch_gmsc
}

if __name__ == "__main__":
    # main()
    if len(sys.argv) <= 1:
        print("You have to specify which dataset to download. Current options are: LCD, GMSC")

    for i in range(1, len(sys.argv)):
        assert sys.argv[i] in DATASETS.keys(),"The dataset cannot be downloaded. Valid options are: LCD, GMSC"
        DATASETS[sys.argv[i]]()