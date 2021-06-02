import os
import tarfile
import zipfile
from urllib.request import urlretrieve
from multiprocessing import Lock
from functools import wraps
from .file_types import File_type
from .constants import data_url



class SingletonType(type):

    def __new__(mcs, name, bases, attrs):
        # Assume the target class is created (i.e. this method to be called) in the main thread.
        cls = super(SingletonType, mcs).__new__(mcs, name, bases, attrs)
        cls.__shared_instance_lock__ = Lock()
        return cls

    def __call__(cls, *args, **kwargs):
        with cls.__shared_instance_lock__:
            try:
                return cls.__shared_instance__
            except AttributeError:
                cls.__shared_instance__ = super(SingletonType, cls).__call__(*args, **kwargs)
                return cls.__shared_instance__

class DownloardAndExtractFile(metaclass=SingletonType):

    """
    A singleton class to download and extract file from the internet
    """
    def download_and_extract_file(self, url, file_type=File_type.ZIP):
        base_name = os.path.basename(url)
        file_name = os.path.splitext(base_name)[0]

        if file_type == File_type.TAR:
            file_name = os.path.splitext(file_name)[0]

        if os.path.isdir(file_name):
            print(f'{file_name} folder already exit')
        else:
            print(f" About to Download {base_name}  ...")
            try:
                file_tmp = urlretrieve(url,filename=None)[0]
            except IOError:
                print(f"Can't retrieve url {url}")
                return
            print(f"File downlaoded to a temporary file {file_tmp}")
            if file_type == File_type.ZIP:
                self.extract_zip_file(file_tmp, file_name)
            elif file_type == File_type.TAR:
                self.extract_tar_file(file_tmp,file_name)
            else:
                print("Unknown File type")

    def extract_zip_file(self,file_tmp, file_name):

        print(f"Extracting file: {file_name}")
        with zipfile.ZipFile(file_tmp, 'r') as zip_ref:
            zip_ref.extractall(file_name)

    def extract_tar_file(self,file_tmp,file_name):
        print(f"Extracting file: {file_name}")
        try:
            tar = tarfile.open(file_tmp)
            tar.extractall(file_name)
        except:
            print(f"Can't tar the {file_name}")
            return
        print(f"Successfully extracted to folder {file_name}")

def main():
    DownloardAndExtractFile().download_and_extract_file(data_url.POWER_PLANT_DATASET_URL)

if __name__ == '__main__':
    main()
