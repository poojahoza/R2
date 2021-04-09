# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

def retrieve_file_names_from_folder(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles

