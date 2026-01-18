import os
import shutil
import time

import pyautogui
import pyscreeze
import zipfile

import domain
from functools import lru_cache

from calc.const import zip_extract, zip_file


def reinstall_express():
    time.sleep(0.1) # In case it's not closed yet
    shutil.rmtree(zip_extract)
    os.makedirs(zip_extract)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(zip_extract)
    for root, dirs, files in os.walk(zip_extract, topdown=False):
        # rename files first
        for name in files + dirs:
            old = os.path.join(root, name)
            new_name = name.encode("cp437").decode("utf-8", errors="strict")
            new = os.path.join(root, new_name)
            if old != new and not os.path.exists(new):
                os.rename(old, new)


def click(asset_path: str):
    button_location = locate_with_cache(asset_path)
    if button_location is None:
        raise domain.ObjectNotFoundError
    pyautogui.click(pyautogui.center(tuple[int, int, int, int](button_location)))


def field_type(asset_path: str, data: str):
    # Find field
    field_location = locate_with_cache(asset_path)
    if field_location is None:
        raise domain.ObjectNotFoundError

    # Clear field
    pyautogui.click(field_location.left + 10, field_location.top + 10)
    pyautogui.click(field_location.left + 10, field_location.top + 10)
    pyautogui.press("backspace")

    # Type value
    pyautogui.typewrite(data, interval=0.005)


@lru_cache(maxsize=128)
def locate_with_cache(asset_path: str) -> pyscreeze.Box:
    return pyautogui.locateOnScreen(asset_path)
