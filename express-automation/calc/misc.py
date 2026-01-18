import os
import shutil
import time

import pyautogui
import pyscreeze
import zipfile

import domain
from typing import Optional

from calc.const import zip_extract, zip_file

_LOCATE_CACHE: dict[str, pyscreeze.Box] = {}


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


def click(asset_path: str, confidence: float = 0.9, grayscale: bool = True, retries: int = 3,
          retry_interval: float = 0.2):
    button_location = locate_with_cache(
        asset_path,
        confidence=confidence,
        grayscale=grayscale,
        retries=retries,
        retry_interval=retry_interval,
    )
    if button_location is None:
        raise domain.ObjectNotFoundError
    pyautogui.click(pyautogui.center(tuple[int, int, int, int](button_location)))


def field_type(asset_path: str, data: str, confidence: float = 0.9, grayscale: bool = True, retries: int = 3,
               retry_interval: float = 0.2):
    # Find field
    field_location = locate_with_cache(
        asset_path,
        confidence=confidence,
        grayscale=grayscale,
        retries=retries,
        retry_interval=retry_interval,
    )
    if field_location is None:
        raise domain.ObjectNotFoundError

    # Clear field
    pyautogui.click(field_location.left + 10, field_location.top + 10)
    pyautogui.click(field_location.left + 10, field_location.top + 10)
    pyautogui.press("backspace")

    # Type value
    pyautogui.typewrite(data, interval=0.005)


def _locate_once(asset_path: str, confidence: float, grayscale: bool) -> Optional[pyscreeze.Box]:
    try:
        return pyautogui.locateOnScreen(asset_path, confidence=confidence, grayscale=grayscale)
    except Exception:
        return pyautogui.locateOnScreen(asset_path, grayscale=grayscale)


def locate_with_cache(asset_path: str, confidence: float = 0.9, grayscale: bool = True, retries: int = 3,
                      retry_interval: float = 0.2) -> Optional[pyscreeze.Box]:
    cached = _LOCATE_CACHE.get(asset_path)
    if cached is not None:
        return cached

    for attempt in range(retries):
        location = _locate_once(asset_path, confidence=confidence, grayscale=grayscale)
        if location is not None:
            _LOCATE_CACHE[asset_path] = location
            return location
        if attempt < retries - 1:
            time.sleep(retry_interval)
    return None
