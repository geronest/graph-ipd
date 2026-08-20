import os
import shutil

from utils.results import create_results_dir


def test_directory_creation():
    target_dir = "./results/test_utils/"
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    create_results_dir("test_utils", "sampleconfig", "sample0")
    assert os.path.isdir(target_dir)
    shutil.rmtree(target_dir, ignore_errors=True)
