#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
import tempfile
import requests


def download_image(url):
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()

        suffix = ".jpg"
        if "content-type" in response.headers:
            if "png" in response.headers["content-type"]:
                suffix = ".png"
            elif "jpeg" in response.headers["content-type"]:
                suffix = ".jpg"

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_file.close()

        return tmp_file.name

    except Exception as e:
        print(f"Failed to download image: {e}")
        sys.exit(1)


def run_imagenet(input_image, output_image):
    cmd = [
        "imagenet",
        "--model=models/my_model/resnet18.onnx",
        "--labels=models/my_model/labels.txt",
        "--input-blob=input_0",
        "--output-blob=output_0",
        input_image,
        output_image
    ]

    print("Running command:")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: imagenet failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def is_url(path):
    return path.startswith("http://") or path.startswith("https://")


def main():
    parser = argparse.ArgumentParser(description="Run imagenet on local file or image URL.")
    parser.add_argument("input", help="Image path or URL")
    parser.add_argument("-o", "--output", default="output_predicted.jpg")

    args = parser.parse_args()

    temp_file = None

    if is_url(args.input):
        print(f"Downloading image from URL: {args.input}")
        temp_file = download_image(args.input)
        input_path = temp_file
    else:
        input_path = args.input

    run_imagenet(input_path, args.output)

    if temp_file:
        os.remove(temp_file)
        print("Cleaned up temporary file")


if __name__ == "__main__":
    main()
