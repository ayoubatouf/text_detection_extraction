# Text Detection and Extraction for Mixed Image Datasets

## Overview

This project provides an efficient solution for processing large collections of mixed images, such as personal photos alongside text-heavy images (e.g., quotes, notes). It combines two key functionalities: detecting and filtering images based on text content using the `EAST` text detection model and extracting text from images using OCR. The system automatically detects text regions, draws bounding boxes around them, and extracts the text using `PaddleOCR`. Images with detected text are saved in a designated folder with bounding boxes, while the extracted text is saved in separate text files. Non-text images (e.g., personal photos) are skipped. By organizing images with and without text, the system simplifies managing and processing mixed-image datasets, making it ideal for scenarios where text-based images need to be separated from visual-only images.

## Usage

To separate photos from text-containing images, run the `filter.py` script, ensuring that you specify the source and destination folders. For extracting text and drawing bounding boxes on images, run the `extract.py` script. These functionalities allow for easy management of mixed image datasets by filtering out personal photos and processing images with text for further analysis.


## Requirements

- OpenCV : 4.10.0
- PaddleOCR : 2.9.1
- NumPy : 1.26.4
