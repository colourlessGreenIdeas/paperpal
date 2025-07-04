#!/bin/bash

# Install Playwright
pip install playwright

# Install browsers
playwright install chromium

# On Linux/WSL, also run:
sudo apt-get install libnspr4 libnss3
sudo playwright install-deps