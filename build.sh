#!/bin/bash

set -ex

rm -rf out

jupyter nbconvert --config jupyter_nbconvert_config.py
sed -i "" -e "s/\.png)$/\.png#center)/g" out/*.md
sed -i "" -e "s/^!\[png\](images/!\[png\](\/images/g" out/*.md
