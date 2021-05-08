#!/usr/bin/env bash

VENVNAME=venv_imagesearch 

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# problems when installing from requirements.txt
pip install ipython
pip install jupyter

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"