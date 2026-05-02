python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements\base.txt
pip install -r requirements\dev.txt
pip install -r requirements\ml.txt
