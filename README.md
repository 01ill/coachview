# coachview

This project provides some tools for giving insight into football data

## Setup
### Getting the data
This project relies on the open data repository from Impect. To get the data just clone their repository into the working directory.
```shell
git clone https://github.com/ImpectAPI/open-data.git
```

If the data is cloned into another directory the path in `settings.py` has to be adjusted.

### Running
Tested with python 3.13
```shell
pip install -r requirements.txt
streamlit run app.py
```