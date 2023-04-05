## Regarding data

1. 10k.json is the actual json file without any labels and reason.
2. I wrote a code which will extract text rows from 10k.json file and will save them into a csv which is named as 10k.csv. The code is Json to Pandas.ipynb
3. Then run the main_script.py code on 10k.csv data.
4. The code i.e. main_script.py will return a output i.e. file named labeled_sentences.csv which contains the text, label and reason.
5. We need to assign the label and reason to the original 10k.json which is base file. To do that you have to run Assigning label & reason columns to 10k json.ipynb code which will get you the output which will be named as updated_10.json


## Steps to run the code

1. Download all the files to local pc and save it in a folder.
2. Install the required libraries
3. To run the code, you have to insert your OpenAI API-KEY in main_script.py
4. Open the command prompt or anaconda prompt (Wherever you have installed the libraries), change directory to the folder where the downloaded files are present, run rerun.bat


#### updated_10k.json is the updated file with annotations (GPT-4)
