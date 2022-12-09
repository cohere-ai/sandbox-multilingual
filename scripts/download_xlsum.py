import os
import datasets
import pandas as pd
from typing import List

LANGUAGES: List[str] = [
    'amharic', 'arabic', 'azerbaijani', 'bengali', 'burmese', 'chinese_simplified', 'chinese_traditional', 'english',
    'french', 'gujarati', 'hausa', 'hindi', 'igbo', 'indonesian', 'japanese', 'kirundi', 'korean', 'kyrgyz', 'marathi',
    'nepali', 'oromo', 'pashto', 'persian', 'pidgin', 'portuguese', 'punjabi', 'russian', 'scottish_gaelic',
    'serbian_cyrillic', 'serbian_latin', 'sinhala', 'somali', 'spanish', 'swahili', 'tamil', 'telugu', 'thai',
    'tigrinya', 'turkish', 'ukrainian', 'urdu', 'uzbek', 'vietnamese', 'welsh', 'yoruba'
]


def main():
    dfs = []
    for language in LANGUAGES:
        dataset = datasets.load_dataset("csebuetnlp/xlsum", language, split="train")
        df = dataset.to_pandas()
        df["language"] = language
        dfs.append(df)
    dfs_combined = pd.concat(dfs)
    os.makedirs("./data", exist_ok=True)
    dfs_combined.to_csv("./data/data.csv")


if __name__ == '__main__':
    main()
