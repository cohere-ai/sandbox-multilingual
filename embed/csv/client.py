# Copyright (c) 2022 Cohere Inc. and its affiliates.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License in the LICENSE file at the top
# level of this repository.

import glob
import os
import pandas as pd
from tqdm import tqdm

from ..client import Article


class Client:

    def __init__(self, rootdir):
        self.rootdir = rootdir

    def get_articles(self):
        """Get list of indexable blocks contained in all documents in the root directory."""
        globstr = os.path.join(self.rootdir, '*.csv')
        files = glob.glob(globstr, recursive=True)
        articles = []
        for file in files:
            df = pd.read_csv(file)
            for _, row in tqdm(df.iterrows(), total=len(df)):
                articles.append(
                    Article(id=row['id'],
                            url=row['url'],
                            title=row['title'],
                            summary=row['summary'],
                            text=row['text'],
                            language=row['language']))

        return articles
