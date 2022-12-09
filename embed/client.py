# Copyright (c) 2022 Cohere Inc. and its affiliates.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License in the LICENSE file at the top
# level of this repository.

import dataclasses
import time
from typing import List

import cohere
import numpy as np
import requests

from tqdm import tqdm

COHERE_BATCH_SIZE = 1024
COHERE_N_RETRIES = 5


@dataclasses.dataclass
class Article():
    id: str
    url: str
    title: str
    summary: str
    text: str
    language: str


class Client:
    """This class wraps around a Cohere client to facilitate embedding blocks of text."""

    def __init__(self, api_token, model_name: str = "multilingual-22-12") -> None:

        self._co = cohere.Client(api_token)
        self._model_name = model_name
        self._embeddings = []
        self._embed_texts = []
        self._article_titles = []
        self._article_texts = []
        self._article_urls = []
        self._article_summaries = []
        self._article_languages = []

    def embed_articles(self, articles: List[Article]) -> int:
        """Given a list of articles, embed each article using the Cohere client."""

        for article in articles:
            self._embed_texts.append(f"Title: {article.title}\nText:{article.text}")
            self._article_titles.append(article.title)
            self._article_texts.append(article.text)
            self._article_urls.append(article.url)
            self._article_summaries.append(article.summary)
            self._article_languages.append(article.language)

        embs = []
        for i in tqdm(range(0, len(self._embed_texts), COHERE_BATCH_SIZE)):
            for _ in range(COHERE_N_RETRIES):
                try:
                    x = self._co.embed(self._embed_texts[i:i + COHERE_BATCH_SIZE],
                                       model=self._model_name,
                                       truncate='RIGHT').embeddings
                    embs.extend(x)
                    break
                except requests.exceptions.ConnectionError:
                    print('Connection dropped... retrying...')
                    time.sleep(1)
                except cohere.error.CohereError:
                    # This is most likely going to happen when people get rate limited due to using a trial key.
                    # Waiting and retrying should solve the problem.
                    time.sleep(60)
            else:
                raise RuntimeError(
                    'Hit maximum number of retries connecting to the Cohere API: is there a problem with your network?')

        self._embeddings = np.array(embs)
        self._article_titles = np.array(self._article_titles)
        self._article_texts = np.array(self._article_texts)
        self._article_urls = np.array(self._article_urls)
        self._article_summaries = np.array(self._article_summaries)
        self._article_languages = np.array(self._article_languages)

        return len(self._embeddings)

    def save_embeddings(self, output_file):
        """Save embeddings as an npz file."""
        np.savez(
            output_file,
            embeddings=self._embeddings,
            article_title=self._article_titles,
            article_text=self._article_texts,
            article_url=self._article_urls,
            article_summary=self._article_summaries,
            article_language=self._article_languages,
        )
