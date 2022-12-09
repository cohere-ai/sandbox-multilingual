# Copyright (c) 2022 Cohere Inc. and its affiliates.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License in the LICENSE file at the top
# level of this repository.
"""Fetch embeddings from cohere and write out to different formats."""

import os
import typing

import click
import jsonlines

from embed.client import Article, Client
from embed.csv.client import Client as LocalCSVClient


@click.group(invoke_without_command=False)
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.option('--root-dir', help='path to search recursively for csv files')
@click.option('--output-file', help='path to output file of embeddings')
@click.option('--api_token', help='Cohere API token', default=None)
@click.option('--model_name', help='Cohere model name', default='multilingual-22-12')
def csv(root_dir,
        output_file,
        api_token: typing.Optional[str] = None,
        model_name: typing.Optional[str] = 'multilingual-22-12'):
    """Retrieve all articles from text documents in a root dir, and embed them."""

    text_client = LocalCSVClient(root_dir)

    # Get all articles from text files in root_dir
    click.secho("Scanning local filesystem for csv...")
    articles = text_client.get_articles()
    click.secho(f"Found {len(articles)} articles")

    if api_token or "COHERE_TOKEN" in os.environ:
        api_token = api_token if api_token else os.environ["COHERE_TOKEN"]
    else:
        raise KeyError("Could not find Cohere API key in kwargs or environment.")

    client = Client(api_token, model_name=model_name)

    # Embed all articles using the Cohere client
    click.secho(f'fetching embeddings from discovered articles', fg='cyan')
    n_embeddings = client.embed_articles(articles)
    click.secho(f'fetched {n_embeddings} block embeddings', fg='cyan')

    if output_file:
        click.secho(f'writing {n_embeddings} embeddings to local storage.', fg='green')
        client.save_embeddings(output_file)
        click.secho(f'Done. Output is in {output_file}.', fg='green')
