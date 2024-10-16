from openai import OpenAI
import configparser
def embeddings(words):
    # gets API Key from environment variable OPENAI_API_KEY
    config = configparser.ConfigParser()
    config.read('../config/config.ini')
    key = config.get('gpt4', 'embedding_key')
    model = config.get('gpt4', 'embedding_model')
    base_url = config.get('gpt4', 'embedding_url')

    client = OpenAI(
        api_key=key,
        base_url=base_url
    )

    resp = client.embeddings.create(
        model=model,
        input=words,
        encoding_format="float"
    )
    res = [item.embedding for item in resp.data]
    return res

