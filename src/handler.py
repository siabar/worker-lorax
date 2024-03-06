import runpod
from lorax import AsyncClient, Client
from loguru import logger
import time
from typing import Generator
import os

os.environ['HF_HUB_OFFLINE'] = 1


client_async = AsyncClient("http://127.0.0.1:8000")
client = Client("http://127.0.0.1:8000")

# Wait for the LoRAX worker to start running.
while True:
    try:
        x = client.generate("This is just for loading test", max_new_tokens=1).generated_text
        print("Successfully cold booted the LoRAX server!")
        # Break from the while loop
        break

    except Exception as e:
        print("The Lorax server is still cold booting...")
        time.sleep(5)


async def handler(job: dict):
    '''
    This is the handler function that will be called by the serverless.
    '''
    logger.info("Starting job...")
    job_input = job['input']

    prompt = job_input.get('inputs')
    parameters = job_input.get('parameters')

    max_new_tokens = parameters.get('max_new_tokens')
    temperature = parameters.get('temperature')
    top_p = parameters.get('top_p')
    do_sample = parameters.get('do_sample')
    seed = parameters.get('seed')


    resp = await client_async.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                           seed=seed,
                           do_sample=do_sample)

    return resp.generated_text


runpod.serverless.start({"handler": handler, "concurrency_modifier": lambda x: 128})