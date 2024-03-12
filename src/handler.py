import runpod
from lorax import AsyncClient, Client
from loguru import logger
import time
from typing import Generator
import os

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "true"
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = "true"

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
        print(str(e))
        print("The Lorax server is still cold booting...")
        time.sleep(5)


async def handler(job: dict):
    '''
    This is the handler function that will be called by the serverless.
    '''
    logger.info("Starting job...")
    job_input = job['input']

    prompt = job_input.get('inputs', {})
    
    # Sampling params and adapter_id
    params = job_input.get('parameters', {})


    resp = await client_async.generate(prompt, **params)

    return resp.generated_text


runpod.serverless.start({"handler": handler, "concurrency_modifier": lambda x: 128})