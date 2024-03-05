#!/bin/bash
# Start the lorax server
nohup lorax-launcher --model-id "TheBloke/Llama-2-13B-chat-AWQ" --quantize awq --max-input-length=4096 --max-total-tokens=5096 --huggingface-hub-cache=/data --hostname=127.0.0.1 --port=8000 &

#sleep 60
# Start the handler using python 3.10
python3.10 -u /handler.py