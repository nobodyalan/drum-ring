import uuid
import requests
import json
from loguru import logger

from core.asr.config import load_doubao_asr_config


class DoubaoASRFile:
    def __init__(self):
        asr_config = load_doubao_asr_config()
        self.submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
        self.appid = asr_config.app_key
        self.token = asr_config.access_key
        self.resource_id = asr_config.file_resource_id

    def submit(self, file_url: str):
        logger.info(f"file_url: {file_url}")
        submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"

        task_id = str(uuid.uuid4())

        headers = {
            "X-Api-App-Key": self.appid,
            "X-Api-Access-Key": self.token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": task_id,
            "X-Api-Sequence": "-1"
        }

        request = {
            "user": {
                "uid": "fake_uid"
            },
            "audio": {
                "url": file_url,
                # "format": "mp3",
                # "rate": 48000,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_channel_split": True, 
                "enable_ddc": True, 
                "enable_punc": True, 
                "enable_itn": True,
                # "show_utterances": True,
                "corpus": {
                    "correct_table_name": "",
                    "context": ""
                }
            }
        }
        response = requests.post(submit_url, data=json.dumps(request), headers=headers)
        if 'X-Api-Status-Code' in response.headers and response.headers["X-Api-Status-Code"] == "20000000":
            return task_id
        else:
            raise Exception(f"[Doubao asr file] Submit task failed.")

    def query(self, task_id: str):
        query_url = "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/query"

        headers = {
            "X-Api-App-Key": self.appid,
            "X-Api-Access-Key": self.token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": task_id,
        }

        response = requests.post(query_url, json.dumps({}), headers=headers)

        if 'X-Api-Status-Code' not in response.headers:
            raise Exception(f"[Doubao asr file] Query task failed.")
        return response
