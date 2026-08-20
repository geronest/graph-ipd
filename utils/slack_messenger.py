import ssl

import certifi
from omegaconf import OmegaConf
from slack_sdk import WebClient

# from slack_sdk.errors import SlackApiError


class SlackMessenger:
    def __init__(self, name_config):
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        self.config = OmegaConf.load(f"comms/{name_config}.yaml")
        self.client = WebClient(token=self.config.token, ssl=ssl_context)
        self.channel = self.config.channel
        print(f"slack messenger established: {name_config}")

    def post_message(self, msg):
        """
        post a message to predefined channel in the cfg.
        """
        try:
            self.client.chat_postMessage(channel="#" + self.channel, text=msg)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    slmsg = SlackMessenger("seongho")
    slmsg.post_message("testing message")
