from dotenv import dotenv_values


def get_config() -> dict:
    config = dotenv_values("./.env")
    return config


def get_user_config() -> dict:
    config = dotenv_values("./.secret")
    return config
