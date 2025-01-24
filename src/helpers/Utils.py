from pathlib import Path

import requests


def download_raw_file(source: str, destination_dir: str | Path, filename: str) -> bool:
    # create the directory if it does not exist, and add the directory to the filename
    if destination_dir is not None:
        Path(destination_dir).mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(
            url=source,
        )
        if r.status_code != 200: return False

        with open("notebook_utils.py", "w") as f: f.write(r.text)
        r.close()
        return True
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        raise Exception(error)
    except requests.exceptions.Timeout:
        raise Exception("Connection timed out")
    except requests.exceptions.RequestException as error:
        raise Exception(f"File downloading failed with error: {error}")


class Utils:

    def __init__(self):
        pass
