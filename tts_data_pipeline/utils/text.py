import httpx
import string

from tts_data_pipeline import constants


def remove_punctuations(sentence: str):
    translator = str.maketrans("", "", string.punctuation)
    return sentence.translate(translator)


async def check_exists(name: str) -> bool:
    """Check if a book exists in the text source."""
    url = constants.TEXT_BASE_URL + name

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.status_code == 200
