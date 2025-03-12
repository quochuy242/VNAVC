# Crawler saving path
AUDIO_SAVE_PATH = "./data/audio/"
ALL_AUDIOBOOK_URLS_SAVE_PATH = "./data/all_audiobook_urls.txt"
TEXT_BOOK_URLS_SAVE_PATH = "./data/text_book_urls.txt"
METADATA_SAVE_PATH = "./data/metadata/"
TEXT_SAVE_PATH = "./data/text/"

# Base url
AUDIO_CATEGORY_URL = "https://sachnoiviet.net/danh-muc-sach/"
# TEXT_BASE_URL = "https://tiemsach.org/"
TEXT_DOWNLOAD_URL = lambda x: f"https://cloud.thuviensachpdf.com/pdf/vi/{x}.pdf"

# Crawler config
USER_AGENTS = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
FETCH_METADATA_LIMIT = 10
DOWNLOAD_BOOK_LIMIT = 5

# Pre-processing config
MIN_SAMPLE_RATE = 24000

# Pre-processing saving path
SENTENCE_DIR = "./data/text/sentences/"
PDF_DIR = "./data/text/pdf/"
RAW_DIR = "./data/audio/raw/"
QUALIFIED_DIR = "./data/audio/qualified/"
UNQUALIFIED_DIR = "./data/audio/unqualified/"
