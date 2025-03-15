# Crawler saving path
AUDIO_SAVE_PATH = "./data/audio/"
ALL_AUDIOBOOK_URLS_SAVE_PATH = "./data/all_audiobook_urls.txt"
TEXT_BOOK_URLS_SAVE_PATH = "./data/text_book_urls.txt"
METADATA_SAVE_PATH = "./data/metadata/book/"
TEXT_SAVE_PATH = "./data/text/"

# Base url
AUDIO_CATEGORY_URL = "https://sachnoiviet.net/danh-muc-sach/"
TEXT_BASE_URL = "https://thuviensachpdf.com/"
TEXT_DOWNLOAD_URL = "https://cloud.thuviensachpdf.com/pdf/vi/"

# Crawler config
USER_AGENTS = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
FETCH_METADATA_LIMIT = 20
DOWNLOAD_BOOK_LIMIT = 10

# Pre-processing config
MIN_SAMPLE_RATE = 24000

# Pre-processing saving path
TEXT_SENTENCE_DIR = "./data/text/sentences/"
TEXT_PDF_DIR = "./data/text/pdf/"
TEXT_TXT_DIR = "./data/text/txt/"
AUDIO_RAW_DIR = "./data/audio/raw/"
AUDIO_QUALIFIED_DIR = "./data/audio/qualified/"
AUDIO_UNQUALIFIED_DIR = "./data/audio/unqualified/"
METADATA_BOOK_PATH = "./data/metadata/metadata_book.csv"
METADATA_NARRATOR_PATH = "./data/metadata/metadata_narrator.csv"
PROCESSED_METADATA_PATH = "./data/metadata/processed/"
