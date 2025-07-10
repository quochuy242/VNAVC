LOG_DIR = "./logs/"
FORMAT_LOG = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# Crawler path
AUDIO_SAVE_PATH = "./data/audio/"
VALID_BOOK_URL_SAVE_PATH = "./data/valid_book_urls.txt"
DOWNLOAD_URL_SAVE_PATH = "./data/download_urls.json"
METADATA_SAVE_PATH = "./data/metadata/book/"
TEXT_SAVE_PATH = "./data/text/"

# Base url
AUDIO_CATEGORY_URL = "https://sachnoiviet.net/danh-muc-sach/"
TEXT_BASE_URL = "https://thuviensachpdf.com/"
TEXT_DOWNLOAD_URL = "https://cloud.thuviensachpdf.com/pdf/vi/"
NARRATOR_DOWNLOAD_URL = "https://docs.google.com/spreadsheets/d/1kaZ4PRYq_JvJAwWJN89Gcw_BuWHUogvDA43IZOd6MbE/export?format=csv&gid=0"

# Crawler config
USER_AGENTS = [
  # Chrome trên Windows
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
  # Chrome trên macOS
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
  # Chrome trên Ubuntu Linux
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.208 Safari/537.36",
  # Firefox trên Windows
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
  # Firefox trên macOS
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 13.4; rv:126.0) Gecko/20100101 Firefox/126.0",
  # Safari trên macOS
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
  # Edge trên Windows
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.208 Safari/537.36 Edg/124.0.2478.97",
  # Android Chrome
  "Mozilla/5.0 (Linux; Android 13; Pixel 7 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.208 Mobile Safari/537.36",
  # iPhone Safari
  "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1",
]

FETCH_METADATA_LIMIT = 8
DOWNLOAD_BOOK_LIMIT = 10
FETCH_URL_LIMIT = 30
CHECK_URL_LIMIT = 30

# Pre-processing config
MIN_SAMPLE_RATE = 24000  # Hz, always larger than 16000

# Pre-processing path
MIN_WORD_THRESHOLD = 20
TEXT_PROCESSED_DIR = "./data/text/processed/"
TEXT_PDF_DIR = "./data/text/pdf/"
TEXT_TXT_DIR = "./data/text/txt/"
AUDIO_RAW_DIR = "./data/audio/raw/"
AUDIO_PROCESSED_DIR = "./data/audio/processed/"
AUDIO_UNQUALIFIED_DIR = "./data/audio/unqualified/"
METADATA_BOOK_PATH = "./data/metadata/metadata_book.csv"
METADATA_NARRATOR_PATH = "./data/metadata/metadata_narrator.csv"
TEST_DATA_PATH = "./data/test/"
AFTER_DOWNLOADING_METADATA_PATH = "./data/metadata/after_download_metadata.csv"

# Align config
AENEAS_OUTPUT_EXT = "tsv"
AENEAS_CONFIG = f"task_language=vie|is_text_type=plain|os_task_file_format={AENEAS_OUTPUT_EXT}|is_txt_unparsed_id_regex=[0-9]+|is_text_unparsed_id_sort=numeric"

# Align path
AENEAS_OUTPUT_DIR = "./data/alignment/"
DATASET_DIR = "./dataset/"
STANDARD_SAMPLE_RATE = None
