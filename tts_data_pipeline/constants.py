LOG_DIR = "./logs/"
FORMAT_LOG = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# Crawler path
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
MIN_SAMPLE_RATE = 24000  # Hz, always larger than 16000

# Pre-processing path
MIN_WORD_THRESHOLD = 20
TEXT_SENTENCE_DIR = "./data/text/sentence/"
TEXT_PDF_DIR = "./data/text/pdf/"
TEXT_TXT_DIR = "./data/text/txt/"
AUDIO_RAW_DIR = "./data/audio/raw/"
AUDIO_QUALIFIED_DIR = "./data/audio/qualified/"
AUDIO_UNQUALIFIED_DIR = "./data/audio/unqualified/"
METADATA_BOOK_PATH = "./data/metadata/metadata_book.csv"
METADATA_NARRATOR_PATH = "./data/metadata/metadata_narrator.csv"
TEST_DATA_PATH = "./data/test/"

# Align config
AENEAS_CONFIG = "task_language=vie|is_text_type=plain|os_task_file_format=tsv|is_txt_unparsed_id_regex=[0-9]+|is_text_unparsed_id_sort=numeric"

# Align path
AENEAS_OUTPUT_DIR = "./data/alignment/aeneas_output"
DATASET_DIR = "./dataset/"
