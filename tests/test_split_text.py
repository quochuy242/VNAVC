from tts_data_pipeline import Book
from tts_data_pipeline.alignment.utils import split_text


def main():
  book = Book.from_json("./data/metadata/book/mat-biec.json")
  print(book.narrator.id)
  book.update_paths(
    text_path="./data/text/sentence/mat-biec.txt",
    alignment_path="./data/alignment/mat-biec/output.tsv",
  )
  split_text(book, max_workers=4)


if __name__ == "__main__":
  main()
