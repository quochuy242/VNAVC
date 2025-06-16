import asyncio
import pandas as pd
from tts_data_pipeline.crawler.utils import (
  get_text_download_url,
  double_check_url,
  get_all_book_url,
)
from rich.console import Console

console = Console()


async def test_get_text_download_url():
  await get_text_download_url(name="nha-lanh-dao-khong-chuc-danh", source="taisachhay")


async def test_double_check_url():
  source = await double_check_url(
    url="https://sachnoiviet.net/sach-noi/co-phieu-thuong-loi-nhuan-phi-thuong",
    console=console,
    semaphore=asyncio.Semaphore(1),
  )
  console.print(source)


async def test_get_all_book_urls():
  urls: pd.DataFrame = await get_all_book_url()
  print(urls.head())


async def main():
  # await test_get_text_download_url()
  await test_double_check_url()
  # await test_get_all_book_urls()


if __name__ == "__main__":
  asyncio.run(main())
