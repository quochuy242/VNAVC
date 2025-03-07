import asyncio
from tts_data_pipeline import crawler


async def main():
    await crawler.main()


if __name__ == "__main__":
    asyncio.run(main())
