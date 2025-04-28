import aiohttp
import subprocess
import asyncio
from tts_data_pipeline.crawler.utils import logger


async def check_playwright_server():
  try:
    async with aiohttp.ClientSession() as session:
      async with session.get("http://localhost:3000") as resp:
        if resp.status == 200 or resp.status == 404:
          return True
  except aiohttp.ClientConnectionError:
    return False
  return False


async def start_playwright_server():
  # You may want to customize the command if needed
  cmd = [
    "docker",
    "run",
    "-p",
    "3000:3000",
    "--rm",
    "--init",
    "-d",
    "--workdir",
    "/home/pwuser",
    "--user",
    "pwuser",
    "mcr.microsoft.com/playwright:v1.51.1-noble",
    "/bin/sh",
    "-c",
    "npx -y playwright@1.51.0 run-server --port 3000 --host 0.0.0.0",
  ]
  subprocess.run(cmd)
  await asyncio.sleep(2)  # Wait for the server to start


async def ensure_playwright_server_running():
  if not await check_playwright_server():
    logger.info("Playwright server not running. Starting new container...")
    await start_playwright_server()
    if not await check_playwright_server():
      logger.error("Failed to start Playwright server.")
  else:
    logger.success("Playwright server already running.")
