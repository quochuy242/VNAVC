import aiohttp
import asyncio
import subprocess
from tts_data_pipeline.crawler.utils import logger


async def check_playwright_server():
  try:
    async with aiohttp.ClientSession() as session:
      async with session.get("http://localhost:3000") as resp:
        if resp.status in (200, 404):
          return True
  except aiohttp.ClientConnectionError:
    return False
  except aiohttp.ClientError as e:
    logger.warning(f"Unexpected client error: {e}")
  return False


async def wait_for_server(timeout=15, interval=1):
  """Waits until the server is reachable or timeout occurs."""
  for _ in range(0, timeout, interval):
    if await check_playwright_server():
      return True
    await asyncio.sleep(interval)
  return False


async def check_container_running(name: str) -> bool:
  """Check if a Docker container with the given name is running."""
  try:
    result = subprocess.run(
      ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      text=True,
    )
    return name in result.stdout.strip().splitlines()
  except Exception as e:
    print(f"Error checking container: {e}")
    return False


async def start_playwright_server():
  cmd = [
    "docker",
    "run",
    "--name",
    "playwright-server",
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

  try:
    await asyncio.create_subprocess_exec(*cmd)
  except subprocess.CalledProcessError as e:
    logger.error(f"Failed to start Playwright container: {e}")
    return False

  logger.info("Container started. Waiting for server to become available...")
  return await wait_for_server()


async def ensure_playwright_server_running():
  if await check_playwright_server() and await check_container_running(
    "playwright-server"
  ):
    return

  logger.info("Playwright server not running. Starting new container...")
  if not await start_playwright_server():
    logger.error("Failed to start Playwright server after waiting.")

