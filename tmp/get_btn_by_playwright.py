from playwright.sync_api import sync_playwright

url = "https://sachnoiviet.net/sach-noi/cac-cuoc-chien-tranh-tien-te"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url)

    mp3_links = page.locator("a.ai-track-btn").evaluate_all(
        "elements => elements.map(el => el.href)"
    )

    print("Danh sách link MP3:")
    for link in mp3_links:
        print(link)

    browser.close()
