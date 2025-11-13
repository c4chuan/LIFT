from playwright.sync_api import sync_playwright
import requests

# requests.request(url="http://127.0.0.1:9980",method="GET")

with sync_playwright() as pw:
    success = 0
    sum_times = 0
    while True:
        browser = pw.chromium.launch(
            # proxy={
            #     "server":"http://81.70.105.191:7891",
            #     "username":"wangzhenchuan",
            #     "password":"144987"
            # },
            headless=True,
            # args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context()
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        page = context.new_page()
        # point at your host’s reachable address
        url = "http://192.168.1.6:9980/index.php?page=login"

        # page.on("requestfailed", lambda req: print("❌ FAILED", req.url, req.failure))
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        try:
            page.goto(url,
                      timeout=60000,              # extend timeout
                      wait_until="domcontentloaded")
            print("✅ Loaded!")
            # context.tracing.stop(path="normal_trace.zip")
            browser.close()
            # 查看page的内容
            success += 1
        except PlaywrightTimeoutError:
            print("❌ Timeout")
            context.tracing.stop(path="trace.zip")
            browser.close()
        sum_times += 1
        if sum_times >= 1000:
            break

print(f"成功率为{success/sum_times}")