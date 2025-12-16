import asyncio
import os
from datetime import datetime
from playwright.async_api import async_playwright, Page, Download

# -----------------------
# CONFIG
# -----------------------
DRY_RUN = False  # True to skip downloads
SCROLL_WAIT = 0.2
SCROLL_STEP = 300

# -----------------------
# Logging helper
# -----------------------
def log(level, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")

# -----------------------
# JS patch to disable page refresh
# -----------------------
JS_DISABLE_REFRESH = """
window.onbeforeunload = null;
window.onunload = null;
history.pushState = ()=>{};
history.replaceState = ()=>{};
window.location.assign = ()=>{};
window.location.replace = ()=>{};
Object.defineProperty(window, 'location', { value: window.location });

const originalSetTimeout = window.setTimeout;
window.setTimeout = function(fn, t) {
    const text = fn.toString();
    if (text.includes('location') || text.includes('refresh')) {
        console.log("BLOCKED timeout refresh:", text);
        return 0;
    }
    return originalSetTimeout(fn, t);
};
"""

# -----------------------
# Wait for EXTJS grid rows to appear
# -----------------------
async def wait_extjs_grid(page: Page, retries=60, delay=0.5):
    log("DEBUG", "Waiting for EXTJS grid rows…")
    for attempt in range(1, retries + 1):
        try:
            rows = await page.query_selector_all("div.x-grid3-row")
        except:
            rows = []

        if rows:
            log("DEBUG", f"EXTJS grid detected ({len(rows)} rows) after {attempt} attempts")
            return len(rows)

        await asyncio.sleep(delay)

    log("ERROR", "Timed out waiting for EXTJS grid!")
    return 0

# -----------------------
# Scroll EXTJS grid properly (visual scrolling)
# -----------------------
async def scroll_extjs_grid(page: Page, grid_el, scroll_step=SCROLL_STEP, wait_time=SCROLL_WAIT):
    last_scroll = -1
    scroll_attempts = 0
    MAX_SCROLL_ATTEMPTS = 1000

    while scroll_attempts < MAX_SCROLL_ATTEMPTS:
        scroll_attempts += 1
        scroll_height, scroll_top, client_height = await page.evaluate(
            """(grid) => {
                const body = grid.querySelector('div.x-grid3-body');
                if (!body) return [0,0,0];
                return [body.scrollHeight, body.scrollTop, body.clientHeight];
            }""",
            grid_el
        )

        if scroll_top >= scroll_height - client_height:
            break

        await page.evaluate(
            """(grid, step) => {
                const body = grid.querySelector('div.x-grid3-body');
                if (body) body.scrollTop += step;
            }""",
            grid_el,
            scroll_step
        )

        await asyncio.sleep(wait_time)

        new_scroll_top = await page.evaluate("(grid) => grid.querySelector('div.x-grid3-body').scrollTop", grid_el)
        if new_scroll_top == last_scroll:
            break
        last_scroll = new_scroll_top

# -----------------------
# DOM row collector with global mouse-wheel scrolling
# -----------------------
async def get_rows_via_dom(page: Page, expected_count: int, on_row_detected=None):
    log("DEBUG", "Collecting rows from EXTJS grid via DOM (with global scroll)…")

    collected = {}
    scroll_attempt = 0
    MAX_SCROLL_ATTEMPTS = 1000
    SCROLL_STEP = 200
    SCROLL_WAIT = 0.2

    while True:
        row_elements = await page.query_selector_all("div.x-grid3-row")
        log("DEBUG", f"Rows in DOM this pass: {len(row_elements)}")

        new_rows = 0
        for row in row_elements:
            try:
                cell = await row.query_selector("div.x-grid3-cell-inner")
                if not cell:
                    continue
                text = (await cell.inner_text()).replace("\u00a0", " ").strip()
                if not text or text in collected:
                    continue

                img = await row.query_selector("img")
                src = (await img.get_attribute("src") or "").lower() if img else ""
                ftype = "folder" if "folder" in src else "file"

                collected[text] = ftype
                new_rows += 1
                log("DEBUG", f"Row detected: {text} ({ftype})")

                # Immediately download or handle row if callback provided
                if on_row_detected:
                    await on_row_detected(text, ftype, row)

            except Exception as e:
                log("WARN", f"Failed parsing row: {e}")

        if len(collected) >= expected_count:
            log("DEBUG", f"All expected rows ({expected_count}) collected")
            break

        if scroll_attempt >= MAX_SCROLL_ATTEMPTS:
            log("WARN", f"Reached max scroll attempts ({MAX_SCROLL_ATTEMPTS}), stopping")
            break

        scroll_attempt += 1
        await page.mouse.wheel(0, SCROLL_STEP)
        await asyncio.sleep(SCROLL_WAIT)

        if new_rows == 0:
            log("WARN", "No new rows rendered, assuming end of grid reached")
            break

    log("DEBUG", f"DOM collection complete. Total rows collected: {len(collected)}")
    if len(collected) < expected_count:
        log("WARN", f"Collected fewer rows ({len(collected)}) than expected ({expected_count})")

    return list(collected.items())

# -----------------------
# Disable refresh
# -----------------------
async def disable_refresh(page: Page):
    log("INFO", "Injecting anti-refresh patch")
    await page.add_init_script(JS_DISABLE_REFRESH)
    try:
        await page.evaluate(JS_DISABLE_REFRESH)
    except:
        pass

# -----------------------
# Folder processing
# -----------------------
async def process_folder(page: Page, folder_path: str, root_dir: str):
    log("INFO", f"Processing folder: {folder_path}")
    local_dir = os.path.join(root_dir, folder_path.strip("/"))
    os.makedirs(local_dir, exist_ok=True)

    total_rows = await wait_extjs_grid(page)
    if total_rows == 0:
        log("WARN", "No rows detected, skipping folder")
        return

    # -----------------------
    # Row download callback
    # -----------------------
    async def download_row(filename, ftype, row_el):
        if ftype != "file":
            return

        download_path = os.path.join(local_dir, filename)
        log("INFO", f"Downloading: {filename}")
        if DRY_RUN:
            log("INFO", f"DRY-RUN — skipping download of {filename}")
            return

        try:
            async with page.expect_download() as handle:
                await row_el.click(button="right")
                menu = await page.wait_for_selector(
                    "//span[contains(@class,'x-menu-item-text') and normalize-space(text())='Download']",
                    timeout=5000
                )
                await menu.click()

            download: Download = await handle.value
            await download.save_as(download_path)
            log("INFO", f"Saved: {download_path}")

        except Exception as e:
            log("ERROR", f"Failed to download {filename}: {e}")

        await asyncio.sleep(0.1)

    # -----------------------
    # Collect rows and download immediately
    # -----------------------
    rows = await get_rows_via_dom(page, expected_count=total_rows, on_row_detected=download_row)

    # Separate files/folders for recursion
    files = [n for n, t in rows if t == "file"]
    folders = [n for n, t in rows if t == "folder"]
    log("INFO", f"Found {len(files)} files, {len(folders)} folders")

    # -----------------------
    # Recurse folders
    # -----------------------
    for folder in folders:
        el = await page.query_selector(
            f"div.x-grid3-row:has-text('{folder}') "
            f"td.x-grid3-td-filename div.x-grid3-cell-inner"
        )
        if not el:
            log("WARN", f"Folder row not found: {folder}")
            continue

        log("INFO", f"Entering subfolder: {folder}")
        await el.dblclick()
        await asyncio.sleep(1.5)

        await process_folder(page, f"{folder_path}/{folder}", root_dir)

        back_btn = await page.query_selector("button[aria-label='Back']")
        if back_btn:
            await back_btn.click()
            await asyncio.sleep(1.5)

# -----------------------
# Main
# -----------------------
async def main(URL: str, OUTDIR: str):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False,
            args=["--ignore-certificate-errors"]
        )
        context = await browser.new_context(ignore_https_errors=True)
        page = await context.new_page()

        log("INFO", f"Navigating to: {URL}")
        await page.goto(URL, wait_until="load")

        await disable_refresh(page)
        await asyncio.sleep(1)

        await process_folder(page, "/", OUTDIR)

        log("INFO", "ALL DONE")

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    asyncio.run(main("https://eostore.itc.utwente.nl:5001/fsdownload/1gJRLdQ71/Dataset-1", "Dataset_1"))
    asyncio.run(main("https://eostore.itc.utwente.nl:5001/fsdownload/c4LlTkVjT/Dataset-2", "Dataset_2"))
    asyncio.run(main("https://eostore.itc.utwente.nl:5001/fsdownload/r4o1tdCNv/Dataset-3", "Dataset_3"))