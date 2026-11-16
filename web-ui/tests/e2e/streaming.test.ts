import { expect, test, type Page } from "@playwright/test";

const ASSISTANT = "[data-testid='message-assistant']";

async function gotoStableChat(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("multimodal-input")).toBeEnabled({
    timeout: 30_000,
  });
  // Guest auth can redirect shortly after first render; let it settle so a
  // filled value is not discarded by a remount.
  await page.waitForTimeout(1500);
}

test.describe("Stage 3 token streaming", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => window.localStorage.clear());
  });

  test("streams a direct answer and persists it after reload", async ({
    page,
  }) => {
    await gotoStableChat(page);

    const input = page.getByTestId("multimodal-input");
    await input.fill("Say hello in exactly three short words.");
    await page.getByTestId("send-button").click();

    const assistant = page.locator(ASSISTANT).last();
    await expect(assistant).toBeVisible({ timeout: 90_000 });
    await expect(page.getByTestId("stop-button")).toBeHidden({
      timeout: 90_000,
    });

    const text = await assistant.innerText();
    expect(text.trim().length).toBeGreaterThan(0);

    await page.waitForTimeout(500);
    await page.reload();
    await expect(page.locator(ASSISTANT).last()).toBeVisible({
      timeout: 30_000,
    });
    expect(
      (await page.locator(ASSISTANT).last().innerText()).trim().length
    ).toBeGreaterThan(0);
  });

  test("stop cancels generation and saves no incomplete assistant", async ({
    page,
  }) => {
    test.setTimeout(180_000);
    await gotoStableChat(page);

    const input = page.getByTestId("multimodal-input");
    const prompt =
      "Write a very long, detailed essay about the ocean. At least 500 words.";
    await input.fill(prompt);
    await expect(input).toHaveValue(prompt);
    await page.getByTestId("send-button").click();

    const stop = page.getByTestId("stop-button");
    await expect(stop).toBeVisible({ timeout: 30_000 });

    const assistant = page.locator(ASSISTANT).last();
    await expect(assistant).toBeVisible({ timeout: 30_000 });
    await expect
      .poll(async () => (await assistant.innerText()).trim().length, {
        timeout: 90_000,
      })
      .toBeGreaterThan(0);

    const clicked = await page.evaluate(() => {
      const button = document.querySelector<HTMLElement>(
        '[data-testid="stop-button"]'
      );
      if (!button) {
        return false;
      }
      button.click();
      return true;
    });
    expect(clicked).toBe(true);

    await expect(stop).toBeHidden({ timeout: 30_000 });

    await page.waitForTimeout(1500);
    await page.reload();
    await expect(page.getByTestId("multimodal-input")).toBeVisible({
      timeout: 30_000,
    });

    // Only the user message should be persisted; an interrupted answer must
    // never be stored as a completed assistant message.
    await expect(page.locator(ASSISTANT)).toHaveCount(0, { timeout: 30_000 });
  });

  test("streams a RAG answer with sources and persists citations", async ({
    page,
  }) => {
    test.setTimeout(240_000);
    await page.context().addCookies([
      {
        name: "sidebar_state",
        value: "true",
        url: "http://localhost:3000",
      },
    ]);
    await gotoStableChat(page);

    const pdf = page
      .locator("button")
      .filter({ hasText: "Work_Summary.pdf" })
      .first();
    await expect(pdf).toBeVisible({ timeout: 20_000 });
    await pdf.click({ force: true });
    await expect(page.getByText(/Documents \(1\/\d+\)/)).toBeVisible({
      timeout: 15_000,
    });
    await page.waitForTimeout(500);

    const input = page.getByTestId("multimodal-input");
    await input.fill("What is this document about?");
    await page.getByTestId("send-button").click();

    const assistant = page.locator(ASSISTANT).last();
    await expect(assistant).toBeVisible({ timeout: 90_000 });
    await expect(page.getByTestId("stop-button")).toBeHidden({
      timeout: 120_000,
    });

    await expect(assistant).toContainText("Work_Summary.pdf", {
      timeout: 30_000,
    });

    await page.waitForTimeout(1000);
    await page.reload();
    await expect(page.locator(ASSISTANT).last()).toContainText(
      "Work_Summary.pdf",
      { timeout: 30_000 }
    );
  });
});
