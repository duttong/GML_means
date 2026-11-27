// Render poster.html to a print-ready 24 x 36 in PDF and a 2x PNG.
// Needs Node with Playwright (npm install playwright) and internet access for Google Fonts.
//   node docs/poster/make_pdf.js
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const dir = __dirname;
  const out = path.join(dir, 'The_Hole_Story_poster_24x36');
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 2304, height: 3456 }, deviceScaleFactor: 2 });
  await page.goto('file://' + path.join(dir, 'poster.html'));
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(800);
  await page.screenshot({ path: out + '.png' });
  await page.pdf({ path: out + '.pdf', width: '24in', height: '36in', printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 }, pageRanges: '1' });
  await browser.close();
  console.log('Wrote ' + out + '.pdf and .png');
})();
