import requests
from bs4 import BeautifulSoup
import csv
import os
import time
from urllib.parse import quote_plus, urljoin
from datetime import datetime

class MarketplaceReviews:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.rate_delay = 3
        self.src_folder = 'GSMArenaDataset'
        self.out_folder = 'ReviewsDataset'
        self.features = [
            "Brand",
            "Model Name",
            "Amazon Rating",
            "Amazon Ratings Count",
            "Amazon URL",
            "Flipkart Rating",
            "Flipkart Ratings Count",
            "Flipkart URL",
            "Retrieved At",
        ]
        self._ensure_folder(self.out_folder)

    def _ensure_folder(self, folder):
        abs_path = os.path.join(os.getcwd(), folder)
        os.makedirs(abs_path, exist_ok=True)
        return abs_path

    def _sleep(self):
        time.sleep(self.rate_delay)

    def _get(self, url, timeout=10):
        try:
            self._sleep()
            resp = requests.get(url, headers=self.headers, timeout=timeout)
            if resp.status_code != 200:
                return None
            return resp
        except Exception:
            return None

    # ---------------- Amazon ----------------
    def _amazon_search_url(self, query):
        return f"https://www.amazon.in/s?k={quote_plus(query)}"

    def _parse_amazon_product_link(self, html, base="https://www.amazon.in"):
        soup = BeautifulSoup(html, 'html.parser')
        # Prefer h2 title anchors
        for a in soup.select('h2 a.a-link-normal'):
            href = a.get('href')
            if href and '/dp/' in href or '/gp/' in href:
                return urljoin(base, href)
        # Fallback: any product link with data-asin
        for card in soup.select('div[data-asin] h2 a'):
            href = card.get('href')
            if href:
                return urljoin(base, href)
        return None

    def _parse_amazon_product_page(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        rating = None
        count = None
        # Rating: span#acrPopover has title like "4.3 out of 5 stars"
        acr = soup.select_one('#acrPopover')
        if acr:
            title = acr.get('title') or acr.get('aria-label')
            if title and 'out of 5' in title:
                try:
                    rating = float(title.split(' out of ')[0].strip())
                except Exception:
                    pass
        if rating is None:
            alt = soup.select_one('i.a-icon-star span.a-icon-alt')
            if alt and 'out of 5' in alt.text:
                try:
                    rating = float(alt.text.split(' out of ')[0].strip())
                except Exception:
                    pass
        # Count: span#acrCustomerReviewText like "12,345 ratings"
        cnt = soup.select_one('#acrCustomerReviewText')
        if cnt and cnt.text:
            try:
                count = int(cnt.text.split()[0].replace(',', ''))
            except Exception:
                pass
        if count is None:
            # Sometimes inside #averageCustomerReviews
            cnt2 = soup.select_one('#averageCustomerReviews #acrCustomerReviewText')
            if cnt2 and cnt2.text:
                try:
                    count = int(cnt2.text.split()[0].replace(',', ''))
                except Exception:
                    pass
        return rating, count

    def fetch_amazon(self, model_query):
        search_url = self._amazon_search_url(model_query)
        resp = self._get(search_url)
        if not resp:
            return None, None, None
        product_url = self._parse_amazon_product_link(resp.text)
        if not product_url:
            return None, None, search_url
        prod = self._get(product_url)
        if not prod:
            return None, None, product_url
        rating, count = self._parse_amazon_product_page(prod.text)
        return rating, count, product_url

    # ---------------- Flipkart ----------------
    def _flipkart_search_url(self, query):
        return f"https://www.flipkart.com/search?q={quote_plus(query)}"

    def _parse_flipkart_product_link(self, html, base="https://www.flipkart.com"):
        soup = BeautifulSoup(html, 'html.parser')
        # New grid card for mobiles
        for a in soup.select('a._1fQZEK, a._2rpwqI, a.s1Q9rs, a._2UzuFa'):
            href = a.get('href')
            if href and href.startswith('/'):
                return urljoin(base, href)
        return None

    def _parse_flipkart_product_page(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        rating = None
        count = None
        # Rating typically in div._3LWZlK near title, sometimes aria-label
        r = soup.select_one('div._3LWZlK')
        if r and r.text:
            try:
                rating = float(r.text.strip())
            except Exception:
                pass
        if rating is None:
            r2 = soup.select_one('[itemprop="ratingValue"]')
            if r2 and r2.get('content'):
                try:
                    rating = float(r2.get('content'))
                except Exception:
                    pass
        # Count in span._2_R_DZ like "12,345 Ratings & 1,234 Reviews"
        c = soup.select_one('span._2_R_DZ')
        if c and c.text:
            txt = c.text.replace(',', '')
            try:
                # take first number as ratings count
                n = ''.join(ch for ch in txt if ch.isdigit() or ch.isspace())
                first = n.strip().split()
                if first:
                    count = int(first[0])
            except Exception:
                pass
        return rating, count

    def fetch_flipkart(self, model_query):
        search_url = self._flipkart_search_url(model_query)
        resp = self._get(search_url)
        if not resp:
            return None, None, None
        product_url = self._parse_flipkart_product_link(resp.text)
        if not product_url:
            return None, None, search_url
        prod = self._get(product_url)
        if not prod:
            return None, None, product_url
        rating, count = self._parse_flipkart_product_page(prod.text)
        return rating, count, product_url

    # ---------------- Orchestration ----------------
    def _list_brand_files(self):
        folder = os.path.join(os.getcwd(), self.src_folder)
        if not os.path.isdir(folder):
            return []
        return [f for f in os.listdir(folder) if f.lower().endswith('.csv')]

    def _iter_models(self, brand_csv_path):
        with open(brand_csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                brand = row.get('Brand', '')
                model = row.get('Model Name', '')
                if brand and model:
                    yield brand.strip(), model.strip()

    def _brand_out_path(self, brand_file):
        brand_name = os.path.splitext(os.path.basename(brand_file))[0]
        out_dir = os.path.join(os.getcwd(), self.out_folder)
        return os.path.join(out_dir, f"{brand_name}.csv")

    def _should_skip_brand(self, brand_file):
        out = self._brand_out_path(brand_file)
        return os.path.exists(out)

    def _write_brand_rows(self, brand_file, rows):
        out_path = self._brand_out_path(brand_file)
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.features, extrasaction='ignore')
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def run(self, brand_filter=None, limit_per_brand=None):
        src_dir = os.path.join(os.getcwd(), self.src_folder)
        brand_files = self._list_brand_files()
        if not brand_files:
            print(f"No brand CSVs found in {src_dir}. Run main.py first.")
            return

        for bf in sorted(brand_files):
            if brand_filter and os.path.splitext(bf)[0].lower() != brand_filter.lower():
                continue

            if self._should_skip_brand(bf):
                print(os.path.splitext(bf)[0] + '.csv already exists in ReviewsDataset. Skipping.')
                continue

            print('Working on', os.path.splitext(bf)[0], 'brand for reviews.')
            src_path = os.path.join(src_dir, bf)
            rows = []
            count = 0
            for brand, model in self._iter_models(src_path):
                query = f"{model} {brand}"
                a_rating, a_count, a_url = self.fetch_amazon(query)
                fk_rating, fk_count, fk_url = self.fetch_flipkart(query)
                row = {
                    'Brand': brand,
                    'Model Name': model,
                    'Amazon Rating': a_rating if a_rating is not None else '',
                    'Amazon Ratings Count': a_count if a_count is not None else '',
                    'Amazon URL': a_url if a_url else '',
                    'Flipkart Rating': fk_rating if fk_rating is not None else '',
                    'Flipkart Ratings Count': fk_count if fk_count is not None else '',
                    'Flipkart URL': fk_url if fk_url else '',
                    'Retrieved At': datetime.utcnow().isoformat() + 'Z',
                }
                rows.append(row)
                count += 1
                print(f"Collected {count}: {model}")
                if limit_per_brand and count >= limit_per_brand:
                    break
            self._write_brand_rows(bf, rows)
            print('Saved reviews for', os.path.splitext(bf)[0])


if __name__ == '__main__':
    # Optional: read environment variables for simple control
    import argparse
    parser = argparse.ArgumentParser(description='Fetch Amazon & Flipkart ratings for phones in GSMArenaDataset')
    parser.add_argument('--brand', type=str, help='Single brand name to process (match GSMArenaDataset file name without extension)')
    parser.add_argument('--limit', type=int, help='Limit number of models per brand for quicker runs')
    args = parser.parse_args()

    scraper = MarketplaceReviews()
    try:
        scraper.run(brand_filter=args.brand, limit_per_brand=args.limit)
    except KeyboardInterrupt:
        print('Stopped by user (KeyboardInterrupt).')
