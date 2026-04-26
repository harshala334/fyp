import requests
from bs4 import BeautifulSoup
import json
import hashlib
import os
import time
import pandas as pd

def get_md5(url):
    return hashlib.md5(url.encode()).hexdigest()

def scrape_factcheck_org(pages_to_scrape=2):
    base_url = "https://www.factcheck.org/page/{}/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    scraped_data = []
    
    for page_num in range(1, pages_to_scrape + 1):
        url = base_url.format(page_num)
        print(f"Scanning FactCheck Page {page_num}...")
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('article')
            for art in articles:
                title_tag = art.find('h3', class_='entry-title')
                if not title_tag: continue
                link = title_tag.find('a')['href']
                title = title_tag.get_text(strip=True)
                post_id = get_md5(link)
                
                art_resp = requests.get(link, headers=headers)
                art_soup = BeautifulSoup(art_resp.content, 'html.parser')
                
                content_div = art_soup.find('div', class_='entry-content')
                text_paragraphs = [p.get_text() for p in content_div.find_all('p')] if content_div else []
                full_text = " ".join(text_paragraphs)
                
                img_tag = content_div.find('img') if content_div else None
                img_url = img_tag['src'] if img_tag else None
                
                img_filename = None
                if img_url:
                    try:
                        img_data = requests.get(img_url, headers=headers).content
                        ext = img_url.split('.')[-1].split('?')[0]
                        if len(ext) > 4: ext = "jpg"
                        img_filename = f"dataset/images/{post_id}.{ext}"
                        with open(img_filename, 'wb') as h: h.write(img_data)
                    except: pass
                
                scraped_data.append({
                    "post_id": post_id,
                    "text": full_text,
                    "title": title,
                    "local_image_path": img_filename,
                    "verdict": "factcheck_analysis"
                })
                time.sleep(0.5)
        except Exception as e: print(f"Error: {e}")
    return scraped_data

def scrape_politifact(pages=2):
    base_url = "https://www.politifact.com/factchecks/list/?page={}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    data = []
    for p in range(1, pages + 1):
        print(f"Scanning PolitiFact Page {p}...")
        try:
            response = requests.get(base_url.format(p), headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            items = soup.find_all('li', class_='o-listicle__item')
            for item in items:
                quote_div = item.find('div', class_='m-statement__quote')
                if not quote_div: continue
                text = quote_div.get_text(strip=True)
                
                meter_div = item.find('div', class_='m-statement__meter')
                verdict = meter_div.find('img')['alt'] if meter_div and meter_div.find('img') else "unknown"
                
                img_div = item.find('div', class_='m-statement__image')
                img_url = img_div.find('img')['src'] if img_div and img_div.find('img') else None
                
                post_id = hashlib.md5(text.encode()).hexdigest()
                img_filename = None
                if img_url:
                    try:
                        img_data = requests.get(img_url, headers=headers).content
                        img_filename = f"dataset/images/{post_id}.jpg"
                        with open(img_filename, 'wb') as f: f.write(img_data)
                    except: pass
                
                data.append({
                    "id": post_id,
                    "text": text,
                    "verdict": verdict,
                    "local_img_path": img_filename
                })
            time.sleep(0.5)
        except Exception as e: print(f"Error: {e}")
    return data

def main():
    os.makedirs('dataset/images', exist_ok=True)
    
    # 1. PolitiFact Incremental Scrape
    print("Starting massive scale PolitiFact scraping...")
    pf_file = 'dataset/political_data.jsonl'
    # Clear file if starting fresh
    with open(pf_file, 'w') as f: pass 
    
    for p in range(1, 61):
        print(f"Scraping PolitiFact Page {p}/60...")
        page_data = scrape_politifact(1) # Scrape 1 page at a time
        if not page_data: break
        with open(pf_file, 'a') as f: # Append mode
            for entry in page_data:
                json.dump(entry, f)
                f.write('\n')
        time.sleep(1) # Polite delay
        
    # 2. FactCheck Incremental Scrape
    print("\nStarting massive scale FactCheck scraping...")
    fc_file = 'dataset/output.jsonl'
    with open(fc_file, 'w') as f: pass
    
    for p in range(1, 26):
        print(f"Scraping FactCheck Page {p}/25...")
        page_data = scrape_factcheck_org(1)
        if not page_data: break
        with open(fc_file, 'a') as f:
            for entry in page_data:
                json.dump(entry, f)
                f.write('\n')
        time.sleep(1)

    print(f"✅ Incremental Scraping complete!")

if __name__ == "__main__":
    main()
