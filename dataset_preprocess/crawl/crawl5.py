from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
from tqdm import tqdm


data = pd.read_csv('music_bugs_4.csv')

def setup_driver_and_open_melon():
    options = Options()
    options.add_experimental_option('detach', True)
    options.add_experimental_option("excludeSwitches", ['enable-logging'])
    options.add_argument("User_Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")
    options.add_argument("disable-gpu")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get("https://www.melon.com")
    return driver

driver = setup_driver_and_open_melon()

# 가사 데이터 저장 함수
def save_lyrics_to_csv(lyrics):
    temp_df = data.copy()
    temp_df['lyric'] = lyrics
    temp_df.to_csv(f'music_bugs_4_lines.csv', index=False)

lyrics = []

for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Fetching lyrics"):
    try:

        # 검색어 입력
        input_msg = row['artist'] + " " + row['song_name']
        search_input = driver.find_element(By.XPATH, '//*[@id="top_search"]')
        search_input.clear()
        search_input.send_keys(input_msg + '\n')

        # 첫 번째 검색 결과 클릭
        first_result = driver.find_element(By.CLASS_NAME, 'btn_icon_detail')
        first_result.click()

        # 가사 가져오기
        lyric_box = driver.find_element(By.CLASS_NAME, 'lyric')
        lyric = lyric_box.text
    except NoSuchElementException:
        lyric = "N/A"
    except Exception as e:
        print(f"Error occurred: {e}")
        lyric = "N/A"

    lyrics.append(lyric)
    driver.back()

# 마지막 남은 가사들 저장
save_lyrics_to_csv(lyrics)

driver.quit()