import requests
from bs4 import BeautifulSoup
import psycopg2
import json
import time


DB_NAME = 'your_db_name'
DB_USER = 'your_db_user'
DB_PASSWORD = 'your_db_password'
DB_HOST = 'localhost'
DB_PORT = '5432'

# Базовый URL для раздела "Второе блюдо"
BASE_URL = 'https://1000.menu'
CATALOG_URL = 'https://1000.menu/catalog/vtoroe-bludo'


def get_connection():
    """Создаёт и возвращает соединение с PostgreSQL."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )


def parse_catalog_page(page_url):
    """
    Скачивает одну страницу каталога (вторых блюд) и
    возвращает список ссылок на отдельные рецепты.
    """
    resp = requests.get(page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    recipe_links = []
    # Ищем блоки с рецептами. На 1000.menu обычно у рецептов есть ссылки вида "/recipes/..."
    # Зачастую рецепты отображаются в карточках с классом "recipe-card" или "content-box" и т.д.
    # Ниже - просто пример, нужно смотреть реальный HTML-код
    cards = soup.find_all('a', class_='content-box-image')  # или другой класс
    for card in cards:
        href = card.get('href')
        if href and '/recipes/' in href:
            recipe_links.append(BASE_URL + href)

    return recipe_links


def parse_recipe_page(recipe_url):
    """
    Скачивает страницу конкретного рецепта и извлекает:
    - Название блюда
    - Описание
    - Список ингредиентов (формат JSON)
    - Калории/белки/жиры/углеводы (если есть)
    - Ссылку на картинку
    Возвращает словарь со всеми полями.
    """
    resp = requests.get(recipe_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Название блюда (обычно внутри <h1>)
    title_el = soup.find('h1')
    dish_name = title_el.get_text(strip=True) if title_el else 'Без названия'

    # Описание (краткое)
    # Иногда описание может лежать в теге <div> с классом "recipe-text" или "summary"
    desc_el = soup.find('div', class_='recipe-new desc')
    description = desc_el.get_text(strip=True) if desc_el else ''

    # Изображение
    # Обычно в теге <img> с классом "big_image" или что-то в этом духе
    image_el = soup.find('img', class_='big_image')
    image_url = image_el.get('src') if image_el else ''
    if image_url and image_url.startswith('//'):
        image_url = 'https:' + image_url

    # Ингредиенты
    # Часто ингредиенты на 1000.menu в <table> или <ul> с классом "ingredients-item" и т.д.
    ingredients_table = soup.find('table', class_='ingredients-list')
    ingredients_json = {}
    if ingredients_table:
        rows = ingredients_table.find_all('tr')
        for row in rows:
            # <td>Название ингредиента</td> <td>Количество</td>
            cols = row.find_all('td')
            if len(cols) >= 2:
                ingr_name = cols[0].get_text(strip=True)
                ingr_qty = cols[1].get_text(strip=True)
                ingredients_json[ingr_name] = ingr_qty

    # Калории / БЖУ
    # На 1000.menu бывает блок "Калорийность на 100 г" или "Пищевая ценность"
    # Можем искать div с классом "nutrients" или что-то похожее
    cals, prot, fat, carbs = 0, 0, 0, 0

    nutrients_block = soup.find('div', class_='nutrients')
    # Пример: <div class="nutrients">
    #            <p>Калорийность: 150 ккал</p>
    #            <p>Белки: 10 г, Жиры: 7 г, Углеводы: 12 г</p>
    #         </div>
    if nutrients_block:
        text = nutrients_block.get_text(strip=True).lower()
        # Наивный пример извлечения
        # "калорийность: 150 ккал белки: 10 г, жиры: 7 г, углеводы: 12 г"
        import re
        cal_match = re.search(r'калорийность[:\s]+(\d+)', text)
        if cal_match:
            cals = int(cal_match.group(1))
        prot_match = re.search(r'белки[:\s]+(\d+)', text)
        if prot_match:
            prot = int(prot_match.group(1))
        fat_match = re.search(r'жиры[:\s]+(\d+)', text)
        if fat_match:
            fat = int(fat_match.group(1))
        carb_match = re.search(r'углеводы[:\s]+(\d+)', text)
        if carb_match:
            carbs = int(carb_match.group(1))

    # Собираем всё в словарь
    result = {
        'name': dish_name,
        'description': description,
        'ingredients': ingredients_json,
        'calories': cals,
        'proteins': prot,
        'fats': fat,
        'carbs': carbs,
        'image_url': image_url
    }
    return result


def insert_dish_to_db(dish_data):
    """
    Вставляет блюдо в таблицу "dishes".
    dish_data - это словарь с ключами:
        name, description, ingredients, calories, proteins, fats, carbs, image_url
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO "dishes"
            ("name", "description", "ingredients", "calories", "proteins", "fats", "carbs", "image_url")
        VALUES (%s, %s, %s::json, %s, %s, %s, %s, %s)
        """,
        (
            dish_data["name"],
            dish_data["description"],
            json.dumps(dish_data["ingredients"], ensure_ascii=False),
            dish_data["calories"],
            dish_data["proteins"],
            dish_data["fats"],
            dish_data["carbs"],
            dish_data["image_url"]
        )
    )
    conn.commit()
    cur.close()
    conn.close()


def main():
    # Будем идти по нескольким страницам каталога, если они есть.
    # На сайте 1000.menu/catalog/vtoroe-bludo?page=2, 3, ... — нужно проверить реальную пагинацию.
    for page_num in range(1, 6):  # скажем, первые 5 страниц
        page_url = f"{CATALOG_URL}?page={page_num}"
        recipe_links = parse_catalog_page(page_url)
        print(f"Страница {page_num}, найдено ссылок: {len(recipe_links)}")

        for link in recipe_links:
            try:
                dish_data = parse_recipe_page(link)
                insert_dish_to_db(dish_data)
                print(f"Вставлено блюдо: {dish_data['name']}")
                # Делаем задержку, чтобы не нагружать сайт
                time.sleep(2)
            except Exception as e:
                print(f"Ошибка при обработке {link}: {e}")
                # Можно здесь сделать time.sleep(5) и продолжить

    print("Готово! Парсинг завершён.")


if __name__ == "__main__":
    main()
