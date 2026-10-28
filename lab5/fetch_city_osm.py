# -*- coding: utf-8 -*-
"""
Знімок даних OpenStreetMap для лабораторної роботи № 5.

Запускати на машині з інтернетом (не в Cowork — там OSM недоступний):

    pip install "osmnx>=2.1" "geopandas>=1.0" pyarrow
    python fetch_city_osm.py "Львів, Україна"
    python fetch_city_osm.py "Тернопіль, Україна" --out ternopil_osm

Створює теку зі скриптами .parquet і manifest.json. Повторний запуск
пропускає вже збережені шари, тож перерваний збір можна продовжити.

Чому пішохідна мережа, а не автомобільна: у старому місті музеї стоять
за 100-200 метрів один від одного, а автомобільних вузлів там одиниці.
Кілька пам'яток сідають на ОДИН вузол, і матриця віддалей каже, що між
ними нуль метрів. Пішохідна мережа щільніша і такого не робить.
"""
import argparse, json, pathlib, sys, time, warnings
warnings.filterwarnings('ignore')

TOURISM = ['museum', 'attraction', 'gallery', 'artwork', 'viewpoint']

# Дзеркала Overpass API без ключа, за переліком у вікі OSM.
# Головний сервер FOSSGIS перевантажений і регулярно недоступний, тому
# пробуємо по черзі. Російське дзеркало VK Maps свідомо НЕ включене.
MIRRORS = [
    'https://overpass-api.de/api',
    'https://overpass.private.coffee/api',
    'https://z.overpass-api.de/api',
]

PROBE = '[out:json][timeout:10];node(49.8419,24.0315,49.8420,24.0316);out count;'

# Overpass просить представлятись. Запит із типовим «python-requests/2.x»
# дзеркала відбивають кодом 429, навіть коли вільні.
USER_AGENT = ('LvivPolytechnic-SK26-coursework/1.0 '
              '(https://lpnu.ua; student lab assignment)')
# HTTP-заголовки передаються в latin-1: кирилиця в User-Agent кладе запит
# помилкою UnicodeEncodeError ще до відправлення. Тому рядок тільки латиницею.
assert USER_AGENT.isascii(), 'User-Agent має бути лише з символів ASCII'


def host_of(url):
    import re
    m = re.match(r'https?://([^/:]+)', url)
    return m.group(1) if m else url


def resolve_all(hostname, port=443):
    """Усі адреси, за якими ховається ім\u2019я (overpass-api.de їх кілька)."""
    import socket
    try:
        infos = socket.getaddrinfo(hostname, port, socket.AF_INET, socket.SOCK_STREAM)
    except Exception:
        return []
    return sorted({i[4][0] for i in infos})


def tcp_ok(ip, port=443, timeout=8):
    import socket
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def pin_host(hostname, ip):
    """Примусити весь процес ходити на ЦЮ адресу для цього імені.

    osmnx усередині викликає socket.gethostbyname і закріплює одну адресу
    на всю сесію. За іменем overpass-api.de стоїть кілька машин, і частина
    з них відмовляє у з\u2019єднанні. Якщо osmnx витягне саме таку, впаде
    весь збір. Тому адресу обираємо самі — вже перевірену."""
    import socket
    original = socket.gethostbyname

    def patched(h):
        return ip if h == hostname else original(h)

    socket.gethostbyname = patched


def ask_mirror(base, timeout=20):
    """Повертає (стан, секунди): 'ok', 'busy' (429) або назву помилки."""
    import requests
    url = base.rstrip('/') + '/interpreter'
    t0 = time.time()
    try:
        r = requests.post(url, data={'data': PROBE},
                          headers={'User-Agent': USER_AGENT}, timeout=timeout)
    except Exception as e:
        return type(e).__name__, time.time() - t0
    if r.status_code == 200 and r.text.lstrip().startswith('{'):
        return 'ok', time.time() - t0
    if r.status_code in (429, 504):
        return 'busy', time.time() - t0
    return 'HTTP %s' % r.status_code, time.time() - t0


def probe_overpass(urls, attempts=3, wait=35):
    """Шукає дзеркало. 429 означає «живе, але зайняте» — таке чекаємо й повторюємо."""
    busy = []
    for attempt in range(1, attempts + 1):
        for base in urls:
            if attempt > 1 and base not in busy:
                continue
            state, sec = ask_mirror(base)
            print('   %-42s %-22s (%.1f c)' % (base, state, sec))
            if state == 'ok':
                return base
            if state == 'busy' and base not in busy:
                busy.append(base)
        if not busy:
            return None
        if attempt < attempts:
            print('   зайняті дзеркала: %d, чекаю %d с і пробую ще раз (спроба %d з %d)'
                  % (len(busy), wait, attempt + 1, attempts))
            time.sleep(wait)
    if busy:
        print('   усі дзеркала відповідають 429; беру %s і покладаюсь на' % busy[0])
        print('   вбудоване обмеження частоти osmnx — буде повільно, але має пройти')
        return busy[0]
    return None


def clean(gdf):
    """OSM-теги неоднорідні: в одній колонці трапляються і число, і список.
    Parquet вимагає один тип на колонку, тож неоднорідну колонку зводимо
    до тексту ЦІЛКОМ, а не лише ті клітинки, де трапився список."""
    import pandas as pd
    g = gdf.reset_index()
    for c in g.columns:
        if c == 'geometry':
            continue
        boxed = g[c].map(lambda v: isinstance(v, (list, dict, set, tuple))).any()
        if boxed or g[c].dtype == object:
            g[c] = g[c].map(
                lambda v: None if v is None or (isinstance(v, float) and pd.isna(v))
                else (json.dumps(v, ensure_ascii=False)
                      if isinstance(v, (list, dict, set, tuple)) else str(v))
            ).astype('string')
    return g


def cache_bytes(cache_dir):
    p = pathlib.Path(cache_dir)
    return sum(f.stat().st_size for f in p.glob('*.json')) if p.exists() else 0


def save(gdf, name, out, manifest, seconds, bytes_before, cache_dir):
    import geopandas as gpd
    p = out / (name + '.parquet')
    if p.exists() and p.stat().st_size > 0:
        g = gpd.read_parquet(p)
        print('  %-14s вже збережено, пропускаю (%d записів)' % (name, len(g)))
    else:
        g = clean(gdf)
        g.to_parquet(p, index=False)
    manifest[name] = {
        'записів': int(len(g)),
        'колонок': int(len(g.columns)),
        'CRS': g.crs.to_string() if g.crs is not None else None,
        'типи_геометрії': {k: int(v) for k, v in g.geometry.geom_type.value_counts().items()},
        'файл_МБ': round(p.stat().st_size / 1e6, 3),
        'передано_МБ': round((cache_bytes(cache_dir) - bytes_before) / 1e6, 3),
        'секунд': round(seconds, 1),
    }
    print('  %-14s %6d записів | файл %6.2f МБ | передано %6.2f МБ | %5.1f с'
          % (name, len(g), manifest[name]['файл_МБ'],
             manifest[name]['передано_МБ'], seconds))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('place', help='назва міста, напр. "Львів, Україна"')
    ap.add_argument('--out', default=None, help='тека виводу')
    ap.add_argument('--network', default='walk', choices=['walk', 'drive', 'bike', 'all'])
    ap.add_argument('--no-buildings', action='store_true',
                    help='не тягнути шар усіх будівель (найбільший і найповільніший)')
    ap.add_argument('--overpass', default=None,
                    help='базова адреса дзеркала Overpass, без /interpreter')
    a = ap.parse_args()

    import osmnx as ox
    import geopandas as gpd

    out = pathlib.Path(a.out or ('osm_' + a.place.split(',')[0].strip().lower().replace(' ', '_')))
    out.mkdir(exist_ok=True)
    cache_dir = str(out / 'cache')
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir
    ox.settings.requests_timeout = 300
    ox.settings.http_user_agent = USER_AGENT
    print('osmnx', ox.__version__, '| geopandas', gpd.__version__)

    print()
    print('шукаю доступне дзеркало Overpass:')
    chosen = a.overpass or probe_overpass(MIRRORS)
    if chosen is not None:
        host = host_of(chosen)
        ips = resolve_all(host)
        if len(ips) > 1:
            good = [ip for ip in ips if tcp_ok(ip)]
            print('   за іменем %s стоїть %d машин, відповідають %d: %s'
                  % (host, len(ips), len(good), ', '.join(good) or 'жодна'))
            if good:
                pin_host(host, good[0])
                print('   закріплюю %s — інакше osmnx може витягнути ту, що відмовляє'
                      % good[0])
    if chosen is None:
        sys.exit('\nЖодне дзеркало Overpass недоступне з цієї мережі.\n'
                 'Межа міста тягнеться з Nominatim і працює, а Overpass — ні,\n'
                 'отже справа саме в ньому. Спробуйте пізніше або вкажіть\n'
                 'своє дзеркало: --overpass https://host/api')
    ox.settings.overpass_url = chosen
    # обмеження частоти вмикаємо завжди: дзеркала віддають 429 охоче,
    # а osmnx уміє читати /status і чекати стільки, скільки треба
    ox.settings.overpass_rate_limit = True
    print('обрано дзеркало:', chosen)
    print('місто:', a.place, '| мережа:', a.network, '| тека:', out.resolve())

    manifest = {}
    mp = out / 'manifest.json'
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text(encoding='utf-8'))
            print('підхоплено попередній manifest.json')
        except Exception:
            manifest = {}
    manifest['_місце'] = a.place
    manifest['_мережа'] = a.network
    manifest['_дата'] = time.strftime('%Y-%m-%d %H:%M')
    manifest['_версії'] = {'osmnx': ox.__version__, 'geopandas': gpd.__version__}
    t_all = time.time()

    print('\n1/5 Межа міста')
    t0 = time.time(); b0 = cache_bytes(cache_dir)
    place = ox.geocode_to_gdf(a.place)
    save(place, 'boundary', out, manifest, time.time() - t0, b0, cache_dir)
    poly = place.geometry.iloc[0]

    print('\n2/5 Пам’ятки: historic=* і tourism=' + '/'.join(TOURISM))
    t0 = time.time(); b0 = cache_bytes(cache_dir)
    her = ox.features_from_polygon(poly, {'historic': True, 'tourism': TOURISM})
    save(her, 'heritage', out, manifest, time.time() - t0, b0, cache_dir)

    print('\n3/5 Вокзали та станції')
    t0 = time.time(); b0 = cache_bytes(cache_dir)
    sta = ox.features_from_polygon(poly, {'railway': ['station', 'halt'],
                                          'building': ['train_station'],
                                          'amenity': ['bus_station']})
    save(sta, 'stations', out, manifest, time.time() - t0, b0, cache_dir)

    print('\n4/5 Заклади поруч (кава, їжа, вода)')
    t0 = time.time(); b0 = cache_bytes(cache_dir)
    am = ox.features_from_polygon(poly, {'amenity': ['cafe', 'restaurant',
                                                     'drinking_water', 'toilets', 'pharmacy']})
    save(am, 'amenities', out, manifest, time.time() - t0, b0, cache_dir)

    print('\n5/5 Мережа %s' % a.network)
    t0 = time.time(); b0 = cache_bytes(cache_dir)
    G = ox.graph_from_polygon(poly, network_type=a.network)
    nodes, edges = ox.convert.graph_to_gdfs(G)
    manifest['_граф'] = {'вузлів': int(len(nodes)), 'ребер': int(len(edges)),
                         'тип': a.network}
    el = time.time() - t0
    save(nodes, 'road_nodes', out, manifest, el / 2, b0, cache_dir)
    save(edges, 'road_edges', out, manifest, el / 2, b0, cache_dir)

    if not a.no_buildings:
        print('\n+ Будівлі (необов’язковий шар, найбільший)')
        t0 = time.time(); b0 = cache_bytes(cache_dir)
        try:
            bl = ox.features_from_polygon(poly, {'building': True})
            save(bl, 'buildings', out, manifest, time.time() - t0, b0, cache_dir)
        except Exception as e:
            print('   не вдалось:', str(e)[:120])

    manifest['_секунд_усього'] = round(time.time() - t_all, 1)
    manifest['_передано_МБ_усього'] = round(cache_bytes(cache_dir) / 1e6, 2)
    manifest['_файлів_МБ_усього'] = round(
        sum(f.stat().st_size for f in out.glob('*.parquet')) / 1e6, 2)
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\nГотово за %.0f с.' % manifest['_секунд_усього'])
    print('Передано мережею : %.2f МБ' % manifest['_передано_МБ_усього'])
    print('Збережено файлами: %.2f МБ' % manifest['_файлів_МБ_усього'])
    print('\nТека %s:' % out.resolve())
    for p in sorted(out.glob('*.parquet')):
        print('   %-20s %8.2f МБ' % (p.name, p.stat().st_size / 1e6))


if __name__ == '__main__':
    main()
