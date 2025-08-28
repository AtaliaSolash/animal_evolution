from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler

def download_images(keyword, family, group, max_num=10):
    folder = f"data_pairs/{family}/{group}/{keyword}"
    crawler = GoogleImageCrawler(storage={'root_dir': folder})
    crawler.crawl(keyword=keyword, max_num=max_num)

def download_family_images(family_name, query_dict, max_per_query=10):
    for group, keywords in query_dict.items():
        for keyword in keywords:
            print(f"Downloading {family_name} - {group} - {keyword} ...")
            download_images(keyword, family_name, group, max_num=max_per_query)


felidea_queries = {
    'Domestic_cat_lineage': [
        'domestic cat head',
        'Chinese mountain cat head',
        'sand cat head'
    ],
    'Leopard_cat_lineage': [
        'leopard cat head',
        'fishing cat head',
        'rusty spotted cat head'
    ],
    'Puma_lineage': [
        'cheetah head',
        'jaguarundi head'
    ],
    'Lynx_lineage': [
        'bobcat head',
        'eurasian lynx head',
        'Iberian lynx head'
    ],
    'Bay_cat_lineage': [
        'asian golden cat head',
        'marbled cat head'
    ],
    'Ocelot_lineage': [
        'ocelot head',
        'margay head'
    ],
    'Caracal_lineage': [
        'caracal head',
        'african golden cat head',
        'serval head'
    ],
    'Panthera_lineage': [
        'lion head',
        'tiger head',
        'leopard head',
        'jaguar head',
        'snow leopard head'
    ]
}

caninae_queries = {
    'canina_canis_lupulella': [
        'gray wolf head',
        'red wolf head',
        'golden jackal head',
        'black-backed jackal head'
    ],
    'Cerdocyonina': [
        'maned wolf head',
        'Speothos venaticus head',
        'crab-eating fox head',
        'sechuran fox head',
        'short-eared dog head',
    ],
    'Vulpini': [
        'red fox head',
        'arctic fox head',
        'fennec fox head',
        'blanford fox head',
        'bat-eared fox head',
        'raccoon dog head'
    ],
    'Urocyon': [
        'island fox head',
        'gray fox head'
    ]
}

bovidae_queries = {
    'Bovinae': [
        'Bubalus head',
        'Bison head'
    ],
    'Antilopinae': [
        'Blackbuck head',
        'Red-fronted gazelle head',
        'chinkara head',
        'Nanger granti head',
        'saiga head'
    ],
    'Reduncinae': [
        'Kobus (antelope) head',
        'Reedbuck head',
        'Grey rhebok head'
    ],
    'Caprinae': [
        'Ovibos moschatus head',
        'Bighorn sheep head',
        'Pyrenean chamois head',
        'Barbary sheep head',
        'Himalayan tahr head'
    ],
    'Alcelaphinae': [
        'bontebok head',
        'kongoni head',
        'Wildebeest head'
    ]
}


cervidae_queries = {
    'Muntiacini': [
        'Muntiacus reevesi head',
        'Tufted deer head'
    ],
    'Cervini': [
        'Common fallow deer head',
        'Persian fallow deer head',
        'Sika deer head',
        'Rusa head',
        'Red deer head',
        'Rucervus eldii head',
        'barasingha head',
        'Chital head'

    ],
    'Rangiferini': [
        'Reindeer head',
        'Red brocket head',
        'Mule deer head',
        'Marsh deer head',
        'Southern pudu head',
        'Taruca head'
    ],
    'Capreolini': [
        'Roe deer head',
        'Water deer head'
    ],
    'Alceini': [
        'Moose head'
    ]
}

download_family_images("felidae", felidea_queries)
download_family_images("caninae", caninae_queries)
download_family_images("bovidae", bovidae_queries)
download_family_images("cervidae", cervidae_queries)

