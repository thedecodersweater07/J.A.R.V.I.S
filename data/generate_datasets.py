import numpy as np
import json
import os
from PIL import Image
import wave
import struct
import csv
import random
import requests
from bs4 import BeautifulSoup
import re
import time
import sys

def genereer_tabulaire_data():
    # Genereer 1000 rijen met dummy classificatie gegevens
    np.random.seed(42)
    with open('trainingsdata.csv', 'w') as f:
        f.write('kenmerk1,kenmerk2,kenmerk3,kenmerk4,kenmerk5,doelwaarde,betrouwbaarheid\n')
        for _ in range(1000):
            kenmerken = np.random.randn(5)
            doelwaarde = 1 if sum(kenmerken) > 0 else 0
            betrouwbaarheid = np.random.uniform(0.6, 1.0)
            rij = ','.join(map(str, kenmerken)) + f',{doelwaarde},{betrouwbaarheid:.3f}\n'
            f.write(rij)
    print("Tabulaire data gegenereerd.")

def genereer_nlp_data():
    # Genereer gespreksparen in het Nederlands
    gesprekken = [
        {"vraag": "Hallo, hoe gaat het?", "antwoord": "Het gaat goed, dank je!", "categorie": "begroeting"},
        {"vraag": "Hoe is het weer vandaag?", "antwoord": "Het is een zonnige dag vandaag.", "categorie": "weer"},
        {"vraag": "Kun je me helpen met programmeren?", "antwoord": "Natuurlijk! Met welke programmeertaal werk je?", "categorie": "programmeren"},
        {"vraag": "Hoe laat is het?", "antwoord": "Ik kan de actuele tijd niet zien, maar ik kan je wel helpen met tijdgerelateerde vragen!", "categorie": "tijd"},
        {"vraag": "Vertel eens een grap", "antwoord": "Waarom houden programmeurs niet van de natuur? Er zitten te veel bugs!", "categorie": "entertainment"}
    ]
    with open('nlp_dataset.jsonl', 'w') as f:
        for gesprek in gesprekken:
            f.write(json.dumps(gesprek) + '\n')
    print("NLP data gegenereerd.")

def genereer_tekstcorpus():
    # Genereer dummy tekstcorpus in het Nederlands
    alineas = [
        "Kunstmatige intelligentie revolutioneert de technologie.",
        "Machine learning stelt computers in staat om te leren van gegevens.",
        "Diepe neurale netwerken verwerken complexe patronen.",
        "Natuurlijke taalverwerking helpt computers tekst te begrijpen.",
        "Computervisiesystemen kunnen afbeeldingen effectief analyseren."
    ] * 2000
    tekst = " ".join(alineas)
    with open('tekstcorpus.txt', 'w') as f:
        f.write(tekst)
    print("Tekstcorpus gegenereerd.")

def genereer_afbeeldingsdataset():
    # Maak dummy afbeeldingen en labels
    os.makedirs('afbeeldingen', exist_ok=True)
    afbeeldingen_data = []
    for i in range(10):
        # Maak dummy afbeelding
        afb = Image.new('RGB', (100, 100), color=f'#{i*20:02x}{i*20:02x}{i*20:02x}')
        bestandsnaam = f'afbeeldingen/afbeelding_{i}.png'
        afb.save(bestandsnaam)

        # Voeg toe aan labels
        afbeeldingen_data.append({
            "bestandsnaam": bestandsnaam,
            "label": f"klasse_{i}"
        })
    with open('afbeelding_labels.json', 'w') as f:
        json.dump(afbeeldingen_data, f, indent=2)
    print("Afbeeldingsdataset gegenereerd.")

def genereer_spraakdataset():
    # Maak spraakdataset map binnen data/ai_training_data
    os.makedirs('spraakdataset', exist_ok=True)

    # Genereer dummy WAV bestanden en transcripties
    transcripties = []
    for i in range(5):
        # Maak dummy audio
        bestandsnaam = f'spraakdataset/audio_{i}.wav'
        with wave.open(bestandsnaam, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            for _ in range(16000):  # 1 seconde audio
                waarde = struct.pack('h', int(32767 * np.sin(2 * np.pi * 440 * _ / 16000)))
                f.writeframes(waarde)
        # Voeg transcriptie toe
        transcripties.append(f"audio_{i}.wav: Dit is transcriptie {i}")
    # Schrijf transcripties
    with open('spraakdataset/transcripties.txt', 'w') as f:
        f.write('\n'.join(transcripties))
    print("Spraakdataset gegenereerd.")

def genereer_woordenlijsten():
    # Genereer woordenlijsten voor verschillende categorieën
    categorieën = {
        'nederlandse_woorden.txt': ['huis', 'boom', 'fiets', 'water', 'zon', 'maan', 'computer', 'telefoon', 'boek', 'tafel'] * 100,
        'engelse_woorden.txt': ['house', 'tree', 'bike', 'water', 'sun', 'moon', 'computer', 'phone', 'book', 'table'] * 100,
        'technische_termen.csv': [('API', 'Application Programming Interface'),
                               ('CPU', 'Centrale Verwerkingseenheid'),
                               ('RAM', 'Random Access Memory'),
                               ('GPU', 'Grafische Verwerkingseenheid')] * 25
    }

    for bestandsnaam, woorden in categorieën.items():
        if bestandsnaam.endswith('.csv'):
            with open(bestandsnaam, 'w', newline='') as f:
                schrijver = csv.writer(f)
                schrijver.writerow(['Term', 'Beschrijving'])
                schrijver.writerows(woorden)
        else:
            with open(bestandsnaam, 'w') as f:
                f.write('\n'.join(woorden))
    print("Woordenlijsten gegenereerd.")

def genereer_json_datasets():
    # Genereer verschillende JSON datasets
    datasets = {
        'producten.json': [
            {'id': i, 'naam': f'Product {i}', 'prijs': random.uniform(10, 1000)}
            for i in range(100)
        ],
        'gebruikers.json': [
            {'id': i, 'gebruikersnaam': f'gebruiker_{i}', 'actief': random.choice([True, False])}
            for i in range(200)
        ],
        'config.json': {
            'instellingen': {
                'taal': 'nl',
                'thema': 'donker',
                'meldingen': True
            }
        }
    }

    for bestandsnaam, data in datasets.items():
        with open(bestandsnaam, 'w') as f:
            json.dump(data, f, indent=2)
    print("JSON datasets gegenereerd.")

def genereer_trainingsmatrices():
    # Genereer numpy matrices voor training
    matrices = {
        'kenmerken_matrix.npy': np.random.randn(1000, 50),
        'labels_matrix.npy': np.random.randint(0, 5, size=(1000,)),
        'embeddings.npy': np.random.randn(100, 300)
    }

    for bestandsnaam, data in matrices.items():
        np.save(bestandsnaam, data)
    print("Trainingsmatrices gegenereerd.")

def scrap_nieuws_data(aantal_paginas=3):
    """Scrap nieuwsartikelen van nu.nl"""
    logger = logging.getLogger(__name__)
    
    # Map wordt gemaakt binnen data/ai_training_data
    nieuws_dir = Path('nieuws_data')
    nieuws_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    artikelen = []
    timeout = 10  # Timeout in seconds for requests
    max_retries = 3
    
    for pagina in range(1, aantal_paginas + 1):
        url = f"https://www.nu.nl/algemeen/{pagina}"
        logger.info(f"Bezig met scrapen van pagina {pagina}...")
        
        # Implement retry logic
        for retry in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Zoek alle artikellinks
                artikel_blokken = soup.find_all('a', href=re.compile(r'/\d+/'))
                
                for blok in artikel_blokken:
                    titel = blok.text.strip()
                    link = blok.get('href')
                    
                    if titel and link and len(titel) > 10:
                        # Zorg voor een volledige URL
                        if not link.startswith('http'):
                            link = f"https://www.nu.nl{link}"
                        
                        artikel_info = {
                            'titel': titel,
                            'url': link
                        }
                        
                        # Voeg toe aan lijst als het nog niet bestaat
                        if artikel_info not in artikelen:
                            artikelen.append(artikel_info)
                
                # Success, break the retry loop
                break
                
            except requests.exceptions.Timeout:
                if retry < max_retries - 1:
                    logger.warning(f"Timeout bij het ophalen van {url}. Poging {retry + 1}/{max_retries}. Opnieuw proberen...")
                    time.sleep(2)  # Wait before retrying
                else:
                    logger.error(f"Timeout bij het ophalen van {url} na {max_retries} pogingen.")
                    raise
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP-fout bij het ophalen van {url}: {e}")
                break  # Don't retry for HTTP errors
            except requests.exceptions.RequestException as e:
                if retry < max_retries - 1:
                    logger.warning(f"Fout bij het ophalen van {url}: {e}. Poging {retry + 1}/{max_retries}. Opnieuw proberen...")
                    time.sleep(2)  # Wait before retrying
                else:
                    logger.error(f"Fout bij het ophalen van {url} na {max_retries} pogingen: {e}")
                    raise
        
        # Wacht even om de server niet te overbelasten
        time.sleep(1)
        
        # Sla de geschraapte data op
        with open('nieuws_data/nieuws_artikelen.json', 'w', encoding='utf-8') as f:
            json.dump(artikelen, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Succesvol {len(artikelen)} nieuwsartikelen geschraapt en opgeslagen.")

def scrap_wikipedia_artikelen(onderwerpen=None):
    """Scrap informatie van Nederlandse Wikipedia-artikelen"""
    logger = logging.getLogger(__name__)
    
    if onderwerpen is None:
        onderwerpen = [
            'Nederland', 'Amsterdam', 'Kunstmatige_intelligentie', 
            'Machine_learning', 'Python_(programmeertaal)', 'Data_mining'
        ]
    
    # Map wordt gemaakt binnen data/ai_training_data
    wiki_dir = Path('wiki_data')
    wiki_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    wiki_data = {}
    
    for onderwerp in onderwerpen:
        url = f"https://nl.wikipedia.org/wiki/{onderwerp}"
        logger.info(f"Bezig met scrapen van Wikipedia-artikel: {onderwerp}")
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Verwijder tabellen, navigatie en andere niet-relevante elementen
            for div in soup.find_all(['table', 'div', 'script', 'style']):
                div.decompose()
            
            # Haal titel op
            titel_element = soup.find('h1', {'id': 'firstHeading'})
            titel = titel_element.text if titel_element else onderwerp
            
            # Haal inhoud op
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            
            if content_div:
                paragrafen = []
                for p in content_div.find_all('p'):
                    tekst = p.text.strip()
                    if tekst and len(tekst) > 50:  # Skip korte paragrafen
                        paragrafen.append(tekst)
                
                inhoud = '\n\n'.join(paragrafen)
                
                # Sla op in de dictionary
                wiki_data[onderwerp] = {
                    'titel': titel,
                    'inhoud': inhoud,
                    'url': url
                }
                
                # Sla elk artikel ook op als apart tekstbestand
                with open(f'wiki_data/{onderwerp}.txt', 'w', encoding='utf-8') as f:
                    f.write(f"TITEL: {titel}\n\n")
                    f.write(f"BRON: {url}\n\n")
                    f.write(inhoud)
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Fout bij het scrapen van {onderwerp}: {e}")
    
    # Sla alle data op in één JSON-bestand
    with open('wiki_data/wikipedia_artikelen.json', 'w', encoding='utf-8') as f:
        json.dump(wiki_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Succesvol {len(wiki_data)} Wikipedia-artikelen geschraapt en opgeslagen.")

def genereer_prompt_dataset():
    """Genereer een dataset met voorbeeldprompts voor AI-training"""
    logger = logging.getLogger(__name__)
    
    # Map wordt gemaakt binnen data/ai_training_data
    prompt_dir = Path('prompt_data')
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = [
        {
            "type": "creatief",
            "prompt": "Schrijf een kort verhaal over een robot die menselijke emoties ontwikkelt.",
            "moeilijkheidsgraad": "gemiddeld"
        },
        {
            "type": "zakelijk",
            "prompt": "Maak een persbericht voor de lancering van een nieuw AI-product.",
            "moeilijkheidsgraad": "gemiddeld"
        },
        {
            "type": "technisch",
            "prompt": "Leg uit hoe een neuraal netwerk werkt in eenvoudige taal.",
            "moeilijkheidsgraad": "gemakkelijk"
        },
        {
            "type": "creatief",
            "prompt": "Schrijf een gedicht over de veranderende seizoenen in Nederland.",
            "moeilijkheidsgraad": "gemakkelijk"
        },
        {
            "type": "educatief",
            "prompt": "Maak een lesplan voor het onderwijzen van machine learning aan middelbare scholieren.",
            "moeilijkheidsgraad": "moeilijk"
        },
        {
            "type": "technisch",
            "prompt": "Schrijf pseudocode voor een algoritme dat boekrecensies classificeert op sentiment.",
            "moeilijkheidsgraad": "moeilijk"
        },
        {
            "type": "zakelijk",
            "prompt": "Maak een marketingstrategie voor een nieuw Nederlands tech-bedrijf.",
            "moeilijkheidsgraad": "gemiddeld"
        },
        {
            "type": "creatief",
            "prompt": "Beschrijf een dag in het leven van een AI-assistent in het jaar 2050.",
            "moeilijkheidsgraad": "gemiddeld"
        },
        {
            "type": "educatief",
            "prompt": "Maak een quiz over Nederlandse geschiedenis met 10 vragen en antwoorden.",
            "moeilijkheidsgraad": "gemiddeld"
        },
        {
            "type": "technisch",
            "prompt": "Schrijf een handleiding voor het optimaliseren van een machine learning model.",
            "moeilijkheidsgraad": "moeilijk"
        }
    ]
    
    # Voeg meer variaties toe voor een grote dataset
    extra_prompts = []
    for i in range(40):
        onderwerpen = ['klimaatverandering', 'ruimtevaart', 'gezondheidszorg', 'onderwijs', 
                      'transport', 'energie', 'voedselproductie', 'waterbeheer']
        types = ['beschrijvend', 'analytisch', 'vergelijkend', 'instructief', 'overtuigend']
        
        onderwerp = random.choice(onderwerpen)
        prompt_type = random.choice(types)
        moeilijkheid = random.choice(['gemakkelijk', 'gemiddeld', 'moeilijk'])
        
        extra_prompts.append({
            "type": prompt_type,
            "prompt": f"Schrijf een {prompt_type} tekst over {onderwerp} in Nederland.",
            "moeilijkheidsgraad": moeilijkheid
        })
    
    # Combineer de prompts
    alle_prompts = prompts + extra_prompts
    
    # Sla de dataset op in data/ai_training_data/prompt_data
    with open('prompt_data/nl_prompts.json', 'w', encoding='utf-8') as f:
        json.dump(alle_prompts, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dataset met {len(alle_prompts)} AI-prompts gegenereerd.")

def hoofdfunctie():
    # Configure logging
    logger = logging.getLogger(__name__)
    
    # Maak de hoofdmap structuur 'data/ai_training_data'
    target_dir = Path('data/ai_training_data')
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Beginnen met het genereren van datasets...")
    
    # Oorspronkelijke generators
    genereer_tabulaire_data()
    genereer_nlp_data()
    genereer_tekstcorpus()
    genereer_afbeeldingsdataset()
    genereer_spraakdataset()

    # Nieuwe generators
    genereer_woordenlijsten()
    genereer_json_datasets()
    genereer_trainingsmatrices()
    
    # Prompt dataset generator
    genereer_prompt_dataset()
    
    # Web scraping functies
    logger.info("\nBeginnen met web scraping...")
    scrap_nieuws_data(aantal_paginas=2)
    scrap_wikipedia_artikelen()

    logger.info("\nDataset generatie compleet!")
    logger.info("Alle datasets zijn opgeslagen in de map 'data/ai_training_data'")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Controleer of we al in de juiste directory zijn
        huidige_dir = os.getcwd()
        if huidige_dir.endswith('data/ai_training_data'):
            logger.info(f"Let op: Je bent al in de map '{huidige_dir}'.")
            logger.info("Script zal worden uitgevoerd in de huidige map.")
            # Reset naar de oorspronkelijke directory
            os.chdir(os.path.dirname(os.path.dirname(huidige_dir)))
        
        # Run the main function
        hoofdfunctie()
        
    except FileNotFoundError as e:
        logger.error(f"Bestand of map niet gevonden: {e}")
        logger.info("Controleer of alle benodigde mappen bestaan en toegankelijk zijn.")
        sys.exit(1)
    except PermissionError as e:
        logger.error(f"Geen toegang tot bestand of map: {e}")
        logger.info("Controleer of je de juiste rechten hebt voor de bestanden en mappen.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        logger.error(f"Netwerkfout bij het ophalen van gegevens: {e}")
        logger.info("Controleer je internetverbinding en probeer het opnieuw.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Fout bij het verwerken van JSON-gegevens: {e}")
        logger.info("Een JSON-bestand heeft een ongeldig formaat.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nScript onderbroken door gebruiker.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Er is een onverwachte fout opgetreden: {e}", exc_info=True)
        logger.info("Controleer de logbestanden voor meer informatie.")
        sys.exit(1)