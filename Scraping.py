# Import the libraries.
from pydoc import text
#popolare il dataset con i ristoranti di Bari
import requests
from bs4 import BeautifulSoup
import pandas as pd


# Extract the HTML and create a BeautifulSoup object.
url = ('https://www.tripadvisor.it/Restaurants-g187874-Bari_Province_of_Bari_Puglia.html')


user_agent = ({'User-Agent':
                   'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                   AppleWebKit/537.36 (KHTML, like Gecko) \
                   Chrome/90.0.4430.212 Safari/537.36',
               'Accept-Language': 'en-US, en;q=0.5'})

def get_page_contents(url):
    page = requests.get(url, headers = user_agent)
    return BeautifulSoup(page.text, 'html.parser')

soup = get_page_contents(url)

# Find and extract the data elements.
nome = []
for name in soup.findAll('div',{'class':'RfBGI'}):
    nome.append(name.text.strip())

recensione = []
for recensioni in soup.findAll('span',{'class':'IiChw'}):
    recensione.append(recensioni.text.strip())

descrizione = []
for descrizioni in soup.findAll('div',{'class':'nrKLE PQvPS bAdrM'}):
    for each in descrizioni:
        desc = each.find_next('span',{'class':'qAvoV'})
        for each in descrizioni:
            desc = each.find_next('span',{'class':'ABgbd'})
    descrizione.append(descrizioni.text)

valutazione = []
for valutazioni in soup.findAll('svg',{'class': 'UctUV d H0'}):
    valutazione.append(valutazioni.get('aria-label'))

link = []
for links in soup.findAll('a',{'class': 'Lwqic Cj b'}):
    link.append('https://tripadvisor.com' + links.get('href'))


# Create the dictionary.

dict = {'Nome':nome, 'Valutazioni' :valutazione, 'nRecensioni':recensione, 'Descrizione' :descrizione, 'Link' :link}

# Create the dataframe.
df = pd.DataFrame.from_dict(dict, orient='index')
df = df.transpose()
df.head(10)

# Convert dataframe to xlsx file.
df.to_excel('hotels.xlsx', index=False, header=True)