import copy
from flask import Flask, request, json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestCentroid
import random
import json
import time
from firebase import firebase
import pyrebase
import os
from types import SimpleNamespace


port = int(os.environ.get('PORT', 5000))

FBConnection = firebase.FirebaseApplication(
    "https://botsac-c23a5-default-rtdb.europe-west1.firebasedatabase.app/", None)

#DATABASE CONFIG
config = {
    "apiKey": "AIzaSyCtxYZQCGYjjr6nQVxmm2L_htHcT15fH7U",
    "authDomain": "sacbot-58b6b.firebaseapp.com",
    "databaseURL" : "https://sacbot-58b6b-default-rtdb.europe-west1.firebasedatabase.app/",
    "projectId": "sacbot-58b6b",
    "storageBucket": "sacbot-58b6b.appspot.com",
    "messagingSenderId": "645208476522",
    "appId": "1:645208476522:web:17e01bfdadfe5b9d92b313"
}

#Inizializzazione Database
fb=pyrebase.initialize_app(config)
db=fb.database()




app= Flask(__name__)

class utente:
    chat_id=""
    Nome=""
    attivita=""
    pos=""
    ideale=""
    desc=""
    cambiapos=""
    cambiaidealeper=""
    gusti=""
    primomipiace=""
    secondomipaice=""
    terzomipiace=""
    interessi=""
    ultima_attivita=""
    inseritodautente=""
    prendere_descrizione=""
    presoaltro=""
    k=0
    j=0

    def __init__(self):
        self.TerminidaCercare = []  #per ogni istanza avrà un vettore a se altrimenti viene condiviso
        self.new_elementi_nonrilev=[]
        self.new_elementi_rilev=[]
        self.globalindici=[]
        self.elementi_rilevanti=[]
        self.elementi_nonrilevanti=[]
        self.interessi=[]


global indice
indice=0
global utenti
utenti=[]

df = pd.read_excel(r'https://firebasestorage.googleapis.com/v0/b/sacbot-58b6b.appspot.com/o/DatasetFinale%20(1)%20(3)%20(1)%20(1).xlsx?alt=media&token=9ffe04aa-12ac-4552-8718-5d58113723c1')
# Definisco che le colonne Punteggio e Nume ro recensioni sono stringhe
df['Punteggio'] = df['Punteggio'].astype('string')
df['Numero_Recensioni'] = df['Numero_Recensioni'].astype('string')

# Metto tutto in lowerCase
# df = df.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

lowerify_cols = [col for col in df if
                 col not in ['Link'] and col not in ['Immagini'] and col not in ['Posizioni'] and col not in['Nome']]  # Seleziona le colonne da mettere in lowercase eccetto quella dei Link
df[lowerify_cols] = df[lowerify_cols].apply(
    lambda x: x.str.lower() if (x.dtype == 'object') else x)  # Mette tutto in piccolo

# eliminotutti i caratteri inutili nella colonna Descrizione
df['Descrizione'] = df['Descrizione'].str.replace(',', '')
df['Descrizione'] = df['Descrizione'].str.replace('•', '')

# Elimino gli spazi all'interno delle colonne
df['Paese'] = df['Paese'].str.replace(' ', '')

# Crea la zuppa per fare la vettorizzazione
# il join ogni volta che non trova un carattere di spazio unisce le parole e
# poi mette lo spazio e fa il join dell'altra feature
def creazionezuppa(x):
    return ''.join(x['Descrizione']) + ' ' + ''.join(x['Paese'])


df['soup'] = df.apply(creazionezuppa, axis=1)
df[['Nome', 'soup', 'Descrizione', 'Punteggio', 'Paese', 'Tipo_Attività', 'IdealePer', 'Numero_Recensioni',
    'Link','Immagini','Posizioni']].head()

print(df)


#MODULO DI RICERCA
def fai_raccomandazione(pos,TerminidaCercare, df=df):
    nuova_riga = df.iloc[-1,
                 :].copy()  # crea una copia dell'ultima riga nel dataset per usi successivi ma tiene conto solo della colonna soup
    termini_da_cercare = TerminidaCercare  # prende l'input dell'utente

    nuova_riga.iloc[-1] = " ".join(termini_da_cercare)  # inserisce l'input dell'utente al posto della riga copiata
    df = df.append(
        nuova_riga)  # mette la riga nuova nel dataset all'ultima colonna perchè a prescindere soup si troverà all'ultima colonna di definizione
    # ogni volta che viene richiamato il metodo riprende l'ultima colonna del dataset di partenza per fare la copia non quello che c'era prima

    count = TfidfVectorizer()
    count_matrix = count.fit_transform(
        df[
            'soup'])  # trasformo soup in una matrice che contiene l'idf-tf dei termini [0.2,0,0.4,0.6,0.1,0.3] cioe avro le righe che saranno i documenti e le colonne che saranno tutti i termini

    # calcola la similarità in modo che per ogni elemento della matrice avrà uno score di similarità con tutti gli altri elementi della matrice
    similarita_coseno = cosine_similarity(count_matrix, count_matrix)

    #INIZIO RAFFINAMENTO
    # EFFETTUA IL RAFFINAMENTO DELLE RICERCHE

    if (len(utenti[pos].new_elementi_rilev) > 1 and len(utenti[pos].new_elementi_nonrilev) > 1 ):

        elementidasommarerilevanti = []
        for item in utenti[pos].new_elementi_rilev:
            elementidasommarerilevanti.append(count_matrix[item])

        elementidasommarenonrilevanti = []
        for item in utenti[pos].new_elementi_nonrilev:
            elementidasommarenonrilevanti.append(count_matrix[item])

        queryutente = count_matrix[-1, :]

        #ROCCHIO
        queryutente = 0.90* queryutente + (0.75 / len(elementidasommarerilevanti)) * np.sum(elementidasommarerilevanti) - ( 0.30 / len(elementidasommarenonrilevanti)) * np.sum(elementidasommarenonrilevanti)
        similarita_coseno = cosine_similarity(queryutente, count_matrix)

        #FINE RAFFINAMENTO

    # ordina le similarità dalla piu alta alla piu bassa in base all'ultima riga ovvero l'input utente

    sim_scores = list(enumerate(similarita_coseno[-1,
                                :]))  # numera la lista di similarità dell'input dell'utente in modo da poi ritrovare a che riga facevano riferimento
    sim_scores = sorted(sim_scores, key=lambda x: x[1],
                        reverse=True)  # ordina in base allo score piu alto le similarità con l'input dell'utente

    score_piu_alti=[]
    for i in range(1,20):

        if(sim_scores[i][1]<=1.0 and sim_scores[i][1]>=0.5):
            score_piu_alti.append(int(sim_scores[i][0]))

    numero_corrispondenti = 0

    for element in score_piu_alti:
        if(df['Paese'].iloc[element]==utenti[pos].pos):
            numero_corrispondenti += 1

    # x:x[1] significa che ordinerà in modo che l'elemento alla pos x[1] sarà piu grande di tutti gli altri quindi anche quello alla posizione x[0]

    indiciposizioni = []
    for i in range(1, 20):
        indx = sim_scores[i][
            0]  # prende il primo valore della lista che indica la riga a cui fa riferimento quella similarità grazie al fatto che abbiamo numerato la lista
        # elementipiu_simili.append([df['Link'].iloc[indx]])
        #elementipiu_simili.append(df['Link'].iloc[indx+indice_start])
        #print([df['Paese'].iloc[indx]])
        indiciposizioni.append(int(indx))

    return (numero_corrispondenti,indiciposizioni)


def isNaN(string):
    return string != string

def find_values(id, json_repr):
    results = []

    def _decode_dict(a_dict):
        try:
            results.append(a_dict[id])
        except KeyError:
            pass
        return a_dict

    json.loads(json_repr, object_hook=_decode_dict) # Return value ignored.
    return results

def sort_per_paese(lista,posizioneluogo,df=df):
    posizioneluogo=posizioneluogo.replace(' ', '')
    stesso_paese = []
    altro_paese = []
    ordinati=[]
    num_stesso_paese=0

    for item in lista:
        if (df["Paese"].iloc[item] == posizioneluogo):
            stesso_paese.append(item)
            num_stesso_paese=num_stesso_paese+1

        else:
            altro_paese.append(item)

    ordinati=stesso_paese+altro_paese
    return ordinati,num_stesso_paese

#Per classificare assegna ad ogni elemento del dataset una classe
y = []
i=0
for i in range(len(df)):

    if df['Tipo_Attività'].iloc[i] == 'puntiinteresse':
        y.append(1)
    elif df['Tipo_Attività'].iloc[i] == 'gastronomia':
        y.append(3)
    elif df['Tipo_Attività'].iloc[i] == 'storia':
        y.append(2)

y.append(3)


#MUODULO DI RACCOMANDAZIONE
def rocchio_classifier(interessi, elementi_rilevanti, elementi_nonrilevanti,posizione_utente,df=df,y=y):
    nuova_riga = df.iloc[-1,
                 :].copy()  # crea una copia dell'ultima riga nel dataset per usi successivi ma tiene conto solo della colonna soup
    # termini_da_cercare = TerminidaCercare  # prende l'input dell'utente

    nuova_riga.iloc[-1] = " ".join(interessi)  # inserisce l'input dell'utente al posto della riga copiata
    df = df.append(
        nuova_riga)  # mette la riga nuova nel dataset all'ultima colonna perchè a prescindere soup si troverà all'ultima colonna di definizione
    # ogni volta che viene richiamato il metodo riprende l'ultima colonna del dataset di partenza per fare la copia non quello che c'era prima

    count = TfidfVectorizer()
    count_matrix = count.fit_transform(
        df[
            'soup'])  # trasformo soup in una matrice che contiene l'idf-tf dei termini [0.2,0,0.4,0.6,0.1,0.3] cioe avro le righe che saranno i documenti e le colonne che saranno tutti i termini
    queryutente=count_matrix[-1,:]

    elementidasommarerilevanti = []
    for item in elementi_rilevanti:
        elementidasommarerilevanti.append(count_matrix[item])

    elementidasommarenonrilevanti = []
    for item in elementi_nonrilevanti:
        elementidasommarenonrilevanti.append(count_matrix[item])

    #SommaElemRilev = np.sum(elementidasommarerilevanti) - np.sum(
    #   elementidasommarenonrilevanti)  # fa la differenza degli elementi rilevanti con quelli non rilevanti

    SommaElemRilev= (0.75* queryutente) + ((0.75 / len(elementidasommarerilevanti)) * np.sum(elementidasommarerilevanti)) - ((0.25 / len(elementidasommarenonrilevanti)) * np.sum(elementidasommarenonrilevanti))

    X = count_matrix  # documenti di training
    clf = NearestCentroid()
    clf.fit(X, y)

    indici=df.loc[(df['Paese'] == posizione_utente) & (df['Descrizione'].isin(interessi))].index

    randomiciIndici = random.choices(indici, k=3)

    return randomiciIndici


#MODULO DI GESTIONE DELLE RICHIESTE IN ENTRATA E USCITA
@app.route('/webhook', methods=['POST'])
def webhook():
    req=request.json
    print(request.json)
    query_result = req.get('queryResult')  # prende tutto il contenuto del post inviato da dialogflow

    #prelevare il chat_id dalla richiesta proveniente da dialogflow
    chat_id=find_values('chat', json.dumps(req)) #Ritorna una lista di stringhe di json
    chat_id=json.dumps(chat_id) #converto la lista in un json
    chat_id = json.loads(chat_id)[0]#la lista convertita la faccio diventare un dizionario
    chat_id=chat_id['id']   #prendo il chat_id , questo procedimento solo per ritrovare il chat_id corretto
    print(chat_id)

    global indice
    global utenti
    pos = db.child("utenti").child(chat_id).child("PosizioneInLista").get()
    pos = pos.val()

    #Modulo di recupero dei dati dell'utente
    #Ritrovamento dei dati dal database per l'utente
    if (pos != None):
        print(pos)
        oggetto_utente = db.child("oggetti_utenti").child(chat_id).get()
        oggetto_utente = json.loads(oggetto_utente.val(), object_hook=lambda d: SimpleNamespace(**d))
        utenti.insert(pos,oggetto_utente)

    if(query_result.get('action')=='input.unknown'):
        start = time.time()
        utenti[pos].TerminidaCercare.clear()
        utenti[pos].j = 0
        utenti[pos].k = 0
        utenti[pos].primomipiace = ''
        utenti[pos].secondomipaice = ''
        utenti[pos].terzomipiace = ''
        if (len(utenti[pos].elementi_rilevanti) > 0 and len(utenti[pos].elementi_nonrilevanti) > 0):

            titolo = "Bentornato "+utenti[pos].Nome+" cosa ti piacerebbe fare ? 😀 "
            nuovaricerca = 'Nuova ricerca 🔍'
            TuoiGusti = 'In base ai tuoi gusti 🔖'
            Impostazioni='Modifica la posizione o le modalità di viaggio ⚙️'
            data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                    [{"text": TuoiGusti, "callback_data": "In base ai tuoi gusti 🔖"}],
                                    [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}]]}}},
                                             "platform": "TELEGRAM"},
                                            {"text": {"text": [""]}}]}

            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)


            return payload

        else:

            titolo = "Bentornato "+utenti[pos].Nome+" cosa ti piacerebbe fare ? 😀 "
            nuovaricerca = "Nuova ricerca 🔍"
            Impostazioni = 'Modifica la posizione o le modalità di viaggio ⚙️'
            data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                    [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}],
                                    ]}}},
                                             "platform": "TELEGRAM"},
                                            {"text": {"text": [""]}}]}

            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)
            end = time.time()
            print(end - start)
            return payload


    if (query_result.get('action') == 'help'):
        if (len(utenti[pos].elementi_rilevanti) > 0 and len(utenti[pos].elementi_nonrilevanti) > 0):

            titolo = "HELP🆘\n\n- Puoi utilizzare i bottoni per avere una scelta rapida, situati affianco alla graffetta in basso a destra, ma anche la tastiera per inserire campi più lunghi \n\n -Puoi modificare la posizione in cui ti trovi, dopo che hai concluso una ricerca, premendo il pulsante\n(Modifica la posizione o le modalità di viaggio ⚙️->Modifica la posizione 🌐) \n\n -Puoi modificare le modalità di viaggio, ovvero  se viaggi in coppia, solo o con la famiglia, dopo che hai concluso una ricerca, premendo il pulsante\Modifica la posizione o le modalità di viaggio ⚙️ -> Modifica modalità di viaggio 🧭) \n\n -Fornire dei feedback (👍/👎) sui risultati, ti garantisce la possibilità di ottenere dei consigliati in base ai tuoi gusti, dopo che hai concluso una ricerca, premendo il pulsante\n(In base ai tuoi gusti 🔖) "+"\n\nPer continuare seleziona un opzione qui sotto 👇"
            nuovaricerca = 'Nuova ricerca 🔍'
            TuoiGusti = 'In base ai tuoi gusti 🔖'
            Impostazioni = 'Modifica la posizione o le modalità di viaggio ⚙️'
            data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                    [{"text": TuoiGusti, "callback_data": "In base ai tuoi gusti 🔖"}],
                                    [{"text": Impostazioni, "callback_data": "Modifica la posizione o le modalità di viaggio"}]]}}},
                                             "platform": "TELEGRAM"},
                                            {"text": {"text": [""]}}]}
            payload = json.dumps(data)

            #Modulo di salvataggio dei dati

            temp = copy.deepcopy(utenti[pos])   #con queste operazioni salvo l'oggetto presente nell'array sul database
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag=db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp,etag)

            return payload

        else:

            titolo = "HELP🔰\n\n- Puoi utilizzare i bottoni per avere una scelta rapida, situati affianco alla graffetta in basso a destra, ma anche la tastiera per inserire campi più lunghi \n\n -Puoi modificare la posizione in cui ti trovi, dopo che hai concluso una ricerca, premendo il pulsante\n(Modifica la posizione o le modalità di viaggio ⚙️->Modifica la posizione 🌐) \n\n -Puoi modificare le modalità di viaggio, ovvero  se viaggi in coppia, solo o con la famiglia, dopo che hai concluso una ricerca, premendo il pulsante\n(Modifica la posizione o le modalità di viaggio ⚙ -> Modifica modalità di viaggio 🧭) \n\n -Fornire dei feedback (👍/👎) sui risultati, ti garantisce la possibilità di ottenere dei consigliati in base ai tuoi gusti, dopo che hai concluso una ricerca, premendo il pulsante\n(In base ai tuoi gusti 🔖) " + "\n\nPer continuare seleziona un opzione qui sotto 👇"
            nuovaricerca = "Nuova ricerca 🔍"
            Impostazioni = 'Modifica la posizione o le modalità di viaggio ⚙️'
            data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                    [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}],
                                    ]}}},
                                             "platform": "TELEGRAM"},
                                            {"text": {"text": [""]}}]}
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp,etag)

        return payload

    if (query_result.get('action') == 'input.welcome'):
        key=db.child("utenti").child(chat_id).child("PosizioneInLista").get()
        key=key.val()
        key=str(key)
        if(key=="None"):
            new_user = utente()
            new_user.chat_id = chat_id
            utenti.append(new_user)
            etag1=db.child("utenti").child(chat_id).child("PosizioneInLista").get()
            db.child("utenti").child(chat_id).child("PosizioneInLista").set(indice,etag1)
            indice = indice + 1

            prima_riga = '\n\n-Cercare luoghi di tuo interesse a Bari, Polignano a Mare, Monopoli e Mola di Bari 🔎 ' + "\n(es.luoghi storici, ristoranti, spiagge, bar ecc.)"
            seconda_riga = '\n\n-Consigliarti nuovi luoghi in base ai tuoi gusti 🔖' + '\n\n Se hai bisogno di aiuto 🆘 digita  "/help" '
            terzariga = '\n\n<b>Prima di tutto dimmi come ti chiami?</b> 😊'
            Titolo = "Ciao 👋 ! Io sono SACbot, il tuo chatbot di supporto alla scoperta di nuovi luoghi! 🤖\n\nEcco cosa posso fare: " + prima_riga + seconda_riga + terzariga

            data = {"fulfillmentMessages": [
                {
                    "payload": {
                        "telegram": {
                            "text": Titolo,
                            "parse_mode": "html"
                        }

                    },
                    "platform": "TELEGRAM"
                }
            ]
            }

            payload = json.dumps(data)

            pos = db.child("utenti").child(chat_id).child("PosizioneInLista").get()
            pos = pos.val()
            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            print(payload)
            return payload


        else:
            del utenti[pos]
            new_user = utente()
            new_user.chat_id = chat_id
            utenti.insert(pos,new_user)

            etag=db.child("utenti").child(chat_id).child("PosizioneInLista").get()
            db.child("utenti").child(chat_id).child("PosizioneInLista").set(pos,etag)

            pos = db.child("utenti").child(chat_id).child("PosizioneInLista").get()
            pos = pos.val()
            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag1=db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp,etag1)


            prima_riga = '\n\n-Cercare luoghi di tuo interesse a Bari, Polignano a Mare, Monopoli e Mola di Bari 🔎 ' + "\n(es.luoghi storici, ristoranti, spiagge, bar ecc.)"
            seconda_riga = '\n\n-Consigliarti nuovi luoghi in base ai tuoi gusti 🔖' + '\n\n Se hai bisogno di aiuto 🆘 digita  "/help" '
            terzariga = '\n\n<b>Prima di tutto dimmi come ti chiami?</b> 😊'
            Titolo = "Ciao 👋 ! Io sono SACbot, il tuo chatbot di supporto alla scoperta di nuovi luoghi! 🤖\n\nEcco cosa posso fare: " + prima_riga + seconda_riga + terzariga
            data = {"fulfillmentMessages": [
                {
                    "payload": {
                        "telegram": {
                            "text": Titolo,
                            "parse_mode": "html"
                        }

                    },
                    "platform": "TELEGRAM"
                }
            ]
            }

            payload = json.dumps(data)
            print(payload)
            return payload


    if (query_result.get('action') == 'altra_descrizione'):
        finale = "Dimmi cosa ti interessa? 😀 "
        utenti[pos].presoaltro='true'
        data={
            "fulfillmentMessages": [
                {
                    "text": {
                        "text": [
                            finale
                        ]
                    }
                }
            ]
        }
        payload=json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)

        return payload

    if (query_result.get('action') == 'tiporistorante'):

        utenti[pos].TerminidaCercare.append("gastronomia")
        data=''
        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)


        return payload


    if (query_result.get('action') == 'trovare_direttamente_luogo'):

        descrizione_diretta=query_result['parameters']['prenderedescrizione']

        if isinstance(descrizione_diretta, list):
            descrizione_diretta = descrizione_diretta[0].lower()
        else:
            descrizione_diretta.lower()


        if(utenti[pos].ideale!=''):
            utenti[pos].TerminidaCercare.append(utenti[pos].ideale)

        utenti[pos].TerminidaCercare.append(utenti[pos].pos) #Warning inserisce due volte la stessa posizione quando l'utente fa per la prima volta direttamenteluogo
        utenti[pos].TerminidaCercare.append(descrizione_diretta)
        utenti[pos].interessi.append(descrizione_diretta)

        utenti[pos].TerminidaCercare= list(set(utenti[pos].TerminidaCercare))#elimina i duplicati causati dal problema precedente nella lsita

        print(utenti[pos].TerminidaCercare)
        (num_stesso_paese, indiciposizione) = fai_raccomandazione(pos, utenti[
            pos].TerminidaCercare)
        utenti[pos].gusti = ''
        #(indiciposizione,num_stesso_paese) = sort_per_paese(indiciposizione,utenti[pos].pos)

        utenti[pos].globalindici = indiciposizione

        nome1=df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        linksito1 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        linkimm1 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        descrizione1 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        posizioe1 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        punteggio1 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        indicazioni1 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        numero_rec1=df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        numero_rec1=int(float(numero_rec1))
        testo_prima = "Nome:"+" "+nome1.title()+"\n"+"Descrizione: " + " " + descrizione1.title() + "\n" + "Posizione: " + " " + posizioe1.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio1 + "\n"+"Numero recensioni su Tripadvisor: "+str(numero_rec1)+"\n"
        utenti[pos].j = utenti[pos].j + 1

        nome2 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        linksito2 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        linkimm2 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        descrizione2 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        posizioe2 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        punteggio2 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        indicazioni2 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        numero_rec2 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        numero_rec2=int(float(numero_rec2))
        testo_seconda = "Nome:"+" "+nome2.title()+"\n"+"Descrizione: " + " " + descrizione2.title() + "\n" + "Posizione: " + " " + posizioe2.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio2 + "\n"+"Numero recensioni su Tripadvisor: "+str(numero_rec2)+"\n"
        utenti[pos].j = utenti[pos].j + 1

        nome3 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        linksito3 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        linkimm3 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        descrizione3 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        posizioe3 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        punteggio3 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        indicazioni3 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        numero_rec3 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
        numero_rec3=int(float(numero_rec3))
        testo_terza = "Nome:"+" "+nome3.title()+"\n"+"Descrizione: " + " " + descrizione3.title() + "\n" + "Posizione: " + " " + posizioe3.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio3 + "\n"+"Numero recensioni su Tripadvisor: "+str(numero_rec3)+"\n"
        utenti[pos].j = utenti[pos].j + 1

        if(utenti[pos].pos=="polignanoamare"):
            temp ="Polignano a Mare"
        elif(utenti[pos].pos=="mola"):
            temp="Mola"
        elif (utenti[pos].pos == "monopoli"):
            temp="Monopoli"
        elif (utenti[pos].pos == "bari"):
            temp="Bari"

        text_corrispondenti = query_result['queryText']
        if (text_corrispondenti == 'Non Modificare'):
            text_corrispondenti = utenti[pos].inseritodautente

        testo = "A " + utenti[pos].pos + " ho trovato alcuni risultati  corrispondenti \na :" + '"{}"'.format(text_corrispondenti) + " 🔍 , i restanti si trovano \nnei comuni limitrofi." + " \n\n*Vuoi vedere altri elementi simili in base alla ricerca che hai appena fatto?* 😀"

        data = {"fulfillmentMessages": [{"image": {"imageUri": linkimm1}, "platform": "TELEGRAM"}, {
            "card": {"title": testo_prima, "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                     "buttons": [{"text": "ㅤ👍"}, {"text": "‎ 👎"}, {"text": "info ℹ️", "postback": linksito1},
                                 {"text": "pos 📍 ", "postback": indicazioni1}]}, "platform": "TELEGRAM"},
                                        {"image": {"imageUri": linkimm2}, "platform": "TELEGRAM"}, {
                                            "card": {"title": testo_seconda,
                                                     "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                                                     "buttons": [{"text": "‎ 👍"}, {"text": "👎ㅤ"},
                                                                 {"text": "info ℹ️", "postback": linksito2},
                                                                 {"text": "pos 📍 ",
                                                                  "postback": indicazioni2}]},
                                            "platform": "TELEGRAM"},
                                        {"image": {"imageUri": linkimm3}, "platform": "TELEGRAM"}, {
                                            "card": {"title": testo_terza,
                                                     "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                                                     "buttons": [{"text": "‏‏‎ ‎👍"}, {"text": "👎‏‏‎ ‎‎"},
                                                                 {"text": "info ℹ️", "postback": linksito3},
                                                                 {"text": "pos 📍 ",
                                                                  "postback": indicazioni3}]},
                                            "platform": "TELEGRAM"},
                                        {
                                            "payload": {
                                                "telegram": {
                                                    "text": testo,
                                                    "parse_mode":"markdown",
                                                    "reply_markup": {
                                                        "inline_keyboard": [
                                                            [
                                                                {
                                                                    "text": "Si",
                                                                    "callback_data": "Si"
                                                                }
                                                            ],
                                                            [
                                                                {
                                                                    "text": "No",
                                                                    "callback_data": "No"
                                                                }
                                                            ]
                                                        ]
                                                    }
                                                }

                                            },
                                            "platform": "TELEGRAM"
                                        },
                                        ],
                }

        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)

        return payload



    if (utenti[pos].prendere_descrizione == "attivato"):
        start = time.time()
        if(query_result.get('action') != 'aggiornaredati' and query_result.get('action')!='positionbutton' and query_result.get('action')!='button_cambiaidealeper' and query_result.get('action')!='xgusti' and query_result.get('action')!='cambia_pos_nome_luogo' ):
            if (utenti[pos].presoaltro == 'true'):
                Altro_desc=query_result['parameters']['any'][0].lower()
                Altro_desc = Altro_desc.replace('"', ' ')
                Altro_desc = " ".join([" ".join(n.split()) for n in
                                       Altro_desc.lower().split(',')])
                if(Altro_desc!= "salta"):
                    utenti[pos].TerminidaCercare.append(Altro_desc)
                    utenti[pos].presoaltro = ''
                    utenti[pos].prendere_descrizione = ''
                    utenti[pos].inseritodautente = Altro_desc
                else:
                    utenti[pos].presoaltro = ''
                    utenti[pos].prendere_descrizione = ''
                    utenti[pos].inseritodautente = ''


            else:
                pres=False
                for item in utenti[pos].TerminidaCercare:
                    if(item==utenti[pos].pos):
                        pres=True


                if(pres==False):
                    utenti[pos].TerminidaCercare.append(utenti[pos].pos)

                utenti[pos].prendere_descrizione = ''

                try:
                    descrizione = query_result['parameters']['prenderedescrizione'].lower()
                    descrizione = " ".join([" ".join(n.split()) for n in
                                            descrizione.lower().split(',')])
                except Exception as e:
                    descrizione = query_result['parameters']['prendereideale_per']

                if(descrizione!="salta"):
                    utenti[pos].desc = descrizione
                    utenti[pos].inseritodautente = descrizione
                    utenti[pos].TerminidaCercare.append(descrizione)
                    # utenti[pos].interessi.append(descrizione)
                else:
                    utenti[pos].desc = ''
                    utenti[pos].inseritodautente = ''

            print(utenti[pos].TerminidaCercare)

            # CHIAMA IL METODO PER LA RACCOMANDAZIONE
            if (len(utenti[pos].TerminidaCercare) >= 1):
                (num_stesso_paese, indiciposizione) = fai_raccomandazione(pos, utenti[
                    pos].TerminidaCercare)
                utenti[pos].gusti = ''
                utenti[pos].globalindici = indiciposizione

                nome1 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linksito1 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linkimm1 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                descrizione1 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                posizioe1 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                punteggio1 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                indicazioni1 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                numero_rec1 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                numero_rec1=int(float(numero_rec1))
                testo_prima = "Nome:"+" "+nome1.title()+"\n"+"Descrizione: " + " " + descrizione1.title() + "\n" + "Posizione: " + " " + posizioe1.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio1 + "\n"+"Numero recensioni su Tripadvisor: "+str(numero_rec1)+"\n"
                utenti[pos].j = utenti[pos].j + 1

                nome2 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linksito2 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linkimm2 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                descrizione2 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                posizioe2 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                punteggio2 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                indicazioni2 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                numero_rec2 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                numero_rec2=int(float(numero_rec2))
                testo_seconda = "Nome:"+" "+nome2.title()+"\n"+"Descrizione: " + " " + descrizione2.title() + "\n" + "Posizione: " + " " + posizioe2.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio2 + "\n"+"Numero recensioni su Tripadvisor: "+str(numero_rec2)+"\n"
                utenti[pos].j = utenti[pos].j + 1

                nome3 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linksito3 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linkimm3 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                descrizione3 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                posizioe3 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                punteggio3 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                indicazioni3 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                numero_rec3 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                numero_rec3=int(float(numero_rec3))
                testo_terza = "Nome:"+" "+nome3.title()+"\n"+"Descrizione: " + " " + descrizione3.title() + "\n" + "Posizione: " + " " + posizioe3.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio3 + "\n"+"Numero recensioni su Tripadvisor: "+str(numero_rec3)+"\n"
                utenti[pos].j = utenti[pos].j + 1

                text_corrispondenti = query_result['queryText']
                if (text_corrispondenti == 'Non Modificare'):
                    text_corrispondenti = utenti[pos].inseritodautente

                testo = "A " + utenti[pos].pos + " ho trovato alcuni risultati  corrispondenti \na :" + '"{}"'.format(
                    text_corrispondenti) + " 🔍 , i restanti si trovano\nnei comuni limitrofi."  + " \n\n*Vuoi vedere altri elementi simili in base alla ricerca che hai appena fatto?* 😀"

                data = {"fulfillmentMessages": [{"image": {"imageUri": linkimm1}, "platform": "TELEGRAM"}, {
                    "card": {"title": testo_prima, "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                             "buttons": [{"text": "ㅤ👍"}, {"text": "‎ 👎"}, {"text": "info ℹ️", "postback": linksito1},
                                         {"text": "pos 📍 ", "postback": indicazioni1}]}, "platform": "TELEGRAM"},
                                                {"image": {"imageUri": linkimm2}, "platform": "TELEGRAM"}, {
                                                    "card": {"title": testo_seconda,
                                                             "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                                                             "buttons": [{"text": "‎ 👍"}, {"text": "👎ㅤ"},
                                                                         {"text": "info ℹ️", "postback": linksito2},
                                                                         {"text": "pos 📍 ",
                                                                          "postback": indicazioni2}]},
                                                    "platform": "TELEGRAM"},
                                                {"image": {"imageUri": linkimm3}, "platform": "TELEGRAM"}, {
                                                    "card": {"title": testo_terza,
                                                             "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                                                             "buttons": [{"text": "‏‏‎ ‎👍"}, {"text": "👎‏‏‎ ‎‎"},
                                                                         {"text": "info ℹ️", "postback": linksito3},
                                                                         {"text": "pos 📍 ",
                                                                          "postback": indicazioni3}]},
                                                    "platform": "TELEGRAM"},
                                                {
                                                    "payload": {
                                                        "telegram": {
                                                            "text": testo,
                                                            "parse_mode": "markdown",
                                                            "reply_markup": {
                                                                "inline_keyboard": [
                                                                    [
                                                                        {
                                                                            "text": "Si",
                                                                            "callback_data": "Si"
                                                                        }
                                                                    ],
                                                                    [
                                                                        {
                                                                            "text": "No",
                                                                            "callback_data": "No"
                                                                        }
                                                                    ]
                                                                ]
                                                            }
                                                        }

                                                    },
                                                    "platform": "TELEGRAM"
                                                },
                                                ],
                        }

                payload = json.dumps(data)

                temp = copy.deepcopy(utenti[pos])
                temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
                temp = json.dumps(temp)
                etag = db.child("oggetti_utenti").child(chat_id).get()
                db.child("oggetti_utenti").child(chat_id).set(temp, etag)

                return payload

        else:
            utenti[pos].prendere_descrizione = ""
            utenti[pos].ultima_attivita=""

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)


    if (query_result.get('action') == 'activatericomincia'):
        start = time.time()
        utenti[pos].TerminidaCercare.clear()
        utenti[pos].j = 0
        utenti[pos].k = 0
        utenti[pos].primomipiace=''
        utenti[pos].secondomipaice=''
        utenti[pos].terzomipiace=''

        if (len(utenti[pos].elementi_rilevanti) > 0 and len(utenti[pos].elementi_nonrilevanti) > 0):

            titolo = "Cosa ti piacerebbe fare ? 😀 "
            nuovaricerca = 'Nuova ricerca 🔍'
            TuoiGusti = 'In base ai tuoi gusti 🔖'
            Impostazioni='Modifica la posizione o le modalità di viaggio ⚙️'
            data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                    [{"text": TuoiGusti, "callback_data": "In base ai tuoi gusti 🔖"}],
                                    [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}]]}}},
                                             "platform": "TELEGRAM"},
                                            {"text": {"text": [""]}}]}

            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload

        else:

            titolo = "Cosa ti piacerebbe fare ? 😀 "
            nuovaricerca = "Nuova ricerca 🔍"
            Impostazioni = 'Modifica la posizione o le modalità di viaggio ⚙️'
            data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                    [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}],
                                    ]}}},
                                             "platform": "TELEGRAM"},
                                            {"text": {"text": [""]}}]}

            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)
            end = time.time()
            print(end - start)
            return payload


    if (query_result.get('action') =='nuovaricerca'):
        #utenti[pos].TerminidaCercare.clear()
        utenti[pos].TerminidaCercare =  utenti[pos].TerminidaCercare[4:]
        utenti[pos].j=0
        utenti[pos].k=0

        titolo = utenti[pos].Nome + "sono felice  di aiutarti per una nuova ricerca ! \n" + "\n    <b>Se hai già in mente cosa cercare, digitalo in chat\n                     es:.(bar,ristoranti,spiagge)</b> 🔎    \n"+"\nAltrimenti,ti faró delle domande 👇 per suggerirti attività che potrebbero piacerti"
        Passatempo = 'PuntiInteresse'
        Storia = 'Storia'
        Gastronomia = 'Gastronomia'
        data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo,"parse_mode":"html", "reply_markup": {
            "inline_keyboard": [[{"text": Passatempo, "callback_data": "PuntiInteresse"}],
                                [{"text": Storia, "callback_data": "Storia"}],
                                [{"text": Gastronomia, "callback_data": "Gastronomia"}]]}}}, "platform": "TELEGRAM"},
                                        ]}
        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)

        return payload


    if (query_result.get('action') =='aggiornaredati'):

        titolo='Seleziona cosa vuoi modificare 🛠️'
        cambiaPosizione='Modifica la posizione 🌐'
        cambiaIdealeper='Modifica modalità di viaggio 🧭'
        data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
            "inline_keyboard": [[{"text": cambiaPosizione, "callback_data": "Modifica la posizione"}],
                                [{"text": cambiaIdealeper, "callback_data": "Modifica modalità di viaggio"}],
                                ]}}},
                                         "platform": "TELEGRAM"},
                                        {"text": {"text": [""]}}]}


        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)

        return payload


    if (query_result.get('action') == 'positionbutton'):

        finale='Inserisci una nuova posizione 👇'
        utenti[pos].cambiapos='true'
        utenti[pos].new_elementi_rilev.clear()
        utenti[pos].new_elementi_nonrilev.clear()
        data = {"fulfillmentMessages": [
            {"quickReplies": {"title": finale, "quickReplies": ["Monopoli", "Polignano", "Mola", "Bari"]},
             "platform": "TELEGRAM"}]}
        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)

        return payload


    if (query_result.get('action') == 'button_cambiaidealeper'):
        finale = 'Inserisci se viaggi da solo, in coppia o con la famiglia 👇'
        utenti[pos].cambiaidealeper = 'true'
        utenti[pos].new_elementi_rilev.clear()
        utenti[pos].new_elementi_nonrilev.clear()
        data = {"fulfillmentMessages": [
            {"quickReplies": {"title": finale, "quickReplies": ["Solo", "Coppia", "Famiglia"]},
             "platform": "TELEGRAM"}]}
        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)

        return payload


    if (query_result.get('action') == 'get_name'):

        Nome = json.dumps(query_result['queryText'])
        Nome = Nome.replace('"', ' ')
        Nome = Nome.title()

        utenti[pos].Nome = Nome #Metto il campo nome dell'utente

        Titolo = 'Piacere di conoscerti ' + utenti[pos].Nome + ' 🤝, mi puoi dire in che fascia di età ti trovi?'
        data = {"fulfillmentMessages": [{"quickReplies": {"title": Titolo, "quickReplies": ["18-30", "31-50", "51-70", "Più di 70", "Salta"]},  "platform": "TELEGRAM"}]}
        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)

        return payload


    if (query_result.get('action') == 'geteta'):

        Eta = query_result['parameters']['number']
        if(Eta!="salta"or "Salta"):
            utenti[pos].Eta = Eta

        else:
            utenti[pos].Eta=''


        finale = "Mi puoi dire in che città vuoi fare la tua ricerca? Per ora conosco solo Monopoli, Polignano a Mare, Mola di Bari e Bari 🌐 " +"\n\n(Ricorda che puoi anche utilizzare i bottoni ↘️ affianco alla graffetta 📎)"
        data = {"fulfillmentMessages": [
            {"quickReplies": {"title": finale, "quickReplies": ["Monopoli", "Polignano", "Mola", "Bari"]},
             "platform": "TELEGRAM"}]}

        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)

        return payload


    if(query_result.get('action') == 'getpos'):
        if(utenti[pos].cambiapos=='true'):
            utenti[pos].TerminidaCercare.clear()
            posizione = query_result['parameters']['location']['city']
            posizione = posizione.replace('"', ' ')
            posizione = " ".join([" ".join(n.split()) for n in
                                  posizione.lower().split(',')])
            posizione = posizione.replace(" ", "")

            if (len(utenti[pos].elementi_rilevanti) > 0 and len(utenti[pos].elementi_nonrilevanti) > 0):

                if(posizione==''):
                    titolo = "La posizione non è stata aggiornata correttamente," + "\ncosa ti piacerebbe fare ?  "

                else:
                    titolo = "Perfetto, la posizione è stata aggiornata" + "\ncosa ti piacerebbe fare ? 😀 "
                    utenti[pos].TerminidaCercare.append(posizione)
                    utenti[pos].pos = posizione



                nuovaricerca = 'Nuova ricerca 🔍'
                TuoiGusti = 'In base ai tuoi gusti 🔖'
                Impostazioni = 'Modifica la posizione o le modalità di viaggio ⚙️'
                data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                    "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                        [{"text": TuoiGusti, "callback_data": "In base ai tuoi gusti 🔖"}],
                                        [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}]]}}},
                                                 "platform": "TELEGRAM"},
                                                {"text": {"text": [""]}}]}
            else:
                if(posizione==''):
                    titolo = "La posizione non è stata aggiornata correttamente," + "\ncosa ti piacerebbe fare ?  "
                else:
                    titolo = "Perfetto, la posizione è stata aggiornata" + "\ncosa ti piacerebbe fare ? 😀 "
                    utenti[pos].TerminidaCercare.append(posizione)
                    utenti[pos].pos = posizione




                nuovaricerca = "Nuova ricerca 🔍"
                Impostazioni = 'Modifica la posizione o le modalità di viaggio ⚙'
                data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                    "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                        [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}],
                                        ]}}},
                                                 "platform": "TELEGRAM"},
                                                {"text": {"text": [""]}}]}

            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload

        else:
            posizione = query_result['parameters']['location']['city']
            posizione = posizione.replace('"', ' ')
            posizione = " ".join([" ".join(n.split()) for n in
                                  posizione.lower().split(',')])
            temp=posizione
            posizione=posizione.replace(" ","")


            if(posizione==''):
                titolo = "La posizione inserita non è valida" + ", per cambiarla, digita 'Modifica la posizione o le modalità di viaggio'" + "\n" + "\n <b>Altirmenti se vuoi comunque proseguire nella ricerca e hai già in mente cosa cercare, digitalo in chat.</b> " + "\nOppure, ti faró delle domande 👇 per suggerirti attività che potrebbero piacerti"
            else:
                utenti[pos].TerminidaCercare.append(posizione)
                utenti[pos].pos = posizione
                titolo = "Perfetto, da ora in poi i risultati che ti mostrerò saranno relativi a: " + temp + ", per cambiarla, più in avanti troverai il bottone 'Modifica la posizione o le modalità di viaggio ⚙'" + "\n" + "\n    <b>Se hai già in mente cosa cercare, digitalo in chat\n                     es:.(bar, ristoranti, spiagge)</b> 🔎    \n" + "\nAltrimenti, ti faró delle domande 👇 per suggerirti attività che potrebbero piacerti"


            Passatempo = 'PuntiInteresse'
            Storia = 'Storia'
            Gastronomia = 'Gastronomia'
            data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo,"parse_mode":"html", "reply_markup": {
                "inline_keyboard": [[{"text": Passatempo, "callback_data": "PuntiInteresse"}],
                                    [{"text": Storia, "callback_data": "Storia"}],
                                    [{"text": Gastronomia, "callback_data": "Gastronomia"}]]}}},
                                             "platform": "TELEGRAM"},
                                            ]}
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload


    if (query_result.get('action') == 'Prendereattivita'):

        attivita = query_result.get('queryText')
        attivita = " ".join([" ".join(n.split()) for n in
                             attivita.lower().split(',')])
        utenti[pos].TerminidaCercare.append(attivita)
        utenti[pos].attivita=attivita
        attivita = attivita.title()


        if (utenti[pos].ultima_attivita == utenti[pos].attivita and utenti[pos].ideale and utenti[pos].inseritodautente!=''):  # Se l'utente in precedenza ha gia selezionato attivita e lo fa anche ora e allo stesso tempo ha già fornito->
            utenti[pos].TerminidaCercare.append(utenti[pos].ideale)  # ->l'ideale_per gli fornisco direttamente la raccomandazione
            utenti[pos].TerminidaCercare.append(utenti[pos].pos)

            Titolo="Hai impostato il filtro 🎯:"+utenti[pos].inseritodautente +", vuoi modificarlo ? "
            one_button='Modificare'
            two_button='Non Modificare'

            data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": Titolo, "reply_markup": {
                "inline_keyboard": [[{"text": one_button, "callback_data": 'Modificare'}],
                                    [{"text": two_button, "callback_data": 'Non Modificare'}],
                                    ]}}},
                                             "platform": "TELEGRAM"},
                                            {"text": {"text": [""]}}]}

            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload

        else:

            utenti[pos].ultima_attivita=attivita.lower()
            utenti[pos].new_elementi_rilev.clear()
            utenti[pos].new_elementi_nonrilev.clear()

            finale = "Hai scelto 👉 " + utenti[pos].attivita + " 👈" + ", per poterti consigliare meglio mi diresti se viaggi da solo, in coppia o con la famiglia ?" +'\nScrivi "salta" se non vuoi dirmelo' +"\n\n(Ricorda che puoi anche utilizzare i bottoni ↘️ affianco alla graffetta 📎)"

            if (utenti[pos].ideale):  # se ha già preso dall'utente il fatto che viaggia in coppia è inutile richiederlo
                utenti[pos].TerminidaCercare.append(utenti[pos].ideale)
                utenti[pos].TerminidaCercare.append(utenti[pos].pos)
                if (utenti[pos].attivita == 'puntiinteresse'):
                    esempio = "Luoghi e punti d'interesse,Spiagge,Cinema"
                    one_button = "Luoghi e punti d'interesse"
                    two_button = 'Spiagge'
                    three_button = 'Cinema'
                    eight_button = 'Altro'
                    finale = "Un ultimissima cosa 🙏 ,inserisci una descrizione di cosa cerchi? (es.:" + esempio + ")"
                    data = {"fulfillmentMessages": [
                        {"quickReplies": {"title": finale,
                                          "quickReplies": [one_button, two_button, three_button, eight_button]},
                         "platform": "TELEGRAM"}]}

                if (utenti[pos].attivita == 'gastronomia'):
                    esempio = 'Pizza,Pesce,Steakhoue,Mediterranea,Bar'
                    one_button = 'Pizza'
                    two_button = 'Pesce'
                    three_button = 'Steakhouse'
                    four_button='Mediterranea'
                    fifth_button='Bar'
                    six_button='Sushi'
                    seven_button='Vegetariana'
                    eight_button = 'Top 5'
                    nine_button = 'Altro'
                    finale = "Un'ultima cosa, inserisci una descrizione di cosa cerchi (es.:" + esempio + ")"
                    data = {"fulfillmentMessages": [
                        {"quickReplies": {"title": finale,
                                          "quickReplies": [one_button, two_button, three_button, four_button,
                                                           fifth_button, six_button, seven_button, eight_button, nine_button]},
                         "platform": "TELEGRAM"}]}

                if (utenti[pos].attivita == 'storia'):
                    esempio = "Siti Religiosi e Siti Storici"
                    one_button = 'Siti Religiosi'
                    two_button = 'Siti Storici'
                    eight_button = 'Altro'

                    finale = "Un ultimissima cosa 🙏 ,inserisci una descrizione di cosa cerchi? (es.:" + esempio + ")"
                    data = {"fulfillmentMessages": [
                        {"quickReplies": {"title": finale,
                                          "quickReplies": [one_button, two_button, eight_button]},
                         "platform": "TELEGRAM"}]}


                utenti[pos].prendere_descrizione = "attivato"  # in questo caso deve prendere la descrizione per fare la ricerca

                payload = json.dumps(data)

                temp = copy.deepcopy(utenti[pos])
                temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
                temp = json.dumps(temp)
                etag = db.child("oggetti_utenti").child(chat_id).get()
                db.child("oggetti_utenti").child(chat_id).set(temp, etag)
                return payload


            else:
                data = {"fulfillmentMessages": [
                    {"quickReplies": {"title": finale, "quickReplies": ["Solo", "Coppia", "Famiglia"]},
                     "platform": "TELEGRAM"}]}
                payload = json.dumps(data)

                temp = copy.deepcopy(utenti[pos])
                temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
                temp = json.dumps(temp)
                etag = db.child("oggetti_utenti").child(chat_id).get()
                db.child("oggetti_utenti").child(chat_id).set(temp, etag)
                return payload


    if (query_result.get('action') == 'non_modificare'):

        utenti[pos].new_elementi_rilev = list(utenti[pos].elementi_rilevanti)
        utenti[pos].new_elementi_nonrilev = list(utenti[pos].elementi_nonrilevanti)

        # se ha fatto xgusti non copio gli array e non faccio raffinata
        if (utenti[pos].gusti == ''):

            utenti[pos].TerminidaCercare.append(utenti[pos].inseritodautente)
            print(utenti[pos].TerminidaCercare)
            if (len(utenti[pos].TerminidaCercare) == 4):
                (num_stesso_paese, indiciposizione) = fai_raccomandazione(pos, utenti[
                    pos].TerminidaCercare)
                utenti[pos].gusti = ''
                #(indiciposizione,num_stesso_paese) = sort_per_paese(indiciposizione, utenti[pos].pos)
                utenti[pos].globalindici = indiciposizione

                nome1=df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linksito1 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linkimm1 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                descrizione1 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                posizioe1 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                punteggio1 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                numero_rec1 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                indicazioni1=df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                testo_prima = "Nome:"+" "+nome1.title()+"\n"+"Descrizione: " + " " + descrizione1.title() + "\n" + "Posizione: " + " " + posizioe1.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio1 + "\n"+"Numero recensioni su Tripadvisor: "+numero_rec1
                utenti[pos].j = utenti[pos].j + 1

                nome2 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linksito2 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linkimm2 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                descrizione2 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                posizioe2 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                punteggio2 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                numero_rec2 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                indicazioni2 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                testo_seconda = "Nome:"+" "+nome2.title()+"\n"+"Descrizione: " + " " + descrizione2.title() + "\n" + "Posizione: " + " " + posizioe2.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio2 + "\n"+"Numero recensioni su Tripadvisor: "+numero_rec2
                utenti[pos].j = utenti[pos].j + 1

                nome3 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linksito3 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linkimm3 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                descrizione3 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                posizioe3 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                punteggio3 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                numero_rec3 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                indicazioni3 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                testo_terza = "Nome:"+" "+nome3.title()+"\n"+"Descrizione: " + " " + descrizione3.title() + "\n" + "Posizione: " + " " + posizioe3.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio3 + "\n"+"Numero recensioni su Tripadvisor: "+numero_rec3
                utenti[pos].j = utenti[pos].j + 1

                text_corrispondenti = query_result['queryText']
                if (text_corrispondenti == 'Non Modificare'):
                    text_corrispondenti = utenti[pos].inseritodautente

                testo = "A " + utenti[pos].pos + " ho trovato alcuni risultati corrispondenti \na :" + '"{}"'.format(text_corrispondenti) + " 🔍 , i restanti si trovano nei comuni\nlimitrofi."  + " \n\n*Vuoi vedere altri elementi simili in base alla ricerca che hai appena fatto?* 😀"

                data = {"fulfillmentMessages": [{"image": {"imageUri": linkimm1}, "platform": "TELEGRAM"}, {
                    "card": {"title": testo_prima, "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                             "buttons": [{"text": "ㅤ👍"}, {"text": "‎ 👎"}, {"text": "info ℹ️", "postback": linksito1},
                                         {"text": "pos 📍 ", "postback": indicazioni1}]}, "platform": "TELEGRAM"},
                                                {"image": {"imageUri": linkimm2}, "platform": "TELEGRAM"}, {
                                                    "card": {"title": testo_seconda,
                                                             "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                                                             "buttons": [{"text": "‎ 👍"}, {"text": "👎ㅤ"},
                                                                         {"text": "info ℹ️", "postback": linksito2},
                                                                         {"text": "pos 📍 ",
                                                                          "postback": indicazioni2}]},
                                                    "platform": "TELEGRAM"},
                                                {"image": {"imageUri": linkimm3}, "platform": "TELEGRAM"}, {
                                                    "card": {"title": testo_terza,
                                                             "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                                                             "buttons": [{"text": "‏‏‎ ‎👍"}, {"text": "👎‏‏‎ ‎‎"},
                                                                         {"text": "info ℹ️", "postback": linksito3},
                                                                         {"text": "pos 📍 ",
                                                                          "postback": indicazioni3}]},
                                                    "platform": "TELEGRAM"},
                                                {
                                                    "payload": {
                                                        "telegram": {
                                                            "text": testo,
                                                            "parse_mode": "markdown",
                                                            "reply_markup": {
                                                                "inline_keyboard": [
                                                                    [
                                                                        {
                                                                            "text": "Si",
                                                                            "callback_data": "Si"
                                                                        }
                                                                    ],
                                                                    [
                                                                        {
                                                                            "text": "No",
                                                                            "callback_data": "No"
                                                                        }
                                                                    ]
                                                                ]
                                                            }
                                                        }

                                                    },
                                                    "platform": "TELEGRAM"
                                                },
                                                ],
                        }

                payload = json.dumps(data)

                temp = copy.deepcopy(utenti[pos])
                temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
                temp = json.dumps(temp)
                etag = db.child("oggetti_utenti").child(chat_id).get()
                db.child("oggetti_utenti").child(chat_id).set(temp, etag)

                return payload


        else:
            utenti[pos].new_elementi_rilev.clear()
            utenti[pos].new_elementi_nonrilev.clear()
            utenti[pos].TerminidaCercare.append(utenti[pos].inseritodautente)

            if (len(utenti[pos].TerminidaCercare) == 4):
                (num_stesso_paese, indiciposizione) = fai_raccomandazione(pos, utenti[
                    pos].TerminidaCercare)
                utenti[pos].gusti = ''
                utenti[pos].globalindici = indiciposizione

                nome1=df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linksito1 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linkimm1 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                descrizione1 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                posizioe1 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                punteggio1 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                indicazioni1 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                testo_prima = "Nome:"+" "+nome1.title()+"\n"+"Descrizione: " + " " + descrizione1.title() + "\n" + "Posizione: " + " " + posizioe1.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio1 + "\n"
                utenti[pos].j = utenti[pos].j + 1

                nome2 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linksito2 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linkimm2 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                descrizione2 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                posizioe2 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                punteggio2 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                indicazioni2 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                testo_seconda = "Nome:"+" "+nome2.title()+"\n"+"Descrizione: " + " " + descrizione2.title() + "\n" + "Posizione: " + " " + posizioe2.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio2 + "\n"
                utenti[pos].j = utenti[pos].j + 1

                nome3 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linksito3 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                linkimm3 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                descrizione3 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                posizioe3 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                punteggio3 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                indicazioni3 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
                testo_terza = "Nome:"+" "+nome3.title()+"\n"+"Descrizione: " + " " + descrizione3.title() + "\n" + "Posizione: " + " " + posizioe3.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio3 + "\n"
                utenti[pos].j = utenti[pos].j + 1

                text_corrispondenti=query_result['queryText']
                if(text_corrispondenti=='Non Modificare'):
                    text_corrispondenti=utenti[pos].inseritodautente

                testo = "A " + utenti[pos].pos + " ho trovato alcuni risultati  corrispondenti \na :" + '"{}"'.format(
                    text_corrispondenti) + " 🔍 , i restanti si trovano\nnei comuni limitrofi."  + " \n\n*Vuoi vedere altri elementi simili in base alla ricerca che hai appena fatto?* 😀"

                data = {"fulfillmentMessages": [{"image": {"imageUri": linkimm1}, "platform": "TELEGRAM"}, {
                    "card": {"title": testo_prima, "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                             "buttons": [{"text": "ㅤ👍"}, {"text": "‎ 👎"}, {"text": "info ℹ️", "postback": linksito1},
                                         {"text": "pos 📍 ", "postback": indicazioni1}]}, "platform": "TELEGRAM"},
                                                {"image": {"imageUri": linkimm2}, "platform": "TELEGRAM"}, {
                                                    "card": {"title": testo_seconda,
                                                             "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                                                             "buttons": [{"text": "‎ 👍"}, {"text": "👎ㅤ"},
                                                                         {"text": "info ℹ️", "postback": linksito2},
                                                                         {"text": "pos 📍 ",
                                                                          "postback": indicazioni2}]},
                                                    "platform": "TELEGRAM"},
                                                {"image": {"imageUri": linkimm3}, "platform": "TELEGRAM"}, {
                                                    "card": {"title": testo_terza,
                                                             "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                                                             "buttons": [{"text": "‏‏‎ ‎👍"}, {"text": "👎‏‏‎ ‎‎"},
                                                                         {"text": "info ℹ️", "postback": linksito3},
                                                                         {"text": "pos 📍 ",
                                                                          "postback": indicazioni3}]},
                                                    "platform": "TELEGRAM"},
                                                {
                                                    "payload": {
                                                        "telegram": {
                                                            "text": testo,
                                                            "parse_mode": "markdown",
                                                            "reply_markup": {
                                                                "inline_keyboard": [
                                                                    [
                                                                        {
                                                                            "text": "Si",
                                                                            "callback_data": "Si"
                                                                        }
                                                                    ],
                                                                    [
                                                                        {
                                                                            "text": "No",
                                                                            "callback_data": "No"
                                                                        }
                                                                    ]
                                                                ]
                                                            }
                                                        }

                                                    },
                                                    "platform": "TELEGRAM"
                                                },
                                                ],
                        }

                payload = json.dumps(data)

                temp = copy.deepcopy(utenti[pos])
                temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
                temp = json.dumps(temp)
                etag = db.child("oggetti_utenti").child(chat_id).get()
                db.child("oggetti_utenti").child(chat_id).set(temp, etag)

                return payload


    if (query_result.get('action') == 'ModificaDesc'):
        utenti[pos].new_elementi_rilev = list(utenti[pos].elementi_rilevanti)
        utenti[pos].new_elementi_nonrilev = list(utenti[pos].elementi_nonrilevanti)
        utenti[pos].new_elementi_rilev.clear()
        utenti[pos].new_elementi_nonrilev.clear()

        if (utenti[pos].attivita == 'puntiinteresse'):
            esempio = "Luoghi e punti d'interesse,Spiagge,Cinema"
            one_button = "Luoghi e punti d'interesse"
            two_button = 'Spiagge'
            three_button = 'Cinema'
            four_button = ''
            fifth_button = ''
            six_button = ''
            seven_button = ''
            eight_button = 'Altro'
            nine_button = ''

        if (utenti[pos].attivita == 'gastronomia'):
            esempio = 'Pizza,Pesce,Steakhoue,Mediterranea,Bar'
            one_button = 'Pizza'
            two_button = 'Pesce'
            three_button = 'Steakhouse'
            four_button='Mediterranea'
            fifth_button='Bar'
            six_button='Sushi'
            seven_button='Vegetariana'
            eight_button = 'Top 5'
            nine_button = 'Altro'

        if (utenti[pos].attivita == 'storia'):
            esempio = "Siti Religiosi e Siti Storici"
            one_button = 'Siti Religiosi'
            two_button = 'Siti Storici'
            three_button = ''
            four_button = ''
            fifth_button = ''
            six_button = ''
            seven_button = ''
            eight_button = 'Altro'
            nine_button = ''

        finale = "Inserisci una descrizione di cosa cerchi? 😃 " + "\n (es.:" + esempio + ")"+ '\n Scrivi "salta" se non vuoi dirmelo'
        data = {"fulfillmentMessages": [
            {"quickReplies": {"title": finale, "quickReplies": [one_button, two_button, three_button, four_button,fifth_button,six_button,seven_button,eight_button, nine_button]},
             "platform": "TELEGRAM"}]}
        utenti[pos].prendere_descrizione = "attivato"  # in questo caso deve prendere la descrizione per fare la ricerca

        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)

        return payload


    if (query_result.get('action') == 'Prendereidealeper'):
        if(utenti[pos].cambiaidealeper=='true'):
            idealeper = query_result.get('queryText')
            idealeper = " ".join([" ".join(n.split()) for n in
                                  idealeper.lower().split(',')])
            if (idealeper != "salta"):
                utenti[pos].ideale = idealeper
                utenti[pos].TerminidaCercare.append(idealeper)
            else:
                utenti[pos].ideale= ''


            if (len(utenti[pos].elementi_rilevanti) > 0 and len(utenti[pos].elementi_nonrilevanti) > 0):

                titolo = "Perfetto, le modalità di viaggio è stata aggiornata" + "\ncosa ti piacerebbe fare ? 😀 "
                nuovaricerca = 'Nuova ricerca 🔍'
                TuoiGusti = 'In base ai tuoi gusti 🔖'
                Impostazioni = 'Modifica la posizione o le modalità di viaggio ⚙'
                data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                    "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                        [{"text": TuoiGusti, "callback_data": "In base ai tuoi gusti 🔖"}],
                                        [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}]]}}},
                                                 "platform": "TELEGRAM"},
                                                {"text": {"text": [""]}}]}
            else:
                titolo = "Perfetto, le modalità di viaggio è stata aggiornata" + "\ncosa ti piacerebbe fare ? 😀 "
                nuovaricerca = "Nuova ricerca 🔍"
                Impostazioni = 'Modifica la posizione o le modalità di viaggio ⚙'
                data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                    "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                        [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}],
                                        ]}}},
                                                 "platform": "TELEGRAM"},
                                                {"text": {"text": [""]}}]}

            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload


        else:

            idealeper = query_result.get('queryText')
            idealeper = " ".join([" ".join(n.split()) for n in
                                  idealeper.lower().split(',')])

            if (idealeper != "salta"):

                utenti[pos].ideale = idealeper
                utenti[pos].TerminidaCercare.append(idealeper)
            else:
                utenti[pos].ideale=''


            if (utenti[pos].attivita == 'puntiinteresse'):
                esempio = "Luoghi e punti d'interesse,Spiagge,Cinema"
                one_button = "Luoghi e punti d'interesse"
                two_button = 'Spiagge'
                three_button = 'Cinema'
                four_button = ''
                fifth_button = ''
                six_button = ''
                seven_button = ''
                eight_button = 'Altro'
                nine_button = ''

            if (utenti[pos].attivita == 'gastronomia'):
                esempio = 'Pizza,Pesce,Steakhoue,Mediterranea,Bar'
                one_button = 'Pizza'
                two_button = 'Pesce'
                three_button = 'Steakhouse'
                four_button='Mediterranea'
                fifth_button='Bar'
                six_button='Sushi'
                seven_button='Vegetariana'
                eight_button = 'Top 5'
                nine_button = 'Altro'

            if (utenti[pos].attivita == 'storia'):
                esempio = "Siti Religiosi e Siti Storici"
                one_button = 'Siti Religiosi'
                two_button = 'Siti Storici'
                three_button = ''
                four_button = ''
                fifth_button = ''
                six_button = ''
                seven_button = ''
                eight_button = 'Altro'
                nine_button = ''

            finale = "Un ultimissima cosa 🙏 ,inserisci una descrizione di cosa cerchi? (es.:" + esempio + ")"
            data = {"fulfillmentMessages": [
                {"quickReplies": {"title": finale, "quickReplies": [one_button, two_button, three_button, four_button,fifth_button,six_button,seven_button,eight_button, nine_button]},
                 "platform": "rocchio_classifierTELEGRAM"}]}


            utenti[pos].prendere_descrizione="attivato" #in questo caso deve prendere la descrizione per fare la ricerca


            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload


    if (query_result.get('action') == 'primo_mipiace'):

        if(utenti[pos].primomipiace==''):

            utenti[pos].elementi_rilevanti.append(utenti[pos].globalindici[utenti[pos].k])
            utenti[pos].interessi.append(utenti[pos].desc)

            utenti[pos].primomipiace='True'

            numeroelemento = "Hai votato positivamente l'elemento:  1️⃣"
            #numeroelemento = numeroelemento + "\n\n Vuoi vedere altri elementi simili in base alla ricerca che hai appena fatto? 😀"

            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": numeroelemento, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload
        else:
            alert='Hai già premuto quel pulsante ❗'
            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": alert, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload


    if (query_result.get('action') == 'primo_nonmipiace'):
        if(utenti[pos].primomipiace==''):
            utenti[pos].elementi_nonrilevanti.append(utenti[pos].globalindici[utenti[pos].k])

            utenti[pos].primomipiace='True'

            numeroelemento = "Hai votato negativamente l'elemento:  1️⃣"
            #numeroelemento = numeroelemento + "\n\nVuoi vedere altri elementi simili in base alla ricerca che hai appena fatto?😀"

            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": numeroelemento, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload
        else:
            alert = 'Hai già premuto quel pulsante ❗'
            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": alert, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload


    if (query_result.get('action') == 'secondo_mipiace'):
        if(utenti[pos].secondomipaice==''):

            utenti[pos].elementi_rilevanti.append(utenti[pos].globalindici[utenti[pos].k+1])
            utenti[pos].interessi.append(utenti[pos].desc)

            utenti[pos].secondomipaice='True'

            numeroelemento = "Hai votato positivamente l'elemento:  2️⃣"
            #numeroelemento = numeroelemento + "\n\n Vuoi vedere altri elementi simili in base alla ricerca che hai appena fatto? 😀  "

            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": numeroelemento, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload
        else:
            alert = 'Hai già premuto quel pulsante ❗'
            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": alert, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload


    if (query_result.get('action') == 'secondononmipiace'):

        if(utenti[pos].secondomipaice==''):
            utenti[pos].elementi_nonrilevanti.append(utenti[pos].globalindici[utenti[pos].k + 1])

            utenti[pos].secondomipaice='True'

            numeroelemento = "Hai votato negativamente l'elemento:  2️⃣"
            #numeroelemento = numeroelemento + "\n\n Vuoi vedere altri elementi simili in base alla ricerca che hai appena fatto? 😀"

            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": numeroelemento, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload
        else:
            alert = 'Hai già premuto quel pulsante ❗'
            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": alert, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload



    if (query_result.get('action') == 'terzo_mipiace'):

        if(utenti[pos].terzomipiace==''):
            utenti[pos].elementi_rilevanti.append(utenti[pos].globalindici[utenti[pos].k+ 2])
            utenti[pos].interessi.append(utenti[pos].desc)

            utenti[pos].terzomipiace='True'

            numeroelemento = "Hai votato positivamente l'elemento:  3️⃣"
            #numeroelemento = numeroelemento + "\n\n Vuoi vedere altri elementi simili in base alla ricerca che hai appena fatto? 😀"

            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": numeroelemento, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload

        else:
            alert = 'Hai già premuto quel pulsante ❗'
            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": alert, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload


    if (query_result.get('action') == 'terzo_nonmipiace'):

        if(utenti[pos].terzomipiace==''):

            utenti[pos].elementi_nonrilevanti.append(utenti[pos].globalindici[utenti[pos].k + 2])

            utenti[pos].terzomipiace = 'True'

            numeroelemento = "Hai votato negativamente l'elemento:  3️⃣"
            #numeroelemento = numeroelemento + "\n\n Vuoi vedere altri elementi simili in base alla ricerca che hai appena fatto? 😀"

            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": numeroelemento, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload

        else:
            alert = 'Hai già premuto quel pulsante ❗'
            data = {"fulfillmentMessages": [
                {"payload": {"telegram": {"text": alert, "parse_mode": "Markdown"}},
                 "platform": "TELEGRAM"}], }
            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload


    if (query_result.get('action') == 'actioncontinua'):
        utenti[pos].primomipiace = ''
        utenti[pos].secondomipaice = ''
        utenti[pos].terzomipiace = ''


        if(utenti[pos].j<18):  #permette di scorrere gli elementi simili fino a 18 ovvero il numero di raccomandazioni calcolate

            utenti[pos].k=utenti[pos].k+1
            utenti[pos].k=utenti[pos].k+1
            utenti[pos].k=utenti[pos].k+1

            nome1=df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            linksito1 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            linkimm1 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            descrizione1 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            posizioe1 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            punteggio1 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            indicazioni1 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            numero_rec1 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            numero_rec1=int(float(numero_rec1))
            testo_prima = "Nome:"+" "+nome1.title()+"\n"+"Descrizione: " + " " + descrizione1.title() + "\n" + "Posizione: " + " " + posizioe1.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio1 + "\n"+"Numero recensioni su Tripadvisor: "+str(numero_rec1)+"\n"
            utenti[pos].j = utenti[pos].j + 1

            nome2 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            linksito2 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            linkimm2 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            descrizione2 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            posizioe2 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            punteggio2 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            indicazioni2 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            numero_rec2 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            numero_rec2=int(float(numero_rec2))
            testo_seconda = "Nome:"+" "+nome2.title()+"\n"+"Descrizione: " + " " + descrizione2.title() + "\n" + "Posizione: " + " " + posizioe2.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio2 + "\n"+"Numero recensioni su Tripadvisor: "+str(numero_rec2)+"\n"
            utenti[pos].j = utenti[pos].j + 1

            nome3 = df['Nome'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            linksito3 = df['Link'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            linkimm3 = df['Immagini'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            descrizione3 = df['Descrizione'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            posizioe3 = df['Paese'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            punteggio3 = df['Punteggio'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            indicazioni3 = df['Posizioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            numero_rec3 = df['Numero_Recensioni'].iloc[utenti[pos].globalindici[utenti[pos].j]]
            numero_rec3=int(float(numero_rec3))
            testo_terza = "Nome:"+" "+nome3.title()+"\n"+"Descrizione: " + " " + descrizione3.title() + "\n" + "Posizione: " + " " + posizioe3.title() + "\n" + "Punteggio su Tripadvisor: " + " " + punteggio3 + "\n"+"Numero recensioni su Tripadvisor: "+str(numero_rec3)+"\n"
            utenti[pos].j = utenti[pos].j + 1

            testo = "*Vuoi vedere altri elementi simili in base alla ricerca che hai appena fatto?* 😀"

            data = {"fulfillmentMessages": [{"image": {"imageUri": linkimm1}, "platform": "TELEGRAM"}, {
                "card": {"title": testo_prima, "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                         "buttons": [{"text": "ㅤ👍"}, {"text": "‎ 👎"}, {"text": "info ℹ️", "postback": linksito1},
                                     {"text": "pos 📍 ", "postback": indicazioni1}]}, "platform": "TELEGRAM"},
                                            {"image": {"imageUri": linkimm2}, "platform": "TELEGRAM"}, {
                                                "card": {"title": testo_seconda,
                                                         "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                                                         "buttons": [{"text": "‎ 👍"}, {"text": "👎ㅤ"},
                                                                     {"text": "info ℹ️", "postback": linksito2},
                                                                     {"text": "pos 📍 ",
                                                                      "postback": indicazioni2}]},
                                                "platform": "TELEGRAM"},
                                            {"image": {"imageUri": linkimm3}, "platform": "TELEGRAM"}, {
                                                "card": {"title": testo_terza,
                                                         "subtitle": "Metti 👍 se ti può interessare, altrimenti 👎 ",
                                                         "buttons": [{"text": "‏‏‎ ‎👍"}, {"text": "👎‏‏‎ ‎‎"},
                                                                     {"text": "info ℹ️", "postback": linksito3},
                                                                     {"text": "pos 📍 ",
                                                                      "postback": indicazioni3}]},
                                                "platform": "TELEGRAM"},
                                            {
                                                "payload": {
                                                    "telegram": {
                                                        "text": testo,
                                                        "parse_mode": "markdown",
                                                        "reply_markup": {
                                                            "inline_keyboard": [
                                                                [
                                                                    {
                                                                        "text": "Si",
                                                                        "callback_data": "Si"
                                                                    }
                                                                ],
                                                                [
                                                                    {
                                                                        "text": "No",
                                                                        "callback_data": "No"
                                                                    }
                                                                ]
                                                            ]
                                                        }
                                                    }

                                                },
                                                "platform": "TELEGRAM"
                                            },
                                            ],
                    }

            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload


        else:
            utenti[pos].TerminidaCercare.clear()

            if (len(utenti[pos].elementi_rilevanti) > 0 and len(utenti[pos].elementi_nonrilevanti) > 0):

                titolo = "Mi dispiace 🙁, non ci sono più elementi da visualizzare "+"\n\nCosa ti piacerebbe fare ? 😀 "
                nuovaricerca = 'Nuova ricerca 🔍'
                TuoiGusti = 'In base ai tuoi gusti 🔖'
                Impostazioni = 'Modifica la posizione o le modalità di viaggio ⚙️'
                data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                    "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                        [{"text": TuoiGusti, "callback_data": "In base ai tuoi gusti 🔖"}],
                                        [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}]]}}},
                                                 "platform": "TELEGRAM"},
                                                {"text": {"text": [""]}}]}

            else:

                titolo = "Mi dispiace 🙁, non ci sono più elementi da visualizzare "+"\n\nCosa ti piacerebbe fare ? 😀 "
                nuovaricerca = "Nuova ricerca 🔍"
                Impostazioni = 'Modifica la posizione o le modalità di viaggio ⚙️'
                data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
                    "inline_keyboard": [[{"text": nuovaricerca, "callback_data": 'Nuova ricerca'}],
                                        [{"text": Impostazioni, "callback_data": 'Modifica la posizione o le modalità di viaggio'}],
                                        ]}}},
                                                 "platform": "TELEGRAM"},
                                                {"text": {"text": [""]}}]}

            payload = json.dumps(data)

            temp = copy.deepcopy(utenti[pos])
            temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
            temp = json.dumps(temp)
            etag = db.child("oggetti_utenti").child(chat_id).get()
            db.child("oggetti_utenti").child(chat_id).set(temp, etag)

            return payload


    if(query_result.get('action')== 'cambia_pos_nome_luogo'):

        titolo = 'Se vuoi cambiare la posizione premi sul pulsante "Modifica la posizione" 🌐'
        cambiaPosizione = 'Modifica la posizione 🌐'
        nuovaricerca = 'Nuova ricerca 🔍'
        data = {"fulfillmentMessages": [{"payload": {"telegram": {"text": titolo, "reply_markup": {
            "inline_keyboard": [[{"text": cambiaPosizione, "callback_data": "Modifica la posizione"}],
                                [{"text": nuovaricerca, "callback_data": "Nuova ricerca"}],
                                ]}}},
                                         "platform": "TELEGRAM"},
                                        {"text": {"text": [""]}}]}

        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)


    if (query_result.get('action') == 'xgusti'):
        utenti[pos].gusti='true'
        raccomandazioneGusti = rocchio_classifier(utenti[pos].interessi,utenti[pos].elementi_rilevanti, utenti[pos].elementi_nonrilevanti,utenti[pos].pos)
        #Prima raccomandazione
        linkimm1 = df['Immagini'].iloc[raccomandazioneGusti[0]]
        nome1 = df['Nome'].iloc[raccomandazioneGusti[0]]
        descrizione1 = df['Descrizione'].iloc[raccomandazioneGusti[0]]
        posizioe1 = df['Paese'].iloc[raccomandazioneGusti[0]]
        punteggio1 = df['Punteggio'].iloc[raccomandazioneGusti[0]]
        linksito1 = df['Link'].iloc[raccomandazioneGusti[0]]
        testo_prima="Nome: " + " " + nome1.title() + "\n" "Descrizione: " + " " + descrizione1.title() + "\n" + "Posizione: " + " " + posizioe1.title() + "\n" + "Punteggio: " + " " + punteggio1

        #Seconda raccomandazione
        linkimm2 = df['Immagini'].iloc[raccomandazioneGusti[1]]
        descrizione2 = df['Descrizione'].iloc[raccomandazioneGusti[1]]
        posizioe2 = df['Paese'].iloc[raccomandazioneGusti[1]]
        punteggio2 = df['Punteggio'].iloc[raccomandazioneGusti[1]]
        linksito2 = df['Link'].iloc[raccomandazioneGusti[1]]
        testo_seconda = "Descrizione: " + " " + descrizione2.title() + "\n" + "Posizione: " + " " + posizioe2.title() + "\n" + "Punteggio: " + " " + punteggio2

        # Terza raccomandazione
        linkimm3 = df['Immagini'].iloc[raccomandazioneGusti[2]]
        descrizione3 = df['Descrizione'].iloc[raccomandazioneGusti[2]]
        posizioe3 = df['Paese'].iloc[raccomandazioneGusti[2]]
        punteggio3 = df['Punteggio'].iloc[raccomandazioneGusti[2]]
        linksito3 = df['Link'].iloc[raccomandazioneGusti[2]]
        testo_terza = "Descrizione: " + " " + descrizione3.title() + "\n" + "Posizione: " + " " + posizioe3.title() + "\n" + "Punteggio: " + " " + punteggio3

        titolo = "Cosa ti piacerebbe fare ? 😀 "
        nuovaricerca = 'Nuova ricerca 🔍'
        TuoiGusti = 'In base ai tuoi gusti 🔖'
        Impostazioni='Modifica la posizione o le modalità di viaggio ⚙'

        data={ "fulfillmentMessages": [ { "image": { "imageUri": linkimm1 }, "platform": "TELEGRAM" }, { "card": { "subtitle": testo_prima, "buttons": [ { "text": "Per maggiori informazioni 👈", "postback": linksito1 } ] }, "platform": "TELEGRAM" }, { "image": { "imageUri": linkimm2 }, "platform": "TELEGRAM" }, { "card": { "subtitle": testo_seconda, "buttons": [ { "text": "Per maggiori informazioni 👈", "postback": linksito2 } ] }, "platform": "TELEGRAM" }, { "image": { "imageUri": linkimm3 }, "platform": "TELEGRAM" }, { "card": { "subtitle": testo_terza, "buttons": [ { "text": "Per maggiori informazioni 👈", "postback": linksito3 } ] }, "platform": "TELEGRAM" }, { "card": { "subtitle": titolo, "buttons": [ { "text": nuovaricerca },{ "text":Impostazioni },{"text":TuoiGusti} ] }, "platform": "TELEGRAM" }, { "text": { "text": [ "" ] } } ] }
        payload = json.dumps(data)

        temp = copy.deepcopy(utenti[pos])
        temp = dict((name, getattr(temp, name)) for name in dir(temp) if not name.startswith('__'))
        temp = json.dumps(temp)
        etag = db.child("oggetti_utenti").child(chat_id).get()
        db.child("oggetti_utenti").child(chat_id).set(temp, etag)

    return payload

if __name__ == '__main__':
    app.run(threaded=True,host='0.0.0.0', port=port)
