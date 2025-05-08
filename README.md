# AI Model Pipeline

Una piattaforma automatizzata per il fine-tuning e il retraining di modelli YOLO e RF-DETR con tracciamento MLFlow e orchestrazione Dagster.

## Caratteristiche

- Caricamento e gestione di dataset in formato COCO, YOLO e VOC
- Addestramento e fine-tuning di modelli YOLO (v5, v8) e RF-DETR
- Tracciamento degli esperimenti tramite MLFlow
- Orchestrazione dei workflow con Dagster
- Interfaccia web Flask per la gestione dei job di addestramento
- Supporto GPU per l'addestramento

## Installazione

### Opzione 1: Docker Compose (consigliata)

Questa opzione avvia l'intero stack (app web, MLFlow, Dagster) in container separati.

```bash
# Clona il repository
git clone https://github.com/tuousername/ai-model-pipeline.git
cd ai-model-pipeline

# Avvia i container
docker compose up -d --build
```

L'applicazione sarà disponibile all'indirizzo http://localhost:5000.
MLFlow sarà disponibile all'indirizzo http://localhost:5001.
Dagster sarà disponibile all'indirizzo http://localhost:3000.

### Opzione 2: Installazione locale

```bash
# Clona il repository
git clone https://github.com/tuousername/ai-model-pipeline.git
cd ai-model-pipeline

# Crea e attiva un ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa le dipendenze
pip install -e .

# Avvia l'applicazione web
python -m gunicorn --bind 0.0.0.0:5000 main:app
```

L'applicazione sarà disponibile all'indirizzo http://localhost:5000 e funzionerà in modalità simulazione (senza server MLFlow e Dagster).

## Utilizzo

1. **Upload del dataset**:
   - Carica un dataset nel formato COCO JSON, YOLO o Pascal VOC
   - Il sistema validerà automaticamente il dataset e mostrerà le statistiche

2. **Configurazione del modello**:
   - Scegli il modello (YOLO v5/v8 o RF-DETR)
   - Configura gli iperparametri per l'addestramento
   - Avvia il job di training

3. **Monitoraggio**:
   - Visualizza lo stato e i progressi dell'addestramento
   - Esamina le metriche di performance in tempo reale

4. **Risultati**:
   - Visualizza le metriche finali
   - Scarica i pesi del modello addestrato
   - Esamina i log e le curve di apprendimento

## Struttura della directory

- `app.py`: Configurazione dell'applicazione Flask
- `models.py`: Definizione dei modelli del database
- `routes.py`: Gestione delle rotte dell'applicazione web
- `forms.py`: Form per l'UI
- `ml_pipelines.py`: Funzioni di alto livello per l'addestramento
- `ml_utils.py`: Utilità per la gestione dei dataset
- `dagster_pipelines.py`: Integrazione con Dagster
- `templates/`: Templates HTML
- `static/`: Asset statici (CSS, JS)
- `uploads/`: Directory per i dataset caricati
- `docker-compose.yml`: Configurazione Docker Compose

## Requisiti hardware

- CPU multi-core per il preprocessing dei dati
- GPU NVIDIA con CUDA 11.7+ per l'addestramento (opzionale ma consigliato)
- 8GB+ di RAM
- Spazio su disco sufficiente per i dataset e i checkpoint dei modelli

## Configurazioni

Le impostazioni di configurazione possono essere modificate nei file:
- `config.py`: Configurazioni generali dell'applicazione
- `docker-compose.yml`: Configurazioni dei servizi Docker