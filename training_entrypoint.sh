#!/bin/bash

# Script di entrypoint per il container di training
# Questo script attenderà i job di training dalla coda e li eseguirà

# Debug info
echo "Avvio del servizio di training..."
echo "Directory corrente: $(pwd)"
echo "Contenuto della directory: $(ls -la)"
echo "Utente corrente: $(whoami)"
echo "Contenuto della directory app: $(ls -la /app)"

# Se sono stati forniti argomenti, esegui un job specifico
if [ $# -gt 0 ]; then
    echo "Esecuzione del job: $@"
    python -m ml_pipelines $@
    exit $?
fi

# Altrimenti, entra in modalità servizio
echo "Attivazione della modalità servizio di training..."
while true; do
    # Controlla se ci sono job in attesa di essere eseguiti
    echo "Controllo dei job in attesa..."
    
    # In una implementazione reale, qui avresti una logica per
    # leggere da una coda (Redis, RabbitMQ, ecc.)
    # Per ora, aspettiamo semplicemente
    
    sleep 60
done