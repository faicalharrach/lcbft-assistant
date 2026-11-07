"""Cloud Function : télécharge le registre national des gels et le charge dans BigQuery.

Déclenchement : HTTP (appelée chaque nuit par Cloud Scheduler).
Test local :
    functions-framework --target maj_gels --source functions/maj_gels/main.py --debug
    curl -X POST localhost:8080
"""

import json
import os
from datetime import datetime, timezone

import functions_framework
import requests
from google.cloud import bigquery

from gels import lire

URL_REGISTRE = "https://www.data.gouv.fr/api/1/datasets/r/83ef53df-3e42-4c6a-a83c-29ce16759e57"
DATASET = "lcbft_gels"
LOCATION = "EU"

S = bigquery.SchemaField
SCHEMA_REGISTRE = [
    S("id_registre", "INT64", "REQUIRED"),
    S("nature", "STRING"),
    S("nom", "STRING"),
    S("prenoms", "STRING", "REPEATED"),
    S("alias", "STRING", "REPEATED"),
    S("dates_naissance", "STRING", "REPEATED"),
    S("lieux_naissance", "STRING", "REPEATED"),
    S("nationalites", "STRING", "REPEATED"),
    S("sexe", "STRING"),
    S("identifiants", "STRING", "REPEATED"),
    S("numero_omi", "STRING"),
    S("fondements", "STRING", "REPEATED"),
    S("motifs", "STRING"),
    S("details_json", "STRING"),
    S("date_publication", "TIMESTAMP"),
    S("date_chargement", "TIMESTAMP"),
]
SCHEMA_NOMS = [
    S("id_registre", "INT64", "REQUIRED"),
    S("type_nom", "STRING"),
    S("nom_complet", "STRING"),
    S("date_publication", "TIMESTAMP"),
]


def _sans_vides(ligne: dict) -> dict:
    """BigQuery refuse les NULL dans une colonne REPEATED : on retire None et chaînes vides des listes."""
    return {k: [x for x in v if x not in (None, "")] if isinstance(v, list) else v
            for k, v in ligne.items()}


def _charger(client: bigquery.Client, table: str, lignes: list[dict], schema: list) -> int:
    job = client.load_table_from_json(
        lignes,
        f"{client.project}.{DATASET}.{table}",
        job_config=bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ),
    )
    job.result()
    return job.output_rows


@functions_framework.http
def maj_gels(request):
    """Point d'entrée : télécharge, met à plat, charge. Renvoie un résumé JSON."""
    reponse = requests.get(URL_REGISTRE, timeout=120)
    reponse.raise_for_status()  # erreur HTTP (404, 500...) -> la fonction échoue, Scheduler le verra
    date_pub, lignes, noms = lire(reponse.json())

    # 7 décimales de secondes dans la source : BigQuery n'en accepte que 6
    date_pub = datetime.fromisoformat(date_pub).isoformat(timespec="microseconds")
    maintenant = datetime.now(timezone.utc).isoformat()
    lignes = [_sans_vides(l) | {"date_publication": date_pub, "date_chargement": maintenant} for l in lignes]
    noms = [n | {"date_publication": date_pub} for n in noms]

    client = bigquery.Client(project=os.environ.get("GCP_PROJECT"))
    dataset = bigquery.Dataset(f"{client.project}.{DATASET}")
    dataset.location = LOCATION
    client.create_dataset(dataset, exists_ok=True)

    resume = {
        "date_publication": date_pub,
        "entrees": _charger(client, "registre", lignes, SCHEMA_REGISTRE),
        "noms": _charger(client, "noms", noms, SCHEMA_NOMS),
    }
    print(json.dumps(resume))  # visible dans les journaux Cloud Logging
    return resume, 200