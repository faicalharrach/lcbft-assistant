# Mise à jour du registre des gels des avoirs

Cloud Function planifiée qui télécharge chaque nuit le registre national des gels
(DG Trésor) et le charge dans BigQuery, pour le criblage des clients.

- Source : https://www.data.gouv.fr/fr/datasets/registre-national-des-gels
- Producteur : Direction générale du Trésor
- Licence : Licence Ouverte / Open Licence 2.0 (réutilisation libre, source citée)
- Usage LCB-FT : criblage obligatoire des clients contre les mesures de gel
  (articles L. 562-1 et suivants du Code monétaire et financier)