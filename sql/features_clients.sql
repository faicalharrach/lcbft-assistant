-- ============================================================================
-- Table de features client (une ligne par client, sur l'année 2025)
-- Dialecte : BigQuery Standard SQL.
-- {raw} est remplacé par le dataset brut, {seuil_especes} par le seuil d'alerte.
--
-- Volontairement exclus : nationalité et pays de résidence (biais
-- discriminatoire). Le risque géographique passe par les flux réels.
-- Les étiquettes (dataset lcbft_labels) ne sont jamais lues ici.
-- ============================================================================

WITH tx AS (
  -- Transactions enrichies avec la famille MCC et le risque pays
  SELECT
    t.transaction_id,
    t.numero_dossier,
    t.date_trans,
    CAST(t.montant AS FLOAT64) AS montant,
    t.sens,
    t.type_trans,
    t.code_pays,
    m.famille AS famille_mcc,
    p.niveau_risque AS risque_pays
  FROM {raw}.transactions AS t
  LEFT JOIN {raw}.ref_marchands AS rm ON rm.marchand_id = t.marchand_id
  LEFT JOIN {raw}.ref_mcc AS m ON m.mcc = rm.mcc
  LEFT JOIN {raw}.ref_pays AS p ON p.code_pays = t.code_pays
),

agregats AS (
  SELECT
    numero_dossier,
    COUNT(*) AS nb_trans,
    SUM(IF(sens = 'debit', montant, 0)) AS montant_total_debit,
    SUM(IF(sens = 'credit', montant, 0)) AS montant_total_credit,

    -- Par type d'opération
    COUNTIF(type_trans = 'ACHAT') AS nb_achats,
    SUM(IF(type_trans = 'ACHAT', montant, 0)) AS montant_achats,
    COUNTIF(type_trans = 'DAB') AS nb_dab,
    SUM(IF(type_trans = 'DAB', montant, 0)) AS montant_dab,
    COUNTIF(type_trans = 'DEPOT_ESPECES') AS nb_depots_especes,
    SUM(IF(type_trans = 'DEPOT_ESPECES', montant, 0)) AS montant_depots_especes,
    MAX(IF(type_trans = 'DEPOT_ESPECES', montant, 0)) AS montant_max_depot_especes,
    COUNTIF(type_trans = 'DEPOT_ESPECES'
            AND montant >= 0.8 * {seuil_especes} AND montant < {seuil_especes}) AS nb_depots_proches_seuil,
    MAX(IF(type_trans = 'VIREMENT' AND sens = 'credit', montant, 0)) AS montant_max_virement_entrant,

    -- Géographie
    COUNTIF(risque_pays = 'eleve') AS nb_trans_pays_risque_eleve,
    SUM(IF(risque_pays = 'eleve', montant, 0)) AS montant_pays_risque_eleve,
    COUNTIF(risque_pays = 'moyen') AS nb_trans_pays_risque_moyen,
    COUNT(DISTINCT IF(code_pays != 'FR', code_pays, NULL)) AS nb_pays_etrangers,

    -- Familles MCC à risque
    COUNTIF(famille_mcc = 'jeux') AS nb_trans_jeux,
    SUM(IF(famille_mcc = 'jeux', montant, 0)) AS montant_jeux,
    COUNTIF(famille_mcc = 'crypto_change') AS nb_trans_crypto,
    SUM(IF(famille_mcc = 'crypto_change', montant, 0)) AS montant_crypto,
    COUNTIF(famille_mcc IN ('valeurs', 'gages')) AS nb_trans_valeurs,
    SUM(IF(famille_mcc IN ('valeurs', 'gages'), montant, 0)) AS montant_valeurs,
    COUNTIF(famille_mcc IN ('transfert', 'prepaye')) AS nb_trans_transfert_prepaye
  FROM tx
  GROUP BY numero_dossier
),

allers_retours AS (
  -- Entrée >= 5 000 € suivie dans les 48h d'une sortie de 90 à 100 % du montant
  SELECT
    e.numero_dossier,
    COUNT(DISTINCT e.transaction_id) AS nb_allers_retours_48h
  FROM tx AS e
  JOIN tx AS s
    ON s.numero_dossier = e.numero_dossier
   AND s.sens = 'debit'
   AND s.type_trans = 'VIREMENT'
   AND s.date_trans > e.date_trans
   AND s.date_trans <= TIMESTAMP_ADD(e.date_trans, INTERVAL 48 HOUR)
   AND s.montant BETWEEN 0.9 * e.montant AND e.montant
  WHERE e.sens = 'credit'
    AND e.type_trans = 'VIREMENT'
    AND e.montant >= 5000
  GROUP BY e.numero_dossier
),

occurrences AS (
  -- Montant identique le plus répété (hors DAB et petits montants)
  SELECT
    numero_dossier,
    montant AS montant_max_occu,
    COUNT(*) AS max_occu
  FROM tx
  WHERE type_trans != 'DAB' AND montant >= 100
  GROUP BY numero_dossier, montant
  QUALIFY ROW_NUMBER() OVER (PARTITION BY numero_dossier ORDER BY COUNT(*) DESC, montant DESC) = 1
),

profil AS (
  SELECT
    numero_dossier,
    age,
    ccsp,
    CAST(salaire_declare + pension AS FLOAT64) AS revenu_mensuel,
    CAST(revenu_foyer AS FLOAT64) AS revenu_foyer,
    DATE_DIFF(DATE '2025-12-31', date_ouverture, MONTH) AS anciennete_mois
  FROM {raw}.clients
),

base AS (
  SELECT
    p.*,
    -- Revenu de référence : le revenu déclaré, ou un minimum de 550 €/mois
    -- (aides) quand rien n'est déclaré, pour garder des ratios calculables.
    12 * IF(p.revenu_mensuel > 0, p.revenu_mensuel, 550) AS revenu_annuel_reference,
    a.* EXCEPT (numero_dossier),
    COALESCE(ar.nb_allers_retours_48h, 0) AS nb_allers_retours_48h,
    COALESCE(o.max_occu, 0) AS max_occu,
    COALESCE(o.montant_max_occu, 0) AS montant_max_occu
  FROM profil AS p
  LEFT JOIN agregats AS a ON a.numero_dossier = p.numero_dossier
  LEFT JOIN allers_retours AS ar ON ar.numero_dossier = p.numero_dossier
  LEFT JOIN occurrences AS o ON o.numero_dossier = p.numero_dossier
)

SELECT
  b.*,
  -- Ratios au revenu déclaré
  ROUND(SAFE_DIVIDE(montant_total_credit, revenu_annuel_reference), 4) AS ratio_credits_revenu,
  ROUND(SAFE_DIVIDE(montant_total_debit, revenu_annuel_reference), 4) AS ratio_depenses_revenu,
  ROUND(SAFE_DIVIDE(montant_depots_especes, montant_total_credit), 4) AS part_especes_credits,
  ROUND(SAFE_DIVIDE(montant_crypto + montant_jeux + montant_valeurs, montant_total_debit), 4) AS part_mcc_risque_debits,
  -- Comparaison à la catégorie socio-professionnelle
  ROUND(AVG(montant_total_debit) OVER (PARTITION BY ccsp), 2) AS avg_montant_ccsp,
  ROUND(montant_total_debit - AVG(montant_total_debit) OVER (PARTITION BY ccsp), 2) AS ecart_montant_ccsp,
  ROUND(SAFE_DIVIDE(montant_total_debit, AVG(montant_total_debit) OVER (PARTITION BY ccsp)), 4) AS ratio_depense_ccsp
FROM base AS b
