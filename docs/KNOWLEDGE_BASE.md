# Agricultural source record

Reviewed 2026-10-02. `knowledge_base/diseases.json` contains short paraphrases of
cultural-management guidance with a source and source region for each disease.

| Disease | Source | Scope |
|---|---|---|
| Bacterial leaf blight | [TNAU](https://agritech.tnau.ac.in/expert_system/paddy/cpdisblb.html) | Disease-free seed, tolerant varieties, nitrogen, spacing, weeds, irrigation movement |
| Brown spot | [TNAU](https://agritech.tnau.ac.in/expert_system/paddy/cpdisbrownspot.html) | Seed health, resistance, nutrition and water stress |
| Leaf blast | [TNAU](https://agritech.tnau.ac.in/expert_system/paddy/cpdisblast.html) | Resistance, nitrogen management, weed hosts |
| Leaf scald | [TNAU](https://agritech.tnau.ac.in/expert_system/paddy/cpdisleafscald.html) | Avoid excessive fertilizer |
| Narrow brown spot | [Rice Knowledge Bank Odisha, IRRI and partners](https://rkb-odisha.in/wp-content/uploads/2024/09/narrow-brown-spot-english.pdf) | Resistance, weed hosts and potassium nutrition |

These are extension references, not verified current product labels. The retrieved
pages mention chemicals and some rates, but publication currency, product
formulation, local registration, and all application restrictions have not been
established. Those values have therefore not been enabled as actionable rules.
Tamil Nadu and Odisha recommendations are recorded with their regions rather than
being silently generalized to every location.

Before enabling a chemical rule, record exact product/formulation, active ingredient,
current label source, applicable jurisdiction, rice disease indication, dose and
units, crop-stage/timing conditions, application frequency, maximum use, and relevant
restrictions. Preserve conflicting recommendations as separate sourced entries.
An agricultural review must establish when application is warranted; a predicted
class alone must not activate a spray recommendation.

`rice_disease.quantity.calculate_quantity` is tested unit arithmetic for a future
verified rule. It supports explicit mass/volume per litre or per hectare, rejects
unverified rules and mismatched regions, and requires an established application
decision. It does not derive a dose from confidence or lesion area. No bundled
chemical rule currently enables quantity output in the application.
