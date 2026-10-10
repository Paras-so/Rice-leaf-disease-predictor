"""Unit-aware quantity arithmetic, separate from image classification and spray decisions."""

import math


def disease_product_reference(disease, *, water_ml=100):
    """Match a disease to sourced label arithmetic, without claiming a per-plant dose."""
    import json
    from pathlib import Path
    if isinstance(water_ml, bool) or not isinstance(water_ml, (int, float)) or not math.isfinite(water_ml) or water_ml <= 0:
        raise ValueError('Water volume must be a positive finite number in mL.')
    catalog = json.loads((Path(__file__).resolve().parent.parent /
                          'knowledge_base/product_references.json').read_text(encoding='utf-8'))
    result = {'disease': disease, 'product': None, 'reference_quantity': None,
              'required_quantity_per_plant': None, 'application_authorized': False,
              'affected_area_used_to_scale_dose': False, 'checked_on': catalog['checked_on']}
    if disease == 'healthy':
        return dict(result, status='no_product_suggested',
                    note='No pesticide is suggested from a healthy-class prediction.')
    matches = [r for r in catalog['references'] if disease in r['diseases']]
    if not matches:
        return dict(result, status='no_verified_label_in_catalog',
                    note='No matching foliar product label has been verified in this catalog. '
                         'Use the disease management reference; do not substitute a different disease dose.')
    reference = matches[0]
    concentration = reference['formulation_ml_per_hectare'] / reference['water_litres_per_hectare']
    amount = concentration * (water_ml / 1000)
    if not math.isfinite(amount):
        raise ValueError('Reference quantity is not finite.')
    return dict(result, status='conditional_label_reference', product=reference,
                reference_quantity={'quantity': amount, 'unit': 'mL', 'water_ml': water_ml,
                                    'equivalent_ml_per_litre': concentration,
                                    'basis': 'ratio of label product rate to label water volume'},
                note='This is label dilution arithmetic, not the quantity needed by the photographed plant. '
                     'The photo cannot determine spray coverage or establish treatment suitability.')


def rice_tank_quantity(rule, *, disease, region, tank_ml=100,
                       application_authorized=False, area_hectares=None):
    """Match a rice/disease rule and convert mL tank volume to L; no severity scaling."""
    if not rule or rule.get('crop') != 'rice' or disease not in rule.get('diseases', []):
        raise ValueError('A verified rice product-label rule for this disease is required.')
    if isinstance(tank_ml, bool) or not isinstance(tank_ml, (int, float)) or not math.isfinite(tank_ml) or tank_ml <= 0:
        raise ValueError('Tank volume must be a positive finite number in mL.')
    result = calculate_quantity(rule, region=region, application_authorized=application_authorized,
                                spray_litres=tank_ml/1000, area_hectares=area_hectares)
    result.update(tank_ml=tank_ml, tank_litres=tank_ml/1000,
                  affected_area_used_to_scale_dose=False)
    return result


def calculate_quantity(rule, *, region, application_authorized, spray_litres=None, area_hectares=None):
    """Use only a reviewed formulation-specific rule after its application conditions are met.

    No such chemical rules are enabled in the bundled knowledge base. This helper
    is infrastructure for later label verification, not a pesticide recommender.
    """
    if not rule or rule.get("label_verified") is not True or not rule.get("source_url"):
        raise ValueError("A verified product-label rule is required.")
    if not application_authorized:
        raise ValueError("Application suitability must be established before calculating a quantity.")
    if region != rule.get("region"):
        raise ValueError("The rule is not verified for the requested region.")
    if not rule.get("product") or not rule.get("formulation"):
        raise ValueError("The exact product and formulation must be specified.")
    dose = rule.get("dose")
    if isinstance(dose, bool) or not isinstance(dose, (int, float)) or not math.isfinite(dose) or dose <= 0:
        raise ValueError("Dose must be a positive finite number.")
    unit = rule.get("unit")
    if unit in ("g/L", "mL/L"):
        scale, output_unit = spray_litres, unit.split("/")[0]
    elif unit in ("g/ha", "kg/ha", "mL/ha", "L/ha"):
        scale, output_unit = area_hectares, unit.split("/")[0]
    else:
        raise ValueError("Unsupported dose unit; do not infer percentage or active-ingredient conversions.")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("The corresponding spray volume or area must be a positive finite number.")
    quantity = dose * scale
    if not math.isfinite(quantity):
        raise ValueError('Calculated quantity is not finite.')
    return {"quantity": quantity, "unit": output_unit, "product": rule["product"],
            "formulation": rule["formulation"], "source_url": rule["source_url"],
            "label_rate": dose, "label_rate_unit": unit,
            "calculation_basis": "spray_volume" if unit.endswith('/L') else "treated_area",
            "spray_litres": spray_litres, "area_hectares": area_hectares,
            "affected_area_used_to_scale_dose": False}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__, epilog="No verified rules are bundled with this project.")
    parser.add_argument("--rule", type=Path, help="JSON file containing a verified product-label rule")
    parser.add_argument("--region", help="Region for which the rule is verified")
    parser.add_argument("--application-authorized", action="store_true")
    parser.add_argument("--spray-litres", type=float)
    parser.add_argument("--area-hectares", type=float)
    args = parser.parse_args()
    if args.rule is None:
        parser.print_help()
    else:
        try:
            rule = json.loads(args.rule.read_text(encoding="utf-8"))
            result = calculate_quantity(rule, region=args.region,
                application_authorized=args.application_authorized,
                spray_litres=args.spray_litres, area_hectares=args.area_hectares)
            print(json.dumps(result, indent=2))
        except (OSError, ValueError) as exc:
            parser.exit(1, f"Quantity error: {exc}\n")
