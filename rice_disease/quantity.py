"""Unit-aware quantity arithmetic, separate from image classification and spray decisions."""

import math


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
    return {"quantity": dose * scale, "unit": output_unit, "product": rule["product"],
            "formulation": rule["formulation"], "source_url": rule["source_url"]}


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
