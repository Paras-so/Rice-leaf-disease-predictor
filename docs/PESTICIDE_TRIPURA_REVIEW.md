# Tripura rice: product-label review status

## Single-photo reference update, 9 October 2026

The user clarified that product suggestions should follow the photographed plant's
predicted disease, without asking for field area. The app and classification CLI
now use `knowledge_base/product_references.json` for conditional product references.
Actual required quantity per plant remains unknown from an uncalibrated photo.

[PPQS label, English leaflet page 12](https://ppqs.gov.in/sites/default/files/azoxystrobin_16.7_tricyclazole_33.3_sc93fcrystal_crop_protection_ltd_1-merged.pdf)
provides the source for the blast/brown-spot entry. The stored product/water ratio
supports dilution arithmetic for a chosen water volume, while preserving the
label's per-hectare basis and restrictions. PDF text was retrieved; screenshot
retrieval failed. No commercial brand was inferred from the blank brand field.

This does not verify application to a particular plant, product pack or Tripura
site. No application-authorized rule was enabled. The remaining three disease
classes have no verified foliar label in this catalog, which does not mean that
no treatment exists. The app returns that gap explicitly instead of borrowing
another disease's product.

The older review below documents the previous state.

Checked 8 October 2026. User context: rice leaves, Tripura (India), 100 mL tank.
The chosen product is to depend on the diagnosed disease. No exact product pack,
crop stage, field diagnosis, application history or calibrated coverage was supplied.

The implementation therefore has a disease-specific verified-rule calculator but
does not activate an automatic chemical recommendation. The 100 mL tank is stored
as 0.1 L; affected leaf percentage is never a dose multiplier.

## Sources located

- [PPQS official label/leaflet: tricyclazole 20.4% + azoxystrobin 6.8% SC](https://ppqs.gov.in/sites/default/files/tricyclazole_20.4_azoxystrobin_6.8_sc93fdow_agrosciences_india_p_ltd_1.pdf).
  This lists rice blast among the covered diseases and contains stage/timing,
  spray-volume, re-entry and pre-harvest conditions. It is a source for a future
  formulation-specific review, not a verified choice for every leaf-blast image or
  every rice growth stage. No rule was activated from it.
- [ICAR Kharif advisory 2025](https://icar.gov.in/sites/default/files/Circulars/ICAR-En-Kharif-Agro-Advisories-for-Farmers-2025.pdf).
  The indexed Tripura paddy section mentions blast management in a flowering-stage
  context. Full-document retrieval was unsuccessful during this session; this
  source was not used to extract a dose or enable any rule.
- [US EPA applicator core manual](https://www.epa.gov/system/files/documents/2022-09/national-pesticide-applicator-cert-core-manual-2014.pdf).
  Supports the general arithmetic distinction between application rate, treated
  area and calibrated spray volume. This is not an Indian registration source.

These references do not verify chemical treatment for bacterial leaf blight,
brown spot, leaf scald or narrow brown spot in the user's situation. Existing
cultural-management references remain available in the app.

## Required rule fields

The corresponding disease entry in `knowledge_base/diseases.json` can hold a
reviewed `chemical_rule` containing `crop: rice`, supported `diseases`, exact
`product` and `formulation`, `region: India/Tripura`, `label_verified: true`,
`source_url`, positive `dose` and a supported `unit`. Record label application
conditions alongside that rule for display and verification. These fields are a
review record; setting a boolean alone does not establish registration or suitability.

Supported units are g/L, mL/L, g/ha, kg/ha, mL/ha and L/ha. Rates per hectare require
the area actually treated; tank size alone is insufficient. The helper refuses a
missing or mismatched crop, disease, region, authorization or dose unit.

At a 100 mL scale, confirm that the computed amount can be accurately measured with
the available equipment and that the actual product label allows the intended use.
The calculator does not invent a minimum measurable amount or round up to one.
