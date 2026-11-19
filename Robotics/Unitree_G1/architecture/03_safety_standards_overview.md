# Safety standards by domain: overview for the G1 and the X650

- **Status:** Draft
- **Date:** 2026-11-19
- **Purpose:** Place the risk-rating methods side by side, so that the choice for the G1 (humanoid robot) and the X650 (drone) is clear. This is an overview from public sources. It is not a compliance statement.
- **Used by:** [`04_fmea.md`](04_fmea.md)

## 1 Standards by domain

| Domain | Standard | Levels | How the level is derived | Role in this repository |
|---|---|---|---|---|
| Road vehicle | ISO 26262 | QM, ASIL A to D | Severity × exposure × controllability, per hazard at vehicle level | Pattern only. Not applicable to the G1 or the X650 |
| General, industry | IEC 61508 | SIL 1 to 4 | Probability of dangerous failure per hour (PFH) | Parent standard of several others. Background |
| Machinery, industrial robots | ISO 13849-1, IEC 62061 | PL a to e, SIL 1 to 3 | Risk graph with severity, frequency, possibility of avoidance (ISO 13849-1) | **Main method for the G1** |
| Service robots | ISO 13482 | no level scale of its own | Risk assessment and protective measures; refers to machinery standards for control systems | Whether the G1 EDU falls under it is **open** |
| Drones (EU) | SORA 2.5 (JARUS, adopted by EASA) | SAIL I to VI | Table from final ground risk class (GRC) and residual air risk class (ARC) | **Main method for the X650** (separate document) |
| Aviation, airborne software and hardware | DO-178C, DO-254 | DAL E to A | Effect of a failure on the aircraft | Only for systems that need certification. Background |

Basic standard for the risk assessment of machinery: ISO 12100 (type A). ISO 13849 and IEC 62061 are generic standards (type B).

Equating levels across domains (for example ASIL D with SIL 3) is a rough comparison found in public tables. It is not a conversion. The standards derive their levels in different ways.

## 2 The three questions in each method

| Question | ISO 26262 | ISO 13849-1 | SORA 2.5 |
|---|---|---|---|
| How severe? | S0 to S3 | S1 slight (normally reversible), S2 serious (normally irreversible, or death) | Intrinsic GRC from the size and speed of the aircraft and the population density below |
| How often? | E0 to E4 | F1 seldom to less often and/or short exposure, F2 frequent to continuous and/or long exposure | ARC: encounter rate in the airspace |
| Can it be avoided? | C0 to C3 | P1 possible under specific conditions, P2 scarcely possible | Mitigations M1 to M3 lower the GRC |
| Result | ASIL A to D or QM | PLr a to e | SAIL I to VI |

## 3 Risk graph of ISO 13849-1 (required performance level)

| S | F | P | PLr |
|---|---|---|---|
| S1 | F1 | P1 | a |
| S1 | F1 | P2 | b |
| S1 | F2 | P1 | b |
| S1 | F2 | P2 | c |
| S2 | F1 | P1 | c |
| S2 | F1 | P2 | d |
| S2 | F2 | P1 | d |
| S2 | F2 | P2 | e |

Evidence: S2/F2/P2 = e, S2/F2/P1 = d, S2/F1/P2 = d and "the S1 branch ends between a and c" are confirmed by a manufacturer article. The other rows come from a source that does not show the graph itself. They are **plausible, but not confirmed by the standard**. The standard is the final reference.

The PL measures how reliably a whole safety function (sensor, logic, actuator) works on demand. The PL a to e correspond to ranges of the average probability of dangerous failure per hour:

| PL | PFHd per hour |
|---|---|
| a | 10⁻⁵ to below 10⁻⁴ |
| b | 3 × 10⁻⁶ to below 10⁻⁵ |
| c | 10⁻⁶ to below 3 × 10⁻⁶ |
| d | 10⁻⁷ to below 10⁻⁶ |
| e | 10⁻⁸ to below 10⁻⁷ |

## 4 SORA 2.5: SAIL from GRC and ARC

| Final GRC | ARC a | ARC b | ARC c | ARC d |
|---|---|---|---|---|
| 2 or less | I | I | IV | VI |
| 3 | II | II | IV | VI |
| 4 | III | III | IV | VI |
| 5 | IV | IV | IV | VI |
| 6 | V | V | V | VI |
| 7 | VI | VI | VI | VI |
| above 7 | certified category | | | |

The row for GRC 2 or less differs between the sources found (one source shows I, I, IV, VI, another I, II, IV, VI for ARC a to d). Check against the JARUS SORA 2.5 document before use. The target of SORA for ground risk is below one fatality per million flight hours.

## 5 What applies to the G1 and the X650

| Topic | G1 EDU | X650 |
|---|---|---|
| Legal frame | Machinery. The EU Machinery Regulation (EU) 2023/1230 replaces Directive 2006/42/EC and applies from 20 January 2027. It names autonomous mobile machinery | EU drone regulation 2019/947, SORA for the specific category |
| Risk assessment | ISO 12100 as method and checklist | SORA steps 1 to 10 |
| Rating of the safety functions | Risk graph of ISO 13849-1 (PLr) | SAIL |
| ISO 13482 | Scope: mobile servant, physical assistant and person carrier robots. Excludes industrial robots (ISO 10218), toys, medical and military robots, and robots faster than 20 km/h. A research humanoid is not named. **Whether it applies: open.** Its hazard list includes "localization and navigation errors", which matches R2 and R5 of the FMEA | Not applicable (flying robots excluded) |
| ISO 26262 | Not applicable | Not applicable |
| Status of the standard | ISO 13482:2014, confirmed 2020, listed as to be revised | SORA 2.5, EASA edition September 2025 |

The statements on the G1 and on legal duties are an **assumption from public summaries**. They are not legal advice and not a classification of the product.

## Appendix: ASIL determination table (ISO 26262-3, reference)

| S | E | C1 | C2 | C3 |
|---|---|---|---|---|
| S1 | E1 | QM | QM | QM |
| S1 | E2 | QM | QM | QM |
| S1 | E3 | QM | QM | ASIL A |
| S1 | E4 | QM | ASIL A | ASIL B |
| S2 | E1 | QM | QM | QM |
| S2 | E2 | QM | QM | ASIL A |
| S2 | E3 | QM | ASIL A | ASIL B |
| S2 | E4 | ASIL A | ASIL B | ASIL C |
| S3 | E1 | QM | QM | ASIL A |
| S3 | E2 | QM | ASIL A | ASIL B |
| S3 | E3 | ASIL A | ASIL B | ASIL C |
| S3 | E4 | ASIL B | ASIL C | ASIL D |

If S, E or C is rated 0, no ASIL is assigned (QM).

## Sources

- Automotive Safety Integrity Level, comparison tables across domains: https://www.wikipedia.org/wiki/ASIL_D
- Cross-industry functional safety comparison: https://www.batterydesign.net/battery-management-system/functional-safety/cross-industry-functional-safety-comparison/
- Festo, safety engineering guidelines (risk graph labels of DIN EN ISO 13849-1): https://www.festo.com/PDF_Flip/safety/en/files/assets/basic-html/page13.html
- Fabrico, ISO 13849 performance levels: https://www.fabrico.io/blog/iso-13849-performance-levels/
- Reer, ISO 13849-1 and 2 (PFHd ranges): https://www.reersafety.com/fr/?p=1002215
- TÜV SÜD, ISO 13849 explained: https://www.tuvsud.com/en-us/resource-centre/blogs/iso-13849-explained-practical-guidance-for-risk-assessment-and-performance-levels-in-machinery
- SORA 2.5 calculator, CAA Norway: https://training.caa.no/SORA-2.5-calculator
- SORA process table, Dronetalks: https://dronetalks.online/blog/step-7-of-the-sora-methodology-final-specific-assurance-and-integrity-level-sail-and-operational-safety-objectives-oso-assignment/
- EASA, AMC and GM to Regulation (EU) 2019/947, SORA 2.5, September 2025: https://www.easa.europa.eu/sites/default/files/dfu/annex_to_ed_decision_2025-018-r_1_2.pdf
- ISO 13482:2014: https://www.iso.org/standard/53820.html
- CEMA, Machinery Regulation published: https://www.cema-agri.org/publication/news/1025-the-new-machinery-regulation-is-published-new-requirements-and-upcoming-steps
