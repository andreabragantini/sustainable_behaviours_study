"""Plain-language descriptions of the AVQ 2021 survey columns.

The raw AVQ microdata store every answer as an opaque code (e.g. ``USAGETT``),
which makes the output tables hard to read.  This module maps the column names
that matter for the three research goals to short English descriptions derived
from ``about/AVQ_Tracciato_2021.html``.

It is a shared helper used by the analysis scripts (clustering, explanatory
models) so that every table printed to disk can show what a variable actually
measures.
"""

from __future__ import annotations

import pandas as pd

# -----------------------------------------------------------------------------
# Descriptions
# -----------------------------------------------------------------------------
# The dictionary is intentionally curated: it covers the protected driver
# blocks (SOCIO / ATTITUDES / SPATIAL) plus the features that routinely surface
# in rankings.  Unknown columns fall back to their raw name.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # --- SOCIO block (Goal 1: human characteristics) ------------------------
    "SESSO": "Sex (demographic sheet; 1 = male, 2 = female)",
    "ETAMi": "Age in completed years",
    "ISTRMi": "Educational attainment (highest qualification)",
    "CONDMi": "Employment / professional condition",
    "REDPRMi": "Main source of household income",
    "TIPFA2Mi": "Household type (reconstructed)",
    "STCIVMi": "Marital status (reconstructed)",
    "NCOMP": "Number of household members",
    "LAVPAS": "Worked in the past",
    "POSIZMi": "Position in the profession (employee/self-employed...)",
    "RELPAR": "Kinship relation to the reference person",
    "PROIND": "Order number of the household member",

    # --- ATTITUDES block (Goal 1: environmental concern checklist) ----------
    # All are 0/1 flags: "environmental problem mentioned as worrying".
    "SERRA": "Worried: greenhouse effect / ozone hole",
    "ESTINZ": "Worried: extinction of animal/plant species",
    "CAMCLI": "Worried: climate change",
    "SMARIF": "Worried: waste production and disposal",
    "AMRUM": "Worried: noise",
    "IARIA": "Worried: air pollution",
    "INQSU": "Worried: soil pollution",
    "INQFIU": "Worried: pollution of rivers and seas",
    "DISDR": "Worried: hydrogeological instability (earthquakes, floods)",
    "CATASTR": "Worried: man-made catastrophes",
    "FORES2": "Worried: destruction of forests",
    "INQELET2": "Worried: electromagnetic pollution",
    "PAESAG2": "Worried: landscape degradation (excessive building)",
    "ESRISO2": "Worried: depletion of natural resources",
    "ALTAMB2": "Worried: other environmental problems",

    # --- SPATIAL block (Goal 3: living area and local services) -------------
    "REGMf": "Region of residence",
    "RIPMf": "Geographic macro-area of residence",
    "SPORCO": "Area problem: street litter",
    "PARCH": "Area problem: parking difficulties",
    "COLMP": "Area problem: poor public-transport links",
    "TRAF": "Area problem: heavy traffic",
    "INQAR": "Area problem: air pollution",
    "RUMORE": "Area problem: noise",
    "CRIM": "Area problem: risk of crime",
    "ODSGR": "Area problem: unpleasant odours",
    "ILLSTR": "Area problem: poor street lighting",
    "CONPAV": "Area problem: poor road pavement conditions",
    "POAPO": "Area served by door-to-door waste collection",
    "ECOSTAZ": "Ecological (recycling) stations present in the area",
    "GODAB": "Housing tenure status (owner/tenant...)",

    # --- Sustainable-behaviour battery (Goal 2, excluded as predictors) -----
    "USAGETT": "Uses single-use / disposable products",
    "ARUMOR": "Avoids noisy driving behaviour",
    "ALOCAL": "Buys local food/products",
    "ETICHET": "Reads ingredients on food-product labels",
    "BIOLOG": "Buys organic food/products",
    "TRASPO": "Chooses transport alternatives to the private car",
    "GCARTE": "Throws paper/cardboard in the street",
    "DOPFIL": "Parks the car in double file",

    # --- Energy / water services (Goal 2) -----------------------------------
    "GAS5": "Type of gas supply of the dwelling",
    "TRISC": "Main heating system of the dwelling",
    "RISCAL": "Heating / energy-efficiency service indicator",
    "SENELE": "Satisfaction with the electricity service",
    "REACQ1": "Water source: municipal aqueduct",
    "REACQ2": "Water source: well",
    "REACQ3": "Water source: spring",
    "REACQ4": "Water source: rain water",
    "REACQ5": "Water source: bottled water / other",
    "SIACQ": "Family drinks tap water",
    "ACQBRU": "Drinks tap water only after boiling it",
    "NOACQ": "No family member drinks tap water (given reason)",
    "ABIPIC": "Dwelling served by small water utility",
    "ABICC": "Dwelling served by municipal water utility",
    "ACQUA": "Water utility indicator",
    "GODAB": "Housing tenure status (owner/tenant...)",
    "COMPOST": "Composting practice",

    # --- Commuting modes / distance (Goal 2/3) ------------------------------
    "TRENO": "Commutes by train",
    "TRAM": "Commutes by tram",
    "METRO": "Commutes by metro",
    "BUS": "Commutes by bus",
    "COR": "Commutes by coach",
    "PAZSC": "Commutes as a car passenger (school run)",
    "AUTOC": "Commutes by car",
    "AUTOP": "Commutes by car as passenger",
    "MOTO": "Commutes by motorcycle",
    "BICI": "Commutes by bicycle",
    "ALMEZ": "Commutes by other means",
    "STCOM": "Commutes within the same municipality",
    "STPROV": "Commutes within the same province",
    "STREG": "Commutes within the same region",
    "ALTREG": "Commutes to another region",
    "ESTERO": "Commutes abroad",

    # --- Waste to eco stations (Goal 2, binarised flags) --------------------
    "ECOCAR": "Takes to eco station: paper/cardboard",
    "ECOVET": "Takes to eco station: glass",
    "ECOPLA": "Takes to eco station: plastic",
    "ECOPNE": "Takes to eco station: tyres",
    "ECOMET": "Takes to eco station: metals",
    "ECOLEG": "Takes to eco station: wood",
    "ECOFAR": "Takes to eco station: medicines",
    "ECOBAT": "Takes to eco station: batteries",
    "ECOING": "Takes to eco station: bulky waste",
    "ECOINE": "Takes to eco station: WEEE/appliances",
    "ECOINF": "Takes to eco station: flammables",
    "ECOALT": "Takes to eco station: other waste",

    # --- Dissatisfaction reasons for door-to-door collection ----------------
    "NOSPO1": "Door-to-door collection: too few pick-ups",
    "NOSPO2": "Door-to-door collection: collection time unsuitable",
    "NOSPO3": "Door-to-door collection: service unreliable",
    "NOSPO4": "Door-to-door collection: sorting rules unclear",
    "NOSPO5": "Door-to-door collection: containers inconvenient",
    "NOSPO6": "Door-to-door collection: cost",
    "NOSPO7": "Door-to-door collection: other reason",
    "NOSPO8": "Door-to-door collection: not provided in the area",
    "SODPOAPO": "Satisfied with door-to-door waste collection",

    # --- Trust / social capital ----------------------------------------------
    "PUNTIFI1": "Trust 0-10: Italian Parliament",
    "PUNTIFI5": "Trust 0-10: European Parliament",
    "PUNTIFI8": "Trust 0-10: regional government",
    "PUNTIFI10": "Trust 0-10: municipal government",
    "PUNTIFI2": "Trust 0-10: judicial system",
    "PUNTIFI3": "Trust 0-10: police forces",
    "PUNTIFI4": "Trust 0-10: political parties",
    "PUNTIFI12": "Trust 0-10: fire brigade",
    "PUNTIFI13": "Trust 0-10: other institution",
    "FIDSCIE": "Trust 0-10: scientists",
    "FIDMED": "Trust 0-10: public health (SSN) doctors",
    "FIDINF": "Trust 0-10: other SSN personnel",
    "FIDUCIA": "General trust in other people",
    "FIDU1": "Lost wallet returned by a neighbour (probability)",
    "FIDU2": "Lost wallet returned by a police officer (probability)",
    "FIDU3": "Lost wallet returned by a stranger (probability)",

    # --- Well-being / civic life ---------------------------------------------
    "VOTOVI": "Life satisfaction 0-10",
    "FUTUASP": "Expectations about the future",
    "SICURO": "Feels safe walking alone in the dark in the area",
    "POLITI": "How often follows Italian politics",
    "PWEB": "Informs about politics through the internet",
    "LQUOT": "Reads newspapers weekly",
    "QUONLINE": "Reads newspapers online weekly",
    "RIVSET": "Reads weekly magazines",
    "NOSETT": "Reads non-weekly periodicals",
    "AMICI": "Frequency of meeting friends in free time",
    "VICINI": "Frequency of meeting neighbours",
    "TEMLIB": "Satisfaction with free time (last 12 months)",
    "VOLON": "Volunteered for associations/groups (last 12 months)",
    "FINAS": "Gave money to an association (last 12 months)",
    "DIBPO": "Attended a political debate (last 12 months)",
    "AMBIENTE": "Satisfaction with the environmental situation of the area",
    "SALUT": "Satisfaction with own health (last 12 months)",
    "SITEC": "Satisfaction with the economic situation (last 12 months)",
    "RELFAM": "Satisfaction with family relations",
    "RELAM": "Satisfaction with friendships",
    "CHIES": "Church / place-of-worship attendance frequency",

    # --- Books / ICT ---------------------------------------------------------
    "LIBFAM": "Number of books in the household",
    "LIBRI": "Bought books (last 3 months)",
    "NLIBRIM": "Number of books bought",
    "USOCEL": "Mobile phone use frequency",
    "TELCEL": "Household owns a mobile phone",
    "NTELCELM": "Number of mobile phones owned",
    "PCTEMPO": "Has ever used a personal computer",
    "INTTEMPO": "Internet use frequency",
    "INTATT30A": "Watched on-demand video (last 3 months)",
    "INTATT30B": "Listened to internet radio (last 3 months)",
    "INTATT28B": "Played or downloaded games (last 3 months)",
    "INTATT7BN": "Watched streaming TV (last 3 months)",
    "INTATT14": "Internet banking use (last 3 months)",
    "COM_STUD": "Used communication tools for study",
    "MATDID": "Used online study material",
    "PCOPE_PASTE": "Copied/pasted content between files",
    "PCOPE_SF": "Installed/loaded software or apps (last 3 months)",
    "PCOPE_FILE": "Copied or moved files between folders",
    "PCOPEPH": "Uploaded/copied photos",
    "PCOPEWO": "Used word-processing software (last 3 months)",
    "PCOPECO": "Online file-sharing",
    "COOKIE": "Cookie-related internet behaviour",
    "LIMCOO": "Limited cookie use",
    "INTUSO1": "Internet use: info from public-administration sites",
    "INTUSO2": "Internet use: downloading P.A. forms",
    "INTUSO3": "Internet use: submitting filled-in forms online",
    "INTCOM": "Bought/ordered goods or services online",
    "INTATT8": "Read newspapers/info/magazines online (last 3 months)",
    "INTATT11": "Used online banking services (last 3 months)",
    "INTATT14": "Searched for health information online (last 3 months)",
    "INTATT16": "Searched for info on goods or services (last 3 months)",
    "INTSAL3": "Booked a doctor appointment online (last 3 months)",
    "INTFASC": "Accessed the electronic health record (last 3 months)",
    "INTALTSAL": "Used other online health services (last 3 months)",

    # --- Banks / payments -----------------------------------------------------
    "BANCA": "Has a bank account",
    "BANCM": "Owns a bancomat/ATM card",
    "CCRED": "Owns a credit card",
    "USOPUL": "Uses online banking",
    "PCOPE": "Online payment indicator",

    # --- Food / diet ----------------------------------------------------------
    "FRUTTA": "Consumption frequency: fresh fruit",
    "VERD": "Consumption frequency: leaf vegetables",
    "POMOD": "Consumption frequency: tomatoes",
    "PZVERD": "Portions of vegetables per day",
    "PZFRUTTA": "Portions of fruit per day",
    "PATATE": "Consumption frequency: potatoes",
    "SNACK": "Consumption frequency: salty snacks",
    "DOLCI": "Consumption frequency: sweets",
    "QTSALE": "Pays attention to salt intake",
    "IODIO": "Uses iodised salt",
    "BGAS": "Sugary drinks (excluding mineral water)",
    "VINO": "Wine consumption",
    "BFPAS": "Alcohol consumption outside meals",
    "ALCOL": "Alcohol intake indicator",
    "SALUMI": "Consumption frequency: cured meats",
    "POLLO": "Consumption frequency: poultry",
    "COV": "Consumption frequency: eggs",
    "CBOV": "Consumption frequency: beef",
    "CMAIAL": "Consumption frequency: pork",
    "UOVA": "Consumption frequency: eggs",
    "FARM": "Consumption frequency: pulses",
    "COLAZ": "Breakfast habit",
    "CPESO": "Weight-control behaviour",

    # --- Health ---------------------------------------------------------------
    "LIMITA": "Health limitations lasting >= 6 months",
    "MH": "Mental-health index (SF-36)",
    "BMI": "Body mass index",
    "IPAR": "Chronic pathology indicator",
    "ARTRO": "Arthritis",
    "OSTEO": "Osteoporosis",
    "RADIO": "Chronic pathology: other",
    "FIDSCI": "Trust in science",
    "USOSS": "Went to a health authority (ASL) to book a visit",
    "UFFPOS": "Went to a post office (last 12 months)",

    # --- Public-service opening hours ----------------------------------------
    "GORAR": "Public office opening hours convenient",
    "CORAR": "Public office: preferred hours if changeable",
    "GOUSL": "ASL office opening hours convenient",
    "COUSL": "ASL office: preferred hours if changeable",
    "GOSPO": "Post office opening hours convenient",
    "COSPO": "Post office: preferred hours if changeable",

    # --- Other ----------------------------------------------------------------
    "UFFAN": "Went to a public office (last 12 months)",
    "RACCO": "Frequency of household waste separation",
    "VCOC": "Door-to-door collection use indicator",
    "RPENS": "Pension-related indicator",
    "RPARA": "Income-support indicator",
    "SPSPO": "Frequency of attending sports events (last 12 months)",
    "DISCO": "Frequency of going to discos/clubs (last 12 months)",
    "CINE": "Frequency of going to the cinema (last 12 months)",
    "MUSIC": "Frequency of classical-music concerts (last 12 months)",
    "ACMUS": "Frequency of other-music concerts (last 12 months)",
    "RADIO": "Habit of listening to the radio",
    "TELE": "Habit of watching TV",
    "HHTEL": "Hours of TV watched",
    "VGIOC": "Household owns a video-game console",
    "FOTODIG": "Household owns a digital camera",
    "TELCOL": "Household owns a TV set",
    "NTELCO": "Number of television sets",
    "NAUTOM": "Number of cars owned",
    "TELCIN": "Home cinema/TV equipment",
    "LEZPR": "Private lessons indicator",
    "LING": "Foreign language indicator",
    "USORA": "Working hours use",
    "HHSCLA": "Household social class",
    "DIVCOM": "Residence: commuting municipality",
    "STCPM": "Marital status before marriage",
    "NUMNU2": "Second household unit indicator",
    "RPNUC2": "Kinship relation in household nucleus 2",
    "ANNO": "Survey year",
    "COEFIN": "Reconstructed financial coefficient",
}

# Ordered by convenience for printing block headers.
DRIVER_BLOCKS: dict[str, list[str]] = {
    "SOCIO": [
        "SESSO", "ETAMi", "ISTRMi", "CONDMi", "REDPRMi",
        "TIPFA2Mi", "STCIVMi", "NCOMP",
    ],
    "ATTITUDES": [
        "SERRA", "ESTINZ", "CAMCLI", "SMARIF", "AMRUM", "IARIA", "INQSU",
        "INQFIU", "DISDR", "CATASTR", "FORES2", "INQELET2", "PAESAG2",
        "ESRISO2", "ALTAMB2",
    ],
    "SPATIAL": [
        "REGMf", "RIPMf", "SPORCO", "PARCH", "COLMP", "TRAF", "INQAR",
        "RUMORE", "CRIM", "ODSGR", "ILLSTR", "CONPAV", "POAPO", "ECOSTAZ",
        "GODAB",
    ],
}

# The 8 sibling sustainable-behaviour items.  They are the closest relatives of
# the two targets (same battery) and are deliberately EXCLUDED from the driver
# model so that the analysis focuses on socio-demographic, attitude and spatial
# determinants rather than on mirror behaviours.
SIBLING_BEHAVIOURS: list[str] = [
    "ETICHET", "BIOLOG", "ALOCAL", "GCARTE", "DOPFIL", "ARUMOR", "TRASPO",
    "USAGETT",
]


def describe(feature: str) -> str:
    """Return the plain-language description of a feature (fallback: the code)."""
    return FEATURE_DESCRIPTIONS.get(feature, feature)


def describe_block(features: list[str]) -> list[str]:
    """Return ``[feature, description]`` pairs for a list of features."""
    return [[f, describe(f)] for f in features]


def render_table(
    df,
    title: str | None = None,
    index: bool = False,
    formats: dict | None = None,
    width: int = 140,
) -> str:
    """Render a DataFrame as an aligned plain-text table.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to render.
    title : str, optional
        Optional header line printed above the table.
    index : bool
        Include the DataFrame index as the first column.
    formats : dict, optional
        Column-name -> format-spec map, e.g. {"p": ".4f"}.  Non-matching
        columns are converted to ``str``.
    width : int
        Soft total width used to cap over-long cell values.
    """
    formats = formats or {}
    cols = list(df.columns)
    rows: list[list[str]] = []

    def cell(value, fmt):
        try:
            if fmt and value is not None and not (isinstance(value, float) and (value != value)):
                return format(value, fmt)
        except (ValueError, TypeError):
            pass
        text = str(value)
        return text if len(text) <= width else text[: width - 3] + "..."

    header = []
    data_rows: list[list[str]] = []
    if index:
        header.append(str(df.index.name) if df.index.name is not None else "#")
        for idx_val in df.index:
            data_rows.append([cell(idx_val, None)])
    for col in cols:
        header.append(str(col))
        fmt = formats.get(col)
        for r, val in enumerate(df[col]):
            if len(data_rows) <= r:
                data_rows.append([])
            data_rows[r].append(cell(val, fmt))

    # Left-align text, right-align numeric-looking cells.
    col_widths = [len(h) for h in header]
    numeric = []
    for j, col in enumerate(cols):
        if index and j == 0:
            numeric.append(False)
            continue
        is_num = pd.api.types.is_numeric_dtype(df[col])
        numeric.append(is_num)
    if index:
        numeric = [False] + numeric
    for r in data_rows:
        for j, v in enumerate(r):
            col_widths[j] = max(col_widths[j], len(v))

    sep = "-+-".join("-" * w for w in col_widths)
    lines = []
    if title:
        lines.append(title)
        lines.append("=" * min(max(sum(col_widths) + 3 * (len(col_widths) - 1), len(title)), 160))

    def format_row(values, align):
        parts = []
        for j, v in enumerate(values):
            if align[j]:
                parts.append(v.rjust(col_widths[j]))
            else:
                parts.append(v.ljust(col_widths[j]))
        return " | ".join(parts)

    lines.append(format_row(header, [False] * len(header)))
    lines.append(sep)
    for r in data_rows:
        lines.append(format_row(r, numeric))
    return "\n".join(lines)
