"""
Generates realistic Indian bank-statement narrations with category labels.

Why: the assignment provides no labelled data, but a traditional ML model needs
training examples. Real Indian narrations follow a small number of channel formats
(UPI, NEFT, IMPS, POS, ATM, NACH, BBPS...), so we combine those channel templates
with merchant lists per category to produce a large, varied, labelled dataset.

Merchants are split into TRAIN / TEST groups so the model is evaluated on merchants
it has never seen - which is the real-world situation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

HANDLES = ["ybl", "paytm", "okaxis", "okhdfcbank", "oksbi", "okicici", "ibl", "axl", "upi", "apl"]
BANK_CODES = ["YESB", "HDFC", "ICIC", "SBIN", "UTIB", "PYTM", "KKBK", "BARB", "PUNB", "IDIB"]
IFSC_PREFIX = ["HDFC0", "ICIC0", "SBIN0", "UTIB0", "KKBK0", "BARB0", "PUNB0", "YESB0"]
CITIES = ["MUMBAI", "PUNE", "BANGALORE", "DELHI", "HYDERABAD", "CHENNAI", "AHMEDABAD", "KOLKATA", "NOIDA", "SURAT"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
REMARKS = ["UPI", "Payment from Phone", "NA", "pay", "Sent using Paytm", "Paid via CRED", "ok", "payment", ""]
FIRST = ["RAHUL", "PRIYA", "AMIT", "SNEHA", "VIKRAM", "ANJALI", "ROHIT", "POOJA", "KARAN", "NEHA",
         "SURESH", "KAVITA", "ARJUN", "DIVYA", "MANOJ", "RITU", "SANJAY", "MEERA", "NIKHIL", "SWATI",
         "RAJESH", "ASHA", "DEEPAK", "SUNITA", "HARSH", "TANVI", "GAURAV", "ISHA", "VARUN", "LATA"]
LAST = ["SHARMA", "PATEL", "SHAH", "MEHTA", "IYER", "REDDY", "GUPTA", "SINGH", "KUMAR", "JOSHI",
        "NAIR", "DESAI", "KULKARNI", "BANERJEE", "VERMA", "RAO", "PILLAI", "CHOPRA", "MISHRA", "JAIN"]
COMPANIES = ["INFOSYS LIMITED", "TATA CONSULTANCY SERVICES", "WIPRO LIMITED", "ACCENTURE SOLUTIONS PVT",
             "ACME TECHNOLOGIES PVT LTD", "HCL TECHNOLOGIES", "TECH MAHINDRA LTD", "CAPGEMINI INDIA",
             "ZENSAR TECHNOLOGIES", "PERSISTENT SYSTEMS", "LARSEN AND TOUBRO", "BEFREE SOLUTIONS PVT LTD"]


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _vpa(rng: random.Random, name: str) -> str:
    base = "".join(ch for ch in name.lower() if ch.isalnum())[:14]
    if rng.random() < 0.4:
        base += _digits(rng, rng.randint(2, 4))
    return f"{base}@{rng.choice(HANDLES)}"


def _card(rng: random.Random) -> str:
    return f"{_digits(rng, 4)}XXXXXX{_digits(rng, 4)}"


# ---------------------------------------------------------------- channel templates
def upi_dr(rng, m):
    ref = _digits(rng, 12)
    return rng.choice([
        f"UPI/DR/{ref}/{m}/{rng.choice(BANK_CODES)}/{_vpa(rng, m)}/{rng.choice(REMARKS)}",
        f"UPI-{m}-{_vpa(rng, m).upper()}-{rng.choice(IFSC_PREFIX)}{_digits(rng, 6)}-{ref}-{rng.choice(REMARKS).upper()}",
        f"UPI/{ref}/{rng.choice(REMARKS)}/{_vpa(rng, m)}/{rng.choice(BANK_CODES)}",
        f"MPAY/UPI/TRANSFER/{ref}/{m}",
    ])


def upi_cr(rng, m):
    ref = _digits(rng, 12)
    return rng.choice([
        f"UPI/CR/{ref}/{m}/{rng.choice(BANK_CODES)}/{_vpa(rng, m)}/{rng.choice(REMARKS)}",
        f"UPI-{m}-{_vpa(rng, m).upper()}-{rng.choice(IFSC_PREFIX)}{_digits(rng, 6)}-{ref}-{rng.choice(REMARKS).upper()}",
        f"BY UPI CREDIT {ref} {m}",
    ])


def pos(rng, m):
    return rng.choice([
        f"POS {_card(rng)} {m} {rng.choice(CITIES)}",
        f"POS/{m}/{rng.choice(CITIES)}/{_digits(rng, 6)}",
        f"PCD/{_card(rng)[-4:]}/{m} {rng.choice(CITIES)}/{_digits(rng, 6)}",
        f"ME DC SI {_card(rng)[-4:]} {m}",
    ])


def ecom(rng, m):
    return rng.choice([
        f"PG/{m}/{_digits(rng, 10)}",
        f"BIL/ONL/{_digits(rng, 9)}/{m}/{_digits(rng, 6)}",
        f"{m} PAYMENTS/{_digits(rng, 10)}",
        f"ONLINE PAYMENT {m} {_digits(rng, 8)}",
    ])


def bbps(rng, m):
    return rng.choice([
        f"BIL/BPAY/{_digits(rng, 9)}/{m}/{_digits(rng, 10)}",
        f"BBPS/{m}/{_digits(rng, 12)}",
        f"BILLPAY {m} {_digits(rng, 10)}",
    ])


def nach(rng, m, what="EMI"):
    return rng.choice([
        f"ACH D- {m}-{_digits(rng, 10)}",
        f"NACH DR {m} {what} {_digits(rng, 8)}",
        f"NACH/{m}/{_digits(rng, 12)}",
        f"ECS DR {m} {_digits(rng, 9)}",
    ])


@dataclass
class Category:
    name: str
    direction: str              # "debit" | "credit"
    amount: tuple               # (low, high) rupees
    merchants: list
    channels: list              # functions (rng, merchant) -> str
    extra: list = None          # functions (rng) -> str, merchant-independent


def _rent(rng):
    n = _person(rng)
    mon = rng.choice(MONTHS)
    return rng.choice([
        f"UPI/DR/{_digits(rng, 12)}/{n}/{rng.choice(BANK_CODES)}/{_vpa(rng, n)}/RENT {mon}",
        f"IMPS/P2A/{_digits(rng, 12)}/{n}/HOUSE RENT",
        f"NEFT DR-{rng.choice(IFSC_PREFIX)}{_digits(rng, 6)}-{n}-RENT FOR {mon}",
        f"CRED RENT PAY/{_digits(rng, 10)}",
        f"NOBROKER RENT PAYMENT {_digits(rng, 8)}",
        f"TO RENT {mon} {n}",
    ])


def _p2p_out(rng):
    n = _person(rng)
    return rng.choice([
        f"UPI/DR/{_digits(rng, 12)}/{n}/{rng.choice(BANK_CODES)}/{_vpa(rng, n)}/{rng.choice(REMARKS)}",
        f"IMPS/P2A/{_digits(rng, 12)}/{n}/{rng.choice(BANK_CODES)}",
        f"NEFT DR-{rng.choice(IFSC_PREFIX)}{_digits(rng, 6)}-{n}-NETBANK-{_digits(rng, 10)}",
        f"TO TRANSFER-{n}",
        f"MB:SENT TO {n}",
        f"FT - DR - {_digits(rng, 12)} - {n}",
    ])


def _p2p_in(rng):
    n = _person(rng)
    return rng.choice([
        f"UPI/CR/{_digits(rng, 12)}/{n}/{rng.choice(BANK_CODES)}/{_vpa(rng, n)}/{rng.choice(REMARKS)}",
        f"IMPS/P2A/{_digits(rng, 12)}/{n}/CREDIT",
        f"NEFT CR-{rng.choice(IFSC_PREFIX)}{_digits(rng, 6)}-{n}-{_digits(rng, 10)}",
        f"BY TRANSFER-{n}",
        f"FT - CR - {_digits(rng, 12)} - {n}",
    ])


def _salary(rng, c):
    mon, yr = rng.choice(MONTHS), rng.choice(["2024", "2025", "24", "25"])
    return rng.choice([
        f"NEFT CR-{rng.choice(IFSC_PREFIX)}{_digits(rng, 6)}-{c}-SALARY {mon} {yr}",
        f"SALARY CREDIT {c}",
        f"SAL/{c}/{mon}{yr}",
        f"BY SALARY {c} {mon}",
        f"ACH C- {c} PAYROLL-{_digits(rng, 8)}",
        f"NEFT/{_digits(rng, 12)}/{c}/SAL {mon}",
    ])


def _interest(rng):
    d = f"{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-2024"
    return rng.choice([
        "CREDIT INTEREST", f"INT.PD:{d} TO {d}", "SB INTEREST", "INTEREST CREDIT",
        f"FD INTEREST {_digits(rng, 10)}", f"{_digits(rng, 10)}:INT.PD:{d}", "SAVINGS INTEREST CAPITALISED",
        f"ACH C- {rng.choice(['ITC LTD', 'INFOSYS LTD', 'HDFC BANK LTD', 'RELIANCE INDUSTRIES', 'TCS LTD'])} DIVIDEND",
        f"DIV {rng.choice(['COAL INDIA', 'ONGC', 'NTPC', 'POWERGRID', 'HUL'])} FINAL DIVIDEND {_digits(rng, 6)}",
    ])


def _atm(rng):
    return rng.choice([
        f"ATM WDL/{rng.choice(CITIES)}/{_digits(rng, 6)}",
        f"NWD-{_card(rng)}-{_digits(rng, 8)}-{rng.choice(CITIES)}",
        f"ATW-{_card(rng)}-S1AW{_digits(rng, 4)}-{rng.choice(CITIES)}",
        "CASH WITHDRAWAL SELF",
        f"ATM CASH {_digits(rng, 6)} {rng.choice(CITIES)}",
        f"CHQ PAID-SELF-{_digits(rng, 6)}",
        f"CASH WDL ATM {rng.choice(CITIES)} {_digits(rng, 6)}",
    ])


def _charges(rng):
    return rng.choice([
        "SMS ALERT CHARGES", "DEBIT CARD ANNUAL FEE", "MIN BAL CHGS", "ATM TXN CHARGES",
        "CHQ BOOK ISSUE CHARGES", "NEFT CHARGES", f"CONSOLIDATED CHARGES FOR A/C {_digits(rng, 4)}",
        "DC AMC CHARGES", "IMPS CHARGES INCL GST", "NON MAINTENANCE OF MIN BAL", "SERVICE CHARGES",
        f"DEBIT CARD FEE {_digits(rng, 4)}", "CASH HANDLING CHARGES", "ECS RETURN CHARGES",
    ])


def _tax(rng):
    return rng.choice([
        "INCOME TAX PAYMENT", f"ADVANCE TAX CHALLAN {_digits(rng, 8)}", "TDS ON INTEREST",
        f"GST PAYMENT CHALLAN {_digits(rng, 10)}", "PROFESSIONAL TAX", f"CBDT/ITNS280/{_digits(rng, 10)}",
        "SELF ASSESSMENT TAX", f"TIN/ITNS280/{_digits(rng, 12)}", "PROPERTY TAX MCGM",
    ])


def _refund(rng, m):
    return rng.choice([
        f"REFUND/{m}/{_digits(rng, 10)}", f"REV-UPI-{_digits(rng, 12)}-{m}", f"CASHBACK {m}",
        f"UPI/CR/{_digits(rng, 12)}/{m}/{rng.choice(BANK_CODES)}/{_vpa(rng, m)}/REFUND",
        f"REVERSAL {m} {_digits(rng, 8)}", f"{m} REFUND {_digits(rng, 6)}",
    ])


def _cash_dep(rng):
    c = rng.choice(CITIES)
    return rng.choice([
        f"CASH DEPOSIT {c} BRANCH", f"BY CASH-{c}", f"CDM DEPOSIT {_digits(rng, 6)}", "CASH DEP BY SELF",
        f"CASH DEPOSIT CDM {c} {_digits(rng, 4)}", "BY CASH DEPOSIT",
    ])


def _cc(rng, m):
    return rng.choice([
        f"BIL/ONL/{_digits(rng, 9)}/{m} CREDIT CARD/{_digits(rng, 6)}",
        f"IB BILLPAY DR-{m}-{_card(rng)}",
        f"CC PAYMENT {m} {_digits(rng, 4)}",
        f"CRED CLUB/{_digits(rng, 10)}/CREDIT CARD BILL",
        f"UPI/DR/{_digits(rng, 12)}/CRED/{rng.choice(BANK_CODES)}/cred.club@axisb/CC BILL",
    ])


def _invest(rng, m):
    return rng.choice([
        f"NACH/{m}/SIP/{_digits(rng, 10)}", f"ACH D- {m}-SIP{_digits(rng, 8)}",
        f"BSE LIMITED/{m}/{_digits(rng, 8)}", f"UPI/DR/{_digits(rng, 12)}/{m}/{rng.choice(BANK_CODES)}/{_vpa(rng, m)}/ADD FUNDS",
        f"{m} MUTUAL FUND PURCHASE", f"INDIAN CLEARING CORP/{m}/{_digits(rng, 8)}",
    ])


CATEGORIES = [
    Category("Food & Dining", "debit", (120, 2500),
             ["SWIGGY", "ZOMATO", "DOMINOS PIZZA", "MCDONALDS", "KFC", "STARBUCKS", "HALDIRAMS", "BARBEQUE NATION",
              "CAFE COFFEE DAY", "BURGER KING", "PIZZA HUT", "CHAAYOS", "BEHROUZ BIRYANI", "FAASOS", "SUBWAY",
              "THEOBROMA", "WOW MOMO", "BIKANERVALA", "PARADISE BIRYANI", "EATSURE"],
             [upi_dr, pos, ecom]),
    Category("Groceries", "debit", (200, 6000),
             ["BIGBASKET", "BLINKIT", "ZEPTO", "DMART", "RELIANCE FRESH", "MORE RETAIL", "SPENCERS", "JIOMART",
              "NATURES BASKET", "GROFERS", "STAR BAZAAR", "RATNADEEP SUPERMARKET", "SWIGGY INSTAMART", "DUNZO DAILY"],
             [upi_dr, pos, ecom]),
    Category("Shopping", "debit", (300, 25000),
             ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "NYKAA", "MEESHO", "CROMA", "RELIANCE DIGITAL", "DECATHLON",
              "LIFESTYLE STORES", "SHOPPERS STOP", "IKEA INDIA", "TATA CLIQ", "PANTALOONS", "WESTSIDE", "LENSKART",
              "FIRSTCRY", "PEPPERFRY", "BEWAKOOF", "H AND M"],
             [upi_dr, pos, ecom]),
    Category("Travel & Transport", "debit", (60, 18000),
             ["UBER", "OLA CABS", "RAPIDO", "IRCTC", "MAKEMYTRIP", "GOIBIBO", "INDIGO", "AIR INDIA", "REDBUS",
              "YATRA", "CLEARTRIP", "MUMBAI METRO", "NAMMA YATRI", "VISTARA", "IXIGO", "FASTAG RECHARGE", "BLUSMART"],
             [upi_dr, pos, ecom]),
    Category("Fuel", "debit", (300, 5000),
             ["INDIAN OIL", "HPCL", "BPCL", "BHARAT PETROLEUM", "SHELL INDIA", "NAYARA ENERGY", "HP PETROL PUMP",
              "IOCL FUEL STATION", "JIO BP", "ESSAR PETROL"],
             [upi_dr, pos]),
    Category("Utilities & Bills", "debit", (150, 6000),
             ["MSEDCL", "BESCOM", "TATA POWER", "ADANI ELECTRICITY", "TNEB", "AIRTEL", "JIO PREPAID", "VODAFONE IDEA",
              "BSNL", "ACT FIBERNET", "MAHANAGAR GAS", "INDANE GAS", "HATHWAY BROADBAND", "TATA PLAY", "BWSSB WATER",
              "TORRENT POWER", "BSES RAJDHANI", "EXCITEL BROADBAND"],
             [bbps, upi_dr, ecom]),
    Category("Rent", "debit", (8000, 45000), [], [], [_rent]),
    Category("EMI & Loans", "debit", (2500, 40000),
             ["BAJAJ FINANCE", "HDFC LTD", "ICICI HOME LOAN", "TATA CAPITAL", "HOME CREDIT", "IDFC FIRST LOAN",
              "LIC HOUSING FINANCE", "MUTHOOT FINANCE", "AXIS BANK LOAN", "KOTAK MAHINDRA PRIME", "FULLERTON INDIA",
              "MAHINDRA FINANCE", "SBI CAR LOAN"],
             [lambda r, m: nach(r, m, "EMI"), lambda r, m: f"EMI {_digits(r, 10)} {m}",
              lambda r, m: f"LOAN RECOVERY {m} {_digits(r, 8)}"]),
    Category("Credit Card Payment", "debit", (1500, 60000),
             ["HDFC", "ICICI", "SBI CARD", "AXIS", "KOTAK", "AMEX", "RBL", "INDUSIND", "YES BANK", "ONECARD"],
             [_cc]),
    Category("Insurance", "debit", (800, 35000),
             ["LIC OF INDIA", "HDFC LIFE", "ICICI PRULIFE", "SBI LIFE", "STAR HEALTH", "MAX LIFE", "BAJAJ ALLIANZ",
              "POLICYBAZAAR", "TATA AIG", "NIVA BUPA", "CARE HEALTH INSURANCE", "ACKO GENERAL", "DIGIT INSURANCE"],
             [lambda r, m: nach(r, m, "PREMIUM"), lambda r, m: f"{m} PREMIUM {_digits(r, 10)}", ecom]),
    Category("Investments", "debit", (500, 50000),
             ["ZERODHA", "GROWW", "UPSTOX", "ICICI SECURITIES", "HDFC MUTUAL FUND", "SBI MUTUAL FUND",
              "AXIS MUTUAL FUND", "NIPPON INDIA MF", "KUVERA", "PAYTM MONEY", "ANGEL ONE", "MIRAE ASSET MF",
              "PARAG PARIKH MF", "COIN BY ZERODHA"],
             [_invest],
             [lambda r: f"PPF DEPOSIT {_digits(r, 10)}", lambda r: f"RD INSTALLMENT {_digits(r, 10)}",
              lambda r: f"FD BOOKED {_digits(r, 10)}", lambda r: f"NPS TRUST {_digits(r, 12)}"]),
    Category("Healthcare", "debit", (100, 20000),
             ["APOLLO PHARMACY", "MEDPLUS", "PHARMEASY", "NETMEDS", "TATA 1MG", "FORTIS HOSPITAL", "MAX HEALTHCARE",
              "PRACTO", "DR LAL PATHLABS", "THYROCARE", "MANIPAL HOSPITAL", "WELLNESS FOREVER", "NARAYANA HEALTH",
              "METROPOLIS LABS"],
             [upi_dr, pos, ecom]),
    Category("Education", "debit", (500, 60000),
             ["BYJUS", "UNACADEMY", "COURSERA", "UDEMY", "VEDANTU", "CHEGG", "UPGRAD", "SIMPLILEARN",
              "PHYSICS WALLAH", "DPS SCHOOL FEES", "MIT UNIVERSITY FEE", "GREAT LEARNING", "SCALER ACADEMY"],
             [upi_dr, ecom, lambda r, m: f"NEFT DR-{r.choice(IFSC_PREFIX)}{_digits(r, 6)}-{m}-FEES"]),
    Category("Entertainment & Subscriptions", "debit", (99, 3000),
             ["NETFLIX", "SPOTIFY", "HOTSTAR", "PRIME VIDEO", "BOOKMYSHOW", "PVR CINEMAS", "INOX", "YOUTUBE PREMIUM",
              "SONYLIV", "ZEE5", "APPLE MEDIA SERVICES", "GOOGLE PLAY", "STEAM GAMES", "JIOCINEMA", "AUDIBLE",
              "DISNEY PLUS"],
             [upi_dr, ecom, pos]),
    Category("Cash Withdrawal", "debit", (500, 20000), [], [], [_atm]),
    Category("Bank Charges", "debit", (5, 900), [], [], [_charges]),
    Category("Taxes", "debit", (1000, 80000), [], [], [_tax]),
    Category("Transfer Out", "debit", (100, 30000), [], [], [_p2p_out]),
    Category("Salary", "credit", (25000, 150000), COMPANIES, [_salary]),
    Category("Interest & Dividends", "credit", (10, 5000), [], [], [_interest]),
    Category("Refunds & Cashback", "credit", (20, 8000),
             ["AMAZON", "FLIPKART", "MYNTRA", "SWIGGY", "ZOMATO", "PAYTM", "PHONEPE", "MAKEMYTRIP", "IRCTC", "AJIO",
              "GOOGLEPAY", "CRED", "NYKAA", "UBER"],
             [_refund], [lambda r: "IT REFUND", lambda r: "REVERSAL OF CHARGES", lambda r: f"CHARGES REVERSAL {_digits(r, 6)}"]),
    Category("Cash Deposit", "credit", (1000, 50000), [], [], [_cash_dep]),
    Category("Transfer In", "credit", (100, 40000), [], [], [_p2p_in]),
]

CATEGORY_BY_NAME = {c.name: c for c in CATEGORIES}
ALL_CATEGORY_NAMES = [c.name for c in CATEGORIES] + ["Uncategorized"]


def split_merchants(seed: int = 7, test_frac: float = 0.25):
    """Deterministically split each category's merchants into train/test groups."""
    rng = random.Random(seed)
    train, test = {}, {}
    for c in CATEGORIES:
        ms = c.merchants[:]
        rng.shuffle(ms)
        k = int(round(len(ms) * test_frac)) if len(ms) >= 4 else 0
        test[c.name], train[c.name] = ms[:k], ms[k:]
    return train, test


def narration(cat: Category, rng: random.Random, merchants: list | None = None) -> str:
    """One narration for a category. `merchants` restricts which merchants may be used."""
    pool = merchants if merchants is not None else cat.merchants
    use_extra = cat.extra and (not pool or not cat.channels or rng.random() < 0.3)
    if use_extra:
        return rng.choice(cat.extra)(rng)
    return rng.choice(cat.channels)(rng, rng.choice(pool))


def make_dataset(n_per_category: int = 150, seed: int = 42, split: str = "all"):
    """Return list of (description, direction, category). split: 'train' | 'test' | 'all'."""
    rng = random.Random(seed)
    train_m, test_m = split_merchants()
    rows = []
    for c in CATEGORIES:
        if split == "train":
            pool = train_m[c.name]
        elif split == "test":
            pool = test_m[c.name] or train_m[c.name]
        else:
            pool = c.merchants
        for _ in range(n_per_category):
            text = narration(c, rng, pool if pool else None)
            # random casing noise - some banks print mixed/lower case
            if rng.random() < 0.1:
                text = text.title()
            rows.append((text, c.direction, c.name))
    return rows
