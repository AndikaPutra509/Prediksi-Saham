# 📈 IDX STOCK SCANNER - QUADRUPLE ENGINE (FIXED)
## 4 Engine Terpisah:
## 
## 🔵 ENGINE 1: SWING (Mean Reversion) - EXISTING
## 🟢 ENGINE 2: INTRADAY LIQUID (Momentum) - EXISTING
## 🔴 ENGINE 3: INTRADAY GORENGAN (Early Momentum) - EXISTING
## 🟣 ENGINE 4: INVESTASI (Quality + Trend) - FIXED

# =============================================================================
# 1. INSTALL DEPENDENCIES
# =============================================================================

!pip install yfinance pandas numpy ta tabulate -q

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import time
import math
import pickle
import os
from tabulate import tabulate
from collections import defaultdict
import logging
import matplotlib.pyplot as plt
import random

# Matikan semua logging
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

print("✅ Dependencies installed")

# =============================================================================
# 2. STOCKBIT UNIVERSE (FULL)
# =============================================================================

STOCKBIT_UNIVERSE = [
    "AALI", "ABBA", "ABDA", "ABMM", "ACES", "ADES", "ADHI", "ADMF", "ADMG", "ADRO",
    "AGAR", "AGII", "AGRO", "AHAP", "AIMS", "AISA", "AKRA", "AKSI", "ALDO", "ALKA",
    "ALMI", "AMAG", "AMFG", "AMMN", "AMRT", "ANDI", "ANJT", "ANTM", "APEX", "APIC",
    "APLN", "ARCI", "ARGO", "ARII", "ARNA", "ARTA", "ARTO", "ASBI", "ASDM", "ASGR",
    "ASHA", "ASII", "ASLI", "ASMI", "ASPI", "ASRI", "ASRM", "ASSA", "ATLA", "AUTO",
    "AVIA", "AWAN", "AYLS", "BABP", "BACA", "BALI", "BANK", "BAPA", "BAPI", "BATA",
    "BAYU", "BBCA", "BBHI", "BBKP", "BBLD", "BBNI", "BBRI", "BBRM", "BBSS", "BBTN",
    "BBYB", "BCAP", "BCIC", "BDMN", "BEEF", "BEKS", "BELL", "BEST", "BFIN", "BGTG",
    "BHAT", "BIMA", "BINA", "BIPP", "BIPI", "BIRD", "BISI", "BJBR", "BJTM", "BKSL",
    "BLTA", "BLUE", "BMAS", "BMBL", "BMRI", "BMSR", "BMTR", "BNBA", "BNBR", "BNGA",
    "BNII", "BNLI", "BOBA", "BOLT", "BOSS", "BPFI", "BPII", "BPTR", "BRAM", "BREN",
    "BRIS", "BRMS", "BRNA", "BRPT", "BSDE", "BSIM", "BSSR", "BTEL", "BTON", "BTPN",
    "BTPS", "BUDI", "BULL", "BUMI", "BUVA", "BWPT", "BYAN", "CAMP", "CANI", "CARS",
    "CASA", "CASS", "CBDK", "CBMF", "CCSI", "CDAX", "CEKA", "CENT", "CFIN", "CITA",
    "CITY", "CKRA", "CLEO", "CLPI", "CMNP", "CMPP", "CMRY", "CNKO", "CNTX", "COAL",
    "COCO", "COWL", "CPIN", "CPRO", "CSAP", "CSIS", "CTBN", "CTRA", "CTTH", "CUAN",
    "DAAZ", "DART", "DASA", "DAYA", "DCII", "DEGA", "DEWA", "DFAM", "DGIK", "DIGI",
    "DILD", "DIVA", "DIVN", "DKFT", "DLTA", "DMAS", "DMND", "DNAR", "DNET", "DNLS",
    "DOID", "DOOH", "DPNS", "DPUM", "DSFI", "DSNG", "DSSA", "DUCK", "DUTI", "DVLA",
    "DYAN", "EASI", "EASY", "EBMT", "ECII", "EDGE", "EKAD", "ELBA", "ELSA", "ELTY",
    "EMBR", "EMDE", "EMTK", "ENRG", "ENVY", "ENZO", "EPAC", "EPMT", "ERAA", "ERTX",
    "ESSA", "ESTA", "ESTI", "ETWA", "EXCL", "FAST", "FASW", "FILM", "FISH", "FITT",
    "FKSF", "FLMC", "FMII", "FORE", "FORU", "FORZ", "FPNI", "FREN", "FUJI", "FUTR",
    "GAMA", "GDST", "GDYR", "GEMS", "GGRM", "GGRP", "GHON", "GIDS", "GJTL", "GLVA",
    "GMFI", "GMTD", "GOLD", "GOOD", "GOTO", "GPRA", "GRPH", "GSMF", "GTBO", "GTRA",
    "GTSI", "GULA", "GZCO", "HADE", "HDFA", "HDIT", "HEAL", "HERO", "HITS", "HKMU",
    "HMSP", "HOKI", "HOMI", "HOPE", "HOTL", "HRME", "HRTA", "HRUM", "HSBK", "HSMP",
    "HUMI", "IBFN", "IBOS", "IBST", "ICBP", "ICON", "IDPR", "IFII", "IFSH", "IGAR",
    "IIKP", "IKAI", "IKAN", "IMAS", "IMJS", "IMPC", "INAF", "INAI", "INCF", "INCI",
    "INCO", "INDF", "INDS", "INDX", "INDY", "INET", "INKP", "INPC", "INPP", "INPS",
    "INRU", "INTA", "INTD", "INTP", "IPCC", "IPCM", "IPOL", "IRRA", "ISAT", "ISEA",
    "ISSP", "ITIC", "ITMG", "JAST", "JAWA", "JAYA", "JECC", "JEMB", "JFAS", "JGLE",
    "JHON", "JIHD", "JKON", "JKSW", "JMAS", "JPFA", "JPII", "JPUR", "JRPT", "JSKY",
    "JSMR", "JSPT", "JTNB", "KAEF", "KAQI", "KARW", "KBLI", "KBLM", "KBRT", "KBRI",
    "KDSI", "KDTN", "KEEN", "KETR", "KICI", "KIJA", "KINO", "KIOS", "KJEN", "KKGI",
    "KLBF", "KMTR", "KOBX", "KOIN", "KOLI", "KONI", "KOTA", "KPAL", "KPIG", "KRAS",
    "KREN", "KRYA", "KSEL", "KUAS", "KUIC", "KUVO", "LAND", "LAPD", "LATA", "LBAK",
    "LCGP", "LCKM", "LEAD", "LIFE", "LINK", "LION", "LISA", "LMAS", "LMPI", "LMSH",
    "LPCK", "LPGI", "LPIN", "LPKR", "LPLI", "LPPF", "LPPS", "LSIP", "LSPI", "LTLS",
    "LUCY", "MABA", "MABH", "MAGP", "MAIN", "MAMI", "MAPA", "MAPB", "MAPI", "MARA",
    "MASA", "MAYA", "MBAP", "MBCA", "MBMA", "MBSS", "MBTO", "MCAS", "MCPI", "MCOR",
    "MDIA", "MDKA", "MDKI", "MEDC", "MEDS", "MEGA", "MERK", "META", "MFIN", "MFMI",
    "MGLV", "MGNA", "MGRO", "MIDI", "MIKA", "MINA", "MIRA", "MITI", "MITT", "MKNT",
    "MKPI", "MLBI", "MLIA", "MLPL", "MLPT", "MLSL", "MMIX", "MMLP", "MNCN", "MOLI",
    "MPOW", "MPPA", "MPRO", "MPTJ", "MRAT", "MSIE", "MSIN", "MSKY", "MTDL", "MTFN",
    "MTLA", "MTPS", "MTSM", "MUDA", "MUTU", "MYOH", "MYOR", "MYRX", "MYSX", "NAGA",
    "NASI", "NATO", "NAYZ", "NCKL", "NELY", "NETV", "NFCX", "NICL", "NIKL", "NISP",
    "NITY", "NIYM", "NOBU", "NPGF", "NRCA", "NSSS", "NTBK", "NUSA", "NUSI", "OASA",
    "OCTN", "OKAS", "OMED", "ONIX", "OPMS", "ORNA", "OTBK", "PADA", "PADI", "PAMG",
    "PANR", "PANS", "PANU", "PAPA", "PASA", "PASS", "PBRX", "PBID", "PBSA", "PCAR",
    "PDES", "PDGD", "PDIN", "PEGE", "PGAS", "PGLI", "PGUN", "PICO", "PIDRA", "PJAA",
    "PKPK", "PLAN", "PLAS", "PLIN", "PMJS", "PMMP", "PNBN", "PNBS", "PNIN", "PNLF",
    "PNSE", "POLI", "POLL", "POLU", "POLY", "POOL", "PORT", "POWR", "PPGL", "PPRE",
    "PPRO", "PPSI", "PRAS", "PRDA", "PRIM", "PRIN", "PRLD", "PROD", "PROT", "PRTS",
    "PSAB", "PSBA", "PSDN", "PSGO", "PSKT", "PSSI", "PTBA", "PTDU", "PTIS", "PTMP",
    "PTPP", "PTPW", "PTRO", "PTSN", "PTSP", "PUDP", "PURA", "PURE", "PWON", "PYFA",
    "RACE", "RADIO", "RAFI", "RAJA", "RAKD", "RALS", "RANC", "RATU", "RBMS", "RDTX",
    "REAL", "RELI", "RIGS", "RIMO", "RISE", "RMBA", "RMKE", "ROCK", "RODA", "ROKI",
    "ROTI", "RRMI", "RUIS", "RUMI", "SABA", "SAFE", "SAME", "SAPX", "SARA", "SATO",
    "SBAT", "SBBP", "SBGA", "SBMA", "SBMF", "SCBD", "SCCC", "SCCO", "SCMA", "SCNP",
    "SDPC", "SDRA", "SEAN", "SECR", "SEMA", "SFAN", "SGER", "SGRO", "SHID", "SHIP",
    "SIDO", "SILO", "SIMA", "SIMP", "SIPD", "SIPO", "SKBM", "SKLT", "SKRN", "SLIS",
    "SMAR", "SMDR", "SMGR", "SMIL", "SMMT", "SMSM", "SMRA", "SNLK", "SNMS", "SOFA",
    "SONA", "SOSS", "SOUL", "SPMA", "SPMI", "SPNA", "SPRE", "SPTO", "SQBI", "SQMI",
    "SRAJ", "SRIL", "SRSN", "SSIA", "SSMS", "SSTM", "STAR", "STTP", "SUGI", "SULI",
    "SUPR", "SURI", "SWAT", "SWID", "TALD", "TAMA", "TAMU", "TAPG", "TARA", "TASP",
    "TATA", "TAXI", "TBIG", "TBLA", "TCID", "TDPM", "TELE", "TEMB", "TEMPO", "TIFA",
    "TIGA", "TINS", "TIRA", "TIRT", "TITA", "TKGA", "TKIM", "TLKM", "TMAS", "TMPO",
    "TMSH", "TOBA", "TOOL", "TOPS", "TOSK", "TOTL", "TOTO", "TOWR", "TPIA", "TPMA",
    "TRAM", "TRGU", "TRIO", "TRIS", "TRJA", "TRON", "TRST", "TRUB", "TRUK", "TRUS",
    "TSPC", "TUGU", "TURI", "TUVN", "TYRE", "UANG", "UCID", "UDIJ", "UFNX", "UGRO",
    "UJSN", "ULTJ", "UNIC", "UNIQ", "UNIT", "UNSP", "UNTR", "UNVR", "USFI", "VALU",
    "VICO", "VICI", "VIDI", "VISI", "VIVA", "VKTR", "VOKS", "VRNA", "VTNY", "WAPO",
    "WEGE", "WEHA", "WICO", "WIFI", "WIIM", "WIKA", "WINS", "WMUU", "WMPP", "WOOD",
    "WOWS", "WRKR", "WSBP", "WSKT", "WTON", "YELO", "YULE", "ZBRA", "ZINC", "ZONE"
]

print(f"✅ Universe: {len(STOCKBIT_UNIVERSE)} stocks")

# =============================================================================
# 3. SEKTOR MAPPING
# =============================================================================

ENERGY_SECTOR = ['ADRO', 'PTBA', 'ITMG', 'MEDC', 'PGAS', 'ENRG', 'BUMI', 'DOID']
MINING_GOLD = ['ANTM', 'MDKA', 'BRMS']
EXPORT_SECTOR = ['ANTM', 'INCO', 'TINS', 'HRUM', 'CPIN', 'JPFA', 'MAIN']

SECTOR_MAPPING = {
    'ENERGY': ENERGY_SECTOR + ['BYAN', 'INDY', 'HRUM', 'ARTI'],
    'MINING': MINING_GOLD + ['INCO', 'TINS', 'CITA', 'DKFT'],
    'FINANCE': ['BBCA', 'BBRI', 'BMRI', 'BBNI', 'PNBN', 'BJBR', 'BJTM', 'NISP', 'BDMN', 'BNLI', 'BNGA', 'BNII', 'BSIM'],
    'PROPERTY': ['PWON', 'BSDE', 'LPKR', 'CTRA', 'SMRA', 'ASRI', 'DILD', 'MDLN', 'ELTY', 'BIPP', 'BKSL', 'MTLA', 'MAPI'],
    'CONSUMER': ['UNVR', 'ICBP', 'INDF', 'GGRM', 'HMSP', 'KLBF', 'MYOR', 'SIDO', 'ULTJ', 'DLTA', 'MLBI', 'TCID', 'ROTI', 'SKBM'],
    'INFRASTRUCTURE': ['TLKM', 'JSMR', 'TOWR', 'TBIG', 'WIKA', 'PTPP', 'WSKT', 'ADHI', 'TOTL', 'ACST'],
    'INDUSTRIAL': ['ASII', 'GJTL', 'AUTO', 'BRAM', 'INDS', 'BOLT', 'IMP', 'PRAS', 'PBRX'],
    'TRADE': ['MAPI', 'ACES', 'RALS', 'LPPF', 'ERAA', 'MAPB', 'SONA', 'CSAP', 'MIDI', 'MFMI'],
    'AGRICULTURE': ['AALI', 'LSIP', 'SGRO', 'BWPT', 'SMAR', 'DSNG', 'JAWA'],
    'HEALTHCARE': ['MIKA', 'SILO', 'KLBF', 'KAEF', 'INAF', 'DVLA', 'TSPC', 'MERK', 'SCPI']
}

def get_sector(symbol):
    for sector, stocks in SECTOR_MAPPING.items():
        if symbol in stocks:
            return sector
    return 'OTHER'

# =============================================================================
# 4. QUALITY STOCKS UNIVERSE (IDX30 + LQ45 + IDX80)
# =============================================================================

QUALITY_STOCKS = [
    # IDX30 (Februari - April 2026)
    'ADRO', 'AMRT', 'ANTM', 'ASII', 'BBCA',
    'BBNI', 'BBRI', 'BMRI', 'BREN', 'BUMI',
    'CPIN', 'EMTK', 'GOTO', 'ICBP', 'INCO',
    'INDF', 'INKP', 'ISAT', 'JPFA', 'JSMR',
    'KLBF', 'MDKA', 'MEDC', 'MIKA', 'MTEL',
    'NCKL', 'PGAS', 'TLKM', 'TOWR', 'UNTR',
    
    # LQ45 (tambahan dari yang belum masuk IDX30)
    'AKRA', 'ARTO', 'BBTN', 'BDKR', 'BIRD',
    'BJBR', 'BJTM', 'BRIS', 'BRPT', 'BUKA',
    'EXCL', 'HRUM', 'INTP', 'ITMG', 'MPMX',
    'PTBA', 'SIDO', 'SMGR', 'TPIA', 'UNVR',
    'WIKA',
    
    # IDX80 (tambahan saham likuid lainnya)
    'ACES', 'ADHI', 'AGRO', 'AALI', 'ARNA',
    'ASGR', 'ASRI', 'AUTO', 'BAYU', 'BEST',
    'BFIN', 'BIPP', 'BISI', 'BKSL', 'BLTA',
    'BMAS', 'BMSR', 'BNGA', 'BNII', 'BOBA',
    'BOLT', 'BOSS', 'BPFI', 'BRAM', 'BRPT',
    'BSDE', 'BSSR', 'BTON', 'BUDI', 'BULL',
    'BUVA', 'BWPT', 'BYAN', 'CAMP', 'CANI',
    'CARS', 'CASA', 'CASS', 'CBDK', 'CBMF',
    'CCSI', 'CDAX', 'CEKA', 'CENT', 'CFIN',
    'CITA', 'CITY', 'CKRA', 'CLEO', 'CLPI',
    'CMNP', 'CMPP', 'CMRY', 'CNKO', 'COAL',
    'COCO', 'COWL', 'CPRO', 'CSAP', 'CSIS',
    'CTBN', 'CTRA', 'CTTH', 'CUAN', 'DART',
    'DASA', 'DCII', 'DEGA', 'DEWA', 'DGIK',
    'DIGI', 'DILD', 'DIVA', 'DIVN', 'DLTA',
    'DMAS', 'DMND', 'DNAR', 'DNET', 'DNLS',
    'DOID', 'DOOH', 'DPNS', 'DSFI', 'DSNG',
    'DSSA', 'DUCK', 'DUTI', 'DVLA', 'DYAN',
    'EASI', 'EASY', 'EBMT', 'ECII', 'EDGE',
    'EKAD', 'ELBA', 'ELSA', 'ELTY', 'EMBR',
    'EMDE', 'ENRG', 'ENVY', 'ENZO', 'EPAC',
    'EPMT', 'ERAA', 'ESSA', 'ESTA', 'ESTI',
    'ETWA', 'FAST', 'FASW', 'FILM', 'FISH',
    'FITT', 'FKSF', 'FLMC', 'FMII', 'FORE',
    'FORU', 'FORZ', 'FPNI', 'FREN', 'FUJI',
    'FUTR', 'GAMA', 'GDST', 'GDYR', 'GEMS',
    'GGRM', 'GGRP', 'GHON', 'GIDS', 'GJTL',
    'GLVA', 'GMFI', 'GMTD', 'GOLD', 'GOOD',
    'GPRA', 'GRPH', 'GSMF', 'GTBO', 'GTRA',
    'GTSI', 'GULA', 'HADE', 'HDFA', 'HDIT',
    'HEAL', 'HERO', 'HITS', 'HKMU', 'HMSP',
    'HOKI', 'HOMI', 'HOPE', 'HOTL', 'HRME',
    'HRTA', 'HSBK', 'HSMP', 'HUMI', 'IBFN',
    'IBOS', 'IBST', 'ICON', 'IDPR', 'IFII',
    'IFSH', 'IGAR', 'IIKP', 'IKAI', 'IKAN',
    'IMAS', 'IMJS', 'IMPC', 'INAF', 'INAI',
    'INCF', 'INCI', 'INDS', 'INDX', 'INDY',
    'INET', 'INPC', 'INPP', 'INPS', 'INRU',
    'INTA', 'INTD', 'IPCC', 'IPCM', 'IPOL',
    'IRRA', 'ISEA', 'ISSP', 'ITIC', 'JAST',
    'JAWA', 'JAYA', 'JECC', 'JEMB', 'JFAS',
    'JGLE', 'JHON', 'JIHD', 'JKON', 'JKSW',
    'JMAS', 'JPII', 'JPUR', 'JRPT', 'JSKY',
    'JSPT', 'JTNB', 'KAEF', 'KAQI', 'KARW',
    'KBLI', 'KBLM', 'KBRT', 'KBRI', 'KDSI',
    'KDTN', 'KEEN', 'KETR', 'KICI', 'KIJA',
    'KINO', 'KIOS', 'KJEN', 'KKGI', 'KMTR',
    'KOBX', 'KOIN', 'KOLI', 'KONI', 'KOTA',
    'KPAL', 'KPIG', 'KRAS', 'KREN', 'KRYA',
    'KSEL', 'KUAS', 'KUIC', 'KUVO', 'LAND',
    'LAPD', 'LATA', 'LBAK', 'LCGP', 'LCKM',
    'LEAD', 'LIFE', 'LINK', 'LION', 'LISA',
    'LMAS', 'LMPI', 'LMSH', 'LPCK', 'LPGI',
    'LPIN', 'LPKR', 'LPLI', 'LPPF', 'LPPS',
    'LSIP', 'LSPI', 'LTLS', 'LUCY', 'MABA',
    'MABH', 'MAGP', 'MAIN', 'MAMI', 'MAPA',
    'MAPB', 'MAPI', 'MARA', 'MASA', 'MAYA',
    'MBAP', 'MBCA', 'MBMA', 'MBSS', 'MBTO',
    'MCAS', 'MCPI', 'MCOR', 'MDIA', 'MDKI',
    'MEGA', 'MERK', 'META', 'MFIN', 'MFMI',
    'MGLV', 'MGNA', 'MGRO', 'MIDI', 'MINA',
    'MIRA', 'MITI', 'MITT', 'MKNT', 'MKPI',
    'MLBI', 'MLIA', 'MLPL', 'MLPT', 'MLSL',
    'MMIX', 'MMLP', 'MNCN', 'MOLI', 'MPOW',
    'MPPA', 'MPRO', 'MPTJ', 'MRAT', 'MSIE',
    'MSIN', 'MSKY', 'MTDL', 'MTFN', 'MTLA',
    'MTPS', 'MTSM', 'MUDA', 'MUTU', 'MYOH',
    'MYOR', 'MYRX', 'MYSX', 'NAGA', 'NASI',
    'NATO', 'NAYZ', 'NELY', 'NETV', 'NFCX',
    'NICL', 'NIKL', 'NISP', 'NITY', 'NIYM',
    'NOBU', 'NPGF', 'NRCA', 'NSSS', 'NTBK',
    'NUSA', 'NUSI', 'OASA', 'OCTN', 'OKAS',
    'OMED', 'ONIX', 'OPMS', 'ORNA', 'OTBK',
    'PADA', 'PADI', 'PAMG', 'PANR', 'PANS',
    'PANU', 'PAPA', 'PASA', 'PASS', 'PBRX',
    'PBID', 'PBSA', 'PCAR', 'PDES', 'PDGD',
    'PDIN', 'PEGE', 'PGLI', 'PGUN', 'PICO',
    'PIDRA', 'PJAA', 'PKPK', 'PLAN', 'PLAS',
    'PLIN', 'PMJS', 'PMMP', 'PNBN', 'PNBS',
    'PNIN', 'PNLF', 'PNSE', 'POLI', 'POLL',
    'POLU', 'POLY', 'POOL', 'PORT', 'POWR',
    'PPGL', 'PPRE', 'PPRO', 'PPSI', 'PRAS',
    'PRDA', 'PRIM', 'PRIN', 'PRLD', 'PROD',
    'PROT', 'PRTS', 'PSAB', 'PSBA', 'PSDN',
    'PSGO', 'PSKT', 'PSSI', 'PTDU', 'PTIS',
    'PTMP', 'PTPP', 'PTPW', 'PTRO', 'PTSN',
    'PTSP', 'PUDP', 'PURA', 'PURE', 'PWON',
    'PYFA', 'RACE', 'RADIO', 'RAFI', 'RAJA',
    'RAKD', 'RALS', 'RANC', 'RATU', 'RBMS',
    'RDTX', 'REAL', 'RELI', 'RIGS', 'RIMO',
    'RISE', 'RMBA', 'RMKE', 'ROCK', 'RODA',
    'ROKI', 'ROTI', 'RRMI', 'RUIS', 'RUMI',
    'SABA', 'SAFE', 'SAME', 'SAPX', 'SARA',
    'SATO', 'SBAT', 'SBBP', 'SBGA', 'SBMA',
    'SBMF', 'SCBD', 'SCCC', 'SCCO', 'SCMA',
    'SCNP', 'SDPC', 'SDRA', 'SEAN', 'SECR',
    'SEMA', 'SFAN', 'SGER', 'SGRO', 'SHID',
    'SHIP', 'SILO', 'SIMA', 'SIMP', 'SIPD',
    'SIPO', 'SKBM', 'SKLT', 'SKRN', 'SLIS',
    'SMAR', 'SMDR', 'SMIL', 'SMMT', 'SMSM',
    'SMRA', 'SNLK', 'SNMS', 'SOFA', 'SONA',
    'SOSS', 'SOUL', 'SPMA', 'SPMI', 'SPNA',
    'SPRE', 'SPTO', 'SQBI', 'SQMI', 'SRAJ',
    'SRIL', 'SRSN', 'SSIA', 'SSMS', 'SSTM',
    'STAR', 'STTP', 'SUGI', 'SULI', 'SUPR',
    'SURI', 'SWAT', 'SWID', 'TALD', 'TAMA',
    'TAMU', 'TAPG', 'TARA', 'TASP', 'TATA',
    'TAXI', 'TBIG', 'TBLA', 'TCID', 'TDPM',
    'TELE', 'TEMB', 'TEMPO', 'TIFA', 'TIGA',
    'TINS', 'TIRA', 'TIRT', 'TITA', 'TKGA',
    'TKIM', 'TMAS', 'TMPO', 'TMSH', 'TOBA',
    'TOOL', 'TOPS', 'TOSK', 'TOTL', 'TOTO',
    'TPMA', 'TRAM', 'TRGU', 'TRIO', 'TRIS',
    'TRJA', 'TRON', 'TRST', 'TRUB', 'TRUK',
    'TRUS', 'TSPC', 'TUGU', 'TURI', 'TUVN',
    'TYRE', 'UANG', 'UCID', 'UDIJ', 'UFNX',
    'UGRO', 'UJSN', 'ULTJ', 'UNIC', 'UNIQ',
    'UNIT', 'UNSP', 'USFI', 'VALU', 'VICO',
    'VICI', 'VIDI', 'VISI', 'VIVA', 'VKTR',
    'VOKS', 'VRNA', 'VTNY', 'WAPO', 'WEGE',
    'WEHA', 'WICO', 'WIFI', 'WIIM', 'WINS',
    'WMUU', 'WMPP', 'WOOD', 'WOWS', 'WRKR',
    'WSBP', 'WSKT', 'WTON', 'YELO', 'YULE',
    'ZBRA', 'ZINC', 'ZONE'
]

# Hapus duplikat
QUALITY_STOCKS = list(dict.fromkeys(QUALITY_STOCKS))
print(f"✅ Quality Universe: {len(QUALITY_STOCKS)} stocks (IDX30 + LQ45 + IDX80)")

# =============================================================================
# 5. FEE CONFIGURATION
# =============================================================================

class FeeConfig:
    """Konfigurasi fee dan biaya transaksi"""
    
    BROKER_FEE_BUY = 0.0015  # 0.15%
    BROKER_FEE_SELL = 0.0025  # 0.25% (termasuk pajak)
    EXCHANGE_FEE = 0.0001     # 0.01%
    
    SLIPPAGE_BUY_MIN = 0.0005  # 0.05%
    SLIPPAGE_BUY_MAX = 0.002   # 0.2%
    SLIPPAGE_SELL_MIN = 0.001  # 0.1%
    SLIPPAGE_SELL_MAX = 0.003  # 0.3%
    
    SLIPPAGE_MODE = 'random'
    MIN_FEE = 0
    
    @classmethod
    def get_slippage_buy(cls):
        if cls.SLIPPAGE_MODE == 'random':
            return random.uniform(cls.SLIPPAGE_BUY_MIN, cls.SLIPPAGE_BUY_MAX)
        else:
            return 0.001
    
    @classmethod
    def get_slippage_sell(cls):
        if cls.SLIPPAGE_MODE == 'random':
            return random.uniform(cls.SLIPPAGE_SELL_MIN, cls.SLIPPAGE_SELL_MAX)
        else:
            return 0.002
    
    @classmethod
    def calculate_buy_cost(cls, amount):
        broker = amount * cls.BROKER_FEE_BUY
        exchange = amount * cls.EXCHANGE_FEE
        slippage = amount * cls.get_slippage_buy()
        total = broker + exchange + slippage
        return max(total, cls.MIN_FEE)
    
    @classmethod
    def calculate_sell_cost(cls, amount):
        broker = amount * cls.BROKER_FEE_SELL
        exchange = amount * cls.EXCHANGE_FEE
        slippage = amount * cls.get_slippage_sell()
        total = broker + exchange + slippage
        return max(total, cls.MIN_FEE)
    
    @classmethod
    def calculate_round_trip_cost(cls, buy_amount, sell_amount):
        buy_cost = cls.calculate_buy_cost(buy_amount)
        sell_cost = cls.calculate_sell_cost(sell_amount)
        return buy_cost + sell_cost
    
    @classmethod
    def adjust_return_for_fee(cls, entry_price, exit_price, lot=1):
        buy_amount = entry_price * 100 * lot
        sell_amount = exit_price * 100 * lot
        total_cost = cls.calculate_round_trip_cost(buy_amount, sell_amount)
        net_profit = (sell_amount - buy_amount) - total_cost
        net_return_pct = (net_profit / buy_amount) * 100
        return net_return_pct, total_cost

# =============================================================================
# 6. GLOBAL INDICES CONFIGURATION
# =============================================================================

GLOBAL_INDICES = {
    "IHSG": "^JKSE",
    "DOWJONES": "^DJI",
    "USDIDR": "IDR=X",
    "OIL": "CL=F",
    "GOLD": "GC=F"
}

# =============================================================================
# 7. KONFIGURASI DASAR (UNTUK ENGINE 1,2,3)
# =============================================================================

class Config:
    def __init__(self, modal, mode):
        self.MODAL = modal
        self.MODE = mode
        
        # FILTER KETAT
        self.MIN_PRICE = 100
        self.MAX_PRICE = 50000
        self.MIN_VOLUME = 50000
        self.MAX_SPREAD_PCT = 2.0
        self.MIN_AVG_VOLUME = 500000
        self.MIN_RR = 1.0
        self.MAX_PORTFOLIO_RISK_PCT = 3.0
        self.MIN_EV_PCT = 2.0
        
        self.ENABLE_SECTOR_FILTER = True
        self.ENABLE_ENTRY_DELAY = True
        self.MAX_ENTRY_DELAY = 2
        self.ENTRY_DELAY_PROB = [0.5, 0.3, 0.2]
        self.ENABLE_RANDOM_SLIPPAGE = True
        self.ENABLE_VOLATILITY_SIZING = True
        
        if mode == 'intraday':
            self.INTERVAL = "1h"
            self.PERIOD = "1mo"
            self.MIN_HISTORY = 30
            self.SESSION1_START = "09:00:00"
            self.SESSION1_END = "12:00:00"
            self.SESSION2_START = "14:00:00"
            self.SESSION2_END = "16:00:00"
            self.JKT_TZ = "Asia/Jakarta"
            
            self.BREAKOUT_PERIOD = 10
            self.VOLUME_BREAKOUT = 1.2
            self.MA_SHORT = 20
            self.MA_LONG = 50
            self.ATR_PERIOD = 14
            self.TP_MULTIPLIER = 1.2
            self.SL_MULTIPLIER = 0.8
            self.RISK_PER_TRADE_PCT = 0.5
            self.MAX_TRADES_PER_DAY = 3
            self.FORCE_CLOSE_HOUR = 15
            self.FORCE_CLOSE_MINUTE = 50
            
        else:  # mingguan
            self.INTERVAL = "1d"
            self.PERIOD = "6mo"
            self.MIN_HISTORY = 60
            
            self.RSI_PERIOD = 14
            self.MA_SHORT = 20
            self.MA_LONG = 50
            self.MA200_PERIOD = 200
            self.SUPPORT_PERIOD = 60
            self.ATR_PERIOD = 14
            self.SL_MULTIPLIER = 1.5
            self.TP_MULTIPLIER = 2.5
            self.BASE_THRESHOLD = 5
            self.VOLUME_BOOST = 1.5
        
        if modal < 100000:
            self.MAX_POSITION_PCT = 0.30
            self.MAX_PORTFOLIO = 2
            self.MIN_PRICE = 100
        elif modal < 500000:
            self.MAX_POSITION_PCT = 0.25
            self.MAX_PORTFOLIO = 3
            self.MIN_PRICE = 100
        else:
            self.MAX_POSITION_PCT = 0.20
            self.MAX_PORTFOLIO = 4
            self.MIN_PRICE = 100

# =============================================================================
# 8. KONFIGURASI UNTUK ENGINE INVESTASI (FIXED)
# =============================================================================

class InvestasiConfig:
    def __init__(self, modal):
        self.MODAL = modal
        self.MODE = 'investasi'
        
        # FILTER INVESTASI (lebih longgar)
        self.MIN_PRICE = 100
        self.MAX_PRICE = 50000
        self.MIN_VOLUME = 50000
        self.MAX_SPREAD_PCT = 3.0
        self.MIN_AVG_VOLUME = 200000
        self.MIN_EV_PCT = 1.0
        
        # Quality stocks universe
        self.QUALITY_STOCKS = QUALITY_STOCKS
        
        # Parameter teknikal
        self.MA200_PERIOD = 200
        self.MA50_PERIOD = 50
        self.PULLBACK_THRESHOLD = 0.02  # 2% dari MA50
        
        # Risk management
        self.MAX_PORTFOLIO_RISK_PCT = 10.0
        self.MAX_PORTFOLIO = 5  # Maksimal 5 saham
        
        # Interval dan period (FIXED)
        self.INTERVAL = "1d"
        self.PERIOD = "5y"  # 5 tahun data (lebih reliable dari "2y")
        self.MIN_HISTORY = 200  # FIXED: Minimal 200 hari untuk MA200

# =============================================================================
# 9. GLOBAL INDICES FETCHER
# =============================================================================

class GlobalIndicesFetcher:
    def __init__(self):
        self.data = {}
        self.momentum = {}
        self.status = {}
        self.prices = {}
        
    def _to_scalar(self, value):
        if value is None:
            return 0.0
        if isinstance(value, (np.ndarray, list)):
            if len(value) > 0:
                return float(value[0])
            return 0.0
        return float(value)
        
    def fetch_all(self, config):
        print("\n📡 Fetching global indices...")
        
        end_date = datetime.now()
        days = 365 * 2
        start_date = end_date - timedelta(days=days)
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        for name, ticker in GLOBAL_INDICES.items():
            try:
                df = yf.download(
                    ticker, 
                    start=start_str,
                    end=end_str,
                    interval="1d",
                    auto_adjust=True, 
                    progress=False, 
                    timeout=10
                )
                
                time.sleep(1)
                
                if df.empty or len(df) < 200:
                    self.status[name] = "UNAVAILABLE"
                    self.momentum[name] = 0.0
                    self.prices[name] = 0.0
                else:
                    close = df['Close'].values
                    current_price = float(close[-1])
                    
                    if len(close) >= 5:
                        momentum_array = (close[-1] / close[-5] - 1) * 100
                        momentum_value = self._to_scalar(momentum_array)
                    else:
                        momentum_value = 0.0
                    
                    if name == "IHSG" and len(close) >= 200:
                        ma200 = np.mean(close[-200:])
                        self.data['IHSG_MA200'] = ma200
                        self.data['IHSG_Close'] = current_price
                    
                    self.data[name] = df
                    self.momentum[name] = round(momentum_value, 2)
                    self.prices[name] = round(current_price, 2)
                    self.status[name] = "OK"
                    
            except Exception:
                self.status[name] = "ERROR"
                self.momentum[name] = 0.0
                self.prices[name] = 0.0
                time.sleep(1)
        
        print(f"   ✅ Global indices ready")
    
    def get_momentum(self, name):
        return self.momentum.get(name, 0.0)
    
    def get_price(self, name):
        return self.prices.get(name, 0.0)
    
    def get_price_str(self, name):
        price = self.get_price(name)
        if price == 0:
            return "N/A"
        
        if name in ["IHSG", "DOWJONES"]:
            return f"{price:,.2f}"
        elif name == "USDIDR":
            return f"Rp {price:,.0f}"
        elif name == "OIL":
            return f"US$ {price:.2f}"
        elif name == "GOLD":
            return f"US$ {price:.2f}"
        return f"{price:.2f}"
    
    def get_trend(self, name):
        mom = self.get_momentum(name)
        if mom > 0.5:
            return "🟢 BULLISH"
        elif mom < -0.5:
            return "🔴 BEARISH"
        else:
            return "🟡 NETRAL"
    
    def is_ihsg_bullish(self):
        if 'IHSG_Close' in self.data and 'IHSG_MA200' in self.data:
            return self.data['IHSG_Close'] > self.data['IHSG_MA200']
        return True

# =============================================================================
# 10. SHARED UTILITY FUNCTIONS
# =============================================================================

def normalize_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def apply_idx_to_jakarta(df, tz="Asia/Jakarta"):
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        return out
    if out.index.tz is None:
        out.index = out.index.tz_localize(tz)
    else:
        out.index = out.index.tz_convert(tz)
    return out

def filter_jakarta_sessions(df, config):
    out = apply_idx_to_jakarta(df, config.JKT_TZ)
    sess1 = out.between_time(config.SESSION1_START, config.SESSION1_END, inclusive="both")
    sess2 = out.between_time(config.SESSION2_START, config.SESSION2_END, inclusive="both")
    
    if not sess1.empty or not sess2.empty:
        out = pd.concat([sess1, sess2]).sort_index()
    
    if isinstance(out.index, pd.DatetimeIndex):
        out = out[out.index.dayofweek < 5]
    return out

def calculate_spread_pct(df):
    try:
        spread = ((df['High'] - df['Low']) / df['Close']).tail(10).mean() * 100
        return float(spread)
    except:
        return 999.0

def calculate_return(series, period=5):
    if len(series) < period + 1:
        return 0.0
    return (series.iloc[-1] / series.iloc[-period-1] - 1) * 100

# =============================================================================
# 11. CACHED DATA FETCHER (DIPERBAIKI)
# =============================================================================

class CachedDataFetcher:
    def __init__(self, cache_dir='stock_cache'):
        self.cache_dir = cache_dir
        self.stats = {'total': 0, 'success': 0, 'filtered': 0, 'failed': 0, 'cached': 0}
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, symbol, config):
        today = datetime.now().strftime("%Y-%m-%d")
        return f"{symbol}_{config.MODE}_{today}"
        
    def _load_from_cache(self, cache_key):
        cache_file = f"{self.cache_dir}/{cache_key}.pkl"
        if os.path.exists(cache_file):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < timedelta(days=1):
                    with open(cache_file, 'rb') as f:
                        self.stats['cached'] += 1
                        return pickle.load(f)
            except:
                pass
        return None
        
    def _save_to_cache(self, cache_key, data):
        cache_file = f"{self.cache_dir}/{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
    
    def fetch(self, symbol, config):
        self.stats['total'] += 1
        
        cache_key = self._get_cache_key(symbol, config)
        cached_df = self._load_from_cache(cache_key)
        if cached_df is not None:
            return cached_df
        
        try:
            ticker = f"{symbol}.JK"
            
            # FIX: Gunakan period dan interval dari config
            df = yf.download(
                ticker, 
                period=config.PERIOD, 
                interval=config.INTERVAL,
                auto_adjust=True,
                progress=False,
                timeout=10
            )
            
            if df.empty:
                self.stats['failed'] += 1
                return None
            
            df = normalize_columns(df)
            
            # FIX: Cek MIN_HISTORY (sekarang sudah ada di InvestasiConfig)
            if hasattr(config, 'MIN_HISTORY') and len(df) < config.MIN_HISTORY:
                self.stats['failed'] += 1
                return None
            
            # Filter likuiditas (jika ada)
            if hasattr(config, 'MIN_AVG_VOLUME'):
                avg_volume_20 = df['Volume'].tail(20).mean()
                if avg_volume_20 < config.MIN_AVG_VOLUME:
                    self.stats['filtered'] += 1
                    return None
            
            # Filter sesi intraday (jika ada)
            if hasattr(config, 'JKT_TZ') and config.MODE == 'intraday':
                df_filtered = filter_jakarta_sessions(df, config)
                if not df_filtered.empty and len(df_filtered) >= 10:
                    df = df_filtered
                else:
                    self.stats['filtered'] += 1
                    return None
            
            # Filter harga
            if hasattr(config, 'MIN_PRICE'):
                last_close = float(df['Close'].iloc[-1])
                if last_close < config.MIN_PRICE or (hasattr(config, 'MAX_PRICE') and last_close > config.MAX_PRICE):
                    self.stats['filtered'] += 1
                    return None
            
            # Filter volume
            if hasattr(config, 'MIN_VOLUME'):
                last_volume = int(df['Volume'].iloc[-1])
                if last_volume < config.MIN_VOLUME:
                    self.stats['filtered'] += 1
                    return None
            
            # Filter spread
            if hasattr(config, 'MAX_SPREAD_PCT'):
                spread = calculate_spread_pct(df)
                if spread > config.MAX_SPREAD_PCT:
                    self.stats['filtered'] += 1
                    return None
            
            self._save_to_cache(cache_key, df)
            
            self.stats['success'] += 1
            return df
            
        except Exception as e:
            self.stats['failed'] += 1
            return None
    
    def print_stats(self):
        print(f"\n📊 Download Statistics:")
        print(f"   Total: {self.stats['total']}")
        print(f"   From Cache: {self.stats['cached']}")
        print(f"   Success: {self.stats['success']}")
        print(f"   Filtered: {self.stats['filtered']}")
        print(f"   Failed: {self.stats['failed']}")

# =============================================================================
# 12. BASE STRATEGY ENGINE
# =============================================================================

class StrategyEngine:
    def __init__(self, config, global_fetcher):
        self.config = config
        self.global_fetcher = global_fetcher
    
    def get_signal(self, symbol, df):
        raise NotImplementedError
    
    def calculate_volatility_lot(self, close, atr, risk_per_trade):
        if atr <= 0:
            return None, None
        
        raw_lot = risk_per_trade / (atr * 100)
        lot = int(raw_lot)
        
        max_lot_by_modal = int(self.config.MODAL / (close * 100))
        lot = min(lot, max_lot_by_modal, 5)
        
        if lot >= 1:
            cost = lot * 100 * close
            if cost > self.config.MODAL:
                lot = int(self.config.MODAL / (close * 100))
                if lot >= 1:
                    cost = lot * 100 * close
                    return lot, cost
                else:
                    return None, None
            return lot, cost
        else:
            return None, None
    
    def calculate_fixed_lot(self, close, risk_per_lot, max_amount):
        max_lot_by_modal = int(self.config.MODAL / (close * 100))
        max_lot_by_risk = int(max_amount / (close * 100)) if max_amount >= close * 100 else 0
        max_lot = min(max_lot_by_modal, max_lot_by_risk, 5)
        
        if max_lot >= 1:
            lot = max_lot
            cost = lot * 100 * close
            if cost > self.config.MODAL:
                lot = int(self.config.MODAL / (close * 100))
                if lot >= 1:
                    cost = lot * 100 * close
                    return lot, cost
                else:
                    return None, None
            return lot, cost
        else:
            return None, None

# =============================================================================
# 13. ENGINE 1: SWING (MEAN REVERSION) - EXISTING
# =============================================================================

class OutperformDetector:
    def __init__(self):
        self.RETURN_THRESHOLD = 2.0
        self.BEARISH_THRESHOLD = -2.0
        self.HOLD_THRESHOLD = -1.0
        self.VOLUME_THRESHOLD = 1.5
        
    def is_outperform(self, saham_return, ihsg_return, volume_ratio):
        if saham_return > ihsg_return + self.RETURN_THRESHOLD:
            return True, f"Return > IHSG+2%", 2
        if ihsg_return < self.BEARISH_THRESHOLD and saham_return > self.HOLD_THRESHOLD:
            return True, f"Tahan turun", 2
        if ihsg_return < self.BEARISH_THRESHOLD and volume_ratio > self.VOLUME_THRESHOLD:
            return True, f"Volume tinggi", 2
        return False, "", 0


class InflowOutflowDetector:
    def __init__(self):
        self.INFLOW_BONUS = 1
        self.OUTFLOW_PENALTY = -1
        
    def calculate_money_flow(self, df):
        if len(df) < 2:
            return 0, "NETRAL"
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        last_volume = df['Volume'].iloc[-1]
        price_change = last_close - prev_close
        money_flow = price_change * last_volume
        avg_volume = df['Volume'].tail(10).mean()
        volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1
        if money_flow > 0 and volume_ratio > 1.2:
            return 1, "INFLOW"
        elif money_flow > 0:
            return 1, "INFLOW"
        elif money_flow < 0 and volume_ratio > 1.2:
            return -1, "OUTFLOW"
        elif money_flow < 0:
            return -1, "OUTFLOW"
        else:
            return 0, "NETRAL"
    
    def get_accumulation_distribution(self, df, period=14):
        if len(df) < period:
            return 0, "NETRAL"
        money_flow_values = []
        for i in range(-period, 0):
            if i < -1:
                price_change = df['Close'].iloc[i] - df['Close'].iloc[i-1]
                money_flow = price_change * df['Volume'].iloc[i]
                money_flow_values.append(money_flow)
        if len(money_flow_values) > 5:
            recent = sum(money_flow_values[-3:])
            previous = sum(money_flow_values[-6:-3])
            if recent > previous * 1.5:
                return 2, "AKUMULASI"
            elif recent > previous:
                return 1, "AKUMULASI"
            elif recent < previous * 0.5:
                return -2, "DISTRIBUSI"
            elif recent < previous:
                return -1, "DISTRIBUSI"
        return 0, "NETRAL"


class MultipleTouchDetector:
    def __init__(self, window=60, tolerance=1.0):
        self.window = window
        self.tolerance = tolerance / 100
        self.support_levels = {'kuat': [], 'sedang': []}
        self.resistance_levels = {'kuat': [], 'sedang': []}
        
    def _count_touches(self, prices, level):
        if len(prices) == 0:
            return 0
        distances = np.abs(prices - level)
        touch_threshold = level * self.tolerance
        touches = np.sum(distances < touch_threshold)
        return touches
    
    def _get_candidate_levels(self, low_prices, high_prices):
        all_prices = np.concatenate([low_prices, high_prices])
        if len(all_prices) == 0:
            return []
        avg_price = np.mean(all_prices)
        step = avg_price * 0.005
        min_price = np.min(all_prices) * 0.98
        max_price = np.max(all_prices) * 1.02
        candidates = np.arange(min_price, max_price + step, step)
        return candidates
    
    def detect_levels(self, df):
        high = df['High'].values
        low = df['Low'].values
        dates = df.index.values
        self.support_levels = {'kuat': [], 'sedang': []}
        self.resistance_levels = {'kuat': [], 'sedang': []}
        for i in range(self.window, len(high)):
            window_high = high[i-self.window:i]
            window_low = low[i-self.window:i]
            window_date = dates[i]
            candidates = self._get_candidate_levels(window_low, window_high)
            for level in candidates:
                touches_support = self._count_touches(window_low, level)
                touches_resistance = self._count_touches(window_high, level)
                if touches_support >= 3:
                    self.support_levels['kuat'].append({'price': level, 'touches': touches_support, 'date': window_date, 'strength': 'KUAT'})
                elif touches_support == 2:
                    self.support_levels['sedang'].append({'price': level, 'touches': touches_support, 'date': window_date, 'strength': 'SEDANG'})
                if touches_resistance >= 3:
                    self.resistance_levels['kuat'].append({'price': level, 'touches': touches_resistance, 'date': window_date, 'strength': 'KUAT'})
                elif touches_resistance == 2:
                    self.resistance_levels['sedang'].append({'price': level, 'touches': touches_resistance, 'date': window_date, 'strength': 'SEDANG'})
        self._deduplicate_and_sort()
        return self.support_levels, self.resistance_levels
    
    def _deduplicate_and_sort(self):
        def deduplicate(levels, tolerance_mult=2.0):
            if not levels:
                return []
            sorted_levels = sorted(levels, key=lambda x: x['price'])
            unique = []
            current_group = [sorted_levels[0]]
            for level in sorted_levels[1:]:
                if abs(level['price'] - current_group[-1]['price']) < (level['price'] * self.tolerance * tolerance_mult):
                    current_group.append(level)
                else:
                    best = max(current_group, key=lambda x: x['touches'])
                    unique.append(best)
                    current_group = [level]
            if current_group:
                best = max(current_group, key=lambda x: x['touches'])
                unique.append(best)
            return unique
        self.support_levels['kuat'] = deduplicate(self.support_levels['kuat'])
        self.support_levels['sedang'] = deduplicate(self.support_levels['sedang'])
        self.resistance_levels['kuat'] = deduplicate(self.resistance_levels['kuat'])
        self.resistance_levels['sedang'] = deduplicate(self.resistance_levels['sedang'])
    
    def get_nearest_support(self, price):
        kuat_below = [s for s in self.support_levels['kuat'] if s['price'] < price]
        if kuat_below:
            nearest = max(kuat_below, key=lambda x: x['price'])
            return nearest['price'], nearest['touches'], nearest['strength']
        sedang_below = [s for s in self.support_levels['sedang'] if s['price'] < price]
        if sedang_below:
            nearest = max(sedang_below, key=lambda x: x['price'])
            return nearest['price'], nearest['touches'], nearest['strength']
        return price * 0.95, 1, 'FALLBACK'
    
    def get_nearest_resistance(self, price):
        kuat_above = [r for r in self.resistance_levels['kuat'] if r['price'] > price]
        if kuat_above:
            nearest = min(kuat_above, key=lambda x: x['price'])
            return nearest['price'], nearest['touches'], nearest['strength']
        sedang_above = [r for r in self.resistance_levels['sedang'] if r['price'] > price]
        if sedang_above:
            nearest = min(sedang_above, key=lambda x: x['price'])
            return nearest['price'], nearest['touches'], nearest['strength']
        return price * 1.05, 1, 'FALLBACK'


class SwingEngine(StrategyEngine):
    def __init__(self, config, global_fetcher):
        super().__init__(config, global_fetcher)
        self.outperform_detector = OutperformDetector()
        self.inflow_detector = InflowOutflowDetector()
        self.sr_detector = MultipleTouchDetector(window=config.SUPPORT_PERIOD, tolerance=1.0)
    
    def get_sector_boost(self, symbol):
        boost = 0
        ihsg_mom = self.global_fetcher.get_momentum("IHSG")
        if ihsg_mom > 0.5:
            boost += 1
        elif ihsg_mom < -0.5:
            boost -= 1
        if symbol in EXPORT_SECTOR:
            usd_mom = self.global_fetcher.get_momentum("USDIDR")
            if usd_mom > 0.5:
                boost += 1
            elif usd_mom < -0.5:
                boost -= 1
        if symbol in ENERGY_SECTOR:
            oil_mom = self.global_fetcher.get_momentum("OIL")
            if oil_mom > 1.0:
                boost += 1
            elif oil_mom < -1.0:
                boost -= 1
        if symbol in MINING_GOLD:
            gold_mom = self.global_fetcher.get_momentum("GOLD")
            if gold_mom > 1.0:
                boost += 1
            elif gold_mom < -1.0:
                boost -= 1
        return boost
    
    def calculate_atr_sl_tp(self, close, atr, support_price, resistance_price):
        atr = max(atr, close * 0.005)
        sl = close - (atr * self.config.SL_MULTIPLIER)
        tp_from_atr = close + (atr * self.config.TP_MULTIPLIER)
        tp_from_resistance = resistance_price * 0.98 if resistance_price > close else tp_from_atr
        tp = min(tp_from_atr, tp_from_resistance)
        sl = max(sl, close * 0.90)
        tp = min(tp, close * 1.15)
        fraction = 5 if close < 100 else 10 if close < 500 else 25
        sl = round(sl / fraction) * fraction
        tp = round(tp / fraction) * fraction
        if sl >= close or tp <= close:
            return None, None
        return int(sl), int(tp)
    
    def compute_features(self, df):
        try:
            out = df.copy()
            close = out['Close']
            high = out['High']
            low = out['Low']
            volume = out['Volume']
            
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=self.config.RSI_PERIOD, min_periods=self.config.RSI_PERIOD).mean()
            avg_loss = loss.rolling(window=self.config.RSI_PERIOD, min_periods=self.config.RSI_PERIOD).mean()
            rs = avg_gain / (avg_loss + 1e-6)
            out['RSI'] = 100 - (100 / (1 + rs))
            
            out['MA20'] = close.rolling(window=self.config.MA_SHORT, min_periods=self.config.MA_SHORT).mean()
            out['MA50'] = close.rolling(window=self.config.MA_LONG, min_periods=self.config.MA_LONG).mean()
            out['MA200'] = close.rolling(window=self.config.MA200_PERIOD, min_periods=self.config.MA200_PERIOD).mean()
            out['MA_Trend'] = (out['MA20'] > out['MA50']).astype(float)
            out['TR'] = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
            out['ATR'] = out['TR'].rolling(window=self.config.ATR_PERIOD, min_periods=self.config.ATR_PERIOD).mean()
            out['Volume_MA'] = volume.rolling(window=20, min_periods=20).mean()
            out['Volume_Ratio'] = volume / (out['Volume_MA'] + 1e-6)
            
            out = out.replace([np.inf, -np.inf], np.nan)
            out = out.dropna()
            return out
        except Exception:
            return pd.DataFrame()
    
    def get_signal(self, symbol, df):
        try:
            self.sr_detector.detect_levels(df)
            df_feat = self.compute_features(df)
            if len(df_feat) < self.config.MIN_HISTORY:
                return None
            
            latest = df_feat.iloc[-1]
            close = float(latest['Close'])
            
            ma200 = float(latest['MA200']) if not pd.isna(latest['MA200']) else close
            if close < ma200:
                return None
            
            prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else close
            if close <= prev_close:
                return None
            
            saham_return = calculate_return(df['Close'], 5)
            ihsg_return = self.global_fetcher.get_momentum("IHSG")
            volume_ratio = float(latest['Volume_Ratio']) if not pd.isna(latest['Volume_Ratio']) else 1
            
            is_outperform, outperform_reason, outperform_bonus = self.outperform_detector.is_outperform(
                saham_return, ihsg_return, volume_ratio
            )
            
            inflow_score, inflow_trend = self.inflow_detector.calculate_money_flow(df)
            acc_score, acc_trend = self.inflow_detector.get_accumulation_distribution(df)
            
            support_price, support_touches, support_strength = self.sr_detector.get_nearest_support(close)
            resistance_price, resistance_touches, resistance_strength = self.sr_detector.get_nearest_resistance(close)
            dist_to_support = (close - support_price) / close * 100
            
            rsi = float(latest['RSI']) if not pd.isna(latest['RSI']) else 50
            atr = float(latest['ATR']) if not pd.isna(latest['ATR']) else close * 0.02
            ma20 = float(latest['MA20']) if not pd.isna(latest['MA20']) else close
            ma50 = float(latest['MA50']) if not pd.isna(latest['MA50']) else close
            ma_trend = float(latest['MA_Trend']) if not pd.isna(latest['MA_Trend']) else 0
            
            base_score = 0
            if rsi < 30:
                base_score += 3
            elif rsi < 40:
                base_score += 1
            if volume_ratio > self.config.VOLUME_BOOST:
                base_score += 2
            elif volume_ratio > 1.2:
                base_score += 1
            if dist_to_support < 3:
                if support_strength == 'KUAT':
                    base_score += 2
                elif support_strength == 'SEDANG':
                    base_score += 1
                else:
                    base_score += 1
            if ma_trend > 0.5:
                base_score += 1
            
            global_boost = self.get_sector_boost(symbol)
            inflow_bonus = 1 if inflow_score > 0 else -1 if inflow_score < 0 else 0
            accumulation_bonus = 1 if acc_score > 0 else -1 if acc_score < 0 else 0
            score = base_score + global_boost + outperform_bonus + inflow_bonus + accumulation_bonus
            
            ihsg_momentum = self.global_fetcher.get_momentum("IHSG")
            if ihsg_momentum < -2.0:
                effective_threshold = self.config.BASE_THRESHOLD + 1
                position_multiplier = 0.5
            elif ihsg_momentum < -1.0:
                effective_threshold = self.config.BASE_THRESHOLD
                position_multiplier = 0.75
            else:
                effective_threshold = self.config.BASE_THRESHOLD
                position_multiplier = 1.0
            
            sl, tp = self.calculate_atr_sl_tp(close, atr, support_price, resistance_price)
            if sl is None or tp is None:
                return None
            
            risk = close - sl
            reward = tp - close
            if risk <= 0 or reward <= 0:
                return None
            
            rr = reward / risk
            if rr < self.config.MIN_RR:
                return None
            
            prob_up = 0.5 + (score * 0.03)
            prob_up = min(max(prob_up, 0.3), 0.8)
            expected_value = (prob_up * reward) - ((1 - prob_up) * risk)
            ev_pct = (expected_value / close) * 100
            
            if ev_pct < self.config.MIN_EV_PCT:
                return None
            
            risk_per_trade = self.config.MODAL * (self.config.RISK_PER_TRADE_PCT / 100) * position_multiplier
            lot, cost = self.calculate_volatility_lot(close, atr, risk_per_trade)
            
            if lot is None or cost is None:
                return None
            
            sector = get_sector(symbol)
            
            return {
                'Symbol': symbol,
                'Sector': sector,
                'Price': int(close),
                'RSI': round(rsi, 1),
                'Support': int(support_price),
                'Resistance': int(resistance_price),
                'Stop_Loss': sl,
                'Take_Profit': tp,
                'R/R': round(rr, 2),
                'Prob_Up': round(prob_up, 3),
                'EV': int(expected_value),
                'EV_Pct': round(ev_pct, 2),
                'Score': score,
                'Risk': int(risk),
                'ATR': round(atr, 2),
                'Lot': lot,
                'Cost': cost,
                'Inflow': inflow_trend,
                'Acc': acc_trend,
                'Volume': f"{volume_ratio:.1f}x",
                'Reasons': f"RSI {rsi:.0f}, Vol {volume_ratio:.1f}x, {support_strength}",
                'Chart': "BULLISH" if ma20 > ma50 and rsi > 50 else "BEARISH" if ma20 < ma50 and rsi < 50 else "NETRAL"
            }
        except Exception:
            return None


# =============================================================================
# 14. ENGINE 2: INTRADAY LIQUID (MOMENTUM) - EXISTING
# =============================================================================

class IntradayLiquidEngine(StrategyEngine):
    def __init__(self, config, global_fetcher):
        super().__init__(config, global_fetcher)
    
    def compute_features(self, df):
        try:
            out = df.copy()
            close = out['Close']
            high = out['High']
            low = out['Low']
            volume = out['Volume']
            
            out['Highest_High'] = high.shift(1).rolling(window=self.config.BREAKOUT_PERIOD).max()
            out['MA_Short'] = close.rolling(window=self.config.MA_SHORT).mean()
            out['MA_Long'] = close.rolling(window=self.config.MA_LONG).mean()
            out['MA_Alignment'] = (out['MA_Short'] > out['MA_Long']).astype(int)
            out['Volume_MA'] = volume.rolling(window=20).mean()
            out['Volume_Ratio'] = volume / (out['Volume_MA'] + 1e-6)
            out['TR'] = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
            out['ATR'] = out['TR'].rolling(window=self.config.ATR_PERIOD).mean()
            out['ATR_Pct'] = out['ATR'] / (close + 1e-6) * 100
            out['Body'] = abs(close - out['Open'])
            out['Range'] = high - low
            out['Body_Ratio'] = out['Body'] / (out['Range'] + 1e-6)
            out['Upper_Wick'] = (high - out[['Close', 'Open']].max(axis=1)) / (out['Range'] + 1e-6)
            out = out.replace([np.inf, -np.inf], np.nan)
            out = out.dropna()
            return out
        except Exception:
            return pd.DataFrame()
    
    def check_breakout(self, row):
        if pd.isna(row['Highest_High']) or pd.isna(row['Close']):
            return False
        if row['Close'] <= row['Highest_High']:
            return False
        if row['Volume_Ratio'] < self.config.VOLUME_BREAKOUT:
            return False
        if row['MA_Alignment'] != 1:
            return False
        return True
    
    def get_signal(self, symbol, df):
        try:
            df_feat = self.compute_features(df)
            if len(df_feat) < 30:
                return None
            latest = df_feat.iloc[-1]
            close = float(latest['Close'])
            if not self.check_breakout(latest):
                return None
            atr = float(latest['ATR']) if not pd.isna(latest['ATR']) else close * 0.02
            atr = max(atr, close * 0.005)
            sl = close - (atr * self.config.SL_MULTIPLIER)
            tp = close + (atr * self.config.TP_MULTIPLIER)
            fraction = 5 if close < 100 else 10 if close < 500 else 25
            sl = round(sl / fraction) * fraction
            tp = round(tp / fraction) * fraction
            if sl >= close or tp <= close:
                return None
            risk = close - sl
            reward = tp - close
            rr = reward / risk
            if rr < self.config.MIN_RR:
                return None
            score = 5
            if latest['Volume_Ratio'] > 1.5:
                score += 1
            if latest['Body_Ratio'] > 0.6:
                score += 1
            if latest['Upper_Wick'] < 0.3:
                score += 1
            prob_up = 0.5 + (score * 0.02)
            prob_up = min(prob_up, 0.7)
            expected_value = (prob_up * reward) - ((1 - prob_up) * risk)
            ev_pct = (expected_value / close) * 100
            if ev_pct < self.config.MIN_EV_PCT:
                return None
            
            risk_per_trade = self.config.MODAL * (self.config.RISK_PER_TRADE_PCT / 100)
            lot, cost = self.calculate_volatility_lot(close, atr, risk_per_trade)
            
            if lot is None or cost is None:
                return None
            
            flow = "INFLOW" if latest['Volume_Ratio'] > 1.2 else "NETRAL"
            acc = "AKUMULASI" if latest['Volume_Ratio'] > 1.3 else "NETRAL"
            sector = get_sector(symbol)
            
            return {
                'Symbol': symbol,
                'Sector': sector,
                'Price': int(close),
                'RSI': '-',
                'Support': int(latest['Highest_High']),
                'Resistance': '-',
                'Stop_Loss': int(sl),
                'Take_Profit': int(tp),
                'R/R': round(rr, 2),
                'Prob_Up': round(prob_up, 3),
                'EV': int(expected_value),
                'EV_Pct': round(ev_pct, 2),
                'Score': score,
                'Risk': int(risk),
                'ATR': round(atr, 2),
                'Lot': lot,
                'Cost': cost,
                'Inflow': flow,
                'Acc': acc,
                'Volume': f"{latest['Volume_Ratio']:.1f}x",
                'Body_Ratio': f"{latest['Body_Ratio']:.2f}",
                'Upper_Wick': f"{latest['Upper_Wick']:.2f}",
                'Reasons': f"Breakout {self.config.BREAKOUT_PERIOD}, Vol {latest['Volume_Ratio']:.1f}x",
                'Chart': "BULLISH" if latest['MA_Alignment'] == 1 else "NETRAL"
            }
        except Exception:
            return None


# =============================================================================
# 15. ENGINE 3: INTRADAY GORENGAN (EARLY MOMENTUM) - EXISTING
# =============================================================================

class IntradayGorenganEngine(StrategyEngine):
    def __init__(self, config, global_fetcher):
        super().__init__(config, global_fetcher)
    
    def compute_features(self, df):
        try:
            out = df.copy()
            close = out['Close']
            high = out['High']
            low = out['Low']
            volume = out['Volume']
            open_price = out['Open']
            
            out['Highest_High_5'] = high.shift(1).rolling(window=5).max()
            out['Volume_MA'] = volume.rolling(window=20).mean()
            out['Volume_Ratio'] = volume / (out['Volume_MA'] + 1e-6)
            out['Turnover'] = close * volume
            out['TR'] = np.maximum(high - low, np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()))
            out['ATR'] = out['TR'].rolling(window=self.config.ATR_PERIOD).mean()
            out['Body'] = abs(close - open_price)
            out['Range'] = high - low
            out['Body_Ratio'] = out['Body'] / (out['Range'] + 1e-6)
            out['Day_Change'] = (close / open_price - 1) * 100
            out = out.replace([np.inf, -np.inf], np.nan)
            out = out.dropna()
            return out
        except Exception:
            return pd.DataFrame()
    
    def check_breakout(self, row):
        if pd.isna(row['Highest_High_5']) or pd.isna(row['Close']):
            return False
        if row['Close'] <= row['Highest_High_5']:
            return False
        if row['Volume_Ratio'] < 2.0:
            return False
        min_turnover = self.config.MODAL * 10
        if row['Turnover'] < min_turnover:
            return False
        if row['Day_Change'] > 20:
            return False
        return True
    
    def get_signal(self, symbol, df):
        try:
            df_feat = self.compute_features(df)
            if len(df_feat) < 30:
                return None
            latest = df_feat.iloc[-1]
            close = float(latest['Close'])
            if not self.check_breakout(latest):
                return None
            atr = float(latest['ATR']) if not pd.isna(latest['ATR']) else close * 0.03
            atr = max(atr, close * 0.01)
            sl = close - (atr * 1.0)
            tp = close + (atr * 1.5)
            fraction = 5 if close < 100 else 10 if close < 500 else 25
            sl = round(sl / fraction) * fraction
            tp = round(tp / fraction) * fraction
            if sl >= close or tp <= close:
                return None
            risk = close - sl
            reward = tp - close
            rr = reward / risk
            if rr < self.config.MIN_RR:
                return None
            score = 5
            if latest['Volume_Ratio'] > 3:
                score += 2
            elif latest['Volume_Ratio'] > 2:
                score += 1
            if latest['Body_Ratio'] > 0.7:
                score += 1
            prob_up = 0.5 + (score * 0.02)
            prob_up = min(prob_up, 0.7)
            expected_value = (prob_up * reward) - ((1 - prob_up) * risk)
            ev_pct = (expected_value / close) * 100
            if ev_pct < 1.5:
                return None
            risk_per_trade = self.config.MODAL * 0.003
            lot, cost = self.calculate_volatility_lot(close, atr, risk_per_trade)
            if lot is None or cost is None:
                return None
            sector = get_sector(symbol)
            return {
                'Symbol': symbol,
                'Sector': sector,
                'Price': int(close),
                'RSI': '-',
                'Support': int(latest['Highest_High_5']),
                'Resistance': '-',
                'Stop_Loss': int(sl),
                'Take_Profit': int(tp),
                'R/R': round(rr, 2),
                'Prob_Up': round(prob_up, 3),
                'EV': int(expected_value),
                'EV_Pct': round(ev_pct, 2),
                'Score': score,
                'Risk': int(risk),
                'ATR': round(atr, 2),
                'Volume_Spike': f"{latest['Volume_Ratio']:.1f}x",
                'Turnover': f"Rp {latest['Turnover']/1e6:.0f}Jt",
                'Lot': lot,
                'Cost': cost,
                'Inflow': 'INFLOW' if latest['Volume_Ratio'] > 1.5 else 'NETRAL',
                'Acc': 'AKUMULASI' if latest['Volume_Ratio'] > 2 else 'NETRAL',
                'Volume': f"{latest['Volume_Ratio']:.1f}x",
                'Reasons': f"Breakout 5, Vol {latest['Volume_Ratio']:.1f}x",
                'Chart': "BULLISH" if latest['Day_Change'] > 0 else "NETRAL"
            }
        except Exception:
            return None


# =============================================================================
# 16. ENGINE 4: INVESTASI (QUALITY + TREND) - FIXED
# =============================================================================

class InvestasiEngine:
    """Engine untuk investasi jangka panjang (Quality + Trend)"""
    
    def __init__(self, config, global_fetcher):
        self.config = config
        self.global_fetcher = global_fetcher
        self.quality_stocks = config.QUALITY_STOCKS
        
    def get_signal(self, symbol, df):
        """Quality + Trend: Beli di pullback ke MA50, selama di atas MA200"""
        
        if symbol not in self.quality_stocks:
            return None
        
        if len(df) < self.config.MIN_HISTORY:
            return None
        
        # Hitung MA
        close = df['Close']
        ma200 = close.rolling(window=200).mean()
        ma50 = close.rolling(window=50).mean()
        
        if len(ma200) < 200 or len(ma50) < 50:
            return None
        
        current_price = float(close.iloc[-1])
        current_ma200 = float(ma200.iloc[-1])
        current_ma50 = float(ma50.iloc[-1])
        
        # SYARAT TREND: Harga di atas MA200 (bullish jangka panjang)
        if current_price < current_ma200:
            return None
        
        # SYARAT ENTRY: Pullback ke MA50 (dalam 2% dari MA50)
        price_to_ma50 = (current_price / current_ma50 - 1) * 100
        
        if price_to_ma50 > 2:  # Terlalu jauh dari MA50
            return None
        
        if price_to_ma50 < -5:  # Jatuh terlalu dalam (mungkin breakdown)
            return None
        
        # Hitung lot (alokasi 30% modal, maksimal 5 saham)
        max_amount = self.config.MODAL * 0.3
        lot = int(max_amount / (current_price * 100))
        lot = max(1, min(lot, 5))
        cost = lot * 100 * current_price
        
        if cost > self.config.MODAL:
            lot = int(self.config.MODAL / (current_price * 100))
            if lot < 1:
                return None
            cost = lot * 100 * current_price
        
        # Stop loss: Jika turun di bawah MA200 (trend berbalik)
        stop_loss = current_ma200 * 0.95
        
        sector = get_sector(symbol)
        
        return {
            'Symbol': symbol,
            'Sector': sector,
            'Price': int(current_price),
            'RSI': '-',
            'MA50': int(current_ma50),
            'MA200': int(current_ma200),
            'To_MA50': f"{price_to_ma50:.1f}%",
            'Stop_Loss': int(stop_loss),
            'Take_Profit': 'HOLD',  # Investasi jangka panjang
            'Lot': lot,
            'Cost': cost,
            'Risk': int(current_price - stop_loss),
            'Reasons': f'Pullback ke MA50 ({price_to_ma50:.1f}%), di atas MA200',
            'Chart': 'BULLISH' if current_price > current_ma200 else 'BEARISH',
            'Inflow': 'QUALITY',
            'Acc': 'LONG-TERM'
        }


# =============================================================================
# 17. MONTE CARLO SIMULATOR
# =============================================================================

class MonteCarloSimulator:
    def __init__(self, trades, initial_capital=40000, n_simulations=1000):
        self.trades = trades
        self.initial_capital = initial_capital
        self.n_simulations = n_simulations
        self.results = []
        self.summary = {}
        
    def run(self):
        if not self.trades:
            print("   ❌ No trades to simulate")
            return
        
        print(f"\n🎲 Running Monte Carlo ({self.n_simulations} simulations)...")
        
        returns = [t['return_after_fee_pct'] / 100 for t in self.trades]
        
        for sim in range(self.n_simulations):
            sampled_returns = random.choices(returns, k=len(returns))
            equity = self.initial_capital
            peak = equity
            max_dd = 0
            final_equity = equity
            
            for r in sampled_returns:
                equity = equity * (1 + r)
                if equity > peak:
                    peak = equity
                else:
                    dd = (peak - equity) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
                final_equity = equity
            
            total_return_pct = (final_equity - self.initial_capital) / self.initial_capital * 100
            self.results.append(total_return_pct)
        
        returns_pct = self.results
        
        self.summary = {
            'n_simulations': self.n_simulations,
            'mean_return': np.mean(returns_pct),
            'median_return': np.median(returns_pct),
            'std_return': np.std(returns_pct),
            'percentile_5': np.percentile(returns_pct, 5),
            'percentile_95': np.percentile(returns_pct, 95),
            'min_return': min(returns_pct),
            'max_return': max(returns_pct),
            'pct_positive': np.sum(np.array(returns_pct) > 0) / self.n_simulations * 100
        }
        
        print(f"   ✅ Monte Carlo complete")
        return self.summary
    
    def print_results(self):
        if not self.summary:
            print("\n📊 No Monte Carlo results")
            return
        
        print("\n" + "="*100)
        print("🎲 MONTE CARLO SIMULATION RESULTS")
        print("="*100)
        print(f"Number of simulations: {self.summary['n_simulations']:,}")
        print(f"Based on: {len(self.trades)} historical trades")
        print(f"Initial capital: Rp {self.initial_capital:,.0f}")
        
        print("\n📈 RETURN DISTRIBUTION:")
        print(f"   Mean return: {self.summary['mean_return']:.2f}%")
        print(f"   Median return: {self.summary['median_return']:.2f}%")
        print(f"   Std deviation: {self.summary['std_return']:.2f}%")
        print(f"   Best case (95th percentile): {self.summary['percentile_95']:.2f}%")
        print(f"   Worst case (5th percentile): {self.summary['percentile_5']:.2f}%")
        print(f"   Range: {self.summary['min_return']:.2f}% to {self.summary['max_return']:.2f}%")
        print(f"   Probability of profit: {self.summary['pct_positive']:.1f}%")


# =============================================================================
# 18. BACKTESTER
# =============================================================================

class Backtester:
    def __init__(self, config, global_fetcher, engine):
        self.config = config
        self.global_fetcher = global_fetcher
        self.engine = engine
        self.trades = []
        self.metrics = {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'total_return': 0, 'returns': [], 'returns_after_fee': []
        }
        self.equity_curve = []
        self.equity_curve_after_fee = []
        self.max_drawdown = 0
        self.max_drawdown_after_fee = 0
        self.max_drawdown_pct = 0
        self.max_drawdown_pct_after_fee = 0
        self.total_fees = 0
        self.monte_carlo = None
        self.entry_delay_stats = {'delay_0': 0, 'delay_1': 0, 'delay_2': 0}
    
    def calculate_equity_curve(self, initial_capital=100000):
        if not self.trades:
            return [], [], 0, 0
        sorted_trades = sorted(self.trades, key=lambda x: x['entry_date'])
        equity = initial_capital
        curve = [(sorted_trades[0]['entry_date'], initial_capital)]
        peak = initial_capital
        max_dd = 0
        equity_fee = initial_capital
        curve_fee = [(sorted_trades[0]['entry_date'], initial_capital)]
        peak_fee = initial_capital
        max_dd_fee = 0
        total_fees = 0
        for trade in sorted_trades:
            equity = equity * (1 + trade['return_pct'] / 100)
            curve.append((trade['entry_date'], equity))
            if equity > peak:
                peak = equity
            else:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            return_after_fee = trade.get('return_after_fee_pct', trade['return_pct'])
            equity_fee = equity_fee * (1 + return_after_fee / 100)
            curve_fee.append((trade['entry_date'], equity_fee))
            if equity_fee > peak_fee:
                peak_fee = equity_fee
            else:
                dd_fee = (peak_fee - equity_fee) / peak_fee * 100
                if dd_fee > max_dd_fee:
                    max_dd_fee = dd_fee
            total_fees += trade.get('fee_cost', 0)
        self.equity_curve = curve
        self.equity_curve_after_fee = curve_fee
        self.max_drawdown_pct = max_dd
        self.max_drawdown_pct_after_fee = max_dd_fee
        self.total_fees = total_fees
        return curve, curve_fee, max_dd, max_dd_fee
    
    def run_monte_carlo(self, n_simulations=1000):
        if len(self.trades) < 20:
            print("\n⚠️ Monte Carlo: Minimal 20 trades required")
            return None
        
        self.monte_carlo = MonteCarloSimulator(
            trades=self.trades,
            initial_capital=self.config.MODAL,
            n_simulations=n_simulations
        )
        return self.monte_carlo.run()
    
    def get_entry_price_with_delay(self, df, signal_idx, signal_price):
        if not hasattr(self.config, 'ENABLE_ENTRY_DELAY') or not self.config.ENABLE_ENTRY_DELAY:
            return signal_price, 0
        
        delay = random.choices(
            [0, 1, 2], 
            weights=self.config.ENTRY_DELAY_PROB
        )[0]
        
        self.entry_delay_stats[f'delay_{delay}'] += 1
        
        if delay == 0:
            return signal_price, delay
        
        max_idx = len(df) - 1
        entry_idx = min(signal_idx + delay, max_idx)
        
        if entry_idx == signal_idx:
            return signal_price, 0
        
        next_close = float(df.iloc[entry_idx]['Close'])
        entry_price = max(next_close, signal_price)
        
        return entry_price, delay
    
    def run(self, stocks_data, time_step=1):
        print(f"\n📊 Running backtest...")
        print(f"   📈 Saham: {len(stocks_data)}")
        print(f"   ⏱️  Time step: setiap {time_step} hari")
        
        total_signals = 0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        all_returns = []
        all_returns_after_fee = []
        
        for symbol, df in stocks_data.items():
            if len(df) < 60:
                continue
            
            for i in range(60, len(df) - 5, time_step):
                try:
                    data_hingga_i = df.iloc[:i].copy()
                    signal = self.engine.get_signal(symbol, data_hingga_i)
                    if signal:
                        total_signals += 1
                        
                        signal_price = signal['Price']
                        entry_price, delay_used = self.get_entry_price_with_delay(
                            df, i, signal_price
                        )
                        
                        sl = signal['Stop_Loss']
                        tp = signal['Take_Profit']
                        lot = signal['Lot']
                        
                        data_setelah = df.iloc[i + delay_used : i + delay_used + 5]
                        
                        if len(data_setelah) > 0:
                            hit_sl = False
                            hit_tp = False
                            exit_price = entry_price
                            
                            for j in range(len(data_setelah)):
                                high = data_setelah.iloc[j]['High']
                                low = data_setelah.iloc[j]['Low']
                                
                                if low <= sl and high >= tp:
                                    if random.random() < 0.5:
                                        hit_sl = True
                                        exit_price = sl
                                    else:
                                        hit_tp = True
                                        exit_price = tp
                                    break
                                elif low <= sl:
                                    hit_sl = True
                                    exit_price = sl
                                    break
                                elif high >= tp:
                                    hit_tp = True
                                    exit_price = tp
                                    break
                            
                            if not hit_sl and not hit_tp:
                                exit_price = data_setelah.iloc[-1]['Close']
                            
                            return_pct = (exit_price - entry_price) / entry_price * 100
                            return_after_fee, fee_cost = FeeConfig.adjust_return_for_fee(entry_price, exit_price, lot)
                            
                            self.trades.append({
                                'symbol': symbol,
                                'entry_date': df.index[i + delay_used],
                                'signal_date': df.index[i],
                                'delay': delay_used,
                                'signal_price': signal_price,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'return_pct': round(return_pct, 2),
                                'return_after_fee_pct': round(return_after_fee, 2),
                                'fee_cost': round(fee_cost, 0),
                                'lot': lot,
                                'hit_sl': hit_sl,
                                'hit_tp': hit_tp
                            })
                            all_returns.append(return_pct)
                            all_returns_after_fee.append(return_after_fee)
                            total_trades += 1
                            if return_after_fee > 0:
                                winning_trades += 1
                            else:
                                losing_trades += 1
                except Exception:
                    continue
        
        self.metrics = {
            'total_signals': total_signals,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_return': sum(all_returns) if all_returns else 0,
            'total_return_after_fee': sum(all_returns_after_fee) if all_returns_after_fee else 0,
            'avg_return': np.mean(all_returns) if all_returns else 0,
            'avg_return_after_fee': np.mean(all_returns_after_fee) if all_returns_after_fee else 0,
            'returns': all_returns,
            'returns_after_fee': all_returns_after_fee
        }
        
        if total_trades > 0:
            self.calculate_equity_curve(initial_capital=self.config.MODAL)
        
        print(f"   ✅ Backtest complete: {total_trades} trades")
        print(f"   💰 Total fees: Rp {self.total_fees:,.0f}")
        return self.metrics
    
    def print_results(self):
        if self.metrics['total_trades'] == 0:
            print("\n📊 No backtest results")
            return
        
        print("\n" + "="*100)
        print("📊 BACKTEST RESULTS (Dengan Fee Realism)")
        print("="*100)
        
        total = self.metrics['total_trades']
        win = self.metrics['winning_trades']
        loss = self.metrics['losing_trades']
        win_rate = (win / total * 100) if total > 0 else 0
        
        print(f"Total Trades: {total}")
        print(f"Winning Trades: {win}")
        print(f"Losing Trades: {loss}")
        print(f"Win Rate: {win_rate:.1f}%")
        
        if total > 0:
            avg_return = self.metrics['avg_return']
            avg_return_fee = self.metrics['avg_return_after_fee']
            total_return = self.metrics['total_return']
            total_return_fee = self.metrics['total_return_after_fee']
            
            print(f"\n📈 SEBELUM FEE:")
            print(f"   Average Return: {avg_return:.2f}%")
            print(f"   Total Return (sum): {total_return:.2f}%")
            
            print(f"\n📉 SETELAH FEE (dengan biaya riil):")
            print(f"   Average Return: {avg_return_fee:.2f}%")
            print(f"   Total Return (sum): {total_return_fee:.2f}%")
            print(f"   Total Biaya Fee: Rp {self.total_fees:,.0f}")
            print(f"   Fee sebagai % dari modal: {(self.total_fees/self.config.MODAL)*100:.2f}%")
            
            if loss > 0:
                avg_win = np.mean([r for r in self.metrics['returns'] if r > 0]) if win > 0 else 0
                avg_loss = abs(np.mean([r for r in self.metrics['returns'] if r < 0])) if loss > 0 else 0
                profit_factor = (avg_win * win) / (avg_loss * loss) if avg_loss > 0 else float('inf')
                avg_win_fee = np.mean([r for r in self.metrics['returns_after_fee'] if r > 0]) if win > 0 else 0
                avg_loss_fee = abs(np.mean([r for r in self.metrics['returns_after_fee'] if r < 0])) if loss > 0 else 0
                profit_factor_fee = (avg_win_fee * win) / (avg_loss_fee * loss) if avg_loss_fee > 0 else float('inf')
                print(f"\n📊 PROFIT FACTOR:")
                print(f"   Sebelum Fee: {profit_factor:.2f}")
                print(f"   Setelah Fee: {profit_factor_fee:.2f}")
        
        if self.equity_curve:
            start_equity = self.equity_curve[0][1]
            end_equity = self.equity_curve[-1][1]
            end_equity_fee = self.equity_curve_after_fee[-1][1]
            total_return_pct = (end_equity - start_equity) / start_equity * 100
            total_return_pct_fee = (end_equity_fee - start_equity) / start_equity * 100
            
            print("\n" + "-"*50)
            print("📈 EQUITY CURVE (Compounding)")
            print("-"*50)
            print(f"Start Equity: Rp {start_equity:,.0f}")
            print(f"End Equity (sebelum fee): Rp {end_equity:,.0f} ({total_return_pct:.2f}%)")
            print(f"End Equity (setelah fee): Rp {end_equity_fee:,.0f} ({total_return_pct_fee:.2f}%)")
            print(f"Max Drawdown (sebelum fee): {self.max_drawdown_pct:.2f}%")
            print(f"Max Drawdown (setelah fee): {self.max_drawdown_pct_after_fee:.2f}%")
            print(f"Number of Trades: {len(self.equity_curve)-1}")


# =============================================================================
# 19. STOCK SCANNER
# =============================================================================

class StockScanner:
    def __init__(self, config, global_fetcher, engine):
        self.config = config
        self.global_fetcher = global_fetcher
        self.engine = engine
        self.daily_trade_count = 0
    
    def calculate_portfolio_risk(self, selected_signals):
        total_risk = 0
        for signal in selected_signals:
            if 'Risk' in signal:
                risk_per_lot = signal['Risk'] * 100
                total_risk += risk_per_lot * signal['Lot']
        risk_pct = (total_risk / self.config.MODAL) * 100 if self.config.MODAL > 0 else 0
        return total_risk, risk_pct
    
    def filter_by_ranking(self, signals):
        if not signals:
            return []
        
        if hasattr(self.config, 'MIN_EV_PCT'):
            ev_filtered = [s for s in signals if s.get('EV_Pct', 100) >= self.config.MIN_EV_PCT]
        else:
            ev_filtered = signals
        
        if not ev_filtered:
            return []
        
        if 'Score' in ev_filtered[0]:
            ranked = sorted(ev_filtered, key=lambda x: -x['Score'])
        else:
            ranked = ev_filtered
        
        max_positions = getattr(self.config, 'MAX_PORTFOLIO', 5)
        top_n = ranked[:max_positions]
        
        return top_n
    
    def print_global_summary(self):
        print("\n" + "="*100)
        print("🌍 RINGKASAN PASAR")
        print("="*100)
        data = []
        for name in GLOBAL_INDICES.keys():
            mom = self.global_fetcher.get_momentum(name)
            trend = self.global_fetcher.get_trend(name)
            price_str = self.global_fetcher.get_price_str(name)
            data.append([name, price_str, f"{mom:+.2f}%", trend])
        print(tabulate(data, headers=["Indeks", "Harga", "Momentum", "Trend"], tablefmt="grid"))
    
    def print_signals(self, signals, engine_name):
        if not signals:
            print(f"\n❌ Tidak ada sinyal {engine_name} hari ini")
            return
        
        print("\n" + "="*100)
        print(f"📊 {engine_name} - REKOMENDASI ({len(signals)} sinyal)")
        print("="*100)
        print(f"Modal: Rp {self.config.MODAL:,}")
        
        if hasattr(self.config, 'MAX_PORTFOLIO'):
            print(f"Max posisi: {self.config.MAX_PORTFOLIO}")
        if hasattr(self.config, 'MAX_PORTFOLIO_RISK_PCT'):
            print(f"Portfolio Risk Cap: {self.config.MAX_PORTFOLIO_RISK_PCT}%")
        
        print("-"*100)
        
        display_data = []
        for s in signals:
            risk_amount = s.get('Risk', 0) * s.get('Lot', 1) * 100
            risk_pct = (risk_amount / self.config.MODAL) * 100 if self.config.MODAL > 0 else 0
            
            if engine_name == "INVESTASI ENGINE":
                display_data.append([
                    s['Symbol'],
                    s['Sector'],
                    f"Rp {s['Price']:,}",
                    f"{s['To_MA50']}",
                    f"Rp {s['MA50']:,}",
                    f"Rp {s['MA200']:,}",
                    f"Rp {s['Stop_Loss']:,}",
                    s['Take_Profit'],
                    f"{risk_pct:.1f}%",
                    f"{s['Lot']} lot",
                    f"Rp {s['Cost']:,}"
                ])
                headers = [
                    "Kode", "Sektor", "Harga", "To MA50", "MA50", "MA200",
                    "Stop Loss", "Target", "Risk%", "Lot", "Biaya"
                ]
            else:
                flow_status = "💰 INFLOW" if s.get('Inflow') == "INFLOW" else "💸 OUTFLOW" if s.get('Inflow') == "OUTFLOW" else "⚖️ NETRAL"
                acc_status = "📈 AKUM" if s.get('Acc') == "AKUMULASI" else "📉 DIST" if s.get('Acc') == "DISTRIBUSI" else "➖"
                
                display_data.append([
                    s['Symbol'],
                    s['Sector'],
                    f"Rp {s['Price']:,}",
                    s.get('RSI', '-'),
                    f"{s.get('Volume', '-')}",
                    flow_status,
                    acc_status,
                    f"Rp {s.get('Support', 0):,}",
                    f"Rp {s.get('Resistance', 0):,}" if s.get('Resistance') else '-',
                    f"Rp {s.get('Stop_Loss', 0):,}",
                    f"Rp {s.get('Take_Profit', 0):,}",
                    f"1:{s.get('R/R', 0)}",
                    f"{s.get('Prob_Up', 0):.0%}" if s.get('Prob_Up') else '-',
                    f"{s.get('EV_Pct', 0)}%",
                    f"{s.get('Score', 0)}",
                    s.get('Chart', '-'),
                    f"{risk_pct:.1f}%",
                    f"{s['Lot']} lot",
                    f"Rp {s['Cost']:,}"
                ])
                headers = [
                    "Kode", "Sektor", "Harga", "RSI", "Vol", "Flow", "Acc",
                    "Support", "Resist", "SL", "TP", "R/R", "Prob",
                    "EV%", "Skor", "Trend", "Risk%", "Lot", "Biaya"
                ]
        
        print(tabulate(display_data, headers=headers, tablefmt='grid', stralign='left', numalign='center'))
        
        total_risk, risk_pct = self.calculate_portfolio_risk(signals)
        print(f"\n📊 Total Portfolio Risk: Rp {total_risk:,} ({risk_pct:.2f}% dari modal)")
    
    def print_portfolio_guide(self, signals, engine_name):
        if not signals:
            return
        print("\n" + "="*100)
        print(f"📊 {engine_name} - PANDUAN PORTOFOLIO")
        print("="*100)
        print(f"Anda bisa membeli maksimal {getattr(self.config, 'MAX_PORTFOLIO', 5)} saham")
        total_risk_used = 0
        for i, s in enumerate(signals, 1):
            risk_amount = s.get('Risk', 0) * s.get('Lot', 1) * 100
            print(f"  {i}. {s['Symbol']}: {s['Lot']} lot, Harga Rp {s['Price']:,}, Risk Rp {risk_amount:,}")
            total_risk_used += risk_amount
        
        print(f"\n💰 Total risk digunakan: Rp {total_risk_used:,}")


# =============================================================================
# 20. MAIN PROGRAM - QUADRUPLE ENGINE FIXED
# =============================================================================

def main():
    print("\n" + "="*60)
    print("🏦 IDX STOCK SCANNER - QUADRUPLE ENGINE (FIXED)")
    print("   Swing + Intraday Liquid + Intraday Gorengan + Investasi")
    print("="*60)
    
    print("\nPilih engine trading:")
    print("1. Swing Engine (Mingguan - Mean Reversion)")
    print("2. Intraday Liquid (1 jam - Momentum)")
    print("3. Intraday Gorengan (1 jam - Early Momentum)")
    print("4. Investasi Engine (Quality + Trend - Jangka Panjang)")
    
    while True:
        engine_choice = input("Pilihan (1/2/3/4): ").strip()
        if engine_choice in ['1', '2', '3', '4']:
            break
        print("❌ Pilih 1, 2, 3, atau 4")
    
    while True:
        try:
            modal_input = input("\nModal (Rp 10,000 - 5,000,000): ").strip()
            modal = int(modal_input.replace('.', '').replace(',', ''))
            if 10000 <= modal <= 5000000:
                break
            print("❌ Modal harus 10,000 - 5,000,000")
        except:
            print("❌ Input tidak valid")
    
    # Buat config berdasarkan engine
    if engine_choice == '1':  # Swing
        mode = 'mingguan'
        config = Config(modal, mode)
        engine_name = "SWING ENGINE"
        EngineClass = SwingEngine
    elif engine_choice == '2':  # Intraday Liquid
        mode = 'intraday'
        config = Config(modal, mode)
        engine_name = "INTRADAY LIQUID ENGINE"
        EngineClass = IntradayLiquidEngine
    elif engine_choice == '3':  # Intraday Gorengan
        mode = 'intraday'
        config = Config(modal, mode)
        # Override untuk gorengan
        config.MIN_PRICE = 50
        config.MAX_SPREAD_PCT = 3.0
        config.MIN_AVG_VOLUME = 200000
        config.MIN_EV_PCT = 1.5
        config.RISK_PER_TRADE_PCT = 0.3
        config.MAX_TRADES_PER_DAY = 2
        engine_name = "INTRADAY GORENGAN ENGINE"
        EngineClass = IntradayGorenganEngine
    else:  # Investasi
        config = InvestasiConfig(modal)
        engine_name = "INVESTASI ENGINE"
        EngineClass = InvestasiEngine
    
    if hasattr(config, 'ENABLE_RANDOM_SLIPPAGE') and config.ENABLE_RANDOM_SLIPPAGE:
        FeeConfig.SLIPPAGE_MODE = 'random'
    else:
        FeeConfig.SLIPPAGE_MODE = 'fixed'
    
    global_fetcher = GlobalIndicesFetcher()
    global_fetcher.fetch_all(config)
    
    # Buat engine
    if engine_choice == '4':
        engine = EngineClass(config, global_fetcher)
    else:
        engine = EngineClass(config, global_fetcher)
    
    fetcher = CachedDataFetcher()
    scanner = StockScanner(config, global_fetcher, engine)
    
    print(f"\n💰 Modal: Rp {modal:,}")
    print(f"📊 Engine: {engine_name}")
    print(f"📊 Interval: {getattr(config, 'INTERVAL', '1d')}")
    if hasattr(config, 'MAX_PORTFOLIO'):
        print(f"📊 Max posisi: {config.MAX_PORTFOLIO}")
    if hasattr(config, 'MIN_PRICE'):
        print(f"📊 Min Price: Rp {config.MIN_PRICE}")
    if hasattr(config, 'MAX_SPREAD_PCT'):
        print(f"📊 Max Spread: {config.MAX_SPREAD_PCT}%")
    if hasattr(config, 'MIN_EV_PCT'):
        print(f"📊 Min EV: {config.MIN_EV_PCT}%")
    if hasattr(config, 'MIN_AVG_VOLUME'):
        print(f"📊 Filter likuiditas: Min {config.MIN_AVG_VOLUME:,} lembar")
    if hasattr(config, 'MAX_PORTFOLIO_RISK_PCT'):
        print(f"📊 Portfolio Risk Cap: {config.MAX_PORTFOLIO_RISK_PCT}%")
    
    # Tentukan universe
    if engine_choice == '4':
        # Untuk investasi, gunakan quality universe
        universe = config.QUALITY_STOCKS
        print(f"📊 Quality Universe: {len(universe)} saham (IDX30+LQ45+IDX80)")
    else:
        universe = STOCKBIT_UNIVERSE
        print(f"📊 Universe: {len(universe)} saham")
    
    print(f"\n📥 Download {len(universe)} stocks...")
    stocks_data = {}
    start_time = time.time()
    
    for i, symbol in enumerate(universe):
        df = fetcher.fetch(symbol, config)
        if df is not None:
            stocks_data[symbol] = df
        if (i+1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"   Progress: {i+1}/{len(universe)} - {len(stocks_data)} ditemukan")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Selesai dalam {elapsed:.1f} detik")
    fetcher.print_stats()
    
    if stocks_data:
        print(f"\n📊 Menganalisis {len(stocks_data)} saham...")
        signals = []
        
        for symbol, df in stocks_data.items():
            signal = engine.get_signal(symbol, df)
            if signal:
                signals.append(signal)
        
        print(f"✅ Ditemukan {len(signals)} sinyal mentah")
        
        ranked_signals = scanner.filter_by_ranking(signals)
        print(f"✅ Setelah ranking: {len(ranked_signals)} sinyal terbaik")
        
        scanner.print_global_summary()
        scanner.print_signals(ranked_signals, engine_name)
        scanner.print_portfolio_guide(ranked_signals, engine_name)
        
        # Opsi backtest (hanya untuk engine trading, bukan investasi)
        if engine_choice != '4':
            print("\n" + "="*100)
            print("📊 BACKTEST OPTION")
            print("="*100)
            print("Pilih periode backtest:")
            print("1. 6 bulan (cepat, 10-15 menit)")
            print("2. 5 tahun (lengkap, 2-3 jam)")
            print("0. Lewati")
            
            bt_choice = input("Pilihan (0/1/2): ").strip()
            
            if bt_choice in ['1', '2']:
                time_step = 1
                if bt_choice == '2' and engine_choice == '1':
                    time_step = 3  # Quick mode untuk swing 5 tahun
                
                print("\n" + "="*100)
                print(f"📊 BACKTEST")
                print("="*100)
                
                backtester = Backtester(config, global_fetcher, engine)
                backtester.run(stocks_data, time_step=time_step)
                backtester.print_results()
                
                if len(backtester.trades) >= 20:
                    print("\n" + "="*100)
                    print("🎲 MONTE CARLO SIMULATION")
                    print("="*100)
                    backtester.run_monte_carlo()
                    backtester.monte_carlo.print_results()
                else:
                    print(f"\n⚠️ Monte Carlo: Minimal 20 trades required (current: {len(backtester.trades)})")
            else:
                print("\n✅ Backtest dilewati.")
        else:
            print("\nℹ️  Backtest tidak tersedia untuk Engine Investasi.")
            print("   Investasi jangka panjang menggunakan analisis teknikal sederhana.")
        
    else:
        print(f"\n❌ Tidak ada data")
    
    print("\n" + "="*100)
    print("📝 CARA EKSEKUSI:")
    print("="*100)
    print("1. Pilih saham yang ingin dibeli")
    print("2. Gunakan LIMIT ORDER di harga ≤ rekomendasi")
    print("3. Pasang Stop Loss sesuai level")
    if engine_choice == '4':
        print("4. Target HOLD jangka panjang (exit jika turun di bawah MA200)")
    else:
        print("4. Target Take Profit sesuai level")
    print("5. Patuhi risk management")
    if engine_choice in ['2', '3']:
        print("6. WAJIB CLOSE sebelum jam 15:50 (no overnight)")
        print(f"7. Max {getattr(config, 'MAX_TRADES_PER_DAY', 2)} trade per hari - disiplin!")
        print("8. Jika loss 2x berturut-turut, stop trading hari itu")
    if engine_choice == '3':
        print("⚠️  PERINGATAN: Saham gorengan berisiko tinggi!")
        print("⚠️  Gunakan risk budget kecil (10-20% dari total modal)")
    if engine_choice == '4':
        print("ℹ️  Investasi jangka panjang: rebalancing bulanan cukup.")
    print("\n✅ Selamat trading!")

if __name__ == "__main__":
    main()
