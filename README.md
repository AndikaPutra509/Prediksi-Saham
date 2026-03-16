# =============================================================================
# 📈 IDX STOCK SCANNER - QUADRUPLE ENGINE (AGGRESSIVE VERSION) - FINAL ENHANCED
# Fitur:
# - Semua fitur sebelumnya + peningkatan:
#   ✅ Filter likuiditas berbasis turnover
#   ✅ Optimasi portofolio mean-variance dengan shrinkage
#   ✅ Analisis skenario makro (IHSG, USD/IDR)
#   ✅ News sentiment dengan bobot jumlah artikel
#   ✅ Mitigasi overfitting pada setiap peningkatan
#   ✅ Input modal bebas (hanya modal yang tersedia)
#   ✅ Peringatan konsentrasi sektor & korelasi
#   ✅ Rekomendasi alokasi antar engine
#   ✅ Template export spreadsheet seragam dengan kolom P&L manual
#   ✅ Stop loss investasi longgar untuk target agresif
#   ✅ Data fundamental (PER, PBV, ROE) – opsional, tanpa default
#   ✅ Indikator teknikal tambahan di Swing Engine (MACD, BB, Stochastic, support, gap, divergensi)
#   ✅ Filter timeframe lebih tinggi (weekly trend)
#   ✅ Pengaruh indeks global tambahan: Nikkei 225, Shanghai Composite
#   ✅ Peningkatan Swing Engine: target 9%, max hold 30 hari, filter ADX >=20, weekly trend diperkuat
#   ✅ Peningkatan Investasi Engine: target 20% per tahun, max hold 365 hari, filter ROE>=12% & PBV<=2, target ATR15
# =============================================================================

# =============================================================================
# 1. INSTALL DEPENDENCIES & IMPORTS
# =============================================================================

!pip install -q ta
from google.colab import auth
from google.auth import default
import gspread
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import time
import pickle
import os
import json
from tabulate import tabulate
from collections import defaultdict
import logging
import random
from typing import Optional, Dict, List, Tuple, Union, Any
import hashlib
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.covariance import ledoit_wolf
from scipy.optimize import minimize
import statsmodels.api as sm
import glob
import math
import requests
from textblob import TextBlob
from dotenv import load_dotenv
import ta  # Technical Analysis Library

# Matikan logging yang tidak perlu
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

# Setup logging untuk error handling
logging.basicConfig(
    filename='scanner_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("✅ Dependencies imported")

# =============================================================================
# 2. STOCK UNIVERSE (FULL) - TIDAK BERUBAH
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
    "UJSN", "ULTJ", "UNIC", "UNIQ", "UNIT", "UNSP", "USFI", "VALU", "VICO", "VICI",
    "VIDI", "VISI", "VIVA", "VKTR", "VOKS", "VRNA", "VTNY", "WAPO", "WEGE", "WEHA",
    "WICO", "WIFI", "WIIM", "WINS", "WMUU", "WMPP", "WOOD", "WOWS", "WRKR", "WSBP",
    "WSKT", "WTON", "YELO", "YULE", "ZBRA", "ZINC", "ZONE"
]

# =============================================================================
# 3. QUALITY STOCKS UNIVERSE (IDX30 + LQ45 + IDX80) - TIDAK BERUBAH
# =============================================================================

QUALITY_STOCKS = [
    'ADRO', 'AMRT', 'ANTM', 'ASII', 'BBCA', 'BBNI', 'BBRI', 'BMRI', 'BREN', 'BUMI',
    'CPIN', 'EMTK', 'GOTO', 'ICBP', 'INCO', 'INDF', 'INKP', 'ISAT', 'JPFA', 'JSMR',
    'KLBF', 'MDKA', 'MEDC', 'MIKA', 'MTEL', 'NCKL', 'PGAS', 'TLKM', 'TOWR', 'UNTR',
    'AKRA', 'ARTO', 'BBTN', 'BDKR', 'BIRD', 'BJBR', 'BJTM', 'BRIS', 'BRPT', 'BUKA',
    'EXCL', 'HRUM', 'INTP', 'ITMG', 'MPMX', 'PTBA', 'SIDO', 'SMGR', 'TPIA', 'UNVR',
    'WIKA', 'ACES', 'ADHI', 'AGRO', 'AALI', 'ARNA', 'ASGR', 'ASRI', 'AUTO', 'BAYU',
    'BEST', 'BFIN', 'BIPP', 'BISI', 'BKSL', 'BLTA', 'BMAS', 'BMSR', 'BNGA', 'BNII',
    'BOBA', 'BOLT', 'BOSS', 'BPFI', 'BRAM', 'BSDE', 'BSSR', 'BTON', 'BUDI', 'BULL',
    'BUVA', 'BWPT', 'BYAN', 'CAMP', 'CANI', 'CARS', 'CASA', 'CASS', 'CBDK', 'CBMF',
    'CCSI', 'CDAX', 'CEKA', 'CENT', 'CFIN', 'CITA', 'CITY', 'CKRA', 'CLEO', 'CLPI',
    'CMNP', 'CMPP', 'CMRY', 'CNKO', 'COAL', 'COCO', 'COWL', 'CPRO', 'CSAP', 'CSIS',
    'CTBN', 'CTRA', 'CTTH', 'CUAN', 'DART', 'DASA', 'DCII', 'DEGA', 'DEWA', 'DGIK',
    'DIGI', 'DILD', 'DIVA', 'DIVN', 'DLTA', 'DMAS', 'DMND', 'DNAR', 'DNET', 'DNLS',
    'DOID', 'DOOH', 'DPNS', 'DSFI', 'DSNG', 'DSSA', 'DUCK', 'DUTI', 'DVLA', 'DYAN',
    'EASI', 'EASY', 'EBMT', 'ECII', 'EDGE', 'EKAD', 'ELBA', 'ELSA', 'ELTY', 'EMBR',
    'EMDE', 'EMTK', 'ENRG', 'ENVY', 'ENZO', 'EPAC', 'EPMT', 'ERAA', 'ESSA', 'ESTA',
    'ESTI', 'ETWA', 'FAST', 'FASW', 'FILM', 'FISH', 'FITT', 'FKSF', 'FLMC', 'FMII',
    'FORE', 'FORU', 'FORZ', 'FPNI', 'FREN', 'FUJI', 'FUTR', 'GAMA', 'GDST', 'GDYR',
    'GEMS', 'GGRM', 'GGRP', 'GHON', 'GIDS', 'GJTL', 'GLVA', 'GMFI', 'GMTD', 'GOLD',
    'GOOD', 'GPRA', 'GRPH', 'GSMF', 'GTBO', 'GTRA', 'GTSI', 'GULA', 'HADE', 'HDFA',
    'HDIT', 'HEAL', 'HERO', 'HITS', 'HKMU', 'HMSP', 'HOKI', 'HOMI', 'HOPE', 'HOTL',
    'HRME', 'HRTA', 'HRUM', 'HSBK', 'HSMP', 'HUMI', 'IBFN', 'IBOS', 'IBST', 'ICON',
    'IDPR', 'IFII', 'IFSH', 'IGAR', 'IIKP', 'IKAI', 'IKAN', 'IMAS', 'IMJS', 'IMPC',
    'INAF', 'INAI', 'INCF', 'INCI', 'INDS', 'INDX', 'INDY', 'INET', 'INKP', 'INPC',
    'INPP', 'INPS', 'INRU', 'INTA', 'INTD', 'INTP', 'IPCC', 'IPCM', 'IPOL', 'IRRA',
    'ISEA', 'ISSP', 'ITIC', 'JAST', 'JAWA', 'JAYA', 'JECC', 'JEMB', 'JFAS', 'JGLE',
    'JHON', 'JIHD', 'JKON', 'JKSW', 'JMAS', 'JPII', 'JPUR', 'JRPT', 'JSKY', 'JSPT',
    'JTNB', 'KAEF', 'KAQI', 'KARW', 'KBLI', 'KBLM', 'KBRT', 'KBRI', 'KDSI', 'KDTN',
    'KEEN', 'KETR', 'KICI', 'KIJA', 'KINO', 'KIOS', 'KJEN', 'KKGI', 'KMTR', 'KOBX',
    'KOIN', 'KOLI', 'KONI', 'KOTA', 'KPAL', 'KPIG', 'KRAS', 'KREN', 'KRYA', 'KSEL',
    'KUAS', 'KUIC', 'KUVO', 'LAND', 'LAPD', 'LATA', 'LBAK', 'LCGP', 'LCKM', 'LEAD',
    'LIFE', 'LINK', 'LION', 'LISA', 'LMAS', 'LMPI', 'LMSH', 'LPCK', 'LPGI', 'LPIN',
    'LPKR', 'LPLI', 'LPPF', 'LPPS', 'LSIP', 'LSPI', 'LTLS', 'LUCY', 'MABA', 'MABH',
    'MAGP', 'MAIN', 'MAMI', 'MAPA', 'MAPB', 'MAPI', 'MARA', 'MASA', 'MAYA', 'MBAP',
    'MBCA', 'MBMA', 'MBSS', 'MBTO', 'MCAS', 'MCPI', 'MCOR', 'MDIA', 'MDKA', 'MDKI',
    'MEDC', 'MEDS', 'MEGA', 'MERK', 'META', 'MFIN', 'MFMI', 'MGLV', 'MGNA', 'MGRO',
    'MIDI', 'MIKA', 'MINA', 'MIRA', 'MITI', 'MITT', 'MKNT', 'MKPI', 'MLBI', 'MLIA',
    'MLPL', 'MLPT', 'MLSL', 'MMIX', 'MMLP', 'MNCN', 'MOLI', 'MPOW', 'MPPA', 'MPRO',
    'MPTJ', 'MRAT', 'MSIE', 'MSIN', 'MSKY', 'MTDL', 'MTFN', 'MTLA', 'MTPS', 'MTSM',
    'MUDA', 'MUTU', 'MYOH', 'MYOR', 'MYRX', 'MYSX', 'NAGA', 'NASI', 'NATO', 'NAYZ',
    'NCKL', 'NELY', 'NETV', 'NFCX', 'NICL', 'NIKL', 'NISP', 'NITY', 'NIYM', 'NOBU',
    'NPGF', 'NRCA', 'NSSS', 'NTBK', 'NUSA', 'NUSI', 'OASA', 'OCTN', 'OKAS', 'OMED',
    'ONIX', 'OPMS', 'ORNA', 'OTBK', 'PADA', 'PADI', 'PAMG', 'PANR', 'PANS', 'PANU',
    'PAPA', 'PASA', 'PASS', 'PBRX', 'PBID', 'PBSA', 'PCAR', 'PDES', 'PDGD', 'PDIN',
    'PEGE', 'PGAS', 'PGLI', 'PGUN', 'PICO', 'PIDRA', 'PJAA', 'PKPK', 'PLAN', 'PLAS',
    'PLIN', 'PMJS', 'PMMP', 'PNBN', 'PNBS', 'PNIN', 'PNLF', 'PNSE', 'POLI', 'POLL',
    'POLU', 'POLY', 'POOL', 'PORT', 'POWR', 'PPGL', 'PPRE', 'PPRO', 'PPSI', 'PRAS',
    'PRDA', 'PRIM', 'PRIN', 'PRLD', 'PROD', 'PROT', 'PRTS', 'PSAB', 'PSBA', 'PSDN',
    'PSGO', 'PSKT', 'PSSI', 'PTDU', 'PTIS', 'PTMP', 'PTPP', 'PTPW', 'PTRO', 'PTSN',
    'PTSP', 'PUDP', 'PURA', 'PURE', 'PWON', 'PYFA', 'RACE', 'RADIO', 'RAFI', 'RAJA',
    'RAKD', 'RALS', 'RANC', 'RATU', 'RBMS', 'RDTX', 'REAL', 'RELI', 'RIGS', 'RIMO',
    'RISE', 'RMBA', 'RMKE', 'ROCK', 'RODA', 'ROKI', 'ROTI', 'RRMI', 'RUIS', 'RUMI',
    'SABA', 'SAFE', 'SAME', 'SAPX', 'SARA', 'SATO', 'SBAT', 'SBBP', 'SBGA', 'SBMA',
    'SBMF', 'SCBD', 'SCCC', 'SCCO', 'SCMA', 'SCNP', 'SDPC', 'SDRA', 'SEAN', 'SECR',
    'SEMA', 'SFAN', 'SGER', 'SGRO', 'SHID', 'SHIP', 'SIDO', 'SILO', 'SIMA', 'SIMP',
    'SIPD', 'SIPO', 'SKBM', 'SKLT', 'SKRN', 'SLIS', 'SMAR', 'SMDR', 'SMGR', 'SMIL',
    'SMMT', 'SMSM', 'SMRA', 'SNLK', 'SNMS', 'SOFA', 'SONA', 'SOSS', 'SOUL', 'SPMA',
    'SPMI', 'SPNA', 'SPRE', 'SPTO', 'SQBI', 'SQMI', 'SRAJ', 'SRIL', 'SRSN', 'SSIA',
    'SSMS', 'SSTM', 'STAR', 'STTP', 'SUGI', 'SULI', 'SUPR', 'SURI', 'SWAT', 'SWID',
    'TALD', 'TAMA', 'TAMU', 'TAPG', 'TARA', 'TASP', 'TATA', 'TAXI', 'TBIG', 'TBLA',
    'TCID', 'TDPM', 'TELE', 'TEMB', 'TEMPO', 'TIFA', 'TIGA', 'TINS', 'TIRA', 'TIRT',
    'TITA', 'TKGA', 'TKIM', 'TLKM', 'TMAS', 'TMPO', 'TMSH', 'TOBA', 'TOOL', 'TOPS',
    'TOSK', 'TOTL', 'TOTO', 'TOWR', 'TPIA', 'TPMA', 'TRAM', 'TRGU', 'TRIO', 'TRIS',
    'TRJA', 'TRON', 'TRST', 'TRUB', 'TRUK', 'TRUS', 'TSPC', 'TUGU', 'TURI', 'TUVN',
    'TYRE', 'UANG', 'UCID', 'UDIJ', 'UFNX', 'UGRO', 'UJSN', 'ULTJ', 'UNIC', 'UNIQ',
    'UNIT', 'UNSP', 'USFI', 'VALU', 'VICO', 'VICI', 'VIDI', 'VISI', 'VIVA', 'VKTR',
    'VOKS', 'VRNA', 'VTNY', 'WAPO', 'WEGE', 'WEHA', 'WICO', 'WIFI', 'WIIM', 'WINS',
    'WMUU', 'WMPP', 'WOOD', 'WOWS', 'WRKR', 'WSBP', 'WSKT', 'WTON', 'YELO', 'YULE',
    'ZBRA', 'ZINC', 'ZONE'
]
QUALITY_STOCKS = list(dict.fromkeys(QUALITY_STOCKS))

# =============================================================================
# 4. SECTOR MAPPING - TIDAK BERUBAH
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

def get_sector(symbol: str) -> str:
    """Get sector for a symbol"""
    for sector, stocks in SECTOR_MAPPING.items():
        if symbol in stocks:
            return sector
    return 'OTHER'

# =============================================================================
# 5. GLOBAL INDICES CONFIGURATION - DIPERBARUI (TAMBAHAN NIKKEI & SHANGHAI)
# =============================================================================

GLOBAL_INDICES = {
    "IHSG": {
        "ticker": "^JKSE",
        "nama": "Indeks Harga Saham Gabungan",
        "satuan": "poin",
        "keterangan": "Indeks utama BEI"
    },
    "DOWJONES": {
        "ticker": "^DJI",
        "nama": "Dow Jones Industrial Average",
        "satuan": "poin",
        "keterangan": "Indeks saham AS"
    },
    "USDIDR": {
        "ticker": "IDR=X",
        "nama": "USD/IDR",
        "satuan": "Rp/USD",
        "keterangan": "Nilai tukar Rupiah terhadap Dolar"
    },
    "OIL": {
        "ticker": "CL=F",
        "nama": "Crude Oil WTI",
        "satuan": "USD/barel",
        "keterangan": "Harga minyak dunia"
    },
    "GOLD": {
        "ticker": "GC=F",
        "nama": "Gold Futures",
        "satuan": "USD/ons",
        "keterangan": "Harga emas dunia"
    },
    "NIKKEI": {
        "ticker": "^N225",
        "nama": "Nikkei 225",
        "satuan": "poin",
        "keterangan": "Indeks saham Jepang"
    },
    "SHANGHAI": {
        "ticker": "000001.SS",
        "nama": "Shanghai Composite Index",
        "satuan": "poin",
        "keterangan": "Indeks saham China"
    }
}

# =============================================================================
# 6. GLOBAL INDICES FETCHER (LENGKAP) - VERSI FIX UNTUK OIL/GOLD
# =============================================================================

class GlobalIndicesFetcher:
    """Fetch global indices data for market context - DENGAN FIX UNTUK OIL/GOLD"""

    def __init__(self):
        self.data = {}
        self.momentum = {}
        self.status = {}
        self.prices = {}
        self.detail = {}

    def fetch_all(self) -> None:
        print("\n" + "="*80)
        print("📡 FETCHING GLOBAL INDICES - DETAIL TRANSPARAN")
        print("="*80)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)

        for name, config in GLOBAL_INDICES.items():
            ticker = config["ticker"]
            print(f"\n📊 {name}: {config['nama']}")
            print(f"   Ticker: {ticker}")
            print(f"   Satuan: {config['satuan']}")
            print(f"   Keterangan: {config['keterangan']}")

            try:
                df = yf.download(
                    ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    timeout=10
                )

                time.sleep(0.5)

                if df.empty or len(df) < 200:
                    self.status[name] = "UNAVAILABLE"
                    self.momentum[name] = 0.0
                    self.prices[name] = 0.0
                    print(f"   ⚠️  Status: TIDAK TERSEDIA (data tidak cukup)")
                else:
                    # Normalize columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    # FIX UNTUK OIL & GOLD: handle NaN
                    if name in ["OIL", "GOLD"]:
                        close_col = None
                        for col in ['Close', 'Adj Close']:
                            if col in df.columns and not df[col].isna().all():
                                close_col = col
                                break
                        if close_col is None:
                            for col in df.columns:
                                if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
                                    close_col = col
                                    break
                        if close_col:
                            close = df[close_col].values
                            close = close[~np.isnan(close)]
                        else:
                            close = np.array([])
                    else:
                        close = df['Close'].values

                    if len(close) == 0:
                        self.status[name] = "UNAVAILABLE"
                        self.momentum[name] = 0.0
                        self.prices[name] = 0.0
                        print(f"   ⚠️  Status: TIDAK TERSEDIA (data tidak valid)")
                        continue

                    current_price = float(close[-1])
                    price_1w_ago = float(close[-6]) if len(close) >= 6 else current_price
                    price_1m_ago = float(close[-22]) if len(close) >= 22 else current_price
                    price_3m_ago = float(close[-66]) if len(close) >= 66 else current_price
                    price_1y_ago = float(close[-252]) if len(close) >= 252 else current_price

                    momentum_1w = (current_price / price_1w_ago - 1) * 100 if price_1w_ago > 0 else 0
                    momentum_1m = (current_price / price_1m_ago - 1) * 100 if price_1m_ago > 0 else 0
                    momentum_3m = (current_price / price_3m_ago - 1) * 100 if price_3m_ago > 0 else 0
                    momentum_1y = (current_price / price_1y_ago - 1) * 100 if price_1y_ago > 0 else 0

                    if name == "IHSG" and len(close) >= 200:
                        ma50 = np.mean(close[-50:])
                        ma200 = np.mean(close[-200:])
                        self.data['IHSG_MA50'] = ma50
                        self.data['IHSG_MA200'] = ma200
                        self.data['IHSG_Close'] = current_price

                        if current_price > ma50 > ma200:
                            trend = "BULLISH (di atas MA50 & MA200)"
                        elif current_price < ma50 < ma200:
                            trend = "BEARISH (di bawah MA50 & MA200)"
                        elif current_price > ma50:
                            trend = "NETRAL (di atas MA50, di bawah MA200)"
                        else:
                            trend = "NETRAL (di bawah MA50, di atas MA200)"

                        returns = pd.Series(close).pct_change().dropna() * 100
                        volatility = returns.tail(20).std() * np.sqrt(252) if len(returns) > 20 else 20.0
                    else:
                        trend = "TIDAK TERSEDIA"
                        volatility = 20.0

                    self.detail[name] = {
                        'current_price': current_price,
                        'price_1w_ago': price_1w_ago,
                        'price_1m_ago': price_1m_ago,
                        'price_3m_ago': price_3m_ago,
                        'price_1y_ago': price_1y_ago,
                        'momentum_1w': round(momentum_1w, 2),
                        'momentum_1m': round(momentum_1m, 2),
                        'momentum_3m': round(momentum_3m, 2),
                        'momentum_1y': round(momentum_1y, 2),
                        'trend': trend,
                        'volatility': round(volatility, 2),
                        'data_length': len(df)
                    }

                    self.data[name] = df
                    self.momentum[name] = round(momentum_1m, 2)
                    self.prices[name] = round(current_price, 2)
                    self.status[name] = "OK"

                    print(f"   ✅ Status: TERSEDIA")
                    print(f"   Harga Saat Ini: {self.get_price_str(name)}")
                    print(f"   Momentum 1w: {momentum_1w:+.2f}%")
                    print(f"   Momentum 1m: {momentum_1m:+.2f}%")
                    print(f"   Momentum 3m: {momentum_3m:+.2f}%")
                    print(f"   Momentum 1y: {momentum_1y:+.2f}%")
                    if name == "IHSG":
                        print(f"   MA50: {ma50:,.2f}")
                        print(f"   MA200: {ma200:,.2f}")
                        print(f"   Trend: {trend}")
                        print(f"   Volatility (annualized): {volatility:.1f}%")

            except Exception as e:
                self.status[name] = "ERROR"
                self.momentum[name] = 0.0
                self.prices[name] = 0.0
                logger.error(f"Error fetching {name}: {str(e)}")
                print(f"   ❌ Status: ERROR - {str(e)[:50]}")
                time.sleep(0.5)

        print("\n" + "="*80)
        print("✅ SEMUA INDEKS GLOBAL SELESAI DI-FETCH")
        print("="*80)

    def print_detailed_report(self) -> None:
        if not self.detail:
            print("\n📊 Tidak ada data indeks untuk ditampilkan")
            return

        print("\n" + "="*100)
        print("📊 LAPORAN DETAIL INDEKS GLOBAL")
        print("="*100)

        data = []
        for name, detail in self.detail.items():
            data.append([
                name,
                self.get_price_str(name),
                f"{detail['momentum_1w']:+.2f}%",
                f"{detail['momentum_1m']:+.2f}%",
                f"{detail['momentum_3m']:+.2f}%",
                f"{detail['momentum_1y']:+.2f}%",
                detail.get('trend', 'N/A'),
                detail.get('volatility', 'N/A'),
                self.status[name]
            ])

        headers = ["Indeks", "Harga", "1W", "1M", "3M", "1Y", "Trend", "Vol%", "Status"]
        print(tabulate(data, headers=headers, tablefmt='grid'))

    def get_price_str(self, name: str) -> str:
        price = self.prices.get(name, 0)
        if price == 0:
            return "N/A"
        if name in ["IHSG", "DOWJONES", "NIKKEI", "SHANGHAI"]:
            return f"{price:,.2f}"
        elif name == "USDIDR":
            return f"Rp {price:,.0f}"
        elif name == "OIL":
            return f"US$ {price:.2f}"
        elif name == "GOLD":
            return f"US$ {price:.2f}"
        return f"{price:.2f}"

    def get_trend(self, name: str) -> str:
        mom = self.momentum.get(name, 0)
        if mom > 0.5:
            return "🟢 BULLISH"
        elif mom < -0.5:
            return "🔴 BEARISH"
        else:
            return "🟡 NETRAL"

    def get_momentum(self, name: str) -> float:
        return self.momentum.get(name, 0.0)

    def get_price(self, name: str) -> float:
        return self.prices.get(name, 0.0)

    def is_ihsg_bullish(self) -> bool:
        if 'IHSG_Close' in self.data and 'IHSG_MA200' in self.data:
            return self.data['IHSG_Close'] > self.data['IHSG_MA200']
        return True

# =============================================================================
# 6A. MARKET REGIME DETECTOR (DENGAN CONFIDENCE SCORE DAN NEWS GLOBAL)
# =============================================================================

class MarketRegimeDetector:
    def __init__(self, lookback_days: int = 252, global_news_analyzer=None):
        self.lookback_days = lookback_days
        self.current_regime = "UNKNOWN"
        self.confidence = 0.0
        self.regime_params = {}
        self.regime_history = []
        self.global_news_analyzer = global_news_analyzer
        self.news_sentiment = "netral"  # default

    def detect_regime(self, ihsg_data: pd.DataFrame) -> Tuple[str, float]:
        if ihsg_data is None or len(ihsg_data) < 200:
            return "UNKNOWN", 0.0

        df = ihsg_data.shift(1).copy()
        close = df['Close'].dropna()

        if len(close) < 200:
            return "UNKNOWN", 0.0

        returns = close.pct_change().dropna() * 100

        mom_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
        mom_60d = (close.iloc[-1] / close.iloc[-61] - 1) * 100 if len(close) > 61 else 0

        vol_20d = returns.tail(20).std() * np.sqrt(252)
        vol_60d = returns.tail(60).std() * np.sqrt(252)
        hist_vol = returns.tail(252).std() * np.sqrt(252) if len(returns) > 252 else 20

        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        price_vs_ma50 = (close.iloc[-1] / ma50.iloc[-1] - 1) * 100 if not pd.isna(ma50.iloc[-1]) else 0
        price_vs_ma200 = (close.iloc[-1] / ma200.iloc[-1] - 1) * 100 if not pd.isna(ma200.iloc[-1]) else 0

        high = df['High']
        low = df['Low']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        up_move = high - high.shift()
        down_move = low.shift() - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)

        trend_strength = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 25

        if vol_20d > hist_vol * 1.5:
            regime = "HIGH_VOL"
            vol_ratio = vol_20d / hist_vol
            confidence = min(100, vol_ratio * 50)

        elif mom_20d > 2 and mom_60d > 5 and price_vs_ma200 > 0:
            regime = "BULL"
            mom_strength = min(100, (abs(mom_20d) + abs(mom_60d)) * 5)
            ma_strength = min(100, price_vs_ma200 * 10)
            trend_conf = min(100, trend_strength * 2)
            confidence = (mom_strength * 0.4 + ma_strength * 0.3 + trend_conf * 0.3)

        elif mom_20d < -2 and mom_60d < -5 and price_vs_ma200 < 0:
            regime = "BEAR"
            mom_strength = min(100, (abs(mom_20d) + abs(mom_60d)) * 5)
            ma_strength = min(100, abs(price_vs_ma200) * 10)
            trend_conf = min(100, trend_strength * 2)
            confidence = (mom_strength * 0.4 + ma_strength * 0.3 + trend_conf * 0.3)

        elif abs(mom_20d) < 3 and abs(mom_60d) < 5:
            regime = "SIDEWAYS"
            mom_weakness = 100 - min(100, (abs(mom_20d) + abs(mom_60d)) * 10)
            vol_weakness = 100 - min(100, (vol_20d / hist_vol) * 50)
            trend_weakness = 100 - min(100, trend_strength * 2)
            confidence = (mom_weakness * 0.4 + vol_weakness * 0.3 + trend_weakness * 0.3)

        else:
            regime = "NEUTRAL"
            confidence = 30.0

        # Ambil sentimen global jika tersedia
        if self.global_news_analyzer:
            news = self.global_news_analyzer.get_sentiment('IHSG')
            self.news_sentiment = news['label']
            if news['label'] == 'positif':
                confidence = min(100, confidence * 1.1)   # naik 10%
            elif news['label'] == 'negatif':
                confidence = max(0, confidence * 0.9)     # turun 10%

        confidence = round(min(confidence, 100.0), 1)

        self.current_regime = regime
        self.confidence = confidence
        self.regime_history.append({
            'date': datetime.now(),
            'regime': regime,
            'confidence': confidence,
            'mom_20d': mom_20d,
            'mom_60d': mom_60d,
            'vol_20d': vol_20d,
            'price_vs_ma200': price_vs_ma200,
            'trend_strength': trend_strength,
            'news_sentiment': self.news_sentiment
        })

        if len(self.regime_history) > 30:
            self.regime_history = self.regime_history[-30:]

        return regime, confidence

    def get_regime_parameters(self, base_risk: float = 3.0) -> Dict:
        confidence_factor = self.confidence / 100.0

        if self.current_regime == "BULL":
            base_multiplier = 1.17
            risk_multiplier = 1.0 + (base_multiplier - 1.0) * confidence_factor
            return {
                'risk_multiplier': round(risk_multiplier, 2),
                'entry_tolerance_multiplier': 0.8,
                'min_rr_multiplier': 1.2,
                'max_positions': 5,
                'description': f'BULL MARKET - Aggressive (Confidence: {self.confidence}%)'
            }
        elif self.current_regime == "BEAR":
            base_multiplier = 0.67
            risk_multiplier = 1.0 - (1.0 - base_multiplier) * confidence_factor
            return {
                'risk_multiplier': round(risk_multiplier, 2),
                'entry_tolerance_multiplier': 1.5,
                'min_rr_multiplier': 0.8,
                'max_positions': 3,
                'description': f'BEAR MARKET - Defensive (Confidence: {self.confidence}%)'
            }
        elif self.current_regime == "HIGH_VOL":
            base_multiplier = 0.5
            risk_multiplier = 1.0 - (1.0 - base_multiplier) * confidence_factor
            return {
                'risk_multiplier': round(risk_multiplier, 2),
                'entry_tolerance_multiplier': 2.0,
                'min_rr_multiplier': 1.5,
                'max_positions': 2,
                'description': f'HIGH VOLATILITY - Very Defensive (Confidence: {self.confidence}%)'
            }
        elif self.current_regime == "SIDEWAYS":
            base_multiplier = 0.83
            risk_multiplier = 1.0 - (1.0 - base_multiplier) * confidence_factor
            return {
                'risk_multiplier': round(risk_multiplier, 2),
                'entry_tolerance_multiplier': 1.2,
                'min_rr_multiplier': 1.0,
                'max_positions': 4,
                'description': f'SIDEWAYS - Neutral (Confidence: {self.confidence}%)'
            }
        else:
            return {
                'risk_multiplier': 1.0,
                'entry_tolerance_multiplier': 1.0,
                'min_rr_multiplier': 1.0,
                'max_positions': 4,
                'description': f'NEUTRAL - Standard (Confidence: {self.confidence}%)'
            }

    def print_regime_report(self) -> None:
        print("\n" + "="*70)
        print("📊 MARKET REGIME DETECTION (WITH CONFIDENCE & GLOBAL NEWS)")
        print("="*70)

        params = self.get_regime_parameters()

        confidence_bar = '█' * int(self.confidence / 5) + '░' * (20 - int(self.confidence / 5))

        print(f"Current Regime: {self.current_regime}")
        print(f"Confidence: {self.confidence}% [{confidence_bar}]")
        if self.news_sentiment:
            print(f"📰 Sentimen IHSG: {self.news_sentiment}")
        print(f"Description: {params['description']}")
        print(f"\nAdjusted Parameters:")
        print(f"  - Risk Multiplier: {params['risk_multiplier']:.2f}x")
        print(f"  - Entry Tolerance: {params['entry_tolerance_multiplier']:.1f}x")
        print(f"  - Min R/R Multiplier: {params['min_rr_multiplier']:.1f}x")
        print(f"  - Max Positions: {params['max_positions']}")

        if len(self.regime_history) > 1:
            print(f"\nRegime History (last {len(self.regime_history)}):")
            print(f"{'Date':12} | {'Regime':10} | {'Conf':6} | {'News':7}")
            print("-" * 45)
            for h in self.regime_history[-5:]:
                print(f"{h['date'].strftime('%Y-%m-%d')} | {h['regime']:10} | {h['confidence']:5.1f}% | {h.get('news_sentiment','netral'):7}")

# =============================================================================
# 7. DATA WAREHOUSE (DENGAN ERROR HANDLING AMAN) - DIPERBAIKI (FOLDER DIVIDEN TERPISAH) + FUNDAMENTAL
# =============================================================================

class DataWarehouse:
    def __init__(self, warehouse_dir: str = 'data_warehouse', min_days: int = 400):
        self.warehouse_dir = warehouse_dir
        self.min_days = min_days
        os.makedirs(warehouse_dir, exist_ok=True)
        self.dividend_dir = f'{warehouse_dir}/dividends'
        os.makedirs(self.dividend_dir, exist_ok=True)
        self.fundamental_dir = f'{warehouse_dir}/fundamental'
        os.makedirs(self.fundamental_dir, exist_ok=True)

        self.stats = {
            'total_symbols': 0,
            'downloaded': 0,
            'failed': 0,
            'cached': 0,
            'filtered_min_days': 0,
            'corrupt_files': 0
        }

    def download_complete_history(
        self,
        symbols: List[str],
        start_date: str = '2018-01-01',
        end_date: str = '2026-12-31',
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        print("\n" + "="*80)
        print("🗄️  DATA WAREHOUSE - DOWNLOAD HISTORIS LENGKAP")
        print("="*80)
        print(f"Periode: {start_date} hingga {end_date}")
        print(f"Total saham: {len(symbols)}")
        print(f"Minimal hari: {self.min_days} (saham dengan data kurang akan difilter)")

        results = {}
        self.stats['total_symbols'] = len(symbols)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        for i, symbol in enumerate(symbols):
            cache_file = f"{self.warehouse_dir}/{symbol}_full.parquet"

            if os.path.exists(cache_file) and not force_refresh:
                try:
                    df = pd.read_parquet(cache_file)

                    if len(df) >= self.min_days and df.index[0] <= start_dt and df.index[-1] >= end_dt:
                        results[symbol] = df
                        self.stats['cached'] += 1
                    else:
                        self.stats['filtered_min_days'] += 1

                    if (i + 1) % 50 == 0:
                        print(f"   Progress: {i+1}/{len(symbols)} - {len(results)} dimuat (cache)")
                    continue
                except Exception as e:
                    logger.error(f"Corrupt cache file for {symbol}: {str(e)}")
                    self.stats['corrupt_files'] += 1

            try:
                ticker = f"{symbol}.JK"
                print(f"   Downloading {symbol} ({i+1}/{len(symbols)})...", end=" ")

                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    timeout=30
                )

                time.sleep(0.5)

                if df.empty:
                    print("❌ GAGAL (data kosong)")
                    self.stats['failed'] += 1
                    logger.warning(f"Empty data for {symbol}")
                    continue

                df = normalize_columns(df)

                if len(df) < self.min_days:
                    print(f"❌ GAGAL (hanya {len(df)} hari, minimal {self.min_days})")
                    self.stats['filtered_min_days'] += 1
                    continue

                df.to_parquet(cache_file)
                results[symbol] = df
                self.stats['downloaded'] += 1
                print(f"✅ {len(df)} hari")

            except Exception as e:
                print(f"❌ ERROR: {str(e)[:50]}")
                self.stats['failed'] += 1
                logger.error(f"Error downloading {symbol}: {str(e)}")
                time.sleep(1)

        print("\n" + "="*80)
        print("📊 RINGKASAN DATA WAREHOUSE")
        print("="*80)
        print(f"Total saham: {self.stats['total_symbols']}")
        print(f"Berhasil dimuat: {len(results)}")
        print(f"  - Dari cache: {self.stats['cached']}")
        print(f"  - Download baru: {self.stats['downloaded']}")
        print(f"  - Difilter (< {self.min_days} hari): {self.stats['filtered_min_days']}")
        print(f"  - File corrupt: {self.stats['corrupt_files']}")
        print(f"Gagal: {self.stats['failed']}")
        print("="*80)

        return results

    def get_all_valid_symbols(self) -> List[str]:
        symbols = []
        for f in os.listdir(self.warehouse_dir):
            if f.endswith('_full.parquet'):
                symbol = f.replace('_full.parquet', '')
                try:
                    df = pd.read_parquet(os.path.join(self.warehouse_dir, f))
                    if len(df) >= self.min_days:
                        symbols.append(symbol)
                except Exception as e:
                    logger.error(f"Corrupt file in get_all_valid_symbols: {f} - {str(e)}")
                    continue
        return symbols

    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        cache_file = f"{self.warehouse_dir}/{symbol}_full.parquet"
        if not os.path.exists(cache_file):
            return None
        try:
            df = pd.read_parquet(cache_file)
            if len(df) >= self.min_days:
                return df
            return None
        except Exception as e:
            logger.error(f"Error reading {symbol}: {str(e)}")
            return None

    def get_all_data(self, max_symbols: int = None) -> Dict[str, pd.DataFrame]:
        results = {}
        symbols = self.get_all_valid_symbols()
        if max_symbols:
            symbols = symbols[:max_symbols]
        for symbol in symbols:
            df = self.get_data(symbol)
            if df is not None:
                results[symbol] = df
        return results

    def print_warehouse_stats(self) -> None:
        symbols = self.get_all_valid_symbols()
        print("\n" + "="*80)
        print("🗄️  DATA WAREHOUSE STATISTICS")
        print("="*80)
        print(f"Total saham valid (≥{self.min_days} hari): {len(symbols)}")
        if symbols:
            sample = symbols[0]
            df = pd.read_parquet(f"{self.warehouse_dir}/{sample}_full.parquet")
            print(f"Rentang tanggal: {df.index[0].date()} hingga {df.index[-1].date()}")
            print(f"Rata-rata hari per saham: {len(df)} hari")
        print("="*80)

    def download_dividend_history(self, symbols: List[str], years_back: int = 10) -> Dict[str, pd.DataFrame]:
        print("\n" + "="*80)
        print("💰 DOWNLOADING DIVIDEND HISTORY (FOLDER TERPISAH)")
        print("="*80)

        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years_back)
        start_timestamp = pd.Timestamp(start_date)

        for i, symbol in enumerate(symbols):
            cache_file = f"{self.dividend_dir}/{symbol}_dividends.parquet"
            if os.path.exists(cache_file):
                try:
                    df = pd.read_parquet(cache_file)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    results[symbol] = df
                    if (i+1) % 50 == 0:
                        print(f"   Progress: {i+1}/{len(symbols)} - {len(results)} from cache")
                    continue
                except Exception as e:
                    logger.error(f"Error reading cache for {symbol}: {str(e)}")
            try:
                ticker = yf.Ticker(f"{symbol}.JK")
                dividends = ticker.dividends
                if dividends is not None and len(dividends) > 0:
                    df = pd.DataFrame(dividends)
                    df.columns = ['Dividend']
                    df.index.name = 'Date'
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df_filtered = df[df.index >= start_timestamp]
                    if len(df_filtered) > 0:
                        df_filtered.to_parquet(cache_file)
                        results[symbol] = df_filtered
                        print(f"   ✅ {symbol}: {len(df_filtered)} dividend records")
                    else:
                        print(f"   ⚠️ {symbol}: no dividends in period")
                else:
                    print(f"   ⚠️ {symbol}: no dividend data")
                time.sleep(0.2)
            except Exception as e:
                print(f"   ❌ {symbol}: {str(e)[:50]}")
                logger.error(f"Error downloading dividend for {symbol}: {str(e)}")
        print("\n" + "="*80)
        print(f"✅ Dividend download complete: {len(results)} symbols with dividends")
        print("="*80)
        return results

    def get_dividends(self, symbol: str) -> Optional[pd.DataFrame]:
        cache_file = f"{self.dividend_dir}/{symbol}_dividends.parquet"
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
            except Exception as e:
                logger.error(f"Error reading dividend for {symbol}: {str(e)}")
                return None
        return None

    # -------------------------------------------------------------------------
    # METHOD UNTUK DATA FUNDAMENTAL
    # -------------------------------------------------------------------------
    def download_fundamental_history(self, symbols: List[str], max_age_days: int = 30) -> Dict[str, Dict]:
        """
        Mengunduh data fundamental (PER, PBV, ROE, dll) untuk daftar saham.
        Disimpan di folder fundamental/. Jika data sudah ada dan masih fresh (< max_age_days), tidak diunduh ulang.
        """
        print("\n" + "="*80)
        print("📊 DOWNLOADING FUNDAMENTAL DATA")
        print("="*80)
        print(f"Total saham: {len(symbols)}")
        print(f"Data akan disimpan di: {self.warehouse_dir}/fundamental/")
        print(f"Masa berlaku data: {max_age_days} hari")

        results = {}
        now = datetime.now()

        for i, symbol in enumerate(symbols):
            cache_file = f"{self.fundamental_dir}/{symbol}_fundamental.parquet"
            need_download = True

            # Cek apakah file sudah ada dan masih fresh
            if os.path.exists(cache_file):
                try:
                    df = pd.read_parquet(cache_file)
                    if 'timestamp' in df.columns:
                        timestamp = pd.to_datetime(df['timestamp'].iloc[0])
                        age_days = (now - timestamp).days
                        if age_days < max_age_days:
                            # Data masih fresh, gunakan
                            results[symbol] = df.to_dict(orient='records')[0]
                            need_download = False
                            if (i+1) % 50 == 0:
                                print(f"   Progress: {i+1}/{len(symbols)} - {len(results)} from cache")
                except Exception as e:
                    logger.error(f"Error reading fundamental cache for {symbol}: {e}")

            if need_download:
                try:
                    ticker = yf.Ticker(f"{symbol}.JK")
                    info = ticker.info

                    # Ambil data fundamental yang diinginkan
                    fund_data = {
                        'symbol': symbol,
                        'timestamp': now.isoformat(),
                        'per': info.get('trailingPE'),
                        'forward_per': info.get('forwardPE'),
                        'pbv': info.get('priceToBook'),
                        'roe': info.get('returnOnEquity'),
                        'eps': info.get('trailingEps'),
                        'profit_margin': info.get('profitMargins'),
                        'debt_to_equity': info.get('debtToEquity'),
                        'market_cap': info.get('marketCap'),
                        'sector': info.get('sector'),
                        'industry': info.get('industry')
                    }

                    # Simpan sebagai DataFrame (1 baris) agar mudah dibaca kembali
                    df = pd.DataFrame([fund_data])
                    df.to_parquet(cache_file)

                    results[symbol] = fund_data
                    print(f"   ✅ {symbol}: PER={fund_data['per']}, PBV={fund_data['pbv']}, ROE={fund_data['roe']}")
                except Exception as e:
                    logger.error(f"Error downloading fundamental for {symbol}: {e}")
                    print(f"   ❌ {symbol}: {str(e)[:50]}")
                time.sleep(0.2)  # Hindari rate limit

        print("\n" + "="*80)
        print(f"✅ Fundamental download complete: {len(results)} symbols")
        print("="*80)
        return results

    def get_fundamental(self, symbol: str, max_age_days: int = 30) -> Optional[Dict]:
        """
        Mengambil data fundamental dari cache. Jika tidak ada atau sudah expired, return None.
        """
        cache_file = f"{self.fundamental_dir}/{symbol}_fundamental.parquet"
        if not os.path.exists(cache_file):
            return None
        try:
            df = pd.read_parquet(cache_file)
            if df.empty:
                return None
            row = df.iloc[0].to_dict()
            # Cek umur data
            if 'timestamp' in row:
                timestamp = pd.to_datetime(row['timestamp'])
                age_days = (datetime.now() - timestamp).days
                if age_days < max_age_days:
                    return row
            return None
        except Exception as e:
            logger.error(f"Error reading fundamental for {symbol}: {e}")
            return None

# =============================================================================
# 8. UTILITY FUNCTIONS - TIDAK BERUBAH + FIXED FUNCTIONS
# =============================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_spread_pct(df: pd.DataFrame) -> float:
    try:
        spread = ((df['High'] - df['Low']) / df['Close']).tail(10).mean() * 100
        return float(spread)
    except Exception:
        return 999.0

def calculate_return(series: pd.Series, period: int = 5) -> float:
    if len(series) < period + 1:
        return 0.0
    return (series.iloc[-1] / series.iloc[-period-1] - 1) * 100

# =============================================================================
# 8A. UTILITY FUNCTIONS - FIXED VERSION DENGAN SAFE STOP LOSS
# =============================================================================

def calculate_safe_stop_loss(
    price: float,
    atr: float,
    multiplier: float,
    fraction: int = 5,
    min_distance_pct: float = 1.5,
    engine_type: str = 'swing'
) -> float:
    if price <= 0:
        return 1

    min_distance_map = {
        'swing': max(atr, price * 0.015),
        'gorengan': max(atr, price * 0.02),
        'investasi': max(atr * 2, price * 0.01)
    }
    min_distance = min_distance_map.get(engine_type, max(atr, price * 0.015))

    raw_sl = price - max(atr * multiplier, min_distance)

    if price - raw_sl < min_distance:
        raw_sl = price - min_distance

    if fraction > 0:
        sl = math.floor(raw_sl / fraction) * fraction
    else:
        sl = raw_sl

    if sl >= price:
        forced_sl = price - min_distance
        sl = math.floor(forced_sl / fraction) * fraction if fraction > 0 else forced_sl

    sl = max(sl, 1)
    return sl

def calculate_safe_take_profit(
    price: float,
    atr: float,
    multiplier: float,
    fraction: int = 5,
    engine_type: str = 'swing'
) -> float:
    if price <= 0 or atr <= 0:
        return price * 1.05
    raw_tp = price + (atr * multiplier)
    if fraction > 0:
        tp = math.ceil(raw_tp / fraction) * fraction
    else:
        tp = raw_tp
    return tp

def validate_risk_reward(
    price: float,
    sl: float,
    tp: float,
    min_rr: float = 1.0
) -> Tuple[bool, float, float, float]:
    if sl >= price:
        return False, 0, 0, 0
    if tp <= price:
        return False, 0, 0, 0
    risk = price - sl
    reward = tp - price
    if risk <= 0:
        return False, 0, 0, 0
    rr = reward / risk
    return rr >= min_rr, risk, reward, rr

# =============================================================================
# 9. WALK-FORWARD COLLECTOR (BARU - TANPA DEFAULT VALUES)
# =============================================================================

class WalkForwardCollector:
    def __init__(self, warehouse, collection_dir: str = 'backtest_collection'):
        self.warehouse = warehouse
        self.collection_dir = collection_dir
        os.makedirs(collection_dir, exist_ok=True)
        self.engine_files = {
            'swing': f'{collection_dir}/swing_trades.json',
            'gorengan': f'{collection_dir}/gorengan_trades.json',
            'investasi': f'{collection_dir}/investasi_trades.json'
        }
        self.trades_cache = defaultdict(list)
        self._load_all_trades()

    def _load_all_trades(self):
        for engine_type, file_path in self.engine_files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        for symbol, trades in data.items():
                            key = f"{engine_type}_{symbol}"
                            self.trades_cache[key] = trades
                    print(f"✅ Loaded {engine_type} trades: {sum(len(v) for v in data.values())} total")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

    def _save_trades(self, engine_type: str, symbol: str, trades: List[float]):
        file_path = self.engine_files[engine_type]
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data[symbol] = trades[-100:]
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def collect_for_engine(self, engine, engine_type: str, symbols: List[str],
                          years_back: int = 3, sample_frequency: int = 10):
        print(f"\n📊 WALK-FORWARD COLLECTION untuk {engine_type.upper()}")
        print(f"   Periode: {years_back} tahun ke belakang")
        print(f"   Sample frequency: setiap {sample_frequency} hari")
        print(f"   Total saham: {len(symbols)}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years_back)

        total_collected = 0
        symbols_with_data = 0

        for idx, symbol in enumerate(symbols):
            if idx % 50 == 0:
                print(f"   Progress: {idx}/{len(symbols)} - Collected: {total_collected}")

            df = self.warehouse.get_data(symbol)
            if df is None or len(df) < 500:
                continue

            df_period = df[df.index >= pd.Timestamp(start_date)].copy()
            if len(df_period) < 200:
                continue

            signals_found = []

            for i in range(200, len(df_period) - 20, sample_frequency):
                df_slice = df_period.iloc[:i].copy()
                signal = engine.get_signal(symbol, df_slice)
                if signal:
                    future_idx = min(i + 20, len(df_period) - 1)
                    entry_price = float(df_period.iloc[i-1]['Close'])
                    exit_price = float(df_period.iloc[future_idx]['Close'])
                    return_pct = (exit_price / entry_price - 1) * 100
                    signals_found.append(round(return_pct, 2))

            if len(signals_found) >= 10:
                key = f"{engine_type}_{symbol}"
                existing = self.trades_cache.get(key, [])
                combined = existing + signals_found
                self.trades_cache[key] = combined[-200:]
                self._save_trades(engine_type, symbol, self.trades_cache[key])
                total_collected += len(signals_found)
                symbols_with_data += 1

        print(f"\n✅ Collection complete!")
        print(f"   Total trades collected: {total_collected}")
        print(f"   Symbols with data: {symbols_with_data}")
        return {
            'total_trades': total_collected,
            'symbols_with_data': symbols_with_data
        }

    def get_trades_for_symbol(self, engine_type: str, symbol: str) -> List[float]:
        key = f"{engine_type}_{symbol}"
        return self.trades_cache.get(key, [])

    def get_backtest_metrics(self, engine_type: str, symbol: str) -> Dict:
        trades = self.get_trades_for_symbol(engine_type, symbol)
        if len(trades) < 20:
            return {
                'has_data': False,
                'total_trades': len(trades),
                'trades_needed': 20 - len(trades),
                'message': f'Butuh {20 - len(trades)} trade lagi',
                'display': f"⏳ ({len(trades)}/20)"
            }
        trades_array = np.array(trades)
        winning_trades = trades_array[trades_array > 0]
        losing_trades = trades_array[trades_array <= 0]
        win_rate = len(winning_trades) / len(trades_array) * 100
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
        total_win = winning_trades.sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_win / total_loss if total_loss > 0 else 0
        if trades_array.std() > 0:
            sharpe = (trades_array.mean() / trades_array.std()) * np.sqrt(20)
        else:
            sharpe = 0
        return {
            'has_data': True,
            'total_trades': len(trades),
            'win_rate': round(win_rate, 1),
            'avg_return': round(trades_array.mean(), 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe': round(sharpe, 2),
            'max_win': round(trades_array.max(), 2),
            'max_loss': round(trades_array.min(), 2),
            'display': f"{round(win_rate,1)}% ({len(trades)} trades)"
        }

    def print_summary(self):
        print("\n" + "="*80)
        print("📊 WALK-FORWARD COLLECTION SUMMARY")
        print("="*80)
        total_all = 0
        for engine_type in self.engine_files.keys():
            engine_trades = 0
            engine_symbols = 0
            for key, trades in self.trades_cache.items():
                if key.startswith(engine_type):
                    engine_trades += len(trades)
                    engine_symbols += 1
            print(f"\n{engine_type.upper()}:")
            print(f"   Symbols: {engine_symbols}")
            print(f"   Total trades: {engine_trades}")
            total_all += engine_trades
        print("\n" + "-"*80)
        print(f"GRAND TOTAL: {total_all} historical trades collected")
        print("="*80)

# =============================================================================
# 10. BACKTEST METRICS (TERINTEGRASI DENGAN WALK-FORWARD COLLECTOR)
# =============================================================================

class BacktestMetrics:
    def __init__(self, walkforward_collector=None, engine_type=None, symbol=None):
        self.walkforward_collector = walkforward_collector
        self.engine_type = engine_type
        self.symbol = symbol
        self.live_trades = []

    def add_live_trade(self, return_pct: float) -> None:
        self.live_trades.append(return_pct)
        if len(self.live_trades) > 100:
            self.live_trades = self.live_trades[-100:]

    def calculate_metrics(self) -> Dict:
        historical_trades = []
        if self.walkforward_collector and self.engine_type and self.symbol:
            historical_trades = self.walkforward_collector.get_trades_for_symbol(
                self.engine_type, self.symbol
            )
        all_trades = historical_trades + self.live_trades
        if len(all_trades) < 20:
            return {
                'has_data': False,
                'total_trades': len(all_trades),
                'historical_trades': len(historical_trades),
                'live_trades': len(self.live_trades),
                'trades_needed': 20 - len(all_trades),
                'message': f'Butuh {20 - len(all_trades)} trade lagi',
                'display': f"⏳ ({len(all_trades)}/20)"
            }
        trades_array = np.array(all_trades)
        winning_trades = trades_array[trades_array > 0]
        win_rate = len(winning_trades) / len(trades_array) * 100
        return {
            'has_data': True,
            'total_trades': len(all_trades),
            'historical_trades': len(historical_trades),
            'live_trades': len(self.live_trades),
            'win_rate': round(win_rate, 1),
            'display': f"{round(win_rate,1)}% ({len(all_trades)} trades)"
        }

# =============================================================================
# 11. RISK MANAGER (AGGRESSIVE - 3% BASE RISK) - DIPERBAIKI
# =============================================================================

class RiskManager:
    def __init__(
        self,
        modal: float,
        risk_per_trade_pct: float = 3.0,
        max_risk_portfolio_pct: float = 15.0,
        max_lot_per_position: int = 10,
        engine_type: str = 'swing'
    ):
        self.modal = modal
        self.base_risk_per_trade_pct = risk_per_trade_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_risk_portfolio_pct = max_risk_portfolio_pct
        self.max_lot_per_position = max_lot_per_position
        self.engine_type = engine_type
        self.max_modal_per_position_pct = 40.0

        self._validate_modal_for_engine()

        self.risk_per_trade_rp = modal * (risk_per_trade_pct / 100)
        self.max_risk_portfolio_rp = modal * (max_risk_portfolio_pct / 100)
        self.max_modal_per_position_rp = modal * (self.max_modal_per_position_pct / 100)

        self.current_positions = []
        self.current_risk_rp = 0.0
        self.current_modal_used_rp = 0.0

        self.trade_history = []
        self.win_rate = 50.0
        self.avg_win = 0
        self.avg_loss = 0
        self.regime_params = {}
        self.backtest_metrics = {}

    def _validate_modal_for_engine(self) -> None:
        min_modal_map = {
            'swing': 40000,
            'gorengan': 10000,
            'investasi': 100000
        }
        max_modal_map = {
            'swing': 5000000,
            'gorengan': 500000,
            'investasi': 1000000000
        }
        min_modal = min_modal_map.get(self.engine_type, 10000)
        max_modal = max_modal_map.get(self.engine_type, 5000000)
        if self.modal < min_modal:
            raise ValueError(f"Modal minimal untuk {self.engine_type} adalah Rp {min_modal:,}")
        if self.modal > max_modal:
            raise ValueError(f"Modal maksimal untuk {self.engine_type} adalah Rp {max_modal:,}")

    def calculate_atr_in_rupiah(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            high = df['High'].shift(1)
            low = df['Low'].shift(1)
            close = df['Close'].shift(1)

            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - close).abs()

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=period).mean()

            valid_atr = atr.dropna()
            if len(valid_atr) == 0:
                current_price = df['Close'].iloc[-1]
                return current_price * 0.02

            atr_value = float(valid_atr.iloc[-1])
            current_price = df['Close'].iloc[-1]

            min_atr_pct = {
                'swing': 0.015,
                'gorengan': 0.02,
                'investasi': 0.01
            }.get(self.engine_type, 0.015)

            min_atr = current_price * min_atr_pct
            atr_value = max(atr_value, min_atr)
            return atr_value

        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return df['Close'].iloc[-1] * 0.02

    def calculate_kelly_lot(self, close: float, atr: float, symbol: str = None) -> tuple:
        if atr <= 0 or close <= 0:
            return None, None, None

        self._update_trade_stats()

        if symbol and symbol in self.backtest_metrics:
            bt = self.backtest_metrics[symbol].calculate_metrics()
            if bt.get('has_data', False):
                win_rate = 0.7 * bt['win_rate'] + 0.3 * self.win_rate
                avg_win = bt.get('avg_return', self.avg_win)
                avg_loss = abs(bt.get('avg_return', self.avg_loss)) if bt.get('avg_return', 0) < 0 else self.avg_loss
            else:
                win_rate = self.win_rate
                avg_win = self.avg_win
                avg_loss = self.avg_loss
        else:
            win_rate = self.win_rate
            avg_win = self.avg_win
            avg_loss = self.avg_loss

        if avg_loss > 0 and win_rate > 0:
            b = avg_win / avg_loss if avg_win > 0 else 1
            p = win_rate / 100
            q = 1 - p
            if b > 0:
                kelly = (p * b - q) / b
                kelly = max(0, min(kelly, 0.25))
            else:
                kelly = 0.01
        else:
            kelly = 0.01

        adjusted_risk_pct = self.risk_per_trade_pct * (1 + kelly * 4)
        adjusted_risk_pct = min(adjusted_risk_pct, 6.0)

        risk_per_trade_rp = self.modal * (adjusted_risk_pct / 100)
        risk_per_lot = atr * 100

        raw_lot = risk_per_trade_rp / risk_per_lot
        lot = int(raw_lot)

        lot = min(lot, self.max_lot_per_position)
        max_lot_by_modal = int(self.modal / (close * 100))
        lot = min(lot, max_lot_by_modal)

        if lot < 1:
            return None, None, None

        cost = lot * 100 * close

        if cost > self.max_modal_per_position_rp:
            lot = int(self.max_modal_per_position_rp / (close * 100))
            if lot < 1:
                return None, None, None
            cost = lot * 100 * close
            risk_amount = lot * risk_per_lot
        else:
            risk_amount = lot * risk_per_lot

        return lot, cost, risk_amount

    def calculate_lot(
        self,
        close: float,
        atr: float,
        symbol: str = None,
        use_kelly: bool = True
    ):
        if atr <= 0:
            return None, None, None
        if use_kelly and (len(self.trade_history) >= 10 or (symbol and symbol in self.backtest_metrics)):
            return self.calculate_kelly_lot(close, atr, symbol)
        if atr <= 0 or close <= 0:
            return None, None, None

        risk_per_lot = atr * 100
        raw_lot = self.risk_per_trade_rp / risk_per_lot
        lot = int(raw_lot)

        lot = min(lot, self.max_lot_per_position)
        max_lot_by_modal = int(self.modal / (close * 100))
        lot = min(lot, max_lot_by_modal)

        if lot < 1:
            return None, None, None

        cost = lot * 100 * close

        if cost > self.max_modal_per_position_rp:
            lot = int(self.max_modal_per_position_rp / (close * 100))
            if lot < 1:
                return None, None, None
            cost = lot * 100 * close
            risk_amount = lot * risk_per_lot
        else:
            risk_amount = lot * risk_per_lot

        return lot, cost, risk_amount

    def _update_trade_stats(self):
        if len(self.trade_history) < 5:
            return
        profits = [t['return_pct'] for t in self.trade_history if t['return_pct'] > 0]
        losses = [abs(t['return_pct']) for t in self.trade_history if t['return_pct'] <= 0]
        self.win_rate = (len(profits) / len(self.trade_history)) * 100 if self.trade_history else 50
        self.avg_win = np.mean(profits) if profits else 0
        self.avg_loss = np.mean(losses) if losses else 0

    def add_trade_result(self, trade_result: Dict):
        self.trade_history.append(trade_result)
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

    def add_backtest_metrics(self, symbol: str, metrics) -> None:
        self.backtest_metrics[symbol] = metrics

    def adjust_for_regime(self, regime_params: Dict):
        self.regime_params = regime_params
        multiplier = regime_params.get('risk_multiplier', 1.0)
        self.risk_per_trade_pct = self.base_risk_per_trade_pct * multiplier
        self.risk_per_trade_rp = self.modal * (self.risk_per_trade_pct / 100)

    def can_add_position(self, risk_amount: float, cost: float = None) -> Tuple[bool, str]:
        if self.current_risk_rp + risk_amount > self.max_risk_portfolio_rp:
            return False, f"Portfolio risk limit: Rp {self.max_risk_portfolio_rp:,.0f}"
        if cost and self.current_modal_used_rp + cost > self.modal * 0.9:
            return False, f"Modal tidak cukup: Rp {self.modal - self.current_modal_used_rp:,.0f} sisa"
        if self.regime_params and 'max_positions' in self.regime_params:
            if len(self.current_positions) >= self.regime_params['max_positions']:
                return False, f"Max positions ({self.regime_params['max_positions']}) reached"
        return True, "OK"

    def add_position(self, symbol: str, lot: int, entry_price: float, risk_amount: float, cost: float) -> None:
        position = {
            'symbol': symbol,
            'lot': lot,
            'entry_price': entry_price,
            'risk_amount': risk_amount,
            'cost': cost,
            'entry_date': datetime.now()
        }
        self.current_positions.append(position)
        self.current_risk_rp += risk_amount
        self.current_modal_used_rp += cost

    def remove_position(self, symbol: str) -> None:
        for i, pos in enumerate(self.current_positions):
            if pos['symbol'] == symbol:
                self.current_risk_rp -= pos['risk_amount']
                self.current_modal_used_rp -= pos['cost']
                self.current_positions.pop(i)
                break

    def get_portfolio_risk_pct(self) -> float:
        return (self.current_risk_rp / self.modal) * 100 if self.modal > 0 else 0

    def get_portfolio_modal_used_pct(self) -> float:
        return (self.current_modal_used_rp / self.modal) * 100 if self.modal > 0 else 0

    def reset(self) -> None:
        self.current_positions = []
        self.current_risk_rp = 0.0
        self.current_modal_used_rp = 0.0

# =============================================================================
# 12. PORTFOLIO RISK CALCULATOR (DENGAN CORRELATION CHECK & STRESS TESTING) - DIPERBAIKI
# =============================================================================

class PortfolioRiskCalculator:
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.correlation_cache = {}
        self.stress_scenarios = {
            'covid_2020': {'shock': -0.30, 'correlation_multiplier': 1.5},
            'taper_tantrum_2013': {'shock': -0.15, 'correlation_multiplier': 1.3},
            'normal': {'shock': 0, 'correlation_multiplier': 1.0}
        }

    def calculate_correlation_matrix(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
        use_exp_weight: bool = True
    ) -> pd.DataFrame:
        returns_dict = {}
        for symbol in symbols:
            if symbol in price_data:
                df = price_data[symbol]
                if len(df) >= self.lookback_days:
                    returns = df['Close'].pct_change().shift(1).tail(self.lookback_days)
                    returns_dict[symbol] = returns
        if not returns_dict:
            return pd.DataFrame()
        returns_df = pd.DataFrame(returns_dict)
        if use_exp_weight and len(returns_df) > 30:
            decay_factor = 0.94
            weights = decay_factor ** np.arange(len(returns_df))[::-1]
            weights = weights / weights.sum()
            mean_returns = returns_df.mean()
            demeaned = returns_df - mean_returns
            cov_matrix = np.dot((demeaned.T * weights), demeaned) / (1 - decay_factor)
            std_dev = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
            corr_df = pd.DataFrame(corr_matrix, index=returns_df.columns, columns=returns_df.columns)
        else:
            corr_df = returns_df.corr()
        return corr_df.fillna(0)

    def get_correlation(self, symbol1: str, symbol2: str, price_data: Dict[str, pd.DataFrame]) -> float:
        cache_key = f"{symbol1}_{symbol2}"
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]
        if symbol1 not in price_data or symbol2 not in price_data:
            return 0.0
        df1 = price_data[symbol1]
        df2 = price_data[symbol2]
        if len(df1) < self.lookback_days or len(df2) < self.lookback_days:
            return 0.0
        returns1 = df1['Close'].pct_change().shift(1).tail(self.lookback_days)
        returns2 = df2['Close'].pct_change().shift(1).tail(self.lookback_days)
        corr = returns1.corr(returns2)
        if pd.isna(corr):
            corr = 0.0
        self.correlation_cache[cache_key] = corr
        self.correlation_cache[f"{symbol2}_{symbol1}"] = corr
        return corr

    def calculate_stressed_correlation(self, corr_matrix: pd.DataFrame, scenario: str = 'covid_2020') -> pd.DataFrame:
        if scenario not in self.stress_scenarios:
            scenario = 'normal'
        params = self.stress_scenarios[scenario]
        stressed_corr = corr_matrix.copy()
        np.fill_diagonal(stressed_corr.values, 1)
        mask = ~np.eye(*stressed_corr.shape, dtype=bool)
        stressed_corr.values[mask] = np.minimum(
            stressed_corr.values[mask] * params['correlation_multiplier'],
            0.99
        )
        return stressed_corr

    def check_correlation_risk(
        self,
        new_symbol: str,
        existing_positions: List[Dict],
        price_data: Dict[str, pd.DataFrame],
        max_correlation: float = 0.7,
        use_stressed: bool = True
    ) -> Tuple[bool, str]:
        if not existing_positions:
            return True, "OK"
        for pos in existing_positions:
            symbol = pos['symbol']
            corr = self.get_correlation(new_symbol, symbol, price_data)
            if use_stressed:
                stressed_corr = min(corr * 1.3, 0.99)
            check_corr = stressed_corr if use_stressed else corr
            if check_corr > max_correlation:
                return False, f"Korelasi {corr:.2f} (stress: {stressed_corr:.2f}) dengan {symbol} > {max_correlation}"
        return True, "OK"

    def calculate_portfolio_var(
        self,
        positions: Dict[str, Dict],
        corr_matrix: pd.DataFrame,
        confidence: float = 0.95,
        use_stressed: bool = False
    ) -> Tuple[float, float, float]:
        if not positions or corr_matrix.empty:
            return 0.0, 0.0, 0.0
        symbols = list(positions.keys())
        weights = []
        total_risk = sum([p['risk_amount'] for p in positions.values()])
        for symbol in symbols:
            if symbol in positions and symbol in corr_matrix.index:
                weight = positions[symbol]['risk_amount'] / total_risk if total_risk > 0 else 0
                weights.append(weight)
        if len(weights) < 2:
            z_score = {0.95: 1.645, 0.99: 2.326}[confidence]
            var_amount = total_risk * z_score
            return var_amount, total_risk, 1.0
        try:
            sub_corr = corr_matrix.loc[symbols, symbols].values
            if use_stressed:
                mask = ~np.eye(len(symbols), dtype=bool)
                sub_corr[mask] = np.minimum(sub_corr[mask] * 1.3, 0.99)
        except Exception:
            return total_risk * 1.645, total_risk, 1.0
        weights_array = np.array(weights)
        portfolio_variance = np.dot(np.dot(weights_array, sub_corr), weights_array)
        portfolio_std = np.sqrt(max(portfolio_variance, 0.001))
        z_score = {0.95: 1.645, 0.99: 2.326}[confidence]
        var_amount = total_risk * portfolio_std * z_score
        return var_amount, total_risk, portfolio_std

# =============================================================================
# 13. REALISTIC FEE CONFIG - STOCKBIT VERSION (FIXED)
# =============================================================================

class RealisticFeeConfig:
    BROKER_FEE_BUY = 0.0015
    BROKER_FEE_SELL = 0.0025
    VAT_RATE = 0.11
    TOTAL_FEE_BUY = BROKER_FEE_BUY * (1 + VAT_RATE)
    TOTAL_FEE_SELL = BROKER_FEE_SELL * (1 + VAT_RATE)
    MIN_FEE_PER_TRANSACTION = 0

    def __init__(self, liquidity: str = 'medium'):
        self.liquidity = liquidity
        self.slippage = {
            'buy': 0.0002,
            'sell': 0.0003
        }

    def calculate_buy_cost(self, price: float, lot: int) -> float:
        amount = price * 100 * lot
        broker_fee = amount * self.TOTAL_FEE_BUY
        slippage = amount * self.slippage['buy']
        total_fee = broker_fee + slippage
        return total_fee

    def calculate_sell_cost(self, price: float, lot: int) -> float:
        amount = price * 100 * lot
        broker_fee = amount * self.TOTAL_FEE_SELL
        slippage = amount * self.slippage['sell']
        total_fee = broker_fee + slippage
        return total_fee

    def calculate_round_trip(
        self,
        entry_price: float,
        exit_price: float,
        lot: int
    ) -> Tuple[float, float, float]:
        buy_cost = self.calculate_buy_cost(entry_price, lot)
        sell_cost = self.calculate_sell_cost(exit_price, lot)
        total_fee = buy_cost + sell_cost
        buy_amount = entry_price * 100 * lot
        sell_amount = exit_price * 100 * lot
        gross_profit = sell_amount - buy_amount
        net_profit = gross_profit - total_fee
        net_return_pct = (net_profit / buy_amount) * 100 if buy_amount > 0 else 0
        return total_fee, net_profit, net_return_pct

    def print_fee_info(self, amount: float):
        buy_fee = amount * self.TOTAL_FEE_BUY
        sell_fee = amount * self.TOTAL_FEE_SELL
        print(f"\n💰 FEE INFO (untuk transaksi Rp {amount:,.0f}):")
        print(f"   Fee beli (0.1665%): Rp {buy_fee:,.0f}")
        print(f"   Fee jual (0.2775%): Rp {sell_fee:,.0f}")
        print(f"   Total fee: Rp {buy_fee + sell_fee:,.0f} ({((buy_fee + sell_fee)/amount)*100:.2f}%)")
        print(f"   Break-even return: {((buy_fee + sell_fee)/amount)*100:.2f}%")

# =============================================================================
# 14. ENTRY DELAY SIMULATOR - DIPERBAIKI (HANYA UNTUK INFORMASI)
# =============================================================================

class EntryDelaySimulator:
    def __init__(self, max_delay: int = 2):
        self.max_delay = max_delay

    def simulate_entry(
        self,
        df: pd.DataFrame,
        signal_idx: int,
        signal_price: float
    ) -> Optional[Dict]:
        start_idx = signal_idx + 1
        end_idx = min(start_idx + self.max_delay, len(df) - 1)
        entry_candidates = []
        for idx in range(start_idx, end_idx + 1):
            row = df.iloc[idx]
            open_price = float(row['Open'])
            low_price = float(row['Low'])
            if open_price <= signal_price * 1.01:
                entry_candidates.append({
                    'idx': idx,
                    'price': open_price,
                    'delay': idx - signal_idx,
                    'type': 'open'
                })
            if low_price <= signal_price:
                entry_price = min(signal_price, open_price)
                entry_candidates.append({
                    'idx': idx,
                    'price': entry_price,
                    'delay': idx - signal_idx,
                    'type': 'limit'
                })
        if entry_candidates:
            best = min(entry_candidates, key=lambda x: x['price'])
            best['slippage_pct'] = ((best['price'] - signal_price) / signal_price) * 100
            return best
        return None

# =============================================================================
# 15. DIVIDEN ANALYZER (FIXED VERSION - REALISTIS UNTUK IDX)
# =============================================================================

class DividendAnalyzer:
    def __init__(self):
        self.weights = {
            'yield': 0.30,
            'consistency': 0.25,
            'growth': 0.25,
            'payout': 0.20
        }
        self.thresholds = {
            'aristocrat': 85,
            'quality': 70,
            'decent': 50,
            'speculative': 30
        }
        self.max_reasonable_yield = 12.0
        self.special_dividend_threshold = 2.0

    def detect_special_dividends(self, dividend_series: pd.Series) -> Tuple[pd.Series, List[str]]:
        if len(dividend_series) < 3:
            return dividend_series, []
        mean_dividend = dividend_series.mean()
        std_dividend = dividend_series.std()
        upper_bound = max(mean_dividend + 2 * std_dividend, mean_dividend * 2)
        special_mask = dividend_series > upper_bound
        special_dates = dividend_series[special_mask].index.strftime('%Y-%m-%d').tolist()
        filtered_series = dividend_series[~special_mask]
        if len(special_dates) > 0:
            print(f"      ⚠️  Terdeteksi {len(special_dates)} special dividend")
        return filtered_series, special_dates

    def calculate_annual_dividend(self, dividend_series: pd.Series, method: str = 'conservative') -> float:
        if len(dividend_series) == 0:
            return 0.0
        now = pd.Timestamp.now()
        if method == 'ttm':
            one_year_ago = now - pd.Timedelta(days=365)
            ttm = dividend_series[dividend_series.index >= one_year_ago].sum()
            return ttm
        elif method == 'last_annual':
            last_year = now.year - 1
            last_year_div = dividend_series[dividend_series.index.year == last_year].sum()
            return last_year_div
        elif method == 'avg_3y':
            three_years_ago = now - pd.Timedelta(days=365*3)
            recent = dividend_series[dividend_series.index >= three_years_ago]
            if len(recent) == 0:
                return 0.0
            yearly_totals = recent.resample('Y').sum()
            if len(yearly_totals) == 0:
                return 0.0
            return yearly_totals.mean()
        elif method == 'conservative':
            ttm = self.calculate_annual_dividend(dividend_series, 'ttm')
            last_annual = self.calculate_annual_dividend(dividend_series, 'last_annual')
            avg_3y = self.calculate_annual_dividend(dividend_series, 'avg_3y')
            candidates = []
            if ttm > 0:
                candidates.append(ttm)
            if last_annual > 0:
                candidates.append(last_annual)
            if avg_3y > 0:
                candidates.append(avg_3y)
            if not candidates:
                return 0.0
            return min(candidates)
        return 0.0

    def analyze_dividend_yield(self, yield_pct: float) -> Dict:
        if yield_pct <= 0:
            return {'score': 0, 'grade': 'NO_YIELD', 'action': 'Tidak bagi dividen'}
        capped_yield = min(yield_pct, self.max_reasonable_yield)
        if capped_yield >= 10:
            return {
                'score': 100,
                'grade': 'EXCELLENT',
                'warning': f'Yield {yield_pct:.1f}% sangat tinggi, perlu cek sustainabilitas',
                'action': 'Verifikasi manual - mungkin special dividend'
            }
        elif capped_yield >= 8:
            return {
                'score': 95,
                'grade': 'VERY_HIGH',
                'warning': 'Yield di atas rata-rata, cek konsistensi',
                'action': 'Menarik tapi waspada'
            }
        elif capped_yield >= 6:
            return {
                'score': 90,
                'grade': 'HIGH',
                'action': 'Sangat menarik untuk income'
            }
        elif capped_yield >= 5:
            return {
                'score': 85,
                'grade': 'GOOD',
                'action': 'Di atas rata-rata IHSG'
            }
        elif capped_yield >= 4:
            return {
                'score': 75,
                'grade': 'ABOVE_AVERAGE',
                'action': 'Cukup menarik'
            }
        elif capped_yield >= 3:
            return {
                'score': 65,
                'grade': 'AVERAGE',
                'action': 'Rata-rata pasar'
            }
        elif capped_yield >= 2:
            return {
                'score': 50,
                'grade': 'BELOW_AVERAGE',
                'action': 'Di bawah rata-rata'
            }
        else:
            return {
                'score': 30,
                'grade': 'LOW',
                'action': 'Dividen kecil'
            }

    def analyze_consistency(self, dividend_series: pd.Series) -> Dict:
        if len(dividend_series) < 2:
            return {'score': 0, 'consistent_years': 0, 'grade': 'INSUFFICIENT_DATA'}
        yearly_div = dividend_series.resample('Y').sum()
        years_with_div = (yearly_div > 0).sum()
        total_years = len(yearly_div)
        consistency_ratio = years_with_div / total_years if total_years > 0 else 0
        streak = 0
        max_streak = 0
        for has_div in (yearly_div > 0):
            if has_div:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        if consistency_ratio >= 0.9 and max_streak >= 8:
            return {
                'score': 100,
                'grade': 'DIVIDEND_KING',
                'consistent_years': max_streak,
                'action': 'Sangat konsisten, blue chip'
            }
        elif consistency_ratio >= 0.8 and max_streak >= 5:
            return {
                'score': 85,
                'grade': 'VERY_CONSISTENT',
                'consistent_years': max_streak,
                'action': 'Konsisten dalam 5+ tahun'
            }
        elif consistency_ratio >= 0.7:
            return {
                'score': 70,
                'grade': 'CONSISTENT',
                'consistent_years': max_streak,
                'action': 'Cukup konsisten'
            }
        elif consistency_ratio >= 0.5:
            return {
                'score': 50,
                'grade': 'MODERATE',
                'consistent_years': max_streak,
                'action': 'Kadang tidak bagi dividen'
            }
        else:
            return {
                'score': 25,
                'grade': 'IRREGULAR',
                'consistent_years': max_streak,
                'action': 'Dividen tidak rutin'
            }

    def analyze_growth(self, dividend_series: pd.Series) -> Dict:
        if len(dividend_series) < 4:
            return {'score': 0, 'cagr': 0, 'grade': 'INSUFFICIENT_DATA'}
        yearly_div = dividend_series.resample('Y').sum()
        yearly_div = yearly_div[yearly_div > 0]
        if len(yearly_div) < 3:
            return {'score': 0, 'cagr': 0, 'grade': 'INSUFFICIENT_DATA'}
        first_year = yearly_div.iloc[0]
        last_year = yearly_div.iloc[-1]
        years = len(yearly_div) - 1
        if first_year > 0 and years > 0:
            cagr = (last_year / first_year) ** (1/years) - 1
            cagr_pct = cagr * 100
        else:
            cagr_pct = 0
        is_growing = True
        for i in range(len(yearly_div)-1):
            if yearly_div.iloc[i] > yearly_div.iloc[i+1] * 1.1:
                is_growing = False
                break
        if cagr_pct > 10 and is_growing:
            return {
                'score': 100,
                'cagr': round(cagr_pct, 2),
                'grade': 'EXCELLENT_GROWTH',
                'action': 'Dividen tumbuh cepat dan konsisten'
            }
        elif cagr_pct > 7:
            return {
                'score': 85,
                'cagr': round(cagr_pct, 2),
                'grade': 'GOOD_GROWTH',
                'action': 'Dividen tumbuh baik'
            }
        elif cagr_pct > 4:
            return {
                'score': 70,
                'cagr': round(cagr_pct, 2),
                'grade': 'MODERATE_GROWTH',
                'action': 'Dividen tumbuh moderat'
            }
        elif cagr_pct > 0:
            return {
                'score': 55,
                'cagr': round(cagr_pct, 2),
                'grade': 'SLOW_GROWTH',
                'action': 'Dividen tumbuh lambat'
            }
        elif cagr_pct > -2:
            return {
                'score': 40,
                'cagr': round(cagr_pct, 2),
                'grade': 'STABLE',
                'action': 'Dividen stabil (tidak tumbuh)'
            }
        else:
            return {
                'score': 20,
                'cagr': round(cagr_pct, 2),
                'grade': 'DECLINING',
                'action': 'Dividen menurun'
            }

    def estimate_payout_ratio(self, annual_dividend: float, price: float) -> Dict:
        if annual_dividend <= 0 or price <= 0:
            return {'score': 0, 'payout': 0, 'grade': 'NO_DATA'}
        assumed_per = 12
        estimated_eps = price / assumed_per
        payout = (annual_dividend / estimated_eps) * 100 if estimated_eps > 0 else 0
        if payout > 100:
            return {
                'score': 10,
                'payout': round(payout, 1),
                'grade': 'VERY_HIGH',
                'warning': 'Dividen > laba, tidak sustain'
            }
        elif payout > 80:
            return {
                'score': 30,
                'payout': round(payout, 1),
                'grade': 'HIGH',
                'warning': 'Payout ratio tinggi'
            }
        elif payout > 60:
            return {
                'score': 60,
                'payout': round(payout, 1),
                'grade': 'MODERATE_HIGH',
                'note': 'Cukup sehat'
            }
        elif payout > 40:
            return {
                'score': 85,
                'payout': round(payout, 1),
                'grade': 'HEALTHY',
                'note': 'Payout ratio sehat'
            }
        elif payout > 20:
            return {
                'score': 95,
                'payout': round(payout, 1),
                'grade': 'VERY_HEALTHY',
                'note': 'Ruang untuk naik dividen'
            }
        else:
            return {
                'score': 70,
                'payout': round(payout, 1),
                'grade': 'LOW',
                'note': 'Payout rendah, mungkin growth stock'
            }

    def analyze(self, symbol: str, dividend_df: pd.DataFrame, current_price: float) -> Dict:
        if dividend_df is None or len(dividend_df) == 0:
            return {
                'has_dividend': False,
                'total_score': 0,
                'category': 'NON_DIVIDEND',
                'suitability': ['Growth Investor', 'Aggressive Growth'],
                'display': 'NO_DIVIDEND',
                'metrics': {
                    'dividend_yield': 0,
                    'annual_dividend': 0,
                    'consistency_years': 0,
                    'dividend_growth': 0,
                    'estimated_payout': 0,
                    'special_dividends': 0
                }
            }
        dividend_series = dividend_df['Dividend']
        filtered_series, special_dates = self.detect_special_dividends(dividend_series)
        annual_dividend = self.calculate_annual_dividend(filtered_series, method='conservative')
        if current_price > 0 and annual_dividend > 0:
            raw_yield = (annual_dividend / current_price) * 100
        else:
            raw_yield = 0
        yield_pct = min(raw_yield, self.max_reasonable_yield)
        if raw_yield > self.max_reasonable_yield:
            print(f"      ⚠️  {symbol}: Yield {raw_yield:.1f}% dibatasi menjadi {yield_pct:.1f}%")
        yield_result = self.analyze_dividend_yield(yield_pct)
        consistency_result = self.analyze_consistency(filtered_series)
        growth_result = self.analyze_growth(filtered_series)
        payout_result = self.estimate_payout_ratio(annual_dividend, current_price)
        total_score = (
            yield_result['score'] * self.weights['yield'] +
            consistency_result['score'] * self.weights['consistency'] +
            growth_result['score'] * self.weights['growth'] +
            payout_result['score'] * self.weights['payout']
        )
        if yield_pct >= 8:
            category = 'HIGH_YIELD'
            if consistency_result['score'] >= 80:
                display = f"💰 HIGH YIELD ({yield_pct:.1f}%)"
            else:
                display = f"⚠️ HIGH YIELD ({yield_pct:.1f}%)"
            suitability = ['Income Seeker', 'High Risk']
        elif yield_pct >= 5:
            category = 'GOOD_YIELD'
            display = f"✅ GOOD ({yield_pct:.1f}%)"
            suitability = ['Income Investor', 'Balanced']
        elif yield_pct >= 3:
            category = 'MODERATE'
            display = f"📊 MODERATE ({yield_pct:.1f}%)"
            suitability = ['Growth & Income', 'Moderate Risk']
        elif yield_pct > 0:
            category = 'LOW_YIELD'
            display = f"📈 LOW ({yield_pct:.1f}%)"
            suitability = ['Growth Investor', 'Capital Appreciation']
        else:
            category = 'NON_DIVIDEND'
            display = 'NO_DIVIDEND'
            suitability = ['Growth Investor', 'Aggressive Growth']
        return {
            'has_dividend': True,
            'symbol': symbol,
            'total_score': round(total_score, 1),
            'category': category,
            'display': display,
            'suitability': suitability,
            'metrics': {
                'dividend_yield': round(yield_pct, 2),
                'annual_dividend': round(annual_dividend, 2),
                'raw_yield': round(raw_yield, 2),
                'consistency_years': consistency_result.get('consistent_years', 0),
                'dividend_growth': growth_result.get('cagr', 0),
                'estimated_payout': payout_result.get('payout', 0),
                'special_dividends': len(special_dates)
            }
        }

# =============================================================================
# 16. HOLDING PERIOD ANALYZER (BARU - DENGAN PARAMETER ADAPTIVE) - DIPERBARUI UNTUK TARGET LEBIH TINGGI
# =============================================================================

class HoldingPeriodAnalyzer:
    def __init__(self, df: pd.DataFrame, engine_type: str, atr_pct: float = None):
        self.df = df.copy()
        self.engine_type = engine_type
        self.atr_pct = atr_pct
        self.base_config = {
            'swing': {
                'base_target': 9.0,          # dinaikkan dari 7.5
                'base_max_hold': 30,          # dinaikkan dari 20
                'lookback_days': 252,
                'check_periods': [5, 10, 15, 20, 25, 30],
                'vol_adjust': True
            },
            'gorengan': {
                'base_target': 2.0,
                'base_max_hold': 3,
                'lookback_days': 63,
                'check_periods': [1, 2, 3],
                'vol_adjust': False
            },
            'investasi': {
                'base_target': 20.0,          # dinaikkan dari 10
                'base_max_hold': 365,         # dinaikkan dari 60
                'lookback_days': 504,
                'check_periods': [60, 120, 180, 252, 365],
                'vol_adjust': True
            }
        }
        base = self.base_config.get(engine_type, self.base_config['swing'])
        if base['vol_adjust'] and atr_pct is not None:
            if atr_pct < 2:
                target_mult = 0.8
                hold_mult = 1.3
            elif atr_pct < 4:
                target_mult = 1.0
                hold_mult = 1.0
            elif atr_pct < 6:
                target_mult = 1.2
                hold_mult = 0.8
            else:
                target_mult = 1.1
                hold_mult = 0.6
        else:
            target_mult = 1.0
            hold_mult = 1.0
        self.config = {
            'target_pct': base['base_target'] * target_mult,
            'max_hold_days': int(base['base_max_hold'] * hold_mult),
            'lookback_days': base['lookback_days'],
            'check_periods': base['check_periods']
        }
        self.results = {}

    def calculate_speed_metrics(self) -> Dict:
        close = self.df['Close'].values
        n = len(close)
        lookback = self.config['lookback_days']
        max_check = self.config['max_hold_days'] * 3
        self.results = {
            'days_to_reach': [],
            'reached_in_max': 0,
            'total_signals': 0
        }
        for i in range(lookback, n - max_check):
            current_price = close[i]
            target_price = current_price * (1 + self.config['target_pct']/100)
            days_to_target = 0
            reached = False
            for j in range(i + 1, min(i + max_check + 1, n)):
                if close[j] >= target_price:
                    days_to_target = j - i
                    reached = True
                    break
            self.results['total_signals'] += 1
            if reached:
                self.results['days_to_reach'].append(days_to_target)
                if days_to_target <= self.config['max_hold_days']:
                    self.results['reached_in_max'] += 1
        days_array = np.array(self.results['days_to_reach'])
        if len(days_array) > 0:
            self.results['success_rate'] = len(days_array) / self.results['total_signals'] * 100
            self.results['success_rate_in_max'] = self.results['reached_in_max'] / self.results['total_signals'] * 100
            self.results['avg_days'] = np.mean(days_array)
            self.results['median_days'] = np.median(days_array)
            self.results['p25_days'] = np.percentile(days_array, 25)
            self.results['p75_days'] = np.percentile(days_array, 75)
            self.results['p90_days'] = np.percentile(days_array, 90)
            self.results['min_days'] = np.min(days_array)
            self.results['max_days'] = np.max(days_array)
            self.results['std_days'] = np.std(days_array)
        else:
            self.results['success_rate'] = 0
            self.results['success_rate_in_max'] = 0
            self.results['avg_days'] = self.config['max_hold_days']
            self.results['median_days'] = self.config['max_hold_days']
            self.results['p90_days'] = self.config['max_hold_days'] * 2
        self.results['prob_by_period'] = {}
        for period in self.config['check_periods']:
            if len(days_array) > 0:
                success_in_period = np.sum(days_array <= period) / len(days_array) * 100
            else:
                success_in_period = 0
            self.results['prob_by_period'][period] = round(success_in_period, 1)
        return self.results

    def get_holding_recommendation(self) -> Dict:
        if not self.results:
            self.calculate_speed_metrics()
        if len(self.results['days_to_reach']) > 0:
            safe_days = int(self.results['p90_days'])
            optimal_days = int(self.results['p75_days'])
        else:
            safe_days = self.config['max_hold_days']
            optimal_days = self.config['max_hold_days'] // 2
        safe_days = min(safe_days, self.config['max_hold_days'] * 2)
        optimal_days = min(optimal_days, self.config['max_hold_days'])
        if self.engine_type in ['swing', 'investasi']:
            if self.results['success_rate_in_max'] > 60:
                grade = "⚡ CEPAT"
            elif self.results['success_rate_in_max'] > 40:
                grade = "⏱️  MODERAT"
            else:
                grade = "🐢 LAMBAT"
        else:
            if self.results['success_rate_in_max'] > 50:
                grade = "⚡ CEPAT"
            elif self.results['success_rate_in_max'] > 30:
                grade = "⏱️  MODERAT"
            else:
                grade = "🐢 LAMBAT"
        recommendation = (
            f"{grade}: Target {self.config['target_pct']:.1f}% dalam {optimal_days} hari "
            f"(prob {self.results['success_rate_in_max']:.1f}%). "
            f"Jual jika >{safe_days} hari."
        )
        return {
            'engine_type': self.engine_type,
            'target_pct': round(self.config['target_pct'], 1),
            'optimal_hold_days': optimal_days,
            'max_hold_days': safe_days,
            'success_rate_in_max': round(self.results['success_rate_in_max'], 1),
            'overall_success_rate': round(self.results['success_rate'], 1),
            'avg_days': round(self.results['avg_days'], 1),
            'median_days': round(self.results['median_days'], 1),
            'prob_by_period': self.results['prob_by_period'],
            'recommendation': recommendation,
            'exit_strategy': f"Jual jika >{safe_days} hari atau target {self.config['target_pct']:.1f}% tercapai"
        }

# =============================================================================
# 16A. ADAPTIVE PARAMETER BASE (BARU) - GABUNGAN 5 LEVEL ADAPTASI
# =============================================================================

class AdaptiveParameterBase:
    def __init__(self, engine_type: str, regime_detector=None):
        self.engine_type = engine_type
        self.regime_detector = regime_detector
        self.volatility_thresholds = {
            'swing': {'very_low': 2.0, 'low': 3.0, 'normal': 4.0, 'high': 6.0, 'very_high': 8.0},
            'gorengan': {'very_low': 3.0, 'low': 4.0, 'normal': 5.0, 'high': 7.0, 'very_high': 10.0},
            'investasi': {'very_low': 1.5, 'low': 2.5, 'normal': 3.5, 'high': 5.0, 'very_high': 7.0}
        }
        self.volume_thresholds = {
            'very_low': 0.5, 'low': 0.8, 'normal': 1.0, 'high': 1.5, 'very_high': 2.0, 'extreme': 3.0
        }
        self.rsi_thresholds = {
            'oversold_extreme': 20, 'oversold': 30, 'neutral_low': 40, 'neutral': 50,
            'neutral_high': 60, 'overbought': 70, 'overbought_extreme': 80
        }

    def get_volatility_level(self, atr_pct: float) -> Tuple[str, float]:
        thresholds = self.volatility_thresholds.get(self.engine_type, self.volatility_thresholds['swing'])
        if atr_pct < thresholds['very_low']:
            return 'VERY_LOW', 1.3
        elif atr_pct < thresholds['low']:
            return 'LOW', 1.15
        elif atr_pct < thresholds['normal']:
            return 'NORMAL', 1.0
        elif atr_pct < thresholds['high']:
            return 'HIGH', 0.85
        elif atr_pct < thresholds['very_high']:
            return 'VERY_HIGH', 0.7
        else:
            return 'EXTREME', 0.5

    def get_volume_level(self, volume_ratio: float) -> Tuple[str, float]:
        if volume_ratio > self.volume_thresholds['extreme']:
            return 'EXTREME', 1.3
        elif volume_ratio > self.volume_thresholds['very_high']:
            return 'VERY_HIGH', 1.2
        elif volume_ratio > self.volume_thresholds['high']:
            return 'HIGH', 1.1
        elif volume_ratio > self.volume_thresholds['normal']:
            return 'NORMAL', 1.0
        elif volume_ratio > self.volume_thresholds['low']:
            return 'LOW', 0.9
        else:
            return 'VERY_LOW', 0.7

    def get_rsi_level(self, rsi: float) -> Tuple[str, float, str]:
        if rsi < self.rsi_thresholds['oversold_extreme']:
            return 'OVERSOLD_EXTREME', 1.2, 'BULLISH'
        elif rsi < self.rsi_thresholds['oversold']:
            return 'OVERSOLD', 1.1, 'BULLISH'
        elif rsi < self.rsi_thresholds['neutral_low']:
            return 'NEUTRAL_LOW', 1.0, 'NEUTRAL_BULL'
        elif rsi < self.rsi_thresholds['neutral_high']:
            return 'NEUTRAL', 1.0, 'NEUTRAL'
        elif rsi < self.rsi_thresholds['overbought']:
            return 'NEUTRAL_HIGH', 1.0, 'NEUTRAL_BEAR'
        elif rsi < self.rsi_thresholds['overbought_extreme']:
            return 'OVERBOUGHT', 0.9, 'BEARISH'
        else:
            return 'OVERBOUGHT_EXTREME', 0.8, 'BEARISH'

    def get_regime_multiplier(self) -> float:
        if not self.regime_detector:
            return 1.0
        regime = self.regime_detector.current_regime
        confidence = self.regime_detector.confidence / 100.0
        if regime == "BULL":
            return 1.0 + (0.2 * confidence)
        elif regime == "BEAR":
            return 1.0 - (0.3 * confidence)
        elif regime == "HIGH_VOL":
            return 1.0 - (0.4 * confidence)
        else:
            return 1.0

    def get_trend_bias(self, price: float, ma50: float, ma200: float) -> Tuple[str, float]:
        if price > ma200 and price > ma50:
            if ma50 > ma200:
                return 'STRONG_UPTREND', 1.1
            else:
                return 'UPTREND', 1.05
        elif price < ma200 and price < ma50:
            if ma50 < ma200:
                return 'STRONG_DOWNTREND', 0.9
            else:
                return 'DOWNTREND', 0.95
        else:
            return 'NEUTRAL', 1.0

    def calculate_adaptive_parameters(self,
                                      atr_pct: float,
                                      volume_ratio: float,
                                      rsi: float,
                                      price: float,
                                      ma50: float,
                                      ma200: float,
                                      base_sl_multiplier: float = 1.5,
                                      base_tp_multiplier: float = 2.5,
                                      base_min_rr: float = 1.0) -> Dict:
        vol_level, vol_mult = self.get_volatility_level(atr_pct)
        vol_level_desc, vol_conf_mult = self.get_volume_level(volume_ratio)
        rsi_level, rsi_mult, rsi_bias = self.get_rsi_level(rsi)
        regime_mult = self.get_regime_multiplier()
        trend_bias, trend_mult = self.get_trend_bias(price, ma50, ma200)

        sl_mult = vol_mult * regime_mult
        if vol_level_desc in ['VERY_LOW', 'LOW']:
            sl_mult *= 0.95
        if rsi_level in ['OVERBOUGHT', 'OVERBOUGHT_EXTREME']:
            sl_mult *= 0.95

        tp_mult = vol_mult * regime_mult * vol_conf_mult * trend_mult
        if rsi_bias in ['BULLISH']:
            tp_mult *= 1.1
        elif rsi_bias in ['BEARISH']:
            tp_mult *= 0.9

        min_rr = base_min_rr * regime_mult

        sl_mult = max(0.5, min(3.0, sl_mult))
        tp_mult = max(0.8, min(4.0, tp_mult))
        min_rr = max(0.5, min(2.0, min_rr))

        confidence_mult = vol_conf_mult * (1.0 + (0.1 if rsi_bias == 'BULLISH' else 0))
        confidence_mult = max(0.5, min(1.5, confidence_mult))

        return {
            'sl_multiplier': round(sl_mult, 2),
            'tp_multiplier': round(tp_mult, 2),
            'min_rr': round(min_rr, 2),
            'confidence_multiplier': round(confidence_mult, 2),
            'volatility_level': vol_level,
            'volume_level': vol_level_desc,
            'rsi_level': rsi_level,
            'rsi_bias': rsi_bias,
            'trend_bias': trend_bias,
            'atr_pct': round(atr_pct, 1),
            'volume_ratio': round(volume_ratio, 1),
            'rsi': round(rsi, 1),
            'regime_multiplier': round(regime_mult, 2)
        }

# =============================================================================
# 17. BASE STRATEGY ENGINE (DENGAN CONFIDENCE SCORE DAN HOLDING PERIOD)
# =============================================================================

class BaseStrategyEngine:
    def __init__(self, config: Any, global_fetcher: GlobalIndicesFetcher, engine_type: str):
        self.config = config
        self.global_fetcher = global_fetcher
        self.engine_type = engine_type
        self.risk_manager = None
        self.regime_detector = None
        self.walkforward_collector = None
        self.current_regime_params = {}
        self.confidence_score = 50
        self.confidence_factors = {}
        self.confidence_history = []
        self.holding_analyzers = {}
        self.adaptive_params = None
        self.backtest_metrics = {}

    def set_risk_manager(self, risk_manager: RiskManager) -> None:
        self.risk_manager = risk_manager

    def set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None:
        self.regime_detector = regime_detector
        if regime_detector:
            self.current_regime_params = regime_detector.get_regime_parameters()
            if self.risk_manager:
                self.risk_manager.adjust_for_regime(self.current_regime_params)

    def set_walkforward_collector(self, walkforward_collector: WalkForwardCollector) -> None:
        self.walkforward_collector = walkforward_collector

    def apply_regime_parameters(self, base_params: Dict) -> Dict:
        if not self.current_regime_params:
            return base_params
        multiplier = self.current_regime_params.get('entry_tolerance_multiplier', 1.0)
        rr_multiplier = self.current_regime_params.get('min_rr_multiplier', 1.0)
        adjusted = base_params.copy()
        if 'entry_tolerance' in adjusted:
            adjusted['entry_tolerance'] = adjusted['entry_tolerance'] * multiplier
        if 'min_rr' in adjusted:
            adjusted['min_rr'] = adjusted['min_rr'] * rr_multiplier
        return adjusted

    def get_backtest_metrics(self, symbol: str) -> Dict:
        if symbol in self.backtest_metrics:
            return self.backtest_metrics[symbol].calculate_metrics()
        if self.walkforward_collector:
            metrics = BacktestMetrics(self.walkforward_collector, self.engine_type, symbol)
            self.backtest_metrics[symbol] = metrics
            return metrics.calculate_metrics()
        return {
            'has_data': False,
            'message': 'No collector',
            'display': 'N/A'
        }

    def analyze_holding_period(self, symbol: str, df: pd.DataFrame, atr_pct: float = None) -> Dict:
        key = f"{symbol}_{atr_pct}" if atr_pct else symbol
        if key not in self.holding_analyzers:
            analyzer = HoldingPeriodAnalyzer(df, self.engine_type, atr_pct)
            analyzer.calculate_speed_metrics()
            self.holding_analyzers[key] = analyzer
        return self.holding_analyzers[key].get_holding_recommendation()

    def calculate_confidence_score(self, signal_data: Dict) -> float:
        raise NotImplementedError

    def get_confidence_breakdown(self) -> Dict:
        return {
            'score': self.confidence_score,
            'factors': self.confidence_factors,
            'history': self.confidence_history[-5:] if self.confidence_history else []
        }

    def get_signal(self, symbol: str, df: pd.DataFrame):
        raise NotImplementedError

# =============================================================================
# 18. MODAL ADAPTER (DENGAN KAPASITAS PER ENGINE) - DIPERBAIKI DENGAN TURNOVER FILTER
# =============================================================================

class ModalAdapter:
    def __init__(self, modal: float, engine_type: str):
        self.modal = modal
        self.engine_type = engine_type
        if modal < 50000:
            self.kategori = "ULTRA_MIKRO"
        elif modal < 100000:
            self.kategori = "MIKRO"
        elif modal < 500000:
            self.kategori = "RETAIL"
        elif modal < 2000000:
            self.kategori = "MENENGAH"
        elif modal < 10000000:
            self.kategori = "BESAR"
        elif modal < 100000000:
            self.kategori = "INSTITUSI_MINOR"
        else:
            self.kategori = "INSTITUSI_MAJOR"
        self.max_harga_beli = modal / 100

    def get_filter_harga(self) -> Tuple[float, float]:
        if self.engine_type == 'swing':
            min_price = 50
            max_price = min(5000, self.max_harga_beli)
        elif self.engine_type == 'gorengan':
            min_price = 5
            max_price = min(500, self.max_harga_beli)
        elif self.engine_type == 'investasi':
            min_price = 100
            max_price = min(50000, self.max_harga_beli)
        else:
            min_price = 50
            max_price = min(1000, self.max_harga_beli)
        if self.kategori in ["ULTRA_MIKRO", "MIKRO"]:
            min_price = max(min_price, 5)
        elif self.kategori == "RETAIL":
            min_price = max(min_price, 10)
        elif self.kategori == "MENENGAH":
            min_price = max(min_price, 50)
        elif self.kategori == "BESAR":
            min_price = max(min_price, 100)
        elif self.kategori == "INSTITUSI_MINOR":
            min_price = max(min_price, 500)
        else:
            min_price = max(min_price, 1000)
        return min_price, max_price

    def get_filter_turnover(self) -> Tuple[float, float]:
        """Mengembalikan (min_turnover, min_avg_turnover) dalam rupiah"""
        if self.kategori in ["ULTRA_MIKRO", "MIKRO"]:
            return 50000, 100000   # Rp 50rb - 100rb
        elif self.kategori == "RETAIL":
            return 150000, 200000
        elif self.kategori == "MENENGAH":
            return 500000, 1000000
        elif self.kategori == "BESAR":
            return 2000000, 5000000
        elif self.kategori == "INSTITUSI_MINOR":
            return 10000000, 20000000
        else:
            return 50000000, 100000000

    def get_filter_spread(self) -> float:
        if self.kategori in ["ULTRA_MIKRO", "MIKRO"]:
            return 8.0
        elif self.kategori == "RETAIL":
            return 6.0
        elif self.kategori == "MENENGAH":
            return 4.0
        elif self.kategori == "BESAR":
            return 3.0
        elif self.kategori == "INSTITUSI_MINOR":
            return 2.5
        else:
            return 2.0

    def get_filter_min_history(self) -> int:
        if self.kategori in ["ULTRA_MIKRO", "MIKRO"]:
            return 30
        elif self.kategori == "RETAIL":
            return 60
        elif self.kategori == "MENENGAH":
            return 100
        else:
            return 150

    def get_quality_filter(self) -> str:
        if self.kategori in ["ULTRA_MIKRO", "MIKRO", "RETAIL"]:
            return "ALL"
        elif self.kategori == "MENENGAH":
            return "LQ45"
        elif self.kategori == "BESAR":
            return "LQ45"
        elif self.kategori == "INSTITUSI_MINOR":
            return "QUALITY"
        else:
            return "QUALITY"

    def get_entry_tolerance(self) -> float:
        if self.kategori in ["ULTRA_MIKRO", "MIKRO"]:
            return 15.0
        elif self.kategori == "RETAIL":
            return 10.0
        elif self.kategori == "MENENGAH":
            return 7.0
        elif self.kategori == "BESAR":
            return 5.0
        elif self.kategori == "INSTITUSI_MINOR":
            return 4.0
        else:
            return 3.0

    def get_trend_tolerance(self) -> float:
        if self.kategori in ["ULTRA_MIKRO", "MIKRO"]:
            return 0.85
        elif self.kategori == "RETAIL":
            return 0.90
        elif self.kategori == "MENENGAH":
            return 0.95
        elif self.kategori == "BESAR":
            return 0.97
        elif self.kategori == "INSTITUSI_MINOR":
            return 0.98
        else:
            return 0.99

    def print_info(self) -> None:
        min_price, max_price = self.get_filter_harga()
        min_turn, avg_turn = self.get_filter_turnover()
        print(f"\n📊 ADAPTASI FILTER UNTUK MODAL: {self.kategori} - ENGINE: {self.engine_type}")
        print(f"   Kapasitas modal: Rp {self.modal:,}")
        print(f"   Max harga per saham: Rp {self.max_harga_beli:,.0f}")
        print(f"   Filter harga: Rp {min_price:,.0f} - {max_price:,.0f}")
        print(f"   Filter turnover: Rp {min_turn:,} - {avg_turn:,} (harian)")
        print(f"   Max spread: {self.get_filter_spread()}%")
        print(f"   Minimal history: {self.get_filter_min_history()} hari")
        print(f"   Quality filter: {self.get_quality_filter()}")
        print(f"   Entry tolerance: {self.get_entry_tolerance()}% dari MA50")
        print(f"   Trend tolerance: {self.get_trend_tolerance()*100:.0f}% dari MA200")

# =============================================================================
# 19. NEWS ANALYZER UNTUK SAHAM (DENGAN BOBOT JUMLAH ARTIKEL)
# =============================================================================

class NewsAnalyzer:
    def __init__(self, api_key: str, days_back: int = 7, max_articles: int = 10):
        self.api_key = api_key
        self.days_back = days_back
        self.max_articles = max_articles
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache = {}

    def _fetch_articles(self, symbol: str) -> List[Dict]:
        from_date = (datetime.now() - timedelta(days=self.days_back)).strftime('%Y-%m-%d')
        params = {
            'q': f"{symbol} stock OR saham",
            'from': from_date,
            'language': 'id',
            'sortBy': 'relevancy',
            'pageSize': self.max_articles,
            'apiKey': self.api_key
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'ok':
                return data.get('articles', [])
            else:
                logger.warning(f"NewsAPI error for {symbol}: {data.get('message')}")
                return []
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def _analyze_sentiment(self, text: str) -> float:
        if not text:
            return 0.0
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0

    def get_sentiment(self, symbol: str, use_cache: bool = True) -> Dict:
        if use_cache and symbol in self.cache:
            cached = self.cache[symbol]
            if cached['date'] == datetime.now().date():
                return cached['data']
        articles = self._fetch_articles(symbol)
        if not articles:
            result = {
                'avg_sentiment': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'label': 'netral'
            }
        else:
            sentiments = []
            for art in articles:
                text = f"{art.get('title', '')} {art.get('description', '')}"
                score = self._analyze_sentiment(text)
                sentiments.append(score)
            sentiments = np.array(sentiments)
            avg_sent = np.mean(sentiments)
            pos = np.sum(sentiments > 0.1)
            neg = np.sum(sentiments < -0.1)
            neu = len(sentiments) - pos - neg
            if avg_sent > 0.1:
                label = 'positif'
            elif avg_sent < -0.1:
                label = 'negatif'
            else:
                label = 'netral'
            result = {
                'avg_sentiment': avg_sent,
                'article_count': len(articles),
                'positive_count': pos,
                'negative_count': neg,
                'neutral_count': neu,
                'label': label
            }
        self.cache[symbol] = {
            'date': datetime.now().date(),
            'data': result
        }
        return result

    def get_multiplier_and_label(self, symbol: str) -> Tuple[float, str]:
        """
        Mengembalikan multiplier (untuk confidence) dan label sentimen.
        Memperhitungkan jumlah artikel: semakin banyak artikel, semakin besar pengaruh.
        """
        sent = self.get_sentiment(symbol)
        if sent['article_count'] == 0:
            return 1.0, 'netral'
        avg_sent = sent['avg_sentiment']
        # multiplier dasar berdasarkan sentimen
        if avg_sent > 0.3:
            base_mult = 1.10
        elif avg_sent > 0.1:
            base_mult = 1.05
        elif avg_sent < -0.3:
            base_mult = 0.90
        elif avg_sent < -0.1:
            base_mult = 0.95
        else:
            base_mult = 1.0
        # penyesuaian berdasarkan jumlah artikel
        article_factor = min(1.0 + (sent['article_count'] / 20), 1.1)  # maks +10%
        mult = base_mult * article_factor
        # batasi multiplier
        mult = max(0.8, min(1.2, mult))
        return round(mult, 3), sent['label']

# =============================================================================
# 20. GLOBAL NEWS ANALYZER (UNTUK IHSG, DOWJONES, DLS)
# =============================================================================

class GlobalNewsAnalyzer:
    def __init__(self, api_key: str, days_back: int = 3, max_articles: int = 10):
        self.api_key = api_key
        self.days_back = days_back
        self.max_articles = max_articles
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache = {}

    def get_sentiment(self, query: str) -> Dict:
        if query in self.cache:
            return self.cache[query]

        from_date = (datetime.now() - timedelta(days=self.days_back)).strftime('%Y-%m-%d')
        params = {
            'q': query,
            'from': from_date,
            'language': 'id' if 'IHSG' in query else 'en',
            'sortBy': 'relevancy',
            'pageSize': self.max_articles,
            'apiKey': self.api_key
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])
            sentiments = []
            for art in articles:
                text = f"{art.get('title', '')} {art.get('description', '')}"
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
            avg_sent = np.mean(sentiments) if sentiments else 0.0
            label = 'positif' if avg_sent > 0.1 else ('negatif' if avg_sent < -0.1 else 'netral')
            result = {
                'label': label,
                'avg_sentiment': avg_sent,
                'article_count': len(articles)
            }
            self.cache[query] = result
            return result
        except Exception as e:
            logger.error(f"Error fetching global news for {query}: {e}")
            return {'label': 'netral', 'avg_sentiment': 0.0, 'article_count': 0}

# =============================================================================
# 21. INVESTASI ENGINE (DENGAN CONFIDENCE SCORE DAN ADAPTIVE PARAMETER) - STOP LOSS DIPERLONGAR + FUNDAMENTAL
# =============================================================================

class InvestasiEngine(BaseStrategyEngine):
    def __init__(self, config, global_fetcher, news_analyzer=None):
        super().__init__(config, global_fetcher, engine_type='investasi')
        self.quality_stocks = QUALITY_STOCKS
        self.modal_adapter = ModalAdapter(config.MODAL, self.engine_type)
        self.warehouse = None
        self.dividend_analyzer = DividendAnalyzer()
        self.weights = {
            'teknikal': 0.35,
            'dividen': 0.30,
            'risk': 0.20,
            'volatilitas': 0.15
        }
        self.confidence_weights = {
            'teknikal': 0.30,
            'dividen': 0.30,
            'risk': 0.20,
            'trend': 0.20
        }
        self.adaptive_params = None
        self.news_analyzer = news_analyzer

    def set_warehouse(self, warehouse):
        self.warehouse = warehouse

    def set_regime_detector(self, regime_detector):
        super().set_regime_detector(regime_detector)
        self.adaptive_params = AdaptiveParameterBase('investasi', self.regime_detector)

    def calculate_atr_for_investment(self, df: pd.DataFrame, period: int = 50) -> Tuple[float, float]:
        try:
            high = df['High'].shift(1)
            low = df['Low'].shift(1)
            close = df['Close'].shift(1)
            tr1 = high - low
            tr2 = (high - close).abs()
            tr3 = (low - close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=period).mean()
            valid_atr = atr.dropna()
            if len(valid_atr) == 0:
                current_price = df['Close'].iloc[-1]
                return current_price * 0.02, 2.0
            atr_value = float(valid_atr.iloc[-1])
            current_price = df['Close'].iloc[-1]
            min_atr = current_price * 0.01
            atr_value = max(atr_value, min_atr)
            atr_pct = (atr_value / current_price) * 100
            return atr_value, atr_pct
        except Exception as e:
            logger.error(f"Error calculating ATR for investment: {str(e)}")
            current_price = df['Close'].iloc[-1]
            return current_price * 0.02, 2.0

    def get_volatility_score(self, atr_pct: float) -> float:
        if atr_pct < 2:
            return 100
        elif atr_pct < 3:
            return 85
        elif atr_pct < 4:
            return 70
        elif atr_pct < 5:
            return 55
        elif atr_pct < 6:
            return 40
        elif atr_pct < 8:
            return 25
        else:
            return 10

    def get_volatility_label(self, atr_pct: float) -> str:
        if atr_pct < 2:
            return "🟢 SANGAT RENDAH"
        elif atr_pct < 3:
            return "🟢 RENDAH"
        elif atr_pct < 4:
            return "🟡 NORMAL"
        elif atr_pct < 5:
            return "🟡 CUKUP"
        elif atr_pct < 6:
            return "🟠 TINGGI"
        elif atr_pct < 8:
            return "🔴 SANGAT TINGGI"
        else:
            return "🔴 EKSTREM"

    def calculate_target_prices(self, current_price: float, atr: float, df: pd.DataFrame) -> Dict:
        close = df['Close'].shift(1)
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        current_ma50 = ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else current_price
        current_ma200 = ma200.iloc[-1] if not pd.isna(ma200.iloc[-1]) else current_price
        swing_high = close.tail(60).max()
        swing_low = close.tail(60).min()
        swing_range = swing_high - swing_low
        fib_1272 = current_price + (swing_range * 0.272)
        fib_1618 = current_price + (swing_range * 0.618)
        target_atr3 = current_price + (atr * 3)
        target_atr5 = current_price + (atr * 5)
        target_atr8 = current_price + (atr * 8)
        target_atr10 = current_price + (atr * 10)
        target_atr15 = current_price + (atr * 15)   # target lebih agresif
        ath = close.max()
        max_reasonable_target = max(current_ma200 * 2, current_price + (atr * 20))
        all_targets = [
            ('ma50', current_ma50),
            ('ma200', current_ma200),
            ('atr3', target_atr3),
            ('atr5', target_atr5),
            ('atr8', target_atr8),
            ('atr10', target_atr10),
            ('atr15', target_atr15),
            ('fib1272', fib_1272),
            ('fib1618', fib_1618),
            ('ath', min(ath, max_reasonable_target))
        ]
        valid_targets = [t for t in all_targets if t[1] > current_price * 1.02]
        if len(valid_targets) == 0:
            target_atr15 = current_price + (atr * 15)
            valid_targets = [('atr15', target_atr15)]
        valid_targets.sort(key=lambda x: x[1])

        # Ambil nilai unik, lalu bulatkan ke integer terdekat
        unique_prices = sorted(set([price for _, price in valid_targets]))
        rounded_prices = [round(price) for price in unique_prices]
        unique_rounded = sorted(set(rounded_prices))

        if len(unique_rounded) >= 3:
            target_konservatif = int(unique_rounded[0])
            target_moderat = int(unique_rounded[1])
            target_agresif = int(unique_rounded[-1])
        elif len(unique_rounded) == 2:
            target_konservatif = int(unique_rounded[0])
            target_moderat = int(unique_rounded[1])
            target_agresif = int(unique_rounded[1])
        else:
            target_konservatif = target_moderat = target_agresif = int(unique_rounded[0])

        # ATH tetap dibulatkan ke integer
        ath_rounded = round(min(ath, max_reasonable_target))

        return {
            'target_konservatif': target_konservatif,
            'target_moderat': target_moderat,
            'target_agresif': target_agresif,
            'target_ath': int(ath_rounded)
        }

    def calculate_teknikal_score(self, df, price, ma50, ma200, ma50_series=None, ma200_series=None):
        score = 0
        if price > ma200:
            score += 35
        elif price > ma200 * 0.9:
            score += 20
        elif price > ma200 * 0.8:
            score += 10
        dist_to_ma50 = (price / ma50 - 1) * 100
        if -3 <= dist_to_ma50 <= 3:
            score += 25
        elif -5 <= dist_to_ma50 <= 5:
            score += 20
        elif -8 <= dist_to_ma50 <= 8:
            score += 10
        close = df['Close'].shift(1)
        ma20 = close.rolling(20).mean()
        if len(ma20) > 0 and not pd.isna(ma20.iloc[-1]) and ma20.iloc[-1] > ma50:
            score += 20
        if ma50_series is not None and len(ma50_series) > 20:
            if not pd.isna(ma50_series.iloc[-1]) and not pd.isna(ma50_series.iloc[-20]):
                if ma50_series.iloc[-1] > ma50_series.iloc[-20]:
                    score += 20
        else:
            ma50_calc = df['Close'].shift(1).rolling(50).mean().dropna()
            if len(ma50_calc) > 20:
                if ma50_calc.iloc[-1] > ma50_calc.iloc[-20]:
                    score += 20
        return min(score, 100)

    def calculate_risk_score(self, df, atr, price, spread) -> float:
        score = 0
        atr_pct = (atr / price) * 100
        if atr_pct < 2:
            score += 40
        elif atr_pct < 3:
            score += 35
        elif atr_pct < 4:
            score += 30
        elif atr_pct < 5:
            score += 20
        elif atr_pct < 6:
            score += 10
        if spread < 1:
            score += 30
        elif spread < 2:
            score += 25
        elif spread < 3:
            score += 20
        elif spread < 4:
            score += 15
        elif spread < 5:
            score += 10
        else:
            score += 5
        # Gunakan turnover sebagai pengganti volume
        turnover = (df['Close'] * df['Volume']).tail(60)
        turn_cv = turnover.std() / turnover.mean() if turnover.mean() > 0 else 999
        if turn_cv < 0.5:
            score += 30
        elif turn_cv < 0.8:
            score += 25
        elif turn_cv < 1.0:
            score += 20
        elif turn_cv < 1.5:
            score += 10
        else:
            score += 5
        return score

    def calculate_confidence_score(self, signal_data: Dict) -> float:
        total_score = 0
        teknikal_score = signal_data.get('teknikal_score', 50)
        teknikal_score_norm = min(100, max(0, teknikal_score))
        total_score += teknikal_score_norm * self.confidence_weights['teknikal']
        dividend_score = signal_data.get('dividend_score', 0)
        total_score += dividend_score * self.confidence_weights['dividen']
        risk_score = signal_data.get('risk_score', 50)
        total_score += risk_score * self.confidence_weights['risk']
        price_vs_ma200 = signal_data.get('price_vs_ma200', 0)
        if price_vs_ma200 > 15:
            trend_score = 100
        elif price_vs_ma200 > 10:
            trend_score = 90
        elif price_vs_ma200 > 5:
            trend_score = 80
        elif price_vs_ma200 > 0:
            trend_score = 70
        elif price_vs_ma200 > -5:
            trend_score = 50
        elif price_vs_ma200 > -10:
            trend_score = 35
        elif price_vs_ma200 > -15:
            trend_score = 20
        else:
            trend_score = 10
        total_score += trend_score * self.confidence_weights['trend']
        if self.regime_detector:
            regime_confidence = self.regime_detector.confidence
            if self.regime_detector.current_regime == "BEAR":
                total_score *= (1 - (regime_confidence / 200))
            elif self.regime_detector.current_regime == "BULL":
                total_score *= (1 + (regime_confidence / 200))
        final_score = min(100, max(0, total_score))
        self.confidence_score = final_score
        return final_score

    def get_adaptive_parameters(self, df, current_price, ma50, ma200):
        if self.adaptive_params is None:
            self.adaptive_params = AdaptiveParameterBase('investasi', self.regime_detector)
        atr_value, atr_pct = self.calculate_atr_for_investment(df, period=50)
        params = self.adaptive_params.calculate_adaptive_parameters(
            atr_pct=atr_pct,
            volume_ratio=1.0,
            rsi=50,
            price=current_price,
            ma50=ma50,
            ma200=ma200,
            base_sl_multiplier=3.0,
            base_tp_multiplier=5.0,
            base_min_rr=1.5
        )
        params['sl_multiplier'] = 3.0 * params['regime_multiplier']
        params['tp_multiplier'] = 5.0 * params['confidence_multiplier']
        return params

    def get_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        try:
            quality_filter = self.modal_adapter.get_quality_filter()
            if quality_filter == "QUALITY" and symbol not in self.quality_stocks:
                return None
            elif quality_filter == "LQ45" and symbol not in QUALITY_STOCKS[:45]:
                return None
            min_history = self.modal_adapter.get_filter_min_history()
            if len(df) < min_history:
                return None
            df_logic = df.shift(1).copy()
            close_logic = df_logic['Close']
            ma200_logic = close_logic.rolling(window=200).mean()
            ma50_logic = close_logic.rolling(window=50).mean()
            if len(ma200_logic.dropna()) < 1 or len(ma50_logic.dropna()) < 1:
                return None
            current_price_logic = float(close_logic.iloc[-1])
            current_ma200 = float(ma200_logic.iloc[-1]) if not pd.isna(ma200_logic.iloc[-1]) else current_price_logic
            current_ma50 = float(ma50_logic.iloc[-1]) if not pd.isna(ma50_logic.iloc[-1]) else current_price_logic
            current_price_real = float(df['Close'].iloc[-1])
            base_entry_tolerance = self.modal_adapter.get_entry_tolerance()
            regime_params = self.apply_regime_parameters({
                'entry_tolerance': base_entry_tolerance,
                'trend_tolerance': self.modal_adapter.get_trend_tolerance()
            })
            trend_tolerance = regime_params.get('trend_tolerance', self.modal_adapter.get_trend_tolerance())
            entry_tolerance = regime_params.get('entry_tolerance', base_entry_tolerance)
            if current_price_logic < current_ma200 * trend_tolerance:
                return None
            price_to_ma50 = (current_price_logic / current_ma50 - 1) * 100
            price_to_ma200 = (current_price_logic / current_ma200 - 1) * 100
            if price_to_ma50 > entry_tolerance:
                return None
            if price_to_ma50 < -entry_tolerance * 1.5:
                return None
            min_price, max_price = self.modal_adapter.get_filter_harga()
            if current_price_logic < min_price or current_price_logic > max_price:
                return None
            # Filter turnover
            min_turn, min_avg_turn = self.modal_adapter.get_filter_turnover()
            last_turnover = float(df_logic['Close'].iloc[-1] * df_logic['Volume'].iloc[-1])
            avg_turnover_20 = (df_logic['Close'] * df_logic['Volume']).tail(20).mean()
            if last_turnover < min_turn or avg_turnover_20 < min_avg_turn:
                return None
            spread = calculate_spread_pct(df)
            max_spread = self.modal_adapter.get_filter_spread()
            if spread > max_spread:
                return None
            atr_value, atr_pct = self.calculate_atr_for_investment(df, period=50)
            atr_14 = self.risk_manager.calculate_atr_in_rupiah(df)
            dividend_analysis = {'has_dividend': False, 'display': 'NO_DIVIDEND'}
            if self.warehouse:
                dividend_df = self.warehouse.get_dividends(symbol)
                if dividend_df is not None:
                    dividend_analysis = self.dividend_analyzer.analyze(symbol, dividend_df, current_price_real)
            backtest_metrics = self.get_backtest_metrics(symbol)
            ma50_series = df['Close'].shift(1).rolling(50).mean()
            teknikal_score = self.calculate_teknikal_score(
                df, current_price_logic, current_ma50, current_ma200, ma50_series=ma50_series
            )
            risk_score = self.calculate_risk_score(df, atr_14, current_price_logic, spread)
            adaptive_params = self.get_adaptive_parameters(df, current_price_logic, current_ma50, current_ma200)
            volatility_score = self.get_volatility_score(adaptive_params['atr_pct'])
            volatility_label = self.get_volatility_label(adaptive_params['atr_pct'])
            final_score = (
                teknikal_score * self.weights['teknikal'] +
                dividend_analysis.get('total_score', 0) * self.weights['dividen'] +
                risk_score * self.weights['risk'] +
                volatility_score * self.weights['volatilitas']
            )
            if backtest_metrics.get('has_data', False):
                final_score += backtest_metrics['win_rate'] * 0.10

            # ===== AMBIL DATA FUNDAMENTAL (OPSIONAL) =====
            fundamental = None
            if self.warehouse:
                fundamental = self.warehouse.get_fundamental(symbol)
                if fundamental:
                    # Hitung skor fundamental sederhana
                    fund_score = 0
                    per = fundamental.get('per')
                    pbv = fundamental.get('pbv')
                    roe = fundamental.get('roe')

                    # Filter fundamental: ROE >= 12% dan PBV <= 2
                    if roe is not None and roe > 0:
                        roe_pct = roe * 100
                        if roe_pct < 12:
                            return None  # tidak lolos filter
                    if pbv is not None and pbv > 0:
                        if pbv > 2:
                            return None  # tidak lolos filter

                    if per is not None and per > 0:
                        if per < 15:
                            fund_score += 15
                        elif per < 25:
                            fund_score += 10
                        elif per < 35:
                            fund_score += 5
                    if pbv is not None and pbv > 0:
                        if pbv < 1.5:
                            fund_score += 15
                        elif pbv < 2.5:
                            fund_score += 10
                        elif pbv < 4:
                            fund_score += 5
                    if roe is not None and roe > 0:
                        roe_pct = roe * 100
                        if roe_pct > 20:
                            fund_score += 20
                        elif roe_pct > 15:
                            fund_score += 15
                        elif roe_pct > 10:
                            fund_score += 10
                        elif roe_pct > 5:
                            fund_score += 5

                    # Tambahkan ke final_score (bobot 10% dari total, maks 100)
                    final_score += fund_score * 0.10
                    final_score = min(100, final_score)

            lot, cost, risk_amount = self.risk_manager.calculate_lot(current_price_logic, atr_14, symbol, use_kelly=True)
            if lot is None:
                return None
            can_add, reason = self.risk_manager.can_add_position(risk_amount, cost)
            if not can_add:
                return None

            # ===== STOP LOSS DIPERLONGAR UNTUK INVESTASI =====
            # Gunakan 15% di bawah entry atau 5% di bawah MA200, mana yang lebih tinggi (agar tidak terlalu ketat)
            sl_by_percent = current_price_logic * 0.85
            sl_by_ma200 = current_ma200 * 0.95
            sl_raw = max(sl_by_percent, sl_by_ma200)
            sl = round(sl_raw)  # <-- PERUBAHAN: bulatkan ke integer
            if sl >= current_price_logic:
                sl = round(current_price_logic * 0.85)

            sector = get_sector(symbol)
            risk_pct = (risk_amount / self.risk_manager.modal) * 100
            targets = self.calculate_target_prices(current_price_logic, atr_value, df)
            signal_data = {
                'teknikal_score': teknikal_score,
                'dividend_score': dividend_analysis.get('total_score', 0),
                'risk_score': risk_score,
                'price_vs_ma200': price_to_ma200
            }
            confidence_score = self.calculate_confidence_score(signal_data)
            confidence_score = min(100, confidence_score * adaptive_params['confidence_multiplier'])

            # News multiplier akan diisi di phase 1 setelah semua sinyal terkumpul
            news_multiplier = 1.0
            news_label = 'netral'

            confidence_score = min(100, max(0, confidence_score))
            if confidence_score >= 80:
                confidence_label = "🏆 VERY HIGH"
            elif confidence_score >= 65:
                confidence_label = "⭐ HIGH"
            elif confidence_score >= 50:
                confidence_label = "✅ MODERATE"
            elif confidence_score >= 35:
                confidence_label = "⚠️ LOW"
            else:
                confidence_label = "❌ VERY LOW"
            atr_pct_for_hold = (atr_value / current_price_logic) * 100
            holding_analysis = self.analyze_holding_period(symbol, df, atr_pct_for_hold)

            return {
                'Symbol': symbol,
                'Sector': sector,
                'Price': int(current_price_real),
                'MA50': int(current_ma50),
                'MA200': int(current_ma200),
                'To_MA50': f"{price_to_ma50:.1f}%",
                'Stop_Loss': int(sl),
                'Target_Konservatif': targets['target_konservatif'],
                'Target_Moderat': targets['target_moderat'],
                'Target_Agresif': targets['target_agresif'],
                'Target_ATH': targets['target_ath'],
                'Lot': lot,
                'Cost': int(cost),
                'Risk_Amount': int(risk_amount),
                'Risk_Pct': round(risk_pct, 1),
                'ATR': int(atr_value),
                'ATR_Pct': adaptive_params['atr_pct'],
                'ATR_Label': volatility_label,
                'ATR_Period': 50,
                'Volatility_Score': volatility_score,
                'Volatility_Level': adaptive_params['volatility_level'],
                'Regime_Multiplier': adaptive_params['regime_multiplier'],
                'Trend_Bias': adaptive_params['trend_bias'],
                'Dividend_Display': dividend_analysis.get('display', 'N/A'),
                'Dividend_Yield': dividend_analysis.get('metrics', {}).get('dividend_yield'),
                'Dividend_Years': dividend_analysis.get('metrics', {}).get('consistency_years'),
                'PER': fundamental.get('per') if fundamental else None,
                'PBV': fundamental.get('pbv') if fundamental else None,
                'ROE': fundamental.get('roe') if fundamental else None,
                'Backtest_Display': backtest_metrics.get('display', 'N/A'),
                'Teknikal_Score': teknikal_score,
                'Risk_Score': risk_score,
                'Final_Score': round(final_score, 1),
                'Confidence_Score': round(confidence_score, 1),
                'Confidence_Label': confidence_label,
                'News_Label': news_label,
                'News_Multiplier': news_multiplier,
                'Target_Pct': holding_analysis['target_pct'],
                'Optimal_Hold_Days': holding_analysis['optimal_hold_days'],
                'Max_Hold_Days': holding_analysis['max_hold_days'],
                'Success_Rate': holding_analysis['success_rate_in_max'],
                'Holding_Recommendation': holding_analysis['recommendation'],
                'Exit_Strategy': holding_analysis['exit_strategy']
            }
        except Exception as e:
            logger.error(f"Error in InvestasiEngine.get_signal for {symbol}: {str(e)}")
            return None

# =============================================================================
# 22. SWING ENGINE (DENGAN PARAMETER ADAPTIVE TERBAIK) - VERSI FRAKSI
# =============================================================================

class SwingEngine(BaseStrategyEngine):
    def __init__(self, config, global_fetcher, news_analyzer=None):
        super().__init__(config, global_fetcher, engine_type='swing')
        self.rsi_period = getattr(config, 'RSI_PERIOD', 14)
        self.ma_short = getattr(config, 'MA_SHORT', 20)
        self.ma_long = getattr(config, 'MA_LONG', 50)
        self.ma200_period = getattr(config, 'MA200_PERIOD', 200)
        self.atr_period = getattr(config, 'ATR_PERIOD', 14)
        self.base_sl_multiplier = getattr(config, 'SL_MULTIPLIER', 1.5)
        self.base_tp_multiplier = getattr(config, 'TP_MULTIPLIER', 2.5)
        self.volume_boost = getattr(config, 'VOLUME_BOOST', 1.5)
        self.base_min_rr = getattr(config, 'MIN_RR', 1.0)
        self.min_ev_pct = getattr(config, 'MIN_EV_PCT', 2.0)
        self.modal_adapter = ModalAdapter(config.MODAL, self.engine_type)
        self.adaptive_params = None
        self.confidence_weights = {
            'rsi': 0.25,
            'volume': 0.20,
            'trend': 0.20,
            'risk_reward': 0.15,
            'historical': 0.20
        }
        self.news_analyzer = news_analyzer

    def set_regime_detector(self, regime_detector):
        super().set_regime_detector(regime_detector)
        self.adaptive_params = AdaptiveParameterBase('swing', self.regime_detector)

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            out = df.copy()
            close = out['Close']
            high = out['High']
            low = out['Low']
            volume = out['Volume']
            open_price = out['Open']
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
            rs = avg_gain / (avg_loss + 1e-6)
            out['RSI'] = 100 - (100 / (1 + rs))
            out['MA20'] = close.rolling(window=self.ma_short, min_periods=self.ma_short).mean()
            out['MA50'] = close.rolling(window=self.ma_long, min_periods=self.ma_long).mean()
            out['MA200'] = close.rolling(window=self.ma200_period, min_periods=self.ma200_period).mean()
            out['MA_Trend'] = (out['MA20'] > out['MA50']).astype(float)
            out['TR'] = np.maximum(high - low, np.maximum((high - close.shift()), (low - close.shift()).abs()))
            out['ATR'] = out['TR'].rolling(window=self.atr_period, min_periods=self.atr_period).mean()
            # Turnover
            out['Turnover'] = close * volume
            out['Turnover_MA'] = out['Turnover'].rolling(window=20, min_periods=20).mean()
            out['Turnover_Ratio'] = out['Turnover'] / (out['Turnover_MA'] + 1e-6)

            # ===== TAMBAHAN INDIKATOR =====
            # MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            out['MACD'] = exp1 - exp2
            out['MACD_signal'] = out['MACD'].ewm(span=9, adjust=False).mean()
            out['MACD_hist'] = out['MACD'] - out['MACD_signal']

            # Bollinger Bands (20,2)
            bb_mid = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            out['BB_lower'] = bb_mid - 2 * bb_std
            out['BB_upper'] = bb_mid + 2 * bb_std

            # Stochastic (14,3)
            low_14 = low.rolling(14).min()
            high_14 = high.rolling(14).max()
            out['Stoch_K'] = 100 * (close - low_14) / (high_14 - low_14)
            out['Stoch_D'] = out['Stoch_K'].rolling(3).mean()

            # Swing Low 20 hari
            out['Swing_Low_20'] = low.rolling(window=20).min()

            # Gap Up
            out['Gap_Up'] = (open_price > high.shift(1)) & (volume > volume.rolling(20).mean() * 1.5)

            # Volume Direction Confirmation
            out['Volume_Up'] = (close > close.shift(1)) & (volume > volume.rolling(20).mean())
            out['Volume_Down'] = (close < close.shift(1)) & (volume > volume.rolling(20).mean())

            # ADX (Average Directional Index)
            adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
            out['ADX'] = adx_indicator.adx()

            out = out.replace([np.inf, -np.inf], np.nan)
            return out.shift(1)
        except Exception as e:
            logger.error(f"Error in calculate_features: {str(e)}")
            return pd.DataFrame()

    def get_adaptive_parameters(self, df, current_price, rsi, turnover_ratio, ma50, ma200):
        if self.adaptive_params is None:
            self.adaptive_params = AdaptiveParameterBase('swing', self.regime_detector)
        atr = self.risk_manager.calculate_atr_in_rupiah(df)
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
        params = self.adaptive_params.calculate_adaptive_parameters(
            atr_pct=atr_pct,
            volume_ratio=turnover_ratio,
            rsi=rsi,
            price=current_price,
            ma50=ma50,
            ma200=ma200,
            base_sl_multiplier=self.base_sl_multiplier,
            base_tp_multiplier=self.base_tp_multiplier,
            base_min_rr=self.base_min_rr
        )
        if params['volatility_level'] in ['VERY_LOW', 'LOW']:
            params['min_ev_pct'] = self.min_ev_pct * 1.2
        elif params['volatility_level'] in ['HIGH', 'VERY_HIGH']:
            params['min_ev_pct'] = self.min_ev_pct * 0.9
        else:
            params['min_ev_pct'] = self.min_ev_pct
        return params

    def calculate_confidence_score(self, signal_data: Dict) -> float:
        factors = {}
        total_score = 0
        rsi = signal_data.get('rsi', 50)
        if rsi < 30:
            rsi_score = 100
        elif rsi < 40:
            rsi_score = 80
        elif rsi < 50:
            rsi_score = 60
        elif rsi > 70:
            rsi_score = 20
        else:
            rsi_score = 40
        factors['rsi'] = rsi_score
        total_score += rsi_score * self.confidence_weights['rsi']
        turnover_ratio = signal_data.get('turnover_ratio', 1.0)
        if turnover_ratio > 2.0:
            vol_score = 100
        elif turnover_ratio > 1.5:
            vol_score = 80
        elif turnover_ratio > 1.2:
            vol_score = 60
        elif turnover_ratio > 1.0:
            vol_score = 40
        else:
            vol_score = 20
        factors['volume'] = vol_score
        total_score += vol_score * self.confidence_weights['volume']
        ma20 = signal_data.get('ma20', 0)
        ma50 = signal_data.get('ma50', 0)
        if ma20 > ma50:
            trend_score = 80
        else:
            trend_score = 40
        factors['trend'] = trend_score
        total_score += trend_score * self.confidence_weights['trend']
        rr = signal_data.get('rr', 1.0)
        if rr > 3.0:
            rr_score = 100
        elif rr > 2.0:
            rr_score = 80
        elif rr > 1.5:
            rr_score = 60
        elif rr > 1.0:
            rr_score = 40
        else:
            rr_score = 20
        factors['risk_reward'] = rr_score
        total_score += rr_score * self.confidence_weights['risk_reward']
        historical_win_rate = signal_data.get('historical_win_rate', 50)
        hist_score = min(100, max(0, historical_win_rate))
        factors['historical'] = hist_score
        total_score += hist_score * self.confidence_weights['historical']
        if self.regime_detector:
            regime_confidence = self.regime_detector.confidence
            if self.regime_detector.current_regime == "BEAR":
                total_score *= (1 - (regime_confidence / 200))
            elif self.regime_detector.current_regime == "BULL":
                total_score *= (1 + (regime_confidence / 200))
        final_score = min(100, max(0, total_score))
        self.confidence_score = final_score
        self.confidence_factors = factors
        self.confidence_history.append({
            'timestamp': datetime.now(),
            'score': final_score,
            'factors': factors.copy()
        })
        return final_score

    def get_signal(self, symbol: str, df: pd.DataFrame):
        try:
            min_history = self.modal_adapter.get_filter_min_history()
            if len(df) < min_history:
                return None
            df_feat = self.calculate_features(df)
            if len(df_feat) < self.ma200_period + 10:
                return None
            latest_logic = df_feat.iloc[-1]
            close_logic = float(latest_logic['Close']) if not pd.isna(latest_logic['Close']) else None
            if close_logic is None:
                return None
            close_real = float(df['Close'].iloc[-1])
            regime_params = self.apply_regime_parameters({
                'min_rr': self.base_min_rr
            })
            base_min_rr = regime_params.get('min_rr', self.base_min_rr)
            min_price, max_price = self.modal_adapter.get_filter_harga()
            if close_logic < min_price or close_logic > max_price:
                return None
            # Filter turnover
            min_turn, min_avg_turn = self.modal_adapter.get_filter_turnover()
            last_turnover = float(latest_logic['Turnover']) if not pd.isna(latest_logic['Turnover']) else 0
            avg_turnover_20 = latest_logic['Turnover_MA'] if not pd.isna(latest_logic['Turnover_MA']) else 0
            if last_turnover < min_turn or avg_turnover_20 < min_avg_turn:
                return None
            spread = calculate_spread_pct(df)
            max_spread = self.modal_adapter.get_filter_spread()
            if spread > max_spread:
                return None
            ma200 = float(latest_logic['MA200']) if not pd.isna(latest_logic['MA200']) else close_logic
            trend_tolerance = self.modal_adapter.get_trend_tolerance()
            if close_logic < ma200 * trend_tolerance:
                return None

            # ===== AMBIL NILAI INDIKATOR =====
            rsi = float(latest_logic['RSI']) if not pd.isna(latest_logic['RSI']) else 50
            turnover_ratio = float(latest_logic['Turnover_Ratio']) if not pd.isna(latest_logic['Turnover_Ratio']) else 1
            ma20 = float(latest_logic['MA20']) if not pd.isna(latest_logic['MA20']) else close_logic
            ma50 = float(latest_logic['MA50']) if not pd.isna(latest_logic['MA50']) else close_logic
            macd = latest_logic['MACD'] if not pd.isna(latest_logic['MACD']) else 0
            macd_signal = latest_logic['MACD_signal'] if not pd.isna(latest_logic['MACD_signal']) else 0
            macd_hist = latest_logic['MACD_hist'] if not pd.isna(latest_logic['MACD_hist']) else 0
            bb_lower = latest_logic['BB_lower'] if not pd.isna(latest_logic['BB_lower']) else close_logic * 0.95
            stoch_k = latest_logic['Stoch_K'] if not pd.isna(latest_logic['Stoch_K']) else 50
            stoch_d = latest_logic['Stoch_D'] if not pd.isna(latest_logic['Stoch_D']) else 50
            swing_low_20 = latest_logic['Swing_Low_20'] if not pd.isna(latest_logic['Swing_Low_20']) else close_logic * 0.9
            gap_up = latest_logic['Gap_Up'] if not pd.isna(latest_logic['Gap_Up']) else False
            volume_up = latest_logic['Volume_Up'] if not pd.isna(latest_logic['Volume_Up']) else False
            volume_down = latest_logic['Volume_Down'] if not pd.isna(latest_logic['Volume_Down']) else False
            adx = float(latest_logic['ADX']) if not pd.isna(latest_logic['ADX']) else 0

            # ===== FILTER ADX =====
            if adx < 20:
                return None

            # ===== HITUNG SKOR DENGAN WEIGHTING =====
            score = 0

            # RSI (weighting)
            if rsi < 25:
                score += 3
            elif rsi < 30:
                score += 2
            elif rsi < 35:
                score += 1

            # Volume (weighting)
            if turnover_ratio > 2.0:
                score += 3
            elif turnover_ratio > 1.5:
                score += 2
            elif turnover_ratio > 1.2:
                score += 1

            # Trend MA
            if ma20 > ma50:
                score += 1
                if (ma20/ma50 - 1) > 0.02:  # jarak > 2%
                    score += 1

            # MACD bullish
            if macd > macd_signal and macd_hist > 0:
                score += 1

            # Bollinger Bands – harga menyentuh lower band
            if close_logic <= bb_lower * 1.01:  # dalam 1% dari lower band
                score += 1

            # Stochastic oversold dan mulai naik
            if stoch_k < 20 and stoch_k > stoch_d:
                score += 1

            # Support – harga dekat swing low 20 hari
            if close_logic <= swing_low_20 * 1.02:  # dalam 2% dari swing low
                score += 1

            # Volume konfirmasi arah
            if volume_up:
                score += 1
            if volume_down:
                score -= 1

            # Gap up
            if gap_up:
                score += 1

            # ===== DETEKSI DIVERGENSI RSI =====
            last_20 = df_feat.tail(20)
            if len(last_20) >= 10:
                min_price_idx = last_20['Close'].idxmin()
                min_price_val = last_20.loc[min_price_idx, 'Close']
                rsi_at_min = last_20.loc[min_price_idx, 'RSI']
                current_price = last_20['Close'].iloc[-1]
                current_rsi = last_20['RSI'].iloc[-1]
                if not pd.isna(min_price_val) and not pd.isna(rsi_at_min):
                    if current_price < min_price_val and current_rsi > rsi_at_min:
                        score += 2

            # ===== FILTER TIMEFRAME LEBIH TINGGI =====
            weekly_df = df.resample('W').last()
            if len(weekly_df) >= 50:
                weekly_ma50 = weekly_df['Close'].rolling(50).mean().iloc[-1]
                weekly_close = weekly_df['Close'].iloc[-1]
                if weekly_close > weekly_ma50:
                    score += 2
                else:
                    score -= 1

            # ===== PENGARUH INDEKS GLOBAL =====
            sector = get_sector(symbol)
            sector_to_index = {
                'ENERGY': 'OIL',
                'MINING': 'GOLD',
                'CONSUMER': 'USDIDR',
                'INFRASTRUCTURE': 'IHSG',
                'TRADE': 'USDIDR',
                'AGRICULTURE': 'OIL',
            }
            if sector in sector_to_index:
                idx_name = sector_to_index[sector]
                idx_mom = self.global_fetcher.get_momentum(idx_name)
                if idx_mom > 2:
                    score += 1
                elif idx_mom < -2:
                    score -= 1

            dow_mom = self.global_fetcher.get_momentum('DOWJONES')
            if dow_mom > 2:
                score += 1
            elif dow_mom < -2:
                score -= 1

            nikkei_mom = self.global_fetcher.get_momentum('NIKKEI')
            if nikkei_mom > 2:
                score += 1
            elif nikkei_mom < -2:
                score -= 1

            shanghai_mom = self.global_fetcher.get_momentum('SHANGHAI')
            if shanghai_mom > 2:
                score += 1
            elif shanghai_mom < -2:
                score -= 1

            # ===== HITUNG SL, TP DENGAN FRAKSI =====
            atr = self.risk_manager.calculate_atr_in_rupiah(df)
            adaptive_params = self.get_adaptive_parameters(df, close_logic, rsi, turnover_ratio, ma50, ma200)
            sl_multiplier = adaptive_params['sl_multiplier']
            tp_multiplier = adaptive_params['tp_multiplier']
            min_rr = adaptive_params['min_rr']
            min_ev_pct = adaptive_params.get('min_ev_pct', self.min_ev_pct)

            # Tentukan fraksi berdasarkan harga
            if close_logic < 100:
                fraction = 5
            elif close_logic < 500:
                fraction = 10
            elif close_logic < 1000:
                fraction = 25
            elif close_logic < 5000:
                fraction = 50
            else:
                fraction = 100

            # Gunakan fungsi pembulatan fraksi
            sl = calculate_safe_stop_loss(
                price=close_logic,
                atr=atr,
                multiplier=sl_multiplier,
                fraction=fraction,
                min_distance_pct=1.5,
                engine_type='swing'
            )
            tp = calculate_safe_take_profit(
                price=close_logic,
                atr=atr,
                multiplier=tp_multiplier,
                fraction=fraction,
                engine_type='swing'
            )

            is_valid_rr, risk, reward, rr = validate_risk_reward(
                price=close_logic,
                sl=sl,
                tp=tp,
                min_rr=min_rr
            )
            if not is_valid_rr:
                return None

            prob_up = 0.5 + (score * 0.015)
            prob_up = min(prob_up, 0.75)
            expected_value = (prob_up * reward) - ((1 - prob_up) * risk)
            ev_pct = (expected_value / close_logic) * 100
            if ev_pct < min_ev_pct:
                return None

            backtest_metrics = self.get_backtest_metrics(symbol)
            signal_data = {
                'rsi': rsi,
                'turnover_ratio': turnover_ratio,
                'ma20': ma20,
                'ma50': ma50,
                'rr': rr,
                'historical_win_rate': backtest_metrics.get('win_rate', 50) if backtest_metrics.get('has_data', False) else 50
            }
            confidence_score = self.calculate_confidence_score(signal_data)
            confidence_score = min(100, confidence_score * adaptive_params['confidence_multiplier'])

            news_multiplier = 1.0
            news_label = 'netral'

            confidence_score = min(100, max(0, confidence_score))
            if confidence_score >= 80:
                confidence_label = "🔥 VERY HIGH"
            elif confidence_score >= 65:
                confidence_label = "⭐ HIGH"
            elif confidence_score >= 50:
                confidence_label = "✅ MODERATE"
            elif confidence_score >= 35:
                confidence_label = "⚠️ LOW"
            else:
                confidence_label = "❌ VERY LOW"
            atr_pct = (atr / close_logic) * 100
            holding_analysis = self.analyze_holding_period(symbol, df, atr_pct)
            lot, cost, risk_amount = self.risk_manager.calculate_lot(close_logic, atr, symbol, use_kelly=True)
            if lot is None:
                return None
            can_add, reason = self.risk_manager.can_add_position(risk_amount, cost)
            if not can_add:
                return None
            sector = get_sector(symbol)
            risk_pct = (risk_amount / self.risk_manager.modal) * 100

            return {
                'Symbol': symbol,
                'Sector': sector,
                'Price': int(close_real),
                'RSI': round(rsi, 1),
                'Stop_Loss': int(sl),
                'Take_Profit': int(tp),
                'R/R': round(rr, 2),
                'Prob_Up': round(prob_up, 3),
                'EV_Pct': round(ev_pct, 2),
                'Score': score,
                'ATR': int(atr),
                'Lot': lot,
                'Cost': int(cost),
                'Risk_Amount': int(risk_amount),
                'Risk_Pct': round(risk_pct, 1),
                'Volume': f"{turnover_ratio:.1f}x",
                'Backtest_Display': backtest_metrics.get('display', 'N/A'),
                'Confidence_Score': round(confidence_score, 1),
                'Confidence_Label': confidence_label,
                'Confidence_Factors': self.confidence_factors,
                'ATR_Pct': round(atr_pct, 1),
                'Volatility_Level': adaptive_params['volatility_level'],
                'Volume_Level': adaptive_params['volume_level'],
                'RSI_Level': adaptive_params['rsi_level'],
                'RSI_Bias': adaptive_params['rsi_bias'],
                'Trend_Bias': adaptive_params['trend_bias'],
                'Regime_Multiplier': adaptive_params['regime_multiplier'],
                'News_Label': news_label,
                'News_Multiplier': news_multiplier,
                'Target_Pct': holding_analysis['target_pct'],
                'Optimal_Hold_Days': holding_analysis['optimal_hold_days'],
                'Max_Hold_Days': holding_analysis['max_hold_days'],
                'Success_Rate': holding_analysis['success_rate_in_max'],
                'Prob_Day5': holding_analysis['prob_by_period'].get(5, 0),
                'Prob_Day10': holding_analysis['prob_by_period'].get(10, 0),
                'Prob_Day15': holding_analysis['prob_by_period'].get(15, 0),
                'Prob_Day20': holding_analysis['prob_by_period'].get(20, 0),
                'Prob_Day25': holding_analysis['prob_by_period'].get(25, 0),
                'Prob_Day30': holding_analysis['prob_by_period'].get(30, 0),
                'Holding_Recommendation': holding_analysis['recommendation'],
                'Exit_Strategy': holding_analysis['exit_strategy'],
                'Reasons': (f"RSI {rsi:.0f} ({adaptive_params['rsi_bias']}), "
                           f"Vol {turnover_ratio:.1f}x ({adaptive_params['volume_level']}), "
                           f"ATR {atr_pct:.1f}% ({adaptive_params['volatility_level']})"),
                'Chart': "BULLISH" if ma20 > ma50 else "NETRAL"
            }
        except Exception as e:
            logger.error(f"Error in SwingEngine.get_signal for {symbol}: {str(e)}")
            return None

# =============================================================================
# 23. INTRADAY GORENGAN ENGINE (VERSI BPJS) - DENGAN FRAKSI
# =============================================================================

class IntradayGorenganEngine(BaseStrategyEngine):
    def __init__(self, config, global_fetcher, news_analyzer=None):
        super().__init__(config, global_fetcher, engine_type='gorengan')
        self.config = config
        self.atr_period = getattr(config, 'ATR_PERIOD', 14)
        self.min_volume_spike = config.MIN_VOLUME_SPIKE
        self.base_min_rr = config.MIN_RR
        self.min_ev_pct = config.MIN_EV_PCT
        self.modal_adapter = ModalAdapter(config.MODAL, self.engine_type)
        self.adaptive_params = None
        self.confidence_weights = {
            'volume_spike': 0.35,
            'momentum': 0.25,
            'turnover': 0.20,
            'candle': 0.10,
            'historical': 0.10
        }
        self.news_analyzer = news_analyzer
        self.warehouse = None

    def set_warehouse(self, warehouse):
        self.warehouse = warehouse

    def set_regime_detector(self, regime_detector):
        super().set_regime_detector(regime_detector)
        self.adaptive_params = AdaptiveParameterBase('gorengan', self.regime_detector)

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            out = df.copy()
            close = out['Close']
            high = out['High']
            low = out['Low']
            volume = out['Volume']
            open_price = out['Open']

            out['MA5'] = close.rolling(window=5, min_periods=5).mean()
            out['MA5_ratio'] = close / out['MA5']
            out['Highest_High_5'] = high.rolling(window=5).max()
            out['Turnover'] = close * volume
            out['Turnover_MA'] = out['Turnover'].rolling(window=20).mean()
            out['Turnover_Ratio'] = out['Turnover'] / (out['Turnover_MA'] + 1e-6)
            out['TR'] = np.maximum(high - low, np.maximum((high - close.shift()), (low - close.shift()).abs()))
            out['ATR'] = out['TR'].rolling(window=self.atr_period).mean()
            out['Body'] = abs(close - open_price)
            out['Range'] = high - low
            out['Body_Ratio'] = out['Body'] / (out['Range'] + 1e-6)
            out['Day_Change'] = (close / open_price - 1) * 100

            out = out.replace([np.inf, -np.inf], np.nan)
            return out.shift(1)
        except Exception as e:
            logger.error(f"Error in calculate_features: {str(e)}")
            return pd.DataFrame()

    def is_bpjs_candidate(self, row: pd.Series, modal: float) -> Tuple[bool, str]:
        day_change = row.get('Day_Change')
        if pd.isna(day_change):
            return False, "Day change tidak tersedia"
        if day_change < self.config.MIN_DAY_CHANGE:
            return False, f"Kenaikan terlalu kecil: {day_change:.1f}% (<{self.config.MIN_DAY_CHANGE}%)"
        if day_change > 12:
            return False, f"Kenaikan terlalu besar: {day_change:.1f}% (>12%)"

        turnover_ratio = row.get('Turnover_Ratio')
        if pd.isna(turnover_ratio):
            return False, "Turnover ratio tidak tersedia"
        if turnover_ratio < self.config.MIN_VOLUME_SPIKE:
            return False, f"Volume spike terlalu rendah: {turnover_ratio:.1f}x (<{self.config.MIN_VOLUME_SPIKE}x)"

        open_price = row.get('Open')
        close_price = row.get('Close')
        if pd.isna(open_price) or pd.isna(close_price):
            return False, "Harga open/close tidak tersedia"
        if close_price < open_price * 1.02:
            return False, f"Harga tutup {close_price} < 1.02× open {open_price}"

        ma5 = row.get('MA5')
        if pd.isna(ma5):
            return False, "MA5 tidak tersedia"
        if close_price < ma5 * self.config.MIN_PRICE_TO_MA5:
            return False, f"Harga {close_price} < {self.config.MIN_PRICE_TO_MA5}× MA5 {ma5:.0f}"

        turnover = row.get('Turnover')
        if pd.isna(turnover):
            return False, "Turnover tidak tersedia"
        min_turnover_modal = max(modal * 100, self.modal_adapter.get_filter_turnover()[0])
        if turnover < min_turnover_modal:
            return False, f"Turnover Rp {turnover/1e6:.1f}Jt < minimal Rp {min_turnover_modal/1e6:.1f}Jt"

        if close_price > 1200:
            return False, f"Harga {close_price} > 1200 (terlalu mahal untuk BPJS)"

        return True, "OK"

    def get_adaptive_parameters(self, df, current_price, latest_logic):
        if self.adaptive_params is None:
            self.adaptive_params = AdaptiveParameterBase('gorengan', self.regime_detector)
        atr = self.risk_manager.calculate_atr_in_rupiah(df)
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
        turnover_ratio = latest_logic['Turnover_Ratio']
        params = self.adaptive_params.calculate_adaptive_parameters(
            atr_pct=atr_pct,
            volume_ratio=turnover_ratio,
            rsi=50,
            price=current_price,
            ma50=current_price,
            ma200=current_price,
            base_sl_multiplier=1.0,
            base_tp_multiplier=2.0,
            base_min_rr=self.base_min_rr
        )
        if turnover_ratio > 4.0:
            params['tp_multiplier'] *= 1.2
        elif turnover_ratio > 3.0:
            params['tp_multiplier'] *= 1.1
        return params

    def calculate_confidence_score(self, signal_data: Dict) -> float:
        factors = {}
        total_score = 0
        turnover_ratio = signal_data.get('turnover_ratio', 1.0)
        if turnover_ratio > 5.0:
            spike_score = 100
        elif turnover_ratio > 4.0:
            spike_score = 90
        elif turnover_ratio > 3.0:
            spike_score = 80
        elif turnover_ratio > 2.5:
            spike_score = 70
        elif turnover_ratio > 2.0:
            spike_score = 60
        else:
            spike_score = 40
        factors['volume_spike'] = spike_score
        total_score += spike_score * self.confidence_weights['volume_spike']

        day_change = signal_data.get('day_change', 0)
        if day_change > 15:
            momentum_score = 90
        elif day_change > 10:
            momentum_score = 80
        elif day_change > 7:
            momentum_score = 70
        elif day_change > 5:
            momentum_score = 60
        elif day_change > 3:
            momentum_score = 50
        else:
            momentum_score = 30
        factors['momentum'] = momentum_score
        total_score += momentum_score * self.confidence_weights['momentum']

        turnover = signal_data.get('turnover', 0)
        modal = signal_data.get('modal', 1)
        turnover_ratio_to_modal = turnover / modal if modal > 0 else 0
        if turnover_ratio_to_modal > 10:
            turnover_score = 100
        elif turnover_ratio_to_modal > 7:
            turnover_score = 85
        elif turnover_ratio_to_modal > 5:
            turnover_score = 70
        elif turnover_ratio_to_modal > 3:
            turnover_score = 55
        else:
            turnover_score = 35
        factors['turnover'] = turnover_score
        total_score += turnover_score * self.confidence_weights['turnover']

        body_ratio = signal_data.get('body_ratio', 0.5)
        if body_ratio > 0.8:
            candle_score = 90
        elif body_ratio > 0.6:
            candle_score = 75
        else:
            candle_score = 50
        factors['candle'] = candle_score
        total_score += candle_score * self.confidence_weights['candle']

        historical_win_rate = signal_data.get('historical_win_rate', 35)
        hist_score = min(100, max(0, historical_win_rate))
        factors['historical'] = hist_score
        total_score += hist_score * self.confidence_weights['historical']

        if self.regime_detector:
            regime_confidence = self.regime_detector.confidence
            if self.regime_detector.current_regime == "BEAR":
                total_score *= (1 - (regime_confidence / 200))
            elif self.regime_detector.current_regime == "BULL":
                total_score *= (1 + (regime_confidence / 200))
        final_score = min(100, max(0, total_score))
        self.confidence_score = final_score
        self.confidence_factors = factors
        self.confidence_history.append({
            'timestamp': datetime.now(),
            'score': final_score,
            'factors': factors.copy()
        })
        return final_score

    def get_signal(self, symbol: str, df: pd.DataFrame):
        try:
            min_history = self.modal_adapter.get_filter_min_history()
            if len(df) < min_history:
                return None
            df_feat = self.calculate_features(df)
            if len(df_feat) < 30:
                return None
            latest_logic = df_feat.iloc[-1]
            close_logic = float(latest_logic['Close']) if not pd.isna(latest_logic['Close']) else None
            if close_logic is None:
                return None
            close_real = float(df['Close'].iloc[-1])
            regime_params = self.apply_regime_parameters({
                'min_rr': self.base_min_rr
            })
            min_rr = regime_params.get('min_rr', self.base_min_rr)

            # Filter harga dari modal adapter
            min_price, max_price = self.modal_adapter.get_filter_harga()
            if close_logic < min_price or close_logic > max_price:
                return None

            # Filter turnover dasar
            min_turn, min_avg_turn = self.modal_adapter.get_filter_turnover()
            last_turnover = float(latest_logic['Turnover']) if not pd.isna(latest_logic['Turnover']) else 0
            avg_turnover_20 = latest_logic['Turnover_MA'] if not pd.isna(latest_logic['Turnover_MA']) else 0
            if last_turnover < min_turn or avg_turnover_20 < min_avg_turn:
                return None

            # Filter spread
            spread = calculate_spread_pct(df)
            max_spread = self.modal_adapter.get_filter_spread()
            if spread > max_spread:
                return None

            # ===== FILTER BPJS =====
            is_candidate, reason = self.is_bpjs_candidate(latest_logic, self.risk_manager.modal)
            if not is_candidate:
                return None

            # ===== HITUNG STOP LOSS DAN TAKE PROFIT DENGAN FRAKSI =====
            if close_logic < 100:
                fraction = 5
            elif close_logic < 500:
                fraction = 10
            elif close_logic < 1000:
                fraction = 25
            elif close_logic < 5000:
                fraction = 50
            else:
                fraction = 100

            sl_price = close_logic * 0.985  # 1.5% loss
            tp_price = close_logic * 1.03   # 3% profit

            sl = math.floor(sl_price / fraction) * fraction
            tp = math.ceil(tp_price / fraction) * fraction

            is_valid_rr, risk, reward, rr = validate_risk_reward(
                price=close_logic,
                sl=sl,
                tp=tp,
                min_rr=min_rr
            )
            if not is_valid_rr:
                return None

            # Hitung skor
            score = 5
            if latest_logic['Turnover_Ratio'] > 3:
                score += 2
            elif latest_logic['Turnover_Ratio'] > 2:
                score += 1
            if latest_logic['Body_Ratio'] > 0.7:
                score += 1
            if latest_logic['Day_Change'] > 8:
                score += 1

            prob_up = 0.5 + (score * 0.02)
            prob_up = min(prob_up, 0.7)
            expected_value = (prob_up * reward) - ((1 - prob_up) * risk)
            ev_pct = (expected_value / close_logic) * 100
            if ev_pct < self.min_ev_pct:
                return None

            backtest_metrics = self.get_backtest_metrics(symbol)
            signal_data = {
                'turnover_ratio': latest_logic['Turnover_Ratio'],
                'day_change': latest_logic['Day_Change'],
                'turnover': latest_logic['Turnover'],
                'body_ratio': latest_logic['Body_Ratio'],
                'modal': self.risk_manager.modal if self.risk_manager else 1,
                'historical_win_rate': backtest_metrics.get('win_rate', 35) if backtest_metrics.get('has_data', False) else 35
            }
            confidence_score = self.calculate_confidence_score(signal_data)
            confidence_score = min(100, confidence_score)

            news_multiplier = 1.0
            news_label = 'netral'

            confidence_score = min(100, max(0, confidence_score))
            if confidence_score >= 80:
                confidence_label = "🔥 VERY HIGH"
            elif confidence_score >= 65:
                confidence_label = "⭐ HIGH"
            elif confidence_score >= 50:
                confidence_label = "✅ MODERATE"
            elif confidence_score >= 35:
                confidence_label = "⚠️ LOW"
            else:
                confidence_label = "❌ VERY LOW"

            atr = self.risk_manager.calculate_atr_in_rupiah(df)
            atr_pct = (atr / close_logic) * 100
            holding_analysis = self.analyze_holding_period(symbol, df, atr_pct)

            lot, cost, risk_amount = self.risk_manager.calculate_lot(close_logic, atr, symbol, use_kelly=True)
            if lot is None:
                return None
            can_add, reason = self.risk_manager.can_add_position(risk_amount, cost)
            if not can_add:
                return None

            sector = get_sector(symbol)
            risk_pct = (risk_amount / self.risk_manager.modal) * 100

            return {
                'Symbol': symbol,
                'Sector': sector,
                'Price': int(close_real),
                'Day_Change': round(latest_logic['Day_Change'], 1),
                'Volume_Spike': f"{latest_logic['Turnover_Ratio']:.1f}x",
                'Turnover': f"Rp {latest_logic['Turnover']/1e6:.0f}Jt",
                'Stop_Loss': int(sl),
                'Take_Profit': int(tp),
                'R/R': round(rr, 2),
                'Prob_Up': round(prob_up, 3),
                'EV_Pct': round(ev_pct, 2),
                'Score': score,
                'ATR': int(atr),
                'Lot': lot,
                'Cost': int(cost),
                'Risk_Amount': int(risk_amount),
                'Risk_Pct': round(risk_pct, 1),
                'Backtest_Display': backtest_metrics.get('display', 'N/A'),
                'Confidence_Score': round(confidence_score, 1),
                'Confidence_Label': confidence_label,
                'Confidence_Factors': self.confidence_factors,
                'ATR_Pct': round(atr_pct, 1),
                'News_Label': news_label,
                'News_Multiplier': news_multiplier,
                'Target_Pct': holding_analysis['target_pct'],
                'Optimal_Hold_Days': holding_analysis['optimal_hold_days'],
                'Max_Hold_Days': holding_analysis['max_hold_days'],
                'Success_Rate': holding_analysis['success_rate_in_max'],
                'Prob_Day1': holding_analysis['prob_by_period'].get(1, 0),
                'Prob_Day2': holding_analysis['prob_by_period'].get(2, 0),
                'Prob_Day3': holding_analysis['prob_by_period'].get(3, 0),
                'Holding_Recommendation': holding_analysis['recommendation'],
                'Exit_Strategy': holding_analysis['exit_strategy'],
                'Reasons': (f"Naik {latest_logic['Day_Change']:.1f}%, Vol {latest_logic['Turnover_Ratio']:.1f}x, "
                           f"Harga {close_logic}, Turnover {latest_logic['Turnover']/1e6:.0f}Jt"),
                'Chart': "BULLISH",
                'Note': 'WATCHLIST UNTUK BESOK PAGI (konfirmasi pre-opening)'
            }
        except Exception as e:
            logger.error(f"Error in IntradayGorenganEngine.get_signal for {symbol}: {str(e)}")
            return None

# =============================================================================
# 24. CONFIGURATION CLASSES (AGGRESSIVE) - TIDAK BERUBAH
# =============================================================================

class SwingConfig:
    def __init__(self, modal: float):
        self.MODAL = modal
        self.MODE = 'swing'
        self.INTERVAL = "1d"
        self.PERIOD = "6mo"
        self.RSI_PERIOD = 14
        self.MA_SHORT = 20
        self.MA_LONG = 50
        self.MA200_PERIOD = 200
        self.ATR_PERIOD = 14
        self.SL_MULTIPLIER = 1.5
        self.TP_MULTIPLIER = 2.5
        self.VOLUME_BOOST = 1.5
        self.MIN_RR = 1.0
        self.MIN_EV_PCT = 2.0

class GorenganConfig:
    def __init__(self, modal: float):
        self.MODAL = modal
        self.MODE = 'gorengan'
        self.INTERVAL = "1h"
        self.PERIOD = "1mo"
        self.ATR_PERIOD = 14
        # Parameter yang bisa dilonggarkan
        self.MIN_DAY_CHANGE = 1.8          # minimal kenaikan harian (%)
        self.MIN_VOLUME_SPIKE = 2.0        # minimal rasio turnover
        self.MIN_PRICE_TO_MA5 = 1.02       # minimal rasio harga terhadap MA5
        self.MIN_RR = 1.5                  # risk/reward minimal
        self.MIN_EV_PCT = 1.0               # minimal expected value

class InvestasiConfig:
    def __init__(self, modal: float):
        self.MODAL = modal
        self.MODE = 'investasi'
        self.INTERVAL = "1d"
        self.PERIOD = "5y"

# =============================================================================
# 25. GOOGLE SHEETS EXPORTER (DENGAN FORMAT SERAGAM)
# =============================================================================

class GoogleSheetsExporter:
    def __init__(self):
        self.initialized = False
        self.spreadsheet = None
        self._url_printed = False
        self.HEADERS = ['Date', 'Engine', 'Symbol', 'Entry', 'SL', 'TP', 'R/R', 'Notes', 'P&L (manual)']

    def _init_sheets(self):
        try:
            auth.authenticate_user()
            creds, _ = default()
            self.gc = gspread.authorize(creds)
            self.initialized = True
            print("✅ Berhasil konek ke Google Sheets")
            return True
        except Exception as e:
            print(f"❌ Gagal konek ke Google Sheets: {e}")
            return False

    def _get_or_create_spreadsheet(self, sheet_name):
        try:
            self.spreadsheet = self.gc.open(sheet_name)
            print(f"✅ Membuka spreadsheet: {sheet_name}")
        except:
            self.spreadsheet = self.gc.create(sheet_name)
            print(f"✅ Membuat spreadsheet baru: {sheet_name}")
            try:
                email = input("\n📧 Share spreadsheet dengan email (optional, tekan Enter untuk skip): ").strip()
                if email:
                    self.spreadsheet.share(email, perm_type='user', role='writer')
                    print(f"✅ Shared with {email}")
            except:
                pass
        return self.spreadsheet

    def _get_or_create_engine_sheet(self, engine_name):
        sheet_title = engine_name.replace(' ', '_').replace('ENGINE', '').strip()[:100]
        try:
            worksheet = self.spreadsheet.worksheet(sheet_title)
            print(f"✅ Membuka sheet: {sheet_title}")
            existing_headers = worksheet.row_values(1)
            if existing_headers != self.HEADERS:
                # Hapus semua baris dan tulis header baru
                worksheet.clear()
                worksheet.append_row(self.HEADERS)
        except:
            worksheet = self.spreadsheet.add_worksheet(title=sheet_title, rows=10000, cols=len(self.HEADERS))
            print(f"✅ Membuat sheet baru: {sheet_title}")
            worksheet.append_row(self.HEADERS)
        return worksheet

    def _format_notes(self, signal, engine_name):
        notes = []
        if engine_name == 'SWING ENGINE':
            notes.append(f"RSI {signal.get('RSI','-')}")
            notes.append(f"Vol {signal.get('Volume','-')}")
            notes.append(f"EV {signal.get('EV_Pct',0)}%")
            notes.append(f"Score {signal.get('Score',0)}")
            notes.append(f"Hold {signal.get('Optimal_Hold_Days','-')}d")
            notes.append(f"ATR {signal.get('ATR_Pct',0)}%")
            if signal.get('Confidence_Factors'):
                # Tambahkan faktor confidence jika ada
                notes.append(f"Conf: {signal['Confidence_Factors']}")
        elif engine_name == 'GORENGAN ENGINE':
            notes.append(f"Spike {signal.get('Volume_Spike','-')}")
            notes.append(f"EV {signal.get('EV_Pct',0)}%")
            notes.append(f"Score {signal.get('Score',0)}")
            notes.append(f"Hold {signal.get('Optimal_Hold_Days','-')}d")
            notes.append(f"ATR {signal.get('ATR_Pct',0)}%")
        elif engine_name == 'INVESTASI ENGINE':
            notes.append(f"Div {signal.get('Dividend_Display','-')}")
            notes.append(f"ToMA50 {signal.get('To_MA50','-')}")
            notes.append(f"Target ATH {signal.get('Target_ATH','-')}")
            notes.append(f"Hold {signal.get('Optimal_Hold_Days','-')}d")
            notes.append(f"ATR {signal.get('ATR_Pct',0)}%")
            notes.append(f"Score {signal.get('Final_Score',0)}")
            if signal.get('PER') and signal.get('PBV') and signal.get('ROE'):
                notes.append(f"PER {signal['PER']:.1f} PBV {signal['PBV']:.1f} ROE {signal['ROE']*100:.1f}%")
        # Tambahkan peringatan korelasi jika ada (nanti diisi di phase1)
        if signal.get('correlation_warning'):
            notes.append(signal['correlation_warning'])
        return " | ".join(notes)

    def ensure_spreadsheet_exists(self):
        if not self.initialized:
            if not self._init_sheets():
                return False
        if not self.spreadsheet:
            sheet_name = "IDX_Scanner_All_Engines"
            self._get_or_create_spreadsheet(sheet_name)
        if not self._url_printed:
            print(f"\n📊 Google Sheets URL: {self.spreadsheet.url}")
            self._url_printed = True
        return True

    def export_signals(self, signals, engine_name, modal):
        if not signals:
            print(f"ℹ️ Tidak ada sinyal untuk {engine_name}")
            return False
        if not self.initialized:
            if not self._init_sheets():
                return False
        try:
            if not self.spreadsheet:
                sheet_name = "IDX_Scanner_All_Engines"
                self._get_or_create_spreadsheet(sheet_name)
            worksheet = self._get_or_create_engine_sheet(engine_name)
            for signal in signals:
                notes = self._format_notes(signal, engine_name)
                row = [
                    datetime.now().strftime('%Y-%m-%d'),
                    engine_name.replace(' ENGINE', ''),
                    signal['Symbol'],
                    signal['Price'],
                    signal['Stop_Loss'],
                    signal.get('Take_Profit', signal.get('Target_Agresif', '-')),
                    signal.get('R/R', '-'),
                    notes,
                    ''  # kolom P&L manual kosong
                ]
                worksheet.append_row(row)
            print(f"✅ Berhasil export {len(signals)} sinyal ke {engine_name} sheet")
            if not self._url_printed:
                print(f"\n📊 Google Sheets URL: {self.spreadsheet.url}")
                self._url_printed = True
            return True
        except Exception as e:
            print(f"❌ Gagal export ke Google Sheets: {e}")
            return False

# =============================================================================
# 26. PORTFOLIO GUIDE UNTUK INVESTASI ENGINE (TETAP, HANYA PERBAIKAN KECIL)
# =============================================================================

def print_investasi_portfolio_guide(signals, modal, risk_manager,
                                   portfolio_risk_calculator=None, price_data=None):
    if not signals:
        return
    qualified_signals = [s for s in signals if s.get('Final_Score', 0) > 50]
    if len(qualified_signals) < 3:
        qualified_signals = signals
    sorted_signals = sorted(qualified_signals,
                           key=lambda x: x.get('Final_Score', 0),
                           reverse=True)
    top_3 = sorted_signals[:3]
    print("\n" + "="*100)
    print("📊 INVESTASI ENGINE - 3 REKOMENDASI TERBAIK")
    print("="*100)
    print(f"Modal: Rp {modal:,} | Risk: {risk_manager.risk_per_trade_pct:.1f}%")
    print("-"*100)
    total_cost = 0
    total_dividend = 0
    div_count = 0
    total_fee_impact = 0
    for i, signal in enumerate(top_3, 1):
        cost = signal.get('Cost', 0)
        div_yield = signal.get('Dividend_Yield', 0)
        if div_yield and div_yield != 'N/A':
            try:
                div_yield = float(div_yield)
                total_dividend += div_yield
                div_count += 1
            except:
                div_yield = 0
        fee_config = RealisticFeeConfig(liquidity='medium')
        total_fee, net_profit, net_return = fee_config.calculate_round_trip(
            signal['Price'], 
            signal['Target_Agresif'],   # dulu 'Target_ATH'
            signal['Lot']
        )
        fee_impact_pct = (total_fee / cost) * 100 if cost > 0 else 0
        total_fee_impact += total_fee
        confidence = signal.get('Confidence_Score', 0)
        conf_bar = '█' * int(confidence/10) + '░' * (10 - int(confidence/10))
        print(f"\n{i}. {signal['Symbol']} - {signal.get('Dividend_Display', 'GROWTH')}")
        print(f"   Harga: Rp {signal['Price']:,} | Lot: {signal['Lot']} | Biaya: Rp {cost:,}")
        print(f"   Target Agresif: {signal.get('Target_Agresif', 'HOLD')} | Stop: Rp {signal['Stop_Loss']:,}")
        print(f"   Hold: {signal['Optimal_Hold_Days']} hari | Sukses: {signal['Success_Rate']}%")
        print(f"   Confidence: {confidence}% [{conf_bar}] {signal['Confidence_Label']}")
        print(f"   Volatilitas: {signal.get('ATR_Pct', 0)}% ({signal.get('Volatility_Level', 'N/A')})")
        if div_yield > 0:
            print(f"   Dividen: {div_yield}%")
        if signal.get('News_Label'):
            print(f"   📰 Sentimen Berita: {signal['News_Label']}")
        if signal.get('PER') and signal.get('PBV') and signal.get('ROE'):
            print(f"   📊 PER: {signal['PER']:.1f} | PBV: {signal['PBV']:.1f} | ROE: {signal['ROE']*100:.1f}%")
        print(f"   💰 Fee Impact: Rp {total_fee:,.0f} ({fee_impact_pct:.1f}% dari biaya)")
        print(f"   📊 Net Return Target: {net_return:.1f}% (setelah fee)")
        total_cost += cost
    print("\n" + "-"*100)
    print(f"TOTAL: Rp {total_cost:,} ({(total_cost/modal)*100:.1f}% modal)")
    print(f"TOTAL FEE: Rp {total_fee_impact:,.0f}")
    if div_count > 0:
        print(f"Rata-rata Dividen: {total_dividend/div_count:.2f}%")
    if total_cost > modal * 1.1:
        print("⚠️  MELEBIHI MODAL! Kurangi lot.")
    print("="*100)

# =============================================================================
# 27. PORTFOLIO GUIDE UNTUK ENGINE LAIN (DENGAN SISA MODAL & RISK)
# =============================================================================

def print_portfolio_guide(signals, modal, risk_manager, portfolio_risk_calculator=None, price_data=None):
    if not signals:
        return
    def calculate_final_score(signal):
        base_score = signal.get('Score', 0)
        confidence = signal.get('Confidence_Score', 0)
        return base_score * (0.7 + 0.3 * confidence / 100)
    sorted_signals = sorted(signals, key=calculate_final_score, reverse=True)
    top_3 = sorted_signals[:3]
    print("\n" + "="*120)
    print("📊 PANDUAN PORTOFOLIO - 3 REKOMENDASI TERBAIK")
    print("="*120)
    print(f"Modal: Rp {modal:,}")
    print(f"Risk per trade: {risk_manager.risk_per_trade_pct:.2f}% (Rp {risk_manager.risk_per_trade_rp:,.0f})")
    print(f"Max portfolio risk: {risk_manager.max_risk_portfolio_pct}% (Rp {risk_manager.max_risk_portfolio_rp:,.0f})")
    print(f"Max modal per posisi: {risk_manager.max_modal_per_position_pct}% (Rp {risk_manager.max_modal_per_position_rp:,.0f})")
    print("-"*120)
    total_cost = 0
    total_risk = 0
    total_fee_impact = 0
    if portfolio_risk_calculator and price_data and len(top_3) > 1:
        print("\n📈 CORRELATION CHECK (NORMAL vs STRESSED):")
        for i in range(len(top_3)):
            for j in range(i+1, len(top_3)):
                corr = portfolio_risk_calculator.get_correlation(
                    top_3[i]['Symbol'],
                    top_3[j]['Symbol'],
                    price_data
                )
                stressed_corr = min(corr * 1.3, 0.99)
                if stressed_corr > 0.7:
                    status = "⚠️ HIGH RISK"
                elif stressed_corr > 0.5:
                    status = "🟡 MEDIUM"
                else:
                    status = "✅ LOW"
                print(f"   {top_3[i]['Symbol']} & {top_3[j]['Symbol']}: {corr:.2f} (stress: {stressed_corr:.2f}) - {status}")
        print("-"*120)
    for i, signal in enumerate(top_3, 1):
        risk_amount = signal.get('Risk_Amount', 0)
        risk_pct = (risk_amount / modal) * 100 if modal > 0 else 0
        cost = signal.get('Cost', 0)
        sl = signal.get('Stop_Loss', 0)
        price = signal.get('Price', 0)
        tp = signal.get('Take_Profit', 'HOLD')
        lot = signal.get('Lot', 1)
        modal_pct = (cost / modal) * 100 if modal > 0 else 0
        fee_config = RealisticFeeConfig(liquidity='medium')
        total_fee, net_profit, net_return = fee_config.calculate_round_trip(price, tp, lot)
        fee_impact_pct = (total_fee / cost) * 100 if cost > 0 else 0
        total_fee_impact += total_fee
        confidence_score = signal.get('Confidence_Score', 0)
        confidence_bar = '█' * int(confidence_score / 10) + '░' * (10 - int(confidence_score / 10))
        print(f"\n{i}. {signal['Symbol']} ({signal.get('Sector', 'OTHER')})")
        print(f"   Harga: Rp {price:,} | Lot: {lot} | Biaya: Rp {cost:,} ({modal_pct:.1f}% modal)")
        print(f"   Stop Loss: Rp {sl:,} | Risk: Rp {risk_amount:,} ({risk_pct:.1f}% modal)")
        print(f"   Target: Rp {tp:,} | R/R: {signal.get('R/R', '-'):.2f}")
        print(f"   🎯 Confidence: {confidence_score}% [{confidence_bar}] {signal.get('Confidence_Label', 'N/A')}")
        print(f"   📊 ATR: {signal.get('ATR_Pct', 0)}% ({signal.get('Volatility_Level', 'N/A')})")
        if signal.get('News_Label'):
            print(f"   📰 Sentimen Berita: {signal['News_Label']}")
        print(f"   ⏱️  Holding: {signal.get('Optimal_Hold_Days', 'N/A')} hari (max {signal.get('Max_Hold_Days', 'N/A')})")
        print(f"   📈 Sukses: {signal.get('Success_Rate', 0)}% dalam {signal.get('Optimal_Hold_Days', 'N/A')} hari")
        print(f"   💡 Exit: {signal.get('Exit_Strategy', 'Target tercapai')}")
        print(f"   📝 Alasan: {signal.get('Reasons', '-')}")
        print(f"   💰 Fee Impact: Rp {total_fee:,.0f} ({fee_impact_pct:.1f}% dari biaya)")
        print(f"   📊 Net Return Target: {net_return:.1f}% (setelah fee)")
        total_cost += cost
        total_risk += risk_amount
    print("\n" + "-"*120)
    print(f"📈 TOTAL 3 POSISI:")
    print(f"   Total Biaya: Rp {total_cost:,} ({(total_cost/modal)*100:.1f}% modal)")
    print(f"   Total Risk: Rp {total_risk:,} ({(total_risk/modal)*100:.1f}% modal)")
    print(f"   Total Fee Impact: Rp {total_fee_impact:,.0f}")
    sisa_modal = modal - total_cost
    sisa_risk = risk_manager.max_risk_portfolio_rp - total_risk
    print(f"   Sisa Modal: Rp {sisa_modal:,} ({(sisa_modal/modal)*100:.1f}%)")
    print(f"   Sisa Risk: Rp {sisa_risk:,} ({(sisa_risk/risk_manager.max_risk_portfolio_rp)*100:.1f}%)")
    if total_cost > modal:
        print(f"\n⚠️  PERINGATAN: Total biaya melebihi modal Rp {total_cost - modal:,}!")
        print(f"   Pilih hanya 1-2 posisi atau kurangi lot.")
    elif total_risk > risk_manager.max_risk_portfolio_rp:
        print(f"\n⚠️  PERINGATAN: Total risk melebihi batas {risk_manager.max_risk_portfolio_pct}%!")
        print(f"   Kurangi lot atau pilih 1-2 posisi saja.")
    else:
        print(f"\n✅ Risk dan modal masih dalam batas aman.")
    print("="*120)

# =============================================================================
# 28. BLOCK BOOTSTRAP MONTE CARLO - FIXED VERSION (VALIDATION ONLY)
# =============================================================================

class BlockBootstrapMonteCarlo:
    def __init__(self, returns_series: pd.Series, engine_type: str = 'swing'):
        self.returns = returns_series.dropna()
        self.engine_type = engine_type.lower()
        self._winsorize_returns()
        self.block_size = self._calculate_optimal_block_size()

    def _winsorize_returns(self):
        if len(self.returns) > 100:
            q1 = self.returns.quantile(0.01)
            q99 = self.returns.quantile(0.99)
            self.winsorize_bounds = (q1, q99)
            self.original_len = len(self.returns)
            self.returns = self.returns.clip(lower=q1, upper=q99)
            print(f"      📊 Winsorize returns: {q1:.2f}% s/d {q99:.2f}%")
            print(f"      📊 {self.original_len - len(self.returns)} outlier dihapus")

    def _calculate_optimal_block_size(self) -> int:
        if len(self.returns) < 100:
            fallback = {'gorengan': 5, 'swing': 20, 'investasi': 60}
            block_size = fallback.get(self.engine_type, 20)
        else:
            try:
                half_life = 10
                for lag in range(1, min(200, len(self.returns)//4)):
                    acf = self.returns.autocorr(lag=lag)
                    if not pd.isna(acf):
                        if acf < 0.5 and lag > 5:
                            half_life = lag
                            break
                base_block = min(120, max(10, half_life * 2))
                engine_multiplier = {'gorengan': 0.5, 'swing': 1.0, 'investasi': 3.0}
                block_size = int(base_block * engine_multiplier.get(self.engine_type, 1.0))
            except Exception as e:
                logger.error(f"Error calculating optimal block size: {str(e)}")
                block_size = {'gorengan': 5, 'swing': 20, 'investasi': 60}.get(self.engine_type, 20)
        if self.engine_type == 'gorengan':
            block_size = max(5, min(10, block_size))
            print(f"      📊 Block size DIPAKSA: {block_size} hari")
        elif self.engine_type == 'swing':
            block_size = max(20, min(40, block_size))
        elif self.engine_type == 'investasi':
            block_size = max(60, min(120, block_size))
        print(f"      📊 Block size untuk {self.engine_type.upper()}: {block_size} hari")
        return block_size

    def _create_blocks(self) -> List[np.ndarray]:
        blocks = []
        step = max(1, self.block_size // 2)
        for i in range(0, len(self.returns) - self.block_size + 1, step):
            block = self.returns.iloc[i:i+self.block_size].values
            blocks.append(block)
        print(f"      📦 Created {len(blocks)} blocks of size {self.block_size}")
        return blocks

    def run_simulation(
        self,
        initial_capital: float,
        n_simulations: int = 1000,
        n_trades: int = 100,
        risk_per_trade_pct: float = 3.0
    ) -> Dict:
        blocks = self._create_blocks()
        if not blocks:
            logger.error("No blocks created for simulation")
            return {}
        results = []
        print(f"\n      🎲 Running {n_simulations} simulations...")
        for sim in range(n_simulations):
            n_blocks_needed = (n_trades + self.block_size - 1) // self.block_size
            sampled_blocks = random.choices(blocks, k=n_blocks_needed)
            sampled_returns = np.concatenate(sampled_blocks)[:n_trades]
            equity = initial_capital
            peak = equity
            max_dd = 0
            loss_streak = 0
            trades_done = 0
            for r in sampled_returns:
                trade_return = equity * (r / 100)
                equity += trade_return
                trades_done += 1
                if equity > peak:
                    peak = equity
                else:
                    dd = (peak - equity) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
                if r < 0:
                    loss_streak += 1
                    if loss_streak >= 5:
                        break
                else:
                    loss_streak = 0
                if equity < initial_capital * 0.3:
                    break
            total_return_pct = (equity / initial_capital - 1) * 100
            results.append({'return': total_return_pct, 'max_dd': max_dd, 'trades': trades_done})
        returns = np.array([r['return'] for r in results])
        max_dds = np.array([r['max_dd'] for r in results])
        trades_counts = np.array([r['trades'] for r in results])
        summary = {
            'engine_type': self.engine_type,
            'block_size': self.block_size,
            'n_simulations': n_simulations,
            'n_trades_max': n_trades,
            'avg_trades_done': np.mean(trades_counts),
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'percentile_5': np.percentile(returns, 5),
            'percentile_25': np.percentile(returns, 25),
            'percentile_75': np.percentile(returns, 75),
            'percentile_95': np.percentile(returns, 95),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'avg_max_dd': np.mean(max_dds),
            'max_dd_95': np.percentile(max_dds, 95),
            'probability_profit': np.mean(returns > 0) * 100,
            'probability_2x': np.mean(returns > 100) * 100,
            'probability_50pct_loss': np.mean(returns < -50) * 100,
            'expected_return_per_trade': np.mean(returns) / np.mean(trades_counts) if np.mean(trades_counts) > 0 else 0
        }
        self.results = results
        self.summary = summary
        self.returns_array = returns
        return summary

    def print_results(self) -> None:
        if not hasattr(self, 'summary'):
            print("\n📊 No simulation results")
            return
        print("\n" + "="*100)
        print(f"🎲 BLOCK BOOTSTRAP MONTE CARLO - {self.engine_type.upper()}")
        print("="*100)
        print(f"Block size: {self.summary['block_size']} hari")
        print(f"Number of simulations: {self.summary['n_simulations']:,}")
        print(f"Max trades per simulation: {self.summary['n_trades_max']}")
        print(f"Rata-rata trades terealisasi: {self.summary['avg_trades_done']:.1f}")
        print(f"Risk per trade: 3.0%")
        print("\n📈 RETURN DISTRIBUTION:")
        print(f"   Mean return: {self.summary['mean_return']:.2f}%")
        print(f"   Median return: {self.summary['median_return']:.2f}%")
        print(f"   Std deviation: {self.summary['std_return']:.2f}%")
        print(f"   Best case (95th percentile): {self.summary['percentile_95']:.2f}%")
        print(f"   Worst case (5th percentile): {self.summary['percentile_5']:.2f}%")
        print("\n📉 DRAWDOWN ANALYSIS:")
        print(f"   Rata-rata max drawdown: {self.summary['avg_max_dd']:.2f}%")
        print(f"   Max drawdown 95th percentile: {self.summary['max_dd_95']:.2f}%")
        print("\n🎯 PROBABILITIES:")
        print(f"   Probability of profit: {self.summary['probability_profit']:.1f}%")
        print(f"   Probability of 2x money: {self.summary['probability_2x']:.1f}%")
        print(f"   Probability of 50% loss: {self.summary['probability_50pct_loss']:.1f}%")

    def plot_distribution(self, save_path: str = None):
        if not hasattr(self, 'returns_array'):
            print("📊 No simulation results to plot")
            return
        plt.figure(figsize=(12, 6))
        plt.hist(self.returns_array, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
        plt.axvline(self.summary['mean_return'], color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {self.summary['mean_return']:.2f}%")
        plt.axvline(self.summary['percentile_5'], color='orange', linestyle=':', linewidth=2,
                    label=f"5%: {self.summary['percentile_5']:.2f}%")
        plt.axvline(self.summary['percentile_95'], color='green', linestyle=':', linewidth=2,
                    label=f"95%: {self.summary['percentile_95']:.2f}%")
        plt.xlabel('Return (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Monte Carlo Simulation - {self.engine_type.upper()} (Block Size: {self.summary["block_size"]} days)',
                 fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        textstr = f'Mean: {self.summary["mean_return"]:.2f}%\nStd: {self.summary["std_return"]:.2f}%\nProfit Prob: {self.summary["probability_profit"]:.1f}%\nBlock Size: {self.summary["block_size"]} days'
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"✅ Chart saved: {save_path}")
        else:
            plt.show()

# =============================================================================
# 29. VALIDATION SUITE (DENGAN HOLDING PERIOD VALIDATION) - FIXED
# =============================================================================

class ValidationSuite:
    def __init__(self, engine_class, base_config, data_dict: Dict[str, pd.DataFrame]):
        self.engine_class = engine_class
        self.base_config = base_config
        self.data_dict = data_dict
        # Normalisasi engine_type dari nama kelas
        raw_name = engine_class.__name__.replace('Engine', '').lower()
        # Mapping untuk gorengan
        if raw_name == 'intradaygorengan':
            self.engine_type = 'gorengan'
        else:
            self.engine_type = raw_name

        self.engine_params = {
            'swing': {
                'name': 'SWING',
                'block_size_range': (20, 40),
                'min_history': 200,
                'holding_period': '5-20 hari',
                'win_rate_threshold': 40,
                'dd_threshold': 50
            },
            'gorengan': {
                'name': 'GORENGAN',
                'block_size_range': (5, 10),
                'min_history': 50,
                'holding_period': '1-3 hari',
                'win_rate_threshold': 38,
                'dd_threshold': 50
            },
            'investasi': {
                'name': 'INVESTASI',
                'block_size_range': (60, 120),
                'min_history': 500,
                'holding_period': '30-90 hari',
                'win_rate_threshold': 45,
                'dd_threshold': 45
            }
        }
        self.params = self.engine_params.get(
            self.engine_type,
            {
                'name': 'UNKNOWN',
                'block_size_range': (20, 40),
                'min_history': 200,
                'win_rate_threshold': 40,
                'dd_threshold': 50
            }
        )

    def run_all(self,
                walk_forward_params: Optional[Dict] = None,
                sensitivity_params: Optional[Dict] = None,
                monte_carlo_params: Optional[Dict] = None) -> Dict:
        results = {}
        if monte_carlo_params is None:
            monte_carlo_params = {'n_simulations': 500, 'n_trades': 100}
        print("\n" + "="*80)
        print(f"PHASE 2: VALIDATION SUITE - {self.params['name']} ENGINE")
        print(f"Holding Period: {self.params['holding_period']}")
        print("="*80)
        print("\n📦 BLOCK BOOTSTRAP MONTE CARLO")
        print("-"*40)
        print(f"   Engine: {self.params['name']}")
        print(f"   Block size range: {self.params['block_size_range'][0]}-{self.params['block_size_range'][1]} hari")
        print(f"   Min history: {self.params['min_history']} hari")
        print(f"\n   📥 Mengumpulkan returns dari {len(self.data_dict)} saham...")
        all_returns = []
        symbols_used = 0
        for symbol, df in self.data_dict.items():
            if len(df) >= self.params['min_history']:
                returns = df['Close'].pct_change().dropna() * 100
                sample_size = min(1000, len(returns))
                all_returns.extend(returns.values[:sample_size])
                symbols_used += 1
        print(f"   ✅ Menggunakan {symbols_used} saham dengan data >= {self.params['min_history']} hari")
        print(f"   📊 Total returns points: {len(all_returns):,}")
        if len(all_returns) < 100:
            print("   ❌ Data tidak cukup untuk Monte Carlo")
            results['monte_carlo'] = {'error': 'Insufficient data'}
        else:
            returns_series = pd.Series(all_returns)
            # Gunakan self.engine_type yang sudah dinormalisasi
            mc = BlockBootstrapMonteCarlo(returns_series=returns_series, engine_type=self.engine_type)
            mc_results = mc.run_simulation(
                initial_capital=self.base_config.MODAL,
                n_simulations=monte_carlo_params['n_simulations'],
                n_trades=monte_carlo_params['n_trades'],
                risk_per_trade_pct=3.0
            )
            mc.print_results()
            plot_path = f'monte_carlo_{self.engine_type}.png'
            mc.plot_distribution(save_path=plot_path)
            print(f"   ✅ Chart saved: {plot_path}")
            results['monte_carlo'] = mc_results
        if walk_forward_params:
            print("\n" + "="*80)
            print("📊 WALK-FORWARD VALIDATION")
            print("="*80)
            wf_results = self.run_walk_forward(walk_forward_params)
            results['walk_forward'] = wf_results
        if sensitivity_params:
            print("\n" + "="*80)
            print("📈 SENSITIVITY ANALYSIS")
            print("="*80)
            sa_results = self.run_sensitivity_analysis(sensitivity_params)
            results['sensitivity'] = sa_results
        print("\n" + "="*80)
        print("📋 VALIDATION SUMMARY")
        print("="*80)
        if 'monte_carlo' in results and 'error' not in results['monte_carlo']:
            mc = results['monte_carlo']
            print(f"\n{self.params['name']} ENGINE - MONTE CARLO RESULTS:")
            print(f"   Block size: {mc.get('block_size', 'N/A')} hari")
            print(f"   Mean return: {mc.get('mean_return', 0):.2f}%")
            print(f"   Profit probability: {mc.get('probability_profit', 0):.1f}%")
            print(f"   Risk (95% DD): {mc.get('max_dd_95', 0):.2f}%")
            win_rate = mc.get('probability_profit', 0)
            threshold = self.params['win_rate_threshold']
            dd_95 = mc.get('max_dd_95', 100)
            dd_threshold = self.params['dd_threshold']
            if win_rate > threshold + 5:
                print(f"   ✅ Win Rate: EXCELLENT (> {threshold+5}%)")
            elif win_rate > threshold:
                print(f"   ✅ Win Rate: ACCEPTABLE ({threshold}-{threshold+5}%)")
            elif win_rate > threshold - 5:
                print(f"   ⚠️ Win Rate: MODERATE ({threshold-5}-{threshold}%)")
            else:
                print(f"   ❌ Win Rate: NEED IMPROVEMENT (< {threshold-5}%)")
            if dd_95 < 30:
                print(f"   ✅ Drawdown: LOW RISK (<30%)")
            elif dd_95 < dd_threshold:
                print(f"   ⚠️ Drawdown: MODERATE RISK (30-{dd_threshold}%)")
            else:
                print(f"   ❌ Drawdown: HIGH RISK (> {dd_threshold}%)")
            print(f"\n   📋 OVERALL ASSESSMENT FOR {self.params['name']}:")
            if win_rate > threshold and dd_95 < dd_threshold:
                print(f"      ✅ READY FOR PRODUCTION")
            elif win_rate > threshold - 5 and dd_95 < dd_threshold + 5:
                print(f"      ⚠️ ACCEPTABLE WITH CAUTION")
            else:
                print(f"      🔧 NEEDS IMPROVEMENT")
        print("\n" + "="*80)
        print(f"✅ PHASE 2 VALIDATION COMPLETE - {self.params['name']}")
        print("="*80)
        return results

    def run_walk_forward(self, params: Dict) -> Dict:
        from sklearn.model_selection import TimeSeriesSplit
        results = {'windows': [], 'avg_metrics': {}}
        n_splits = params.get('n_splits', 5)
        test_size = params.get('test_size', 60)
        print(f"\n📊 Walk-Forward dengan {n_splits} split")
        sample_symbols = list(self.data_dict.keys())[:50]
        all_metrics = []
        for symbol in sample_symbols:
            df = self.data_dict.get(symbol)
            if df is None or len(df) < 500:
                continue
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
            for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                engine = self.engine_class(self.base_config, None)
                signals = []
                for i in range(len(test_df) - 20):
                    signal = engine.get_signal(symbol, test_df.iloc[:i+200])
                    if signal:
                        future_price = test_df.iloc[i+20]['Close']
                        entry = test_df.iloc[i]['Close']
                        ret = (future_price / entry - 1) * 100
                        signals.append(ret)
                if signals:
                    win_rate = np.mean(np.array(signals) > 0) * 100
                    all_metrics.append({
                        'symbol': symbol,
                        'fold': fold,
                        'win_rate': win_rate,
                        'n_trades': len(signals)
                    })
        if all_metrics:
            df_metrics = pd.DataFrame(all_metrics)
            results['avg_metrics'] = {
                'avg_win_rate': df_metrics['win_rate'].mean(),
                'avg_trades_per_fold': df_metrics['n_trades'].mean(),
                'total_folds': len(df_metrics)
            }
            print(f"\n   ✅ Walk-Forward Results:")
            print(f"      Avg Win Rate: {results['avg_metrics']['avg_win_rate']:.1f}%")
            print(f"      Avg Trades/Fold: {results['avg_metrics']['avg_trades_per_fold']:.1f}")
        return results

    def run_sensitivity_analysis(self, params: Dict) -> Dict:
        results = {}
        parameters = params.get('parameters', ['risk_per_trade', 'sl_multiplier', 'tp_multiplier'])
        variations = params.get('variations', [0.8, 0.9, 1.0, 1.1, 1.2])
        print(f"\n📈 Sensitivity Analysis pada {len(parameters)} parameter")
        sample_symbols = list(self.data_dict.keys())[:20]
        for param in parameters:
            results[param] = []
            base_value = getattr(self.base_config, param.upper(), 1.0)
            for var in variations:
                modified_value = base_value * var
                total_returns = []
                for symbol in sample_symbols[:5]:
                    df = self.data_dict.get(symbol)
                    if df is None:
                        continue
                    modified_config = self.base_config
                    setattr(modified_config, param.upper(), modified_value)
                    engine = self.engine_class(modified_config, None)
                    for i in range(200, len(df) - 20, 20):
                        signal = engine.get_signal(symbol, df.iloc[:i])
                        if signal:
                            future = df.iloc[i+20]['Close']
                            entry = df.iloc[i]['Close']
                            ret = (future / entry - 1) * 100
                            total_returns.append(ret)
                if total_returns:
                    results[param].append({
                        'multiplier': var,
                        'value': modified_value,
                        'mean_return': np.mean(total_returns),
                        'win_rate': np.mean(np.array(total_returns) > 0) * 100
                    })
            print(f"\n   {param.upper()}:")
            for r in results[param]:
                print(f"      {r['multiplier']:.1f}x -> Return: {r['mean_return']:.2f}%, WR: {r['win_rate']:.1f}%")
        return results

# =============================================================================
# 30. DATA WAREHOUSE INITIALIZATION
# =============================================================================

def initialize_data_warehouse():
    print("\n" + "="*80)
    print("🗄️  INISIALISASI DATA WAREHOUSE")
    print("="*80)
    print("1. Download data harga saham historis (jika belum ada)")
    print("2. Download data dividen (opsional, folder terpisah)")
    print("3. Download data fundamental (opsional, folder terpisah)")
    print("="*80)
    print(f"Total saham: {len(STOCKBIT_UNIVERSE)}")
    print(f"Minimal hari: 400 (saham dengan data kurang akan difilter)")

    warehouse = DataWarehouse(warehouse_dir='data_warehouse', min_days=400)

    existing_data = warehouse.get_all_valid_symbols()
    print(f"\n📊 Data harga yang sudah tersedia: {len(existing_data)} saham")

    print("\n" + "="*80)
    print("📈 DATA HARGA SAHAM")
    print("="*80)

    download_harga = 'n'
    if len(existing_data) >= 400:
        print(f"✅ Data harga sudah mencukupi ({len(existing_data)} saham)")
        download_harga = input("   Tetap download ulang data harga? (y/n): ").strip().lower()
    else:
        print("⚠️  Data harga belum lengkap")
        download_harga = input("   Download data harga? (y/n): ").strip().lower()

    if download_harga == 'y':
        print("\n⚠️  PERINGATAN: Download data harga akan memakan waktu BEBERAPA JAM!")
        confirm = input("Lanjutkan download data harga? (y/n): ").strip().lower()
        if confirm == 'y':
            print("\n📥 Mendownload data harga saham...")
            data = warehouse.download_complete_history(
                symbols=STOCKBIT_UNIVERSE,
                start_date='2018-01-01',
                end_date='2026-12-31'
            )
            print(f"✅ Selesai! Data harga tersedia untuk {len(data)} saham")
        else:
            print("⏩ Lewati download data harga")
    else:
        print("⏩ Menggunakan data harga yang sudah ada")

    print("\n" + "="*80)
    print("💰 DIVIDEN DATA")
    print("="*80)
    print("Data dividen disimpan di folder terpisah:")
    print("   data_warehouse/dividends/")
    print("TIDAK MENGGANGGU data harga saham utama")

    import glob
    existing_div = glob.glob(f"{warehouse.dividend_dir}/*_dividends.parquet")
    if existing_div:
        print(f"✅ Data dividen sudah tersedia untuk {len(existing_div)} saham")

    div_confirm = input("\nDownload data dividen untuk SEMUA saham? (y/n): ").strip().lower()

    if div_confirm == 'y':
        print(f"\n📥 Mendownload data dividen untuk {len(STOCKBIT_UNIVERSE)} saham...")
        print("   Proses ini akan memakan waktu ~30-60 menit...")
        dividend_results = warehouse.download_dividend_history(
            symbols=STOCKBIT_UNIVERSE,
            years_back=10
        )
        print(f"\n✅ Selesai! Data dividen tersedia untuk {len(dividend_results)} saham")
        print(f"   Saham tanpa data dividen: {len(STOCKBIT_UNIVERSE) - len(dividend_results)}")
    else:
        print("⏩ Lewati download dividen")

    print("\n" + "="*80)
    print("📊 FUNDAMENTAL DATA")
    print("="*80)
    print("Data fundamental disimpan di folder terpisah:")
    print("   data_warehouse/fundamental/")
    print("TIDAK MENGGANGGU data harga saham utama")

    existing_fund = glob.glob(f"{warehouse.fundamental_dir}/*_fundamental.parquet")
    if existing_fund:
        print(f"✅ Data fundamental sudah tersedia untuk {len(existing_fund)} saham")

    fund_confirm = input("\nDownload data fundamental (PER, PBV, ROE) untuk SEMUA saham? (y/n): ").strip().lower()

    if fund_confirm == 'y':
        print(f"\n📥 Mendownload data fundamental untuk {len(STOCKBIT_UNIVERSE)} saham...")
        print("   Proses ini akan memakan waktu ~30-60 menit...")
        fundamental_results = warehouse.download_fundamental_history(
            symbols=STOCKBIT_UNIVERSE,
            max_age_days=30
        )
        print(f"\n✅ Selesai! Data fundamental tersedia untuk {len(fundamental_results)} saham")
        print(f"   Saham tanpa data fundamental: {len(STOCKBIT_UNIVERSE) - len(fundamental_results)}")
    else:
        print("⏩ Lewati download fundamental")

    print("\n" + "="*80)
    print("📊 RINGKASAN AKHIR DATA WAREHOUSE")
    print("="*80)

    final_data = warehouse.get_all_valid_symbols()
    print(f"Data harga: {len(final_data)} saham valid (≥{warehouse.min_days} hari)")

    final_div = glob.glob(f"{warehouse.dividend_dir}/*_dividends.parquet")
    print(f"Data dividen: {len(final_div)} saham")

    final_fund = glob.glob(f"{warehouse.fundamental_dir}/*_fundamental.parquet")
    print(f"Data fundamental: {len(final_fund)} saham")

    if final_data:
        sample = final_data[0]
        df = warehouse.get_data(sample)
        if df is not None:
            print(f"Rentang tanggal harga: {df.index[0].date()} hingga {df.index[-1].date()}")

    print("="*80)

    if 'data' in locals():
        return data
    else:
        return warehouse.get_all_data()

# =============================================================================
# 30A. FUNGSI OPTIMASI PORTOFOLIO (BARU - PASCA SCANNING)
# =============================================================================

def optimize_portfolio(signals, price_data, modal, risk_manager, max_positions=5):
    """
    Optimasi alokasi portofolio menggunakan mean-variance dengan shrinkage.
    Dilakukan setelah scanning, tidak mempengaruhi kecepatan real-time.
    """
    if len(signals) < 2:
        return signals  # tidak bisa optimasi

    # Ambil sinyal dengan skor tertinggi (misal top 10)
    if signals[0].get('Final_Score') is not None:
        sorted_sigs = sorted(signals, key=lambda x: x.get('Final_Score', 0), reverse=True)[:10]
    else:
        sorted_sigs = sorted(signals, key=lambda x: x.get('Score', 0), reverse=True)[:10]

    symbols = [s['Symbol'] for s in sorted_sigs]
    # Ambil data harga untuk menghitung return
    returns_dict = {}
    for symbol in symbols:
        df = price_data.get(symbol)
        if df is not None and len(df) > 60:
            ret = df['Close'].pct_change().dropna().tail(60)
            returns_dict[symbol] = ret
    if len(returns_dict) < 2:
        return sorted_sigs[:3]  # fallback

    ret_df = pd.DataFrame(returns_dict)
    mean_returns = ret_df.mean() * 252  # annualized
    cov_matrix = ret_df.cov() * 252

    # Shrinkage Ledoit-Wolf sederhana
    try:
        lw_cov, _ = ledoit_wolf(ret_df)
        cov_matrix = pd.DataFrame(lw_cov, index=ret_df.columns, columns=ret_df.columns)
    except:
        pass

    # Optimasi: maksimalkan Sharpe ratio dengan batasan bobot
    from scipy.optimize import minimize

    num_assets = len(symbols)
    args = (mean_returns, cov_matrix)

    def neg_sharpe(weights, mean_returns, cov_matrix):
        port_return = np.sum(mean_returns * weights)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_std

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 0.4) for _ in range(num_assets))  # maks 40% per aset
    init_guess = [1/num_assets] * num_assets

    opt_results = minimize(neg_sharpe, init_guess, args=args,
                           method='SLSQP', bounds=bounds, constraints=constraints)

    if not opt_results.success:
        return sorted_sigs[:3]  # fallback

    weights = opt_results.x
    # Konversi bobot ke lot berdasarkan modal
    recommended = []
    for i, sig in enumerate(sorted_sigs):
        w = weights[i]
        if w < 0.05:  # minimal 5%
            continue
        # Hitung lot yang sesuai dengan bobot
        max_cost = modal * w
        price = sig['Price']
        lot = int(max_cost / (price * 100))
        if lot < 1:
            continue
        # Sesuaikan risk
        cost = lot * price * 100
        risk_amount = sig.get('Risk_Amount', 0)
        # Update sinyal dengan lot baru
        sig_copy = sig.copy()
        sig_copy['Lot'] = lot
        sig_copy['Cost'] = cost
        sig_copy['Risk_Amount'] = risk_amount * (lot / sig.get('Lot', 1))
        recommended.append(sig_copy)
        if len(recommended) >= max_positions:
            break
    return recommended

# =============================================================================
# 30B. ANALISIS SKENARIO MAKRO (UNTUK MENYESUAIKAN RISK MULTIPLIER)
# =============================================================================

def apply_macro_scenario(regime_detector, global_fetcher):
    """
    Menyesuaikan risk multiplier berdasarkan kondisi makro.
    Dipanggil setelah fetch global indices.
    """
    base_mult = 1.0
    if not regime_detector or not global_fetcher:
        return base_mult

    # Jika IHSG turun >10% dalam sebulan, kurangi risk
    ihsg_mom = global_fetcher.get_momentum('IHSG')
    if ihsg_mom < -10:
        base_mult *= 0.8
    elif ihsg_mom < -5:
        base_mult *= 0.9

    # Jika USD/IDR melemah >5% dalam sebulan, kurangi risk untuk sektor tertentu?
    # Tapi kita tidak perlu sampai ke level itu, cukup global
    usd_mom = global_fetcher.get_momentum('USDIDR')
    if usd_mom > 5:  # rupiah melemah
        # Biasanya sektor ekspor diuntungkan, tapi risiko makro meningkat
        base_mult *= 0.95

    return base_mult

# =============================================================================
# 30C. FUNGSI PERINGATAN KONSENTRASI SEKTOR & KORELASI
# =============================================================================

def check_sector_concentration(signals, price_data, portfolio_risk, threshold=0.7):
    """
    Memeriksa korelasi antar sinyal dan memberi peringatan jika terlalu tinggi.
    Menambahkan field 'correlation_warning' pada sinyal yang berisiko.
    """
    if len(signals) < 2:
        return signals
    # Hitung matriks korelasi
    symbols = [s['Symbol'] for s in signals]
    corr_matrix = portfolio_risk.calculate_correlation_matrix(symbols, price_data)
    if corr_matrix.empty:
        return signals
    # Beri peringatan untuk pasangan dengan korelasi > threshold
    warnings = defaultdict(list)
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            corr = corr_matrix.iloc[i, j]
            if not pd.isna(corr) and abs(corr) > threshold:
                warnings[symbols[i]].append(f"{symbols[j]}: {corr:.2f}")
                warnings[symbols[j]].append(f"{symbols[i]}: {corr:.2f}")
    # Tambahkan ke notes sinyal
    for sig in signals:
        if sig['Symbol'] in warnings:
            warning_str = "⚠️ Korelasi tinggi dengan " + ", ".join(warnings[sig['Symbol']])
            sig['correlation_warning'] = warning_str
    return signals

# =============================================================================
# 31. PHASE 1 - MAIN PROGRAM (DENGAN SEMUA PENINGKATAN)
# =============================================================================

def run_phase1(sheets_exporter):
    print("\n" + "="*60)
    print("🏦 IDX STOCK SCANNER - QUADRUPLE ENGINE (AGGRESSIVE ENHANCED)")
    print("   Modal-Adaptive Filters | Risk 3% per trade | Target 15-20% per tahun")
    print("   Fitur Baru: Turnover Filter, Optimasi Portofolio, Makro Scenario, Fundamental, Indeks Asia")
    print("="*60)

    warehouse = DataWarehouse(warehouse_dir='data_warehouse', min_days=400)
    symbols = warehouse.get_all_valid_symbols()

    if not symbols:
        print("\n⚠️  Data warehouse kosong atau tidak ditemukan!")
        print("   Jalankan opsi 3 (Inisialisasi Data Warehouse) terlebih dahulu.")
        return

    print(f"\n📊 Data warehouse siap: {len(symbols)} saham dengan data >= 400 hari")

    print("\nPilih engine trading:")
    print("1. Swing Engine (Mingguan - Mean Reversion) - Modal: Rp 40rb - 5jt")
    print("2. Gorengan Engine (Intraday - Early Momentum) - Modal: Rp 10rb - 500rb")
    print("3. Investasi Engine (Quality + Trend - Jangka Panjang) - Modal: Rp 100rb - 1M")

    while True:
        engine_choice = input("Pilihan (1/2/3): ").strip()
        if engine_choice in ['1', '2', '3']:
            break
        print("❌ Pilih 1, 2, atau 3")

    modal_ranges = {
        '1': (40000, 5000000, 'swing'),
        '2': (10000, 500000, 'gorengan'),
        '3': (100000, 1000000000, 'investasi')
    }

    min_modal, max_modal, engine_type = modal_ranges[engine_choice]

    # ===== INPUT MODAL BEBAS =====
    while True:
        try:
            modal_input = input(f"\nModal bebas (Rp {min_modal:,} - {max_modal:,}): ").strip()
            modal = int(modal_input.replace('.', '').replace(',', ''))
            if min_modal <= modal <= max_modal:
                break
            print(f"❌ Modal harus {min_modal:,} - {max_modal:,}")
        except Exception:
            print("❌ Input tidak valid")

    if engine_choice == '1':
        config = SwingConfig(modal)
        engine_name = "SWING ENGINE"
        EngineClass = SwingEngine
        timeframe = '1d'
    elif engine_choice == '2':
        config = GorenganConfig(modal)
        engine_name = "GORENGAN ENGINE"
        EngineClass = IntradayGorenganEngine
        timeframe = '1h'
    else:
        config = InvestasiConfig(modal)
        engine_name = "INVESTASI ENGINE"
        EngineClass = InvestasiEngine
        timeframe = '1d'

    modal_adapter = ModalAdapter(modal, engine_type)
    modal_adapter.print_info()

    print("\n🚀 Initializing Phase 1 Components...")

    risk_manager = RiskManager(
        modal=modal,
        risk_per_trade_pct=3.0,
        max_risk_portfolio_pct=15.0,
        max_lot_per_position=10,
        engine_type=engine_type
    )
    print(f"   ✅ Risk Manager: Rp {risk_manager.risk_per_trade_rp:,.0f} risk/trade (3.0%)")

    fee_config = RealisticFeeConfig(liquidity='medium')
    print(f"   ✅ Fee Config: min fee Rp {fee_config.MIN_FEE_PER_TRANSACTION:,} (termasuk VAT 11%)")

    global_fetcher = GlobalIndicesFetcher()
    global_fetcher.fetch_all()
    global_fetcher.print_detailed_report()

    # Inisialisasi news analyzer untuk indeks global (IHSG)
    NEWS_API_KEY = "75e932cca3c44eb68edf44aa81453bab"
    global_news_analyzer = GlobalNewsAnalyzer(api_key=NEWS_API_KEY, days_back=3, max_articles=10)

    regime_detector = MarketRegimeDetector(global_news_analyzer=global_news_analyzer)
    if 'IHSG' in global_fetcher.data:
        ihsg_data = global_fetcher.data['IHSG']
        current_regime, confidence = regime_detector.detect_regime(ihsg_data)
        # Terapkan skenario makro
        macro_mult = apply_macro_scenario(regime_detector, global_fetcher)
        if macro_mult != 1.0:
            regime_detector.confidence *= macro_mult
            print(f"\n🌍 Macro scenario applied: risk multiplier {macro_mult:.2f}")
        regime_detector.print_regime_report()
    else:
        print("\n⚠️  IHSG data tidak tersedia, regime detection skipped")

    delay_simulator = EntryDelaySimulator(max_delay=2)
    print(f"   ✅ Entry Delay Simulator: max delay {delay_simulator.max_delay} days (informational only)")

    portfolio_risk = PortfolioRiskCalculator(lookback_days=60)
    print(f"   ✅ Portfolio Risk Calculator: {portfolio_risk.lookback_days} days lookback with stress testing")

    # Inisialisasi news analyzer untuk saham
    stock_news_analyzer = NewsAnalyzer(api_key=NEWS_API_KEY, days_back=7, max_articles=10)

    # Inisialisasi engine dengan parameter baru
    engine = EngineClass(config, global_fetcher, news_analyzer=None)
    engine.set_risk_manager(risk_manager)
    engine.set_regime_detector(regime_detector)

    if engine_choice == '3':
        engine.set_warehouse(warehouse)

    print("\n" + "="*60)
    print(f"📊 {engine_name} - READY (AGGRESSIVE ENHANCED)")
    print("="*60)
    print(f"Modal: Rp {modal:,}")
    print(f"Risk per trade: Rp {risk_manager.risk_per_trade_rp:,.0f} (3.0% AGGRESSIVE)")
    print(f"Max portfolio risk: Rp {risk_manager.max_risk_portfolio_rp:,.0f} (15.0% AGGRESSIVE)")
    print(f"Min fee: Rp {fee_config.MIN_FEE_PER_TRANSACTION:,}")
    print(f"Universe: {len(symbols)} saham valid dari warehouse")
    print(f"📰 News Analyzer (saham): AKTIF (hanya untuk sinyal terbaik, hemat kuota)")

    print(f"\n📥 Menganalisis SEMUA {len(symbols)} saham...")

    stocks_data = {}
    for i, symbol in enumerate(symbols):
        df = warehouse.get_data(symbol)
        if df is not None:
            stocks_data[symbol] = df
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(symbols)} - {len(stocks_data)} dimuat")

    print(f"\n✅ Berhasil memuat {len(stocks_data)} saham dari warehouse")

    if stocks_data:
        print(f"\n📊 Menganalisis {len(stocks_data)} saham...")
        signals = []
        for symbol, df in stocks_data.items():
            signal = engine.get_signal(symbol, df)
            if signal:
                signals.append(signal)
        print(f"✅ Ditemukan {len(signals)} sinyal")

        if signals:
            # ===== REKOMENDASI ALOKASI ENGINE =====
            print("\n📊 REKOMENDASI ALOKASI ANTAR ENGINE (berdasarkan regime pasar):")
            if regime_detector.current_regime == "BEAR":
                print("   Bear market: Swing 20% | Gorengan 10% | Investasi 70%")
            elif regime_detector.current_regime == "BULL":
                print("   Bull market: Swing 40% | Gorengan 30% | Investasi 30%")
            elif regime_detector.current_regime == "SIDEWAYS":
                print("   Sideways: Swing 30% | Gorengan 20% | Investasi 50%")
            elif regime_detector.current_regime == "HIGH_VOL":
                print("   High volatility: Swing 15% | Gorengan 5% | Investasi 80%")
            else:
                print("   Regime unknown: Swing 33% | Gorengan 33% | Investasi 33%")
            print("   (Anda bebas menentukan sendiri, ini hanya panduan)")

            # ===== OPTIMASI NEWS ANALYZER: hanya untuk sinyal terbaik =====
            print("\n📰 Mengambil sentimen berita untuk sinyal terbaik (maks 20)...")
            # Urutkan sinyal berdasarkan skor
            if engine_choice == '3':
                sorted_signals = sorted(signals, key=lambda x: x.get('Final_Score', 0), reverse=True)
            else:
                sorted_signals = sorted(signals, key=lambda x: x.get('Score', 0), reverse=True)

            # Ambil maksimal 20 sinyal terbaik
            top_news_signals = sorted_signals[:20]

            # Update news untuk sinyal-sinyal tersebut
            for sig in top_news_signals:
                mult, label = stock_news_analyzer.get_multiplier_and_label(sig['Symbol'])
                sig['Confidence_Score'] = min(100, sig['Confidence_Score'] * mult)
                sig['News_Label'] = label
                sig['News_Multiplier'] = mult

            # Untuk sinyal lainnya, set label default 'netral'
            for sig in sorted_signals[20:]:
                sig['News_Label'] = 'netral'
                sig['News_Multiplier'] = 1.0

            # ===== PERINGATAN KONSENTRASI SEKTOR & KORELASI =====
            sorted_signals = check_sector_concentration(sorted_signals, stocks_data, portfolio_risk, threshold=0.7)

            print("\n" + "="*100)
            print("🌍 RINGKASAN PASAR")
            print("="*100)
            market_data = []
            for name in GLOBAL_INDICES.keys():
                mom = global_fetcher.get_momentum(name)
                trend = global_fetcher.get_trend(name)
                price_str = global_fetcher.get_price_str(name)
                market_data.append([name, price_str, f"{mom:+.2f}%", trend])
            print(tabulate(market_data, headers=["Indeks", "Harga", "Momentum", "Trend"], tablefmt="grid"))

            # ===== OPTIMASI PORTOFOLIO (pasca scanning) =====
            if len(sorted_signals) >= 3:
                optimized = optimize_portfolio(sorted_signals, stocks_data, modal, risk_manager, max_positions=3)
            else:
                optimized = sorted_signals[:3]

            print("\n" + "="*100)
            print(f"📊 {engine_name} - REKOMENDASI OPTIMAL (Top {len(optimized)} dari {len(signals)} sinyal)")
            print("="*100)

            # Tampilkan rekomendasi (sama seperti sebelumnya, tapi pakai optimized)
            top_5 = optimized[:5]  # untuk tampilan

            if engine_choice == '3':
                display_data = []
                for s in top_5:
                    target_str = f"{s.get('Target_Konservatif', '-')} | {s.get('Target_Moderat', '-')} | {s.get('Target_Agresif', '-')}"
                    display_data.append([
                        s['Symbol'],
                        s['Sector'],
                        f"Rp {s['Price']:,}",
                        f"{s['To_MA50']}",
                        f"Rp {s['MA50']:,}",
                        f"Rp {s['MA200']:,}",
                        f"Rp {s['Stop_Loss']:,}",
                        target_str,
                        s.get('Dividend_Display', 'N/A'),
                        f"{s['Confidence_Score']}%",
                        f"{s['Optimal_Hold_Days']}",
                        f"{s['Success_Rate']}%",
                        f"{s['Lot']} lot",
                        f"Rp {s['Cost']:,}",
                        s.get('Final_Score', 'N/A'),
                        s.get('News_Label', 'netral')
                    ])
                headers = [
                    "Kode", "Sektor", "Harga", "To MA50", "MA50", "MA200",
                    "Stop", "Target", "Div", "Conf", "Hold", "Sks%", "Lot", "Biaya", "Score", "News"
                ]
            else:
                display_data = []
                for s in top_5:
                    display_data.append([
                        s['Symbol'],
                        s['Sector'],
                        f"Rp {s['Price']:,}",
                        s.get('RSI', '-'),
                        s.get('Volume', '-'),
                        f"{s['R/R']:.2f}",
                        f"{s['EV_Pct']}%",
                        f"{s['Score']}",
                        f"{s['Confidence_Score']}%",
                        f"{s['Optimal_Hold_Days']}",
                        f"{s['Success_Rate']}%",
                        f"{s['ATR_Pct']}%",
                        f"Rp {s['Stop_Loss']:,}",
                        f"Rp {s['Take_Profit']:,}",
                        f"{s['Lot']} lot",
                        f"Rp {s['Cost']:,}",
                        s.get('News_Label', 'netral')
                    ])
                headers = [
                    "Kode", "Sektor", "Harga", "RSI", "Vol", "R/R",
                    "EV%", "Skor", "Conf%", "Hold", "Sukses%", "ATR%", "SL", "TP", "Lot", "Biaya", "News"
                ]

            print(tabulate(display_data, headers=headers, tablefmt='grid', stralign='left', numalign='center'))

            if engine_choice == '3':
                print_investasi_portfolio_guide(optimized, modal, risk_manager, portfolio_risk, stocks_data)
            else:
                print_portfolio_guide(optimized, modal, risk_manager, portfolio_risk, stocks_data)

            export_choice = input(f"\n📊 Export {engine_name} signals ke Google Sheets? (y/n): ").strip().lower()
            if export_choice == 'y':
                sheets_exporter.export_signals(optimized, engine_name, modal)

        else:
            print("\n❌ Tidak ada sinyal hari ini")
    else:
        print("\n❌ Tidak ada data yang berhasil dimuat dari warehouse")

    print("\n" + "="*60)
    print("✅ PHASE 1 COMPLETE - AGGRESSIVE ENHANCED")
    print("   Risk Management: 3% per trade, 15% portfolio")
    print("="*60)

# =============================================================================
# 32. PHASE 2 - MAIN PROGRAM (TIDAK BERUBAH)
# =============================================================================

def run_phase2():
    print("\n" + "="*80)
    print("📊 IDX STOCK SCANNER - PHASE 2 VALIDATION (AGGRESSIVE)")
    print("   Monte Carlo FIXED | Walk-Forward | Sensitivity | Risk 3%")
    print("="*80)

    warehouse = DataWarehouse(warehouse_dir='data_warehouse', min_days=400)
    symbols = warehouse.get_all_valid_symbols()

    if not symbols:
        print("\n⚠️  Data warehouse kosong!")
        print("   Jalankan opsi 3 (Inisialisasi Data Warehouse) terlebih dahulu.")
        return

    print(f"\n📥 Loading data dari warehouse ({len(symbols)} saham tersedia)...")

    print(f"\n📊 Pilih jumlah saham untuk validasi:")
    print(f"   1. Semua ({len(symbols)} saham)")
    print(f"   2. Sample 200 saham")
    print(f"   3. Sample 100 saham")

    while True:
        sample_choice = input("Pilihan (1/2/3): ").strip()
        if sample_choice == '1':
            test_symbols = symbols
            break
        elif sample_choice == '2':
            test_symbols = symbols[:200]
            break
        elif sample_choice == '3':
            test_symbols = symbols[:100]
            break
        else:
            print("❌ Pilih 1, 2, atau 3")

    data_dict = {}
    for i, symbol in enumerate(test_symbols):
        df = warehouse.get_data(symbol)
        if df is not None:
            data_dict[symbol] = df
        if (i+1) % 50 == 0:
            print(f"   Loaded {i+1}/{len(test_symbols)} - {len(data_dict)} valid")

    print(f"\n✅ Loaded {len(data_dict)} stocks dengan data lengkap dari warehouse")

    if not data_dict:
        print("❌ No data loaded. Cannot proceed.")
        return

    print("\nPilih engine untuk divalidasi:")
    print("1. Swing Engine")
    print("2. Gorengan Engine")
    print("3. Investasi Engine")

    while True:
        choice = input("Pilihan (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("❌ Pilih 1, 2, atau 3")

    if choice == '1':
        engine_class = SwingEngine
        config = SwingConfig(modal=100000000)
        engine_name = "SWING ENGINE"
    elif choice == '2':
        engine_class = IntradayGorenganEngine
        config = GorenganConfig(modal=100000000)
        engine_name = "GORENGAN ENGINE"
    else:
        engine_class = InvestasiEngine
        config = InvestasiConfig(modal=100000000)
        engine_name = "INVESTASI ENGINE"

    print(f"\n🔧 Validating: {engine_name} dengan {len(data_dict)} saham (AGGRESSIVE 3%)")

    suite = ValidationSuite(engine_class, config, data_dict)
    results = suite.run_all()

    print("\n" + "="*80)
    print("✅ PHASE 2 VALIDATION COMPLETE - AGGRESSIVE")
    print("   File output: monte_carlo_optimal_aggressive.png")
    print("="*80)

# =============================================================================
# 33. MAIN MENU
# =============================================================================

def main():
    print("\n" + "="*70)
    print("🏦 IDX STOCK SCANNER - QUADRUPLE ENGINE (AGGRESSIVE ENHANCED)")
    print("   FULL SYSTEM DENGAN DATA WAREHOUSE")
    print("   Modal-Adaptive Filters | Risk 3% per trade | Target 15-20% per tahun")
    print("   Fitur Baru: Turnover Filter, Optimasi Portofolio, Makro Scenario, Fundamental, Indeks Asia")
    print("="*70)

    sheets_exporter = GoogleSheetsExporter()
    sheets_exporter.ensure_spreadsheet_exists()

    print("\nPilih Mode:")
    print("1. Phase 1 - Trading Scanner (Live Signals) - AGGRESSIVE")
    print("2. Phase 2 - Validation Suite (Monte Carlo Optimal) - AGGRESSIVE")
    print("3. Inisialisasi Data Warehouse (Download historis lengkap)")
    print("4. Exit")

    while True:
        choice = input("\nPilihan (1/2/3/4): ").strip()
        if choice == '1':
            run_phase1(sheets_exporter)
            break
        elif choice == '2':
            run_phase2()
            break
        elif choice == '3':
            initialize_data_warehouse()
            break
        elif choice == '4':
            print("\n✅ Exiting...")
            break
        else:
            print("❌ Pilih 1, 2, 3, atau 4")

if __name__ == "__main__":
    main()
