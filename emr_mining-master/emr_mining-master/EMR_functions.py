# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import sys
sys.path.append('/exports/reum/tdmaarseveen/modules/')
import re
from acronyms_old import ACRONYM_MAP
from acronyms_old import TYPO_ACRONYM_MAP
from acronyms_old import PARENT_MAP
from acronyms_old import MEDICINE, OTHER_MEDICINE, TYPO_MED
import pandas as pd
import datetime
import io
import numpy as np
import unicodedata
import random
import pattern.nl as patNL
from input_files.datecols import DATECOLS # datecols from Tinekes dataset
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool
from bokeh.palettes import viridis
from bokeh.models.sources import ColumnDataSource
import xml.etree.ElementTree
import pyConTextNLP.pyConTextNLP.pyConText as pyConText
import pyConTextNLP.pyConTextNLP.itemData as itemData
from textblob import TextBlob
#For Visualisation:     from IPython.display import display, HTML
#import pyConTextNLP.pyConTextNLP.display.html as html

def overviewPerformanceDict(d_perf):
    """
    generate a summary off the performance
    """
    accuracy = ( d_perf['true_pos'] + d_perf['true_neg']) / sum(d_perf.values())
    ppv = d_perf['true_pos'] / (d_perf['true_pos'] + d_perf['false_pos']) 
    npv = d_perf['true_neg'] / (d_perf['true_neg'] + d_perf['false_neg']) 
    sensitivity = d_perf['true_pos'] / (d_perf['true_pos'] + d_perf['false_neg'])
    specificity = d_perf['true_neg'] / (d_perf['true_neg'] + d_perf['false_pos']) 
    print('PPV:\t\t' + str(ppv) + '\nNPV:\t\t' + str(npv) + '\nSensitivity:\t' + str(sensitivity) + \
         '\nSpecificity:\t' + str(specificity) + '\nAccuracy:\t' + str(accuracy))
    return

def overviewPerformanceList(l_perf):
    """
    generate a summary off the performance
    """
    if (l_perf[0] + l_perf[1]) > 0 :
        accuracy = ( l_perf[0] + l_perf[1]) / sum(l_perf)
    else : 
        accuracy = 0.0
    if (l_perf[0] + l_perf[2]) > 0 :
        ppv = l_perf[0] / (l_perf[0] + l_perf[2]) 
    else : 
        ppv = 0.0
    if (l_perf[1] + l_perf[3]) > 0 :
        npv = l_perf[1] / (l_perf[1] + l_perf[3]) 
    else : 
        npv = 0.0
    if (l_perf[0] + l_perf[3]) > 0 :
        sensitivity = l_perf[0] / (l_perf[0] + l_perf[3])
    else : 
        sensitivity = 0.0
    if (l_perf[1] + l_perf[2])  > 0 :
        specificity = l_perf[1] / (l_perf[1] + l_perf[2]) 
    else :
        specificity = 0.0
    return [ppv, npv, sensitivity, specificity, accuracy]

def getMedicineSubset(df):
    """
    This function creates a medicine subset.
    """
    l_med = [s for s in df['XANTWOORD'] if any(xs in s for xs in MEDICINE)]
    df_new = df[df['XANTWOORD'].isin(l_med)]
    return df_new

def build_convenient_list(f_name, delim, encod):
    tot_list = []
    if encod == "ascii":
        f = io.open(f_name, mode='r')
    else :
        f = io.open(f_name, mode='r', encoding= encod)
    content = f.read()
    for x in content.split('[report_end]'):
        tot_list.append(x.split("|"))
    f.close()
    return tot_list

def makeFile(f_name, content):
    txt_allopt = open(r'output_files/' + f_name, "w")
    txt_allopt.write(content)
    txt_allopt.close()
    return

def convert_to_ascii(f_input, f_output):
    f = io.open(f_input, mode='r', encoding="utf-16")
    content = f.read()
    makeFile(f_output, content)
    f.close()
    return 

def list_to_df(conv_list):
    headers = conv_list.pop(0)
    df = pd.DataFrame(conv_list, columns=headers)
    return df

def determineIntersection(s1, s2):
    # Input = column (Pandas Series)
    a1 = s1.unique()
    a1 = a1[a1 != np.array(None)]
    a2 = s2.unique()
    a2 = a2[a2 != np.array(None)]
    return np.intersect1d(a1, a2)
    
def determineDiff(s1, s2):
    a1 = s1.unique()
    a1 = a1[a1 != np.array(None)]
    a2 = s2.unique()
    a2 = a2[a2 != np.array(None)]
    return np.setdiff1d(a1, a2)

def read_csv(f_name):
    return pd.read_csv(f_name, sep="|")

def appendValueDict(d, key, value):
    """
        d = dict
        value = new value
    """
    if key in d:
        d[key].append(str(value))
    else:
        d[key] = [str(value)]
    return d

def determineFirstDate(df):
    abs_min=''
    for row in range(len(df)):
      s1 = df.loc[row, DATECOLS].values
      if abs_min == '':
        abs_min = min(d for d in s1 if isinstance(d, datetime.date))
      elif abs_min > min(d for d in s1 if isinstance(d, datetime.date)):
        abs_min = min(d for d in s1 if isinstance(d, datetime.date))
    return abs_min

def midasTouch(df, path=r'input_files/EMR_goldenstandard.csv'):
    """
    Seperates the patients from the golden standard from the test & 
    trainingsset (df). Only the candidates with a corresponding follow-up
    will be selected as golden standard material. 
    
    This function also creates a new dataframe consisting of the extracted 
    'golden' patients and deletes the extracted patients
    from the originating dataframe (df).
    
    path = path to the golden standard file
    df_gold = dataframe consisting of all the golden standard patients
    df = dataframe of which the golden patients will be extracted
    df_extract = dataframe consisting of the patients extracted from df
    l_putative_patients = list of patients from the golden standard that
        are also present in the df
    l_golden_patients = list of patients from the golden standard that
        are also present in the df and contain a corresponding follow-up
    """ 
    df_gold = read_csv(path) # Golden Standard
    df_gold = df_gold.replace(r'^\s+$', np.nan, regex=True)
    df_gold['zkhpatnr'] = df_gold['zkhpatnr'].fillna(-1)
    df_gold['zkhpatnr'] = df_gold['zkhpatnr'].astype(int)
    
    l_putative_patients = determineIntersection(df_gold['zkhpatnr'], df['PATNR'])
    df_extract = correspondingFollowUp(df.loc[df['PATNR'].isin(l_putative_patients)], df_gold)
    
    print(determineFirstDate(df_gold))
    l_golden_patients = determineIntersection(df_gold['zkhpatnr'], df_extract['PATNR'])
    #print(determineDiff(pd.Series(l_putative_patients.tolist()), pd.Series(l_golden_patients.tolist())))
    # df = df.loc[~df['PATNR'].isin(l_golden_patients)]
    df = df.loc[~df['PATNR'].isin(l_putative_patients)] # 
    df_extract.to_csv('output_files/DF_gold.csv', sep='|', index=True)
    return df, df_extract

def correspondingFollowUp(df_extract, df_gold):
    """
    This method deletes all rows of entries that are not included in the
    follow up of the golden standard of Tineke but which are included in the
    DDR_A Hix table. So every row where the date in Hix exceeds the last 
    noted date in the EMR_goldenstandard. 
    
    This function also removes some of the typos in the golden standard.
    
    Variables:
    df_gold = the golden standard of Tineke
    df_extract = extract of DDR_A consisting of patients that are also
        found in the golden standard of Tineke
    df_extract_pat = subset of df_extract containing 1 individual patient
    df_extract_corr = a corrected version of df_extract which only
        contains the necessary rows (those within the follow-up of df_gold)
    """
    df_gold.loc[142, 'mtx_start1'] = '3/22/2004' # 3004
    df_gold.loc[561, 'mtx_start1'] = '2/13/2014' # 3014
    df_gold.loc[184, 'mtx_start2'] = '10/21/2015' # 20115
    df_gold.loc[602, 'mtx_start2'] = '2/23/2015' # 3015
    
    for x in DATECOLS:
        df_gold[x].replace(' ', np.nan, inplace=True)
        df_gold[x] = pd.DatetimeIndex(pd.to_datetime(df_gold[x], errors='ignore')).tz_localize('UTC')
    
    df_extract = df_extract.reset_index(drop=True)
    df_extract['DATUM'] = pd.DatetimeIndex(pd.to_datetime(df_extract['DATUM'])).tz_localize('UTC')
    df_extract_corr = df_extract.copy()
    l_exceeding = []
    
    for row in range(len(df_gold)):
        s1 = df_gold.loc[row, DATECOLS].values
        pat = df_gold.loc[row, 'zkhpatnr']
        if pat == " ":
            pat = '0'
        df_extract_pat = df_extract[df_extract['PATNR'] == int(pat)]
        mask = (df_extract_pat['DATUM'] <= max(d for d in s1 if isinstance(d, datetime.date))) & (df_extract_pat['DATUM'] >= min(d for d in s1 if isinstance(d, datetime.date)))
        l_exceeding.extend(list(df_extract_pat.loc[~mask].index.values))
    df_extract_corr = df_extract_corr.drop(df_extract_corr.index[l_exceeding])
    return df_extract_corr

def createSubsets(df_total, seed, perc_exp, perc_typo):
    """
    Create random subset of patients
    Divide df_total into:
        - an exploration set (df_explore)
        - a typo validation set (df_typo)
        - a test set (df_test)
        - a context explortation set (df_context)
    
    The size of df_extract depends on the parameter perc_exp, whereas the
    size of df_typo_val depends on the parameter perc_typo. These 
    parameters store the percentages. 
    
    Variables:
        num_explore = number of patients to select for the explore set
        num_typo = number of patients to select for the typo validation set
        num_context = number of patients to select for the context 
            explore set
        num_cont_val = number of patients to select for the context 
            validation set
    """
    num_explore = round(len(df_total['PATNR'].unique())*perc_exp)
    num_typo = round(len(df_total['PATNR'].unique())*perc_typo)
    total_num = num_explore + num_typo
    random.seed(seed)
    l_pat_random = random.sample(list(df_total['PATNR'].unique()), total_num)
    df_typo = df_total.loc[df_total['PATNR'].isin(l_pat_random[:num_explore])]
    df_explore = df_total.loc[df_total['PATNR'].isin(l_pat_random[
            num_explore: (num_explore + num_typo)])]
    df_test = df_total.loc[~df_total['PATNR'].isin(l_pat_random)]
    
    df_typo.to_csv('output_files/DF_typoValidation.csv', sep='|', index=True)
    df_test.to_csv('output_files/DF_test.csv', sep='|', index=True)
    df_explore.to_csv('output_files/DF_explore.csv', sep='|', index=True)
    return df_test, df_typo, df_explore

def lemmatizingText(sentence):
    return ' '.join(patNL.Sentence(patNL.parse(sentence, lemmata=True)).lemmata)
    

class Processing(object):
    """
        Input: df (dataframe) -> exploration or testset
    """
    def __init__(self, df):
        self.df = df
        self.l_typo = []
        self.l_score_typo = [0, 0, 0, 0, 0, 0]
        self.d_cutoff = {}
        self.getDictCutOff(1)
        self.d_typo_rows = {}

    def getDF(self):
        return self.df
    
    def setDF(self, df):
        self.df = df
        return
    
    def getPatients(self, col='PATNR'):
        return len(self.df[col].unique())
    
    def getDim(self):
        return "Rows: " + str(self.df.shape[0]) + "\tColumns: " + str(self.df.shape[1])

    def getHeader(self):
        return list(self.df.columns.values)

    def getTypos(self):
        return self.l_typo
    
    def getDicTypos(self):
        return self.d_typo_rows
    
    def getDictCutOff(self, init=0):
        try:
            df_CutOff = read_csv(r'input_files/DF_CutOff.csv')
            for x in df_CutOff:
                #print("Cut off " + x + " =\t " + str(df_CutOff[x].idxmin()+1) + " (" + str(df_CutOff[x].min()) + ")")
                self.d_cutoff[x] = int(df_CutOff[x].idxmin()+1)
        except:
            if init == 0:
                print("ERROR: There is no DF_CutOff -> you probably have to" + \
                      " generate a new DF_CutOff with the" + \
                      " generateTypoCuttOffPlot() function")
        return self.d_cutoff
        
    def isDigit(self, string):
        try: 
            int(string)
            return True
        except ValueError:
            return False
    
    def removeRTF(self, column):
        #df_rtf = self.df[self.df[column].str.contains(r"^\{.*\}.*", na=False)]
        self.df = self.df[~self.df[column].str.contains(r"^\{.*\}.*", na=False)]
        #df_rtf.to_csv(r'output_files/DF_REUBEL_RTF.csv', sep='|', index=False)
        return
    
    def removeAccent(self, text):
        """
        This function removes the accent of characters
        """
        try:
            text = unicode(text, 'utf-8')
        except NameError: # unicode is a default on python 3 
            pass
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return str(text)
    
    def processTableREU(self):
        self.df['DATUM'] = pd.DatetimeIndex(pd.to_datetime(self.df['DATUM'])).tz_localize('UTC')
        self.df = self.df.assign(DUURTOT=self.df.sort_values(['PATNR', 'DATUM']).groupby('PATNR')['DATUM'].transform(lambda x: x.iat[-1] - x.iat[0]))
        
        self.df['DUURTOT'] = self.df['DUURTOT'].dt.days.astype(int)
        self.df['XANTWOORD']= self.df['XANTWOORD'].str.lower()
        
        self.df['XANTWOORD'] = self.df['XANTWOORD'].replace('', np.nan)
        self.df['STELLING'] = self.df['STELLING'].replace('', np.nan)
        self.df['DOSSIERID'] = self.df['DOSSIERID'].replace('', np.nan)
        
        self.df = self.df.dropna(subset=['XANTWOORD', 'STELLING', 'DOSSIERID'])
        self.df.to_csv(r'output_files/DF_REU.csv', sep='|', index=False)
        return
    
    def createTableREU(self):
        self.df = self.df.loc[self.df['CATEGORIE'] == 'REU'] 
        self.processTableREU()
        return
    
    def handleAcronym(self, text, acronym_mapping=ACRONYM_MAP):
        """
        This function searches for acronyms in the text by checking
        every word. Once a word is found, the acronym is converted
        to the full medication word.
        
        The updated text (expanded_text) is then returned as output.
        """
        acronyms_pattern = re.compile(r'\b({})\b'.format('|'.join(acronym_mapping.keys())), 
                                          flags=re.IGNORECASE|re.DOTALL)
        def expand_match(acronym):
            match = acronym.group(0)
            expanded_acronym = acronym_mapping.get(match)
            return expanded_acronym
    
        expanded_text = acronyms_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return str(expanded_text)
    
    def handleTypoAcronym(self, text, acronym_mapping, row_nr):
        """
        This function searches for acronyms in the text by checking
        every word. Once a word is found, the acronym is converted
        to the full medication word.
        
        The updated text (expanded_text) is then returned as output.
        
        Variables:
        self.d_typo_rows = dictionary keeping track of the found typos 
            per row
        """
        acronyms_pattern = re.compile(r'\b({})\b'.format('|'.join(acronym_mapping.keys())), 
                                          flags=re.IGNORECASE|re.DOTALL)
        def expand_match(acronym):
            match = acronym.group(0)
            expanded_acronym = acronym_mapping.get(match)
            self.d_typo_rows = appendValueDict(self.d_typo_rows, expanded_acronym, row_nr)
            return expanded_acronym
        
        expanded_text = acronyms_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return str(expanded_text)
    
    def handleStickyChars(self, text):
        """
        At first this function surrounds all sticky characters with spaces.
        This is ultimately to prevent the algorithm from misinterpreting
        text (like medication or negation): mtx.depo -> mtx . depo
        
        After the splitting on sticky characters the algorithm will 
        assemble the special cases where a sticky character is allowed! 
        
        Dots and commas between integers will be put together: 
            7 , 5 mg -> 7,5 mg
            
        Medication with special/sticky characters in them will be assembled
            as well:
            (1) cortico ' s -> cortico's (yes)
            (2) methotrexaat ' -> methotrexaat ' (no)
            (3) anti - tnf -> anti-tnf (yes)
            (4) mtx - sasp -> mtx - sasp (no)
            
        Only medications that stick to a hyphen will be seperated with spaces: 
            mtx-infliximab -> mtx - infliximab 
        
        The function also appends an whitespace to digits: 5mtx -> 5 mtx. 
        Except when there is an acronym like anti-il1 which consists of an
        integer!
            
        Finally, this function removes all of the duplicate whitespace. 
        """
        sticky_chars = r'([!#?,.:";@\-\+\\/&=$\]\[<>\'\*`â€™\(\)])'
        words = text.split() 
        new_words = []
        for word in words:
            new_words.append(re.sub(sticky_chars, r' \1 ', word))
        new_text = ""
        for word in new_words:
            pat1 = re.compile(r'([a-z]{1,})(\d{1,})')
            for r1, r2 in pat1.findall(word):
                if r1+r2 not in MEDICINE and r1+r2 not in ACRONYM_MAP.keys():
                    word = re.sub(r1+r2," " + r1 + " " + r2 + " ", word)
            pat2 = re.compile(r'([a-z]{1,})\s'+sticky_chars+'\s([a-z]{1,})')
            for r1, r2, r3 in pat2.findall(word):
                if r1+r2+r3 in MEDICINE or r1+r2+r3 in ACRONYM_MAP.keys():
                    word = re.sub(r1 + " " + r2 + " " + r3," " + r1+r2+r3+ " ", word)
            pat3 = re.compile(r'(\d{1,})\s([,.])\s(\d{1,})')
            for r1, r2, r3 in pat3.findall(word):
                word = re.sub(r1 + " " + r2 + " " + r3, " "+ r1+r2+r3 + " ", word)
            word = re.sub(r'(\d{1,})([a-z]{1,})', " " + r'\1' + ' ' + r'\2' " ", word)    
            new_text += word + " "
        new_text = re.sub(r'\s+', ' ', new_text)      
        return new_text
    
    def createTableREUBEL(self):
        self.df = self.df[self.df['STELLING'].isin(['Beleid'])]
        self.removeRTF('XANTWOORD')
        self.df['XANTWOORD'] = self.df['XANTWOORD'].apply(lambda x: self.removeAccent(str(x)))
        self.df['XANTWOORD'] = self.df['XANTWOORD'].apply(lambda x: self.handleStickyChars(str(x)))
        self.df.to_csv(r'output_files/DF_REUBEL.csv', sep='|', index=False)
        return
    
    def createTableREUCON(self):
        self.df = self.df[self.df['STELLING'].isin(['Conclusie'])]
        self.removeRTF('XANTWOORD')
        self.df['XANTWOORD'] = self.df['XANTWOORD'].apply(lambda x: self.removeAccent(str(x)))
        self.df['XANTWOORD'] = self.df['XANTWOORD'].apply(lambda x: self.handleStickyChars(str(x)))
        self.df.to_csv(r'output_files/DF_REUCON.csv', sep='|', index=False)
        return
    
    def splitDatasetIntoCategories(self, frac=0.5, cutoff=0, seed=777):
        """
        This function divides the data into three categories, and 
            creates three random samples (50% of rows) out of those selections:
            
            1. Entries with typos   -> Sample 2
            2. Entries with medication and no typos     -> Sample 4
            3. Entries without medication   -> Sample 3
        
        In the case of sample2 and sample4, there exists both a raw and 
        a corrected version. sample2 consists of the same 
        rows as sample2_raw except for the fact that the typos are corrected.
        In the corrected version of Sample 4 the full name of the 
        supposed medication is written down.
        
        These three samples can be used to evaluate the efficacy of the 
        typo algorithm
        
        Variables:
            self.df = exploration set or testset
            frac = the fraction/percentage of the subset that you 
                want to sample. The default = 50%
            cutoff = indicates wheter or not the cutoffs have to be 
                determined
            seed = random state used to create the samples
            
        """
        if (cutoff == 0):
            self.generateTypoCuttOffPlot(self.df)
            self.getDictCutOff()
        df_corrected, l_typo_found = self.typoProcessing('XANTWOORD')
        df_typo_rows = df_corrected.iloc[l_typo_found]
        
        print(len(df_typo_rows))
        # Sample 2
        sample2 = df_typo_rows.sample(n=round(len(df_typo_rows)*frac), random_state=seed)
        sample2[['PATNR', 'DATUM', 'XANTWOORD']].to_csv(r'output_files/Sample2_corrected.csv', sep='|', index=True)
        sample2_raw = self.df.loc[sample2.index.values][['PATNR', 'DATUM', 'XANTWOORD']]
        sample2_raw.to_csv(r'output_files/Sample2_raw.csv', sep='|', index=True)
        df_corrected.to_csv(r'output_files/DF_corrected_full.csv', sep='|', index=True)
        
        df_remain = df_corrected.drop(df_typo_rows.index.values) 
        l_med = self.getRowsMedicine(df_remain)
        df_nomed = df_remain.drop(l_med) 
        df_notypo = df_remain.loc[l_med]
        
        # Sample 3
        sample3 = df_nomed.sample(n=round(len(df_nomed)*frac), random_state=seed)
        sample3[['PATNR', 'DATUM', 'XANTWOORD']].to_csv('output_files/Sample3.csv', sep='|', index=True)
        
        # Sample 4
        sample4 = df_notypo.sample(n=round(len(df_notypo)*frac), random_state=seed)
        sample4_raw = self.df.loc[sample4.index.values][['PATNR', 'DATUM', 'XANTWOORD']]
        
        sample4[['PATNR', 'DATUM', 'XANTWOORD']].to_csv('output_files/Sample4.csv', sep='|', index=True)
        sample4_raw.to_csv('output_files/Sample4_raw.csv', sep='|', index=True)
        return
    
    def getRowsMedicine(self, df):
        """
        """
        l_med = []
        for index, row in df.iterrows():
            xant = row['XANTWOORD']
            if any(xs in xant for xs in MEDICINE):
                l_med.append(index)
        return l_med
    
    def damerauLevenshtein(self, s, t):
        """
        Calculates the levenshtein score, which is the total sum of
        substitutions, insertions, deletions and (small) transpositions.
        https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
        """
        d = {}
        lenstr1 = len(s)
        lenstr2 = len(t)
        for i in range(-1,lenstr1+1):
            d[(i,-1)] = i+1
        for j in range(-1,lenstr2+1):
            d[(-1,j)] = j+1
     
        for i in range(lenstr1):
            for j in range(lenstr2):
                if s[i] == t[j]:
                    cost = 0
                else:
                    cost = 1
                d[(i,j)] = min(
                               d[(i-1,j)] + 1, # deletion
                               d[(i,j-1)] + 1, # insertion
                               d[(i-1,j-1)] + cost, # substitution
                              )
                if i and j and s[i]==t[j-1] and s[i-1] == t[j]:
                    d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
     
        return d[lenstr1-1,lenstr2-1]
    
    def typosPerCategory(self, text, word):
        """
            For Analyzing:
                Determines the amount of typos per Category
        """
        factor_word = int(len(word)*1/6+1) # vanaf 6 letters naar boven, standaard van 1
        for x in re.split(' |\n',str(text)):
            if len(x) > len(word)-factor_word and len(x) < len(word)+factor_word \
            and self.isDigit(x) == False and str(x) not in ACRONYM_MAP.keys() \
            and str(x) not in MEDICINE and str(x): # NOT in MEDICINE
                score = self.damerauLevenshtein(x, word)
                if score < factor_word+1 and score != 0:
                    #print(str(x) + "; " + str(word) + "; " + str(score))
                    self.l_typo.append(str(x) + " : " + str(word))
        return text
    
    def typoAnalyzing(self, column, medication_list):
        """
            For Analyzing:
                Determines the amount of typos per Medicine
        """
        self.l_typo = []
        for y in medication_list:
            self.df[column] = self.df[column].apply(lambda x : self.typosPerCategory(x, y))
        return
    
    def correctTypos(self, text, row_nr):
        """
            med_list = selection of medication
            
            d_typos = dictionary featuring all of the found typos per
                sentence and is used to determine the closest neighbour:
                
                    galimumab : adalimumab (levenshtein: 2)
                    galimumab : golimumab (levenshtein: 1)
                
                In the example above the function picks golimumab as
                the closest neighbour.
                    
            Not checking for 'goud' currently. -> word is too small
            
            The typos in Acronyms are also corrected
        """
        exceptions = ['dmards', 'biologicals']
        d_typos = {}
        for x in re.split(' |\n',str(text)):
            for word in TYPO_MED:
                factor_word = self.d_cutoff[word]
                #factor_word = int(len(word)*1/6+1) # vanaf 6 letters naar boven, standaard van 1
                if len(x) > len(word)-factor_word and len(x) < len(word)+factor_word \
                and self.isDigit(x) == False and str(x) not in OTHER_MEDICINE \
                and str(x) not in MEDICINE : # NOT in MEDICINE
                    score = self.damerauLevenshtein(x, word)
                    if score < factor_word+1 and score != 0 and str(x) not in exceptions:
                        if str(x) in d_typos:
                            if word != d_typos[str(x)] and score < self.damerauLevenshtein(x, d_typos[str(x)]):
                                d_typos[str(x)] = str(word)
                        else:
                            d_typos[str(x)] = str(word)
                        #if score == factor_word or score == factor_word+1: # LATER VERWIJDEREN
                        #print(str(x) + "; " + str(word) + "; " + str(score) + "; " + str(score-factor_word) + "\n")
        for x in d_typos.keys():
            text = text.replace(str(x), d_typos[str(x)])
            self.d_typo_rows = appendValueDict(self.d_typo_rows, d_typos[str(x)], row_nr)
        text = self.handleTypoAcronym(text, TYPO_ACRONYM_MAP, row_nr)
        text = self.handleAcronym(text)
        text = self.handleAcronym(text, PARENT_MAP) # convert to Parent
        return text


    def typoProcessing(self, column):
        """
            d_typo_rows = dictionary featuring the row numbers where a typo
                is corrected per medication (key)
            l_typo_found = list with all rows where a typo was found 
                (for creating sample 1 (and ultimatelly sample 3 and 4))
            Uiteindelijk wil je gaan selecteren op de rijnummers
        """
        self.d_typo_rows = {}
        l_typo_found = []
        df_corrected = self.df.copy()
        #for row in df_corrected[column].iterrows():
        row_nr = 0
        for row in df_corrected[column]:
            df_corrected.iat[row_nr, df_corrected.columns.get_loc('XANTWOORD')] = self.correctTypos(row, row_nr)
            row_nr += 1
        #df_corrected[column], d_typo_rows = self.df[column].apply(
        #    lambda x : self.correctTypos(x, d_typo_rows, x.name))
        df_typoCat = pd.DataFrame(columns = MEDICINE)
        df_typoCat.append(pd.Series([np.nan]), ignore_index = True)
        df_typoCat.loc[0] = [0 for n in range(len(MEDICINE))]
        for key in self.d_typo_rows:
            df_typoCat[key][0] =str(len(self.d_typo_rows[key]))
            df_typoCat.to_csv(r'output_files/DF_TyposPerCat.csv', sep='|', index=False)
            l_typo_found.extend(self.d_typo_rows[key])
            print(str(key) + " Calculated amount of Typos: " + str(len(self.d_typo_rows[key])))
        l_typo_found = set(l_typo_found)
        l_typo_found = list(l_typo_found)
        return df_corrected, l_typo_found

    def calculateTyposInAll(self, text, word):
        """
            Count all posibilities of variations -> no stringent threshold.
            This can be useful to determine the cut-off for typos
        """
        for x in re.split(' |\n',str(text)):
            if self.isDigit(x) == False and str(x) not in MEDICINE : # NOT in MEDICINE
                score = self.damerauLevenshtein(x, word)
                if score > 0 and score < 7:
                    self.l_score_typo[(score-1)] = int(self.l_score_typo[(score-1)])+1
        return
    
    def variationsPerScore(self, df, column, med): # column, medication_list
        self.l_score_typo = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        df[column].apply(lambda x : self.calculateTyposInAll(x, med))
        return self.l_score_typo
    
    def normalizeScores(self, l_score_typo):
        n_list = []
        if sum(l_score_typo) != 0:
            for y in l_score_typo:
                n_list.append(int(y) / sum(l_score_typo))
        elif len(l_score_typo) == 6:
            n_list =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else : 
            n_list =[0.0, 0.0, 0.0, 0.0, 0.0]
        return n_list 
    
    def createCategoryPlot(self, df, label, sum_list, p_title, lbl_x, lbl_y):
        p = figure(plot_width=1000, plot_height=600, title=p_title, x_axis_label=lbl_x, y_axis_label=lbl_y)
        numlines=len(df.columns)
        mypalette=viridis(len(TYPO_MED))
        for (leg, x, y, colnr ) in zip(label,[df.index.values]*numlines, [df[name].values for name in df], range(0,numlines)):
            source = ColumnDataSource(data=dict(
                x=x+1,
                y=y,
                freq=y*sum_list[colnr],
                desc=[leg]*len(df.index.values),
            ))
            p.line('x', 'y', line_color=mypalette[colnr], line_width=5, legend=leg + " : " + str(sum_list[colnr]), source=source)
        p.add_tools(HoverTool(
            tooltips = [
            ("index", '@desc'),
            ("freq", '@freq'),
            ("(x,y)", "(@x, @y)")
            ]
        ))
        p.legend.click_policy="hide"
        return p
    
    def generateTypoCuttOffPlot(self, df):
        typo_df = pd.DataFrame(columns = TYPO_MED)
        typo_df2 = pd.DataFrame(columns = TYPO_MED)
        sum_list = []
        sum_list2 = []
        for x in TYPO_MED:
            l_score_typo = self.variationsPerScore(df, 'XANTWOORD', x)
            print(l_score_typo)
            sum_list.append(sum(l_score_typo[0:6]))
            sum_list2.append(sum(l_score_typo[0:5]))
            typo_df[x] = self.normalizeScores(l_score_typo[0:6])
            typo_df2[x] = self.normalizeScores(l_score_typo[0:5])
        p1 = self.createCategoryPlot(typo_df, TYPO_MED, sum_list, 'Occurences of variations per Medication calculated with the Damerau-Levenshtein ', 'Damerau-Levenshtein score', 'Occurence of medication (normalized by unity)')
        p2 = self.createCategoryPlot(typo_df2, TYPO_MED, sum_list2,  'Occurences of variations per Medication calculated with the Damerau-Levenshtein ', 'Damerau-Levenshtein score', 'Occurence of medication (normalized by unity)')
        output_file(r'output_files/MedTypoDist6.html')
        save(p1)
        output_file(r'output_files/MedTypoDist5.html')
        save(p2)
        typo_df.to_csv(r'input_files/DF_CutOff.csv', sep='|', index=False)
        return
    
    def scopeMed(self, sentence, scope=5):
        """
        """
        sticky_chars = r'([!#?,.:";@\-\+\\/&=$\]\[<>\'\*`â€™\(\)\d])'
        sentence = re.sub(sticky_chars, r' ', sentence)
        words = sentence.split()
        counter = 0
        window = []
        # windows = [window1, window2]
        for x in words:
            if x in MEDICINE:
                window = []
                for y in range(-scope,scope+1):
                    try:
                        window.append(words[counter+y])
                    except (IndexError, ValueError):
                        window.append(np.nan)
                
            counter += 1
        if window == []:
            window.extend([np.nan]*scope*2+[np.nan])
        return window
    
    def frequencyWordsInScope(self, scope=5):
        col_list = []
        for x in range(-scope, scope+1):
            if x != 0:
                col_list.append('pos ' + str(x))
            else: 
                col_list.append('medicine')
        scope_df = self.df.copy()
        scope_df['XANTWOORD'] = self.df['XANTWOORD'].apply(lambda x: self.handleAcronym(str(x)))  
        scope_df['XANTWOORD'] = scope_df['XANTWOORD'].apply(lambda x: self.handleAcronym(str(x), PARENT_MAP))
        scope_df = getMedicineSubset(scope_df)
        window = scope_df['XANTWOORD'].apply(lambda x: self.scopeMed(str(x), scope))
        df_scope = pd.DataFrame(columns=['Woord', 'Position'])
        count = 0
        for row in window:
            ix = -5
            for y in row:
                if ix != 0:
                    df_scope.loc[count] = [y , ix]
                    count += 1
                ix += 1
        df_top_freq = df_scope.groupby(["Woord", "Position"])['Woord'].agg(
            {"code_count": len}).sort_values(
            "code_count", ascending=False).reset_index()
        df_top_freq.to_csv('output_files/FreqWordScope.csv', sep='|', index=False)
        return df_top_freq
    
    

class ValidateTypoAlgorithm(object):
    def __init__(self, df_nomed, df_notypo, df_typo, df_gold):
        self.df_nomed = df_nomed
        self.df_notypo = df_notypo
        self.df_typo = df_typo
        self.df_gold = df_gold
        self.total_med = 0
        self.lines_med = 0
        col_list = ['Index', 'XANTWOORD', 'MED_FOUND_ALGO']
        col_list.extend([col for col in self.df_gold.columns if 'Drug' in col])
        col_list.append('Score')
        self.col_list = col_list
        self.df_mismatches = pd.DataFrame(columns=self.col_list)
        self.performance = {'true_pos' : 0, 'true_neg': 0, 'false_pos': 0, 'false_neg': 0}
        self.perf_line = {'true_pos' : 0, 'true_neg': 0, 'false_pos': 0, 'false_neg': 0}
        self.createTokens()
        
    def getPerformance(self):
        return self.performance
    
    def getPerformanceMedicine(self):
        return self.d_med_perf
    
    def getPerformancePerLine(self):
        return self.perf_line
        
    def getMismatches(self):
        return self.df_mismatches
    
    def getSumMed(self):
        return self.total_med
    
    def getLinesMed(self):
        return self.lines_med
    
    def createTokens(self):
        """
        create dictionary where every med is associated with a token (char)
        """
        self.d_token_med = {np.nan : ' ', 'triamcinolon' : 'z'}
        for x in range(len(MEDICINE)):
            self.d_token_med[MEDICINE[x]] = chr(x+97)
        self.d_med_token = {' ' : np.nan, 'z' : 'triamcinolon'}
        for x in range(len(MEDICINE)):
            self.d_med_token[chr(x+97)] = MEDICINE[x]
        self.d_med_perf = {'triamcinolon' : [0, 0, 0, 0], 'methylprednisolon' : [0, 0, 0, 0]}
        for x in range(len(MEDICINE)):
            # {'true_pos' : 0, 'true_neg': 0, 'false_pos': 0, 'false_neg': 0}
            self.d_med_perf[MEDICINE[x]] = [0, 0, 0, 0] 
        
    def calculatePerformanceTotal(self):
        self.total_med = 0
        self.df_mismatches = pd.DataFrame(columns=self.col_list)
        self.performance = {'true_pos' : 0, 'true_neg': 0, 'false_pos': 0, 'false_neg': 0}
        for x in [self.df_nomed, self.df_notypo, self.df_typo]:
            self.calculateMatchesSample(x)
        return
    
    def listMeds(self, text):
        """
        Creates a list consisting of all the meds found in the sentence. 
        This function also appends Nan if no med is found (this is
        an essential step for comparison).
        """
        med = re.compile(r'\b({})\b'.format('|'.join(MEDICINE)), 
                                              flags=re.IGNORECASE|re.DOTALL)
        l_meds = re.findall(med, text) 
        return l_meds
    
    def calculateMatchesSample(self, df):
        for ix, row in df.iterrows():
            l_meds = self.listMeds(row['XANTWOORD'])
            self.matchWithGold(l_meds, ix)
        return
    
    def exceptionsGold(self, ix, med_cols):
        # Group names not taken into account
        d_exception = {'pred' : 'prednison', 'sulfasalzine' : 'sulfasalazine', 'methylprednison' : 'depomedrol', 
                       'methotrexaaat' : 'methotrexaat', 's' : np.nan, 'metho' : 'methotrexaat', 
                      'azathioprine ': 'azathioprine', 'cetroluzimab': 'certolizumab', 'depo': 'depomedrol', 
                      'pre' : 'prednison', 'mehto' : 'methotrexaat','anit-tnf' : 'anti-tnf', 'predn' : 'prednison', 
                       'gou' : 'goud', 'corticostreoiden' : 'corticosteroiden', 'jak' : np.nan,
                       'plaquenil' : 'hydroxychloroquine', 'dmard' : np.nan, 'anti-tnf': np.nan, 'jak-inhibitor' : np.nan}
        
        gold_meds = self.df_gold.loc[ix][med_cols]        
        for n, i in enumerate(gold_meds):
            if i in d_exception.keys():
                gold_meds[n] = d_exception[i]
        return gold_meds
    
    def matchWithGold(self, l_meds, ix):
        """
        This function matches the row corrected by the typoAlgorithm
        with the same row from the golden standard. All of the 
        found medications are compared. 
        
        The medications are first tokenized with a single char
        which results in two strings: one for the typoAlgorithm and
        one for the golden standard. When both strings are exactly 
        the same: every medication is counted as a true positive. 
        
        If the strings differ in size then a damerau-Levenshtein
        is applied on the strings, which results in a dataframe. 
        This dataframe is then used to optimize the strings in 
        order to compare them fairly.
        
        During this process the performance dictionary,
        containing the sum of true- & false positives and the 
        true- & false negatives is updated consistently. The 
        dataframe with the mismatches (df_mismatches) is updated
        aswell. 
        
        ix = index of the row
        l_meds = list of medication 
        gold_meds - meds according to golden standard
        """ 
        med_cols = [col for col in self.df_gold.columns if 'Drug' in col]
        
        scope=len(med_cols)
        conv_meds = [self.d_token_med[x] for x in l_meds]
        str_med = ''.join(conv_meds)
        conv_gold = []
        gold_meds = self.exceptionsGold(ix, med_cols)
        for x in gold_meds:
            try:
                conv_gold.append(self.d_token_med[x])
            except: 
                print('unrecognized med in gold standard: ', x)
        str_gold = ''.join(conv_gold)
        str_gold = str_gold.replace(' ', '')
        if str_gold != '':
            self.lines_med += 1
        for med in MEDICINE:
            if med not in gold_meds and med not in l_meds: 
                # true negative += 1
                self.d_med_perf[med][1] += 1
        self.total_med += len(str_gold)
        if str_med != str_gold:
            old_performance = self.performance.copy()
            d = self.damerauLevenshteinDictionary(str_med, str_gold)
            new_string = self.alignStringsForComparison(str_med, str_gold, d)
            new_meds = [self.d_med_token[x] for x in new_string]
            if len(new_meds) != scope:
                new_meds.extend([np.nan]*(scope-len(new_meds)))
            self.determineDifferences(new_meds, ix, gold_meds)
            self.calculatePerformanceLine(str_med, str_gold)
            mismatch_row = [ix,  self.df_gold.loc[ix]['XANTWOORD'], str(l_meds)]
            mismatch_row.extend(gold_meds)
            mismatch_row.append(self.calculateLocalPerformance(old_performance))
            print(str_med, new_string, str_gold, str(self.calculateLocalPerformance(old_performance)))
            mismatch = pd.DataFrame([mismatch_row], columns=list(self.df_mismatches.columns.values))
            self.df_mismatches = self.df_mismatches.append(mismatch, ignore_index=True)
        elif str_med != '' and str_gold != '' :
            for x in range(len(str_med)):
                self.d_med_perf[l_meds[x]][0] = self.d_med_perf[l_meds[x]][0] + 1
                self.performance['true_pos'] += 1
            self.perf_line['true_pos'] += 1
        else :
            self.performance['true_neg'] += 1
            self.perf_line['true_neg'] += 1
        return
    
    def calculatePerformanceLine(self, str_med, str_gold):
        """
        This function calculates the performance per line,
        which means that there cant be multiple false positives or
        false negatives per line. There can only be one of each type. 
        """
        false_pos = 0
        false_neg = 0
        if len(str_med) > len(str_gold):
            str_gold += ' ' * (len(str_med) - len(str_gold))
        elif len(str_gold) > len(str_med):
            str_med += ' ' * (len(str_gold) - len(str_med))
        for x in range(len(str_med)):
            if str_med[x] == ' ' and str_gold[x] != ' ':
                false_neg = 1
            elif str_med[x] != ' ' and str_gold[x] == ' ':
                false_pos = 1
            elif str_med[x] != str_gold[x]:
                false_pos = 1
                false_neg = 1
        #if false_pos > 0 or false_neg > 0:
        #print(str_med + '\t' + str_gold + '\t' + str([false_pos, false_neg]))
        self.perf_line['false_pos'] += false_pos
        self.perf_line['false_neg'] += false_neg
        return
    
    def damerauLevenshteinDictionary(self, s, t):
        """
        Calculates the levenshtein score, which is the total sum of
        substitutions, insertions, deletions and (small) transpositions.
        https://www.guyrutenberg.com/2008/12/15/damerau-levenshtein-distance-in-python/
        """
        s = ' ' + s
        t = ' ' + t
        d = {}
        lenstr1 = len(s)
        lenstr2 = len(t)
        for i in range(-1,lenstr1+1):
            d[(i,-1)] = i+1
        for j in range(-1,lenstr2+1):
            d[(-1,j)] = j+1
     
        for i in range(lenstr1):
            for j in range(lenstr2):
                if s[i] == t[j]:
                    cost = 0
                else:
                    cost = 1
                d[(i,j)] = min(
                               d[(i-1,j)] + 1, # deletion
                               d[(i,j-1)] + 1, # insertion
                               d[(i-1,j-1)] + cost, # substitution
                              )
                if i and j and s[i]==t[j-1] and s[i-1] == t[j]:
                    d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
        return d
    
    def alignStringsForComparison(self, string1, string2, dist_matrix):
        """
        This function aligns both strings by deleting characters (meds) and/ or
        adding whitespace. So that both strings can be fairly compared.
        
        This function prevents the following scenario's:
        
        (Scenario 1:)
        [med A] [med C] [med D] [med E]
           |       x       x       x       x
        [med A] [med B] [med C] [med D] [med E]
    
            result: 1 true positive; 3 false positives; 1 false negative
        
        [med A] [     ] [med C] [med D] [med E]
           |       x       |       |       |
        [med A] [med B] [med C] [med D] [med E]
            
            result: 4 true positive; 1 false negative;
            
        (Scenario 2:)
        [med A] [med B] [med C] [med D] [med E]
           |       x       x       x       x
        [med A] [med C] [med D] [med E]
        
            result: 1 true positive; 4 false positives;
            
        [med A] [med C] [med D] [med E]
           |       |       |       |
        [med A] [med C] [med D] [med E]
            
            result: 4 true positive; 1 false positive;
        
        Variables:
            string1 = tokenized list of meds found by typo algorithm
            string2 = tokenized list of meds in golden standard
            dist_matrix = damerau levenshtein matrix between string1 & string2
        """
        string1 = ' ' + string1
        string2 = ' ' + string2
        i, j = len(string1), len(string2)
        i -= 1
        j -= 1
        ops = list()
        
        while i != -1 and j != -1:
            index = np.argmin([dist_matrix[i-1, j-1], dist_matrix[i, j-1], dist_matrix[i-1, j]])
            if index == 0:
                if dist_matrix[i, j] > dist_matrix[i-1, j-1]:
                    ops.insert(0, ('replace', i - 1, j - 1))
                i -= 1
                j -= 1
            elif index == 1:
                ops.insert(0, ('insert', i))
                j -= 1
            elif index == 2:
                ops.insert(0, ('delete', i))
                i -= 1
        new_string = self.createNewStringOps(ops, string1)
        return new_string[1:]
    
    def createNewStringOps(self, ops, string):
        """
        This function edits the string of the typoAlgorithm
        based on the ops ( list of changes ). The ops is 
        based on the results of the damerau levenshtein table.
        The function returns an optimized string that can be aligned
        to the golden standard string.
        
        Variables:
            string = compressed list of medication from the
                typoAlgorithm (a.k.a tokenized)  
        
        Output: 
            new_string: a string that can be aligned to the
                golden standard string.
        """
        shift = 0
        new_string = list(string)
        for op in ops:
            i = op[1]
            if op[0] == 'delete':
                self.d_med_perf[self.d_med_token[new_string[i + shift]]][2] = \
                    self.d_med_perf[self.d_med_token[new_string[i + shift]]][2] + 1
                del new_string[i + shift]
                self.performance['false_pos'] += 1
                shift -= 1
            elif op[0] == 'insert':
                new_string.insert(i + shift + 1, ' ')
                shift += 1
        new_string = ''.join(new_string)
        return new_string
    
    def determineDifferences(self, l_meds, ix, gold_meds):
        """
        Iterates through every medication from the typo algorithm
        chronologically. When this medication matches with the medication
        in the golden standard (must be at same position in sequence) it is
        considered to be a true positive.
        
        Other conditions:
        a) If typoAlgorithm finds an medication that is not present
            in the golden standard: false positive. 
            
            Implication: There is either no medication (None = float) 
            or another medication in the golden standard. 
            
        b) If typoAlgorithm doesn't find the medication found in the 
            golden standard: false negative
        """
        # med_cols = [col for col in self.df_gold.columns if 'Drug' in col]
        for x in range(len(l_meds)):
            med_validated = gold_meds[x] #self.df_gold.loc[ix][med_cols[x]]
            if med_validated not in MEDICINE:
                break
            if type(med_validated) != float:
                med_validated = med_validated.replace(' ', '')
            if l_meds[x] == med_validated:
                self.performance['true_pos'] += 1
                self.d_med_perf[med_validated][0] = self.d_med_perf[med_validated][0] + 1
            elif type(l_meds[x]) == float and type(med_validated) != float:
                self.performance['false_neg'] += 1
                self.d_med_perf[med_validated][3] = self.d_med_perf[med_validated][3] + 1
            elif l_meds[x] != med_validated and type(l_meds[x]) != float and \
                type(med_validated) != float:
                self.performance['false_pos'] += 1
                self.d_med_perf[l_meds[x]][2] = self.d_med_perf[l_meds[x]][2] + 1
            elif type(l_meds[x]) != float and type(med_validated) == float:
                self.performance['false_pos'] += 1
                self.d_med_perf[l_meds[x]][2] = self.d_med_perf[l_meds[x]][2] + 1
        return
    
    def calculateLocalPerformance(self, old_perf):
        perf = self.performance
        stri_perf = "true pos: " + str(int(perf['true_pos']) - int(old_perf['true_pos']))
        stri_perf += "| false pos: " + str(int(perf['false_pos']) - int(old_perf['false_pos']))
        stri_perf += "| true neg: " + str(int(perf['true_neg']) - int(old_perf['true_neg']))
        stri_perf += "| false neg: " + str(int(perf['false_neg']) - int(old_perf['false_neg']))
        return stri_perf

class ContextProcessing(object):
    """
        Input: df (dataframe) -> context exploration or test set
    """
    def __init__(self, df, path_mod='../corpus/modifiersNL.yml', \
                 path_tar='../corpus/targets.yml'):
        self.df = df
        self.modifiers = itemData.get_items(
            path_mod, url=False)
        self.targets = itemData.get_items(
            path_tar, url=False)
    
    def getDF(self):
        return self.df
    
    def setDF(self, df):
        self.df = df
        return
    
    def setModifiers(self, path_mod, path_tar):
        self.modifiers = itemData.get_items(
            path_mod, url=False)
        self.targets = itemData.get_items(
            path_tar, url=False)
        return
    
    def predict(self):
        l_context= self.df['XANTWOORD'].apply(lambda x: self.radiag(str(x))).tolist()
        self.df = self.df.assign(PRED=[l_context[x][0] for x in range(len(l_context))])
        self.df = self.df.assign(CONTEXT=[l_context[x][1] for x in range(len(l_context))])
        return self.df

    def markup_sentence(self, s, modifiers, targets, prune_inactive=True):
        """
        Markup sentence -> based on pyConText example
        """
        markup = pyConText.ConTextMarkup()
        markup.setRawText(s)
        markup.cleanText()
    
        markup.markItems(modifiers, mode="modifier")
        markup.markItems(targets, mode="target")
        markup.pruneMarks()
        markup.dropMarks('Exclusion')
        # apply modifiers to any targets within the modifiers scope
        markup.applyModifiers()
        markup.pruneSelfModifyingRelationships()
        if prune_inactive:
            markup.dropInactiveModifiers()
        return markup
    
    def readContext(self, report):
        """
        Read Context -> based on pyConText example
        """
        context = pyConText.ConTextDocument()
        blob = TextBlob(report.lower())
        rslts = []
        for s in blob.sentences:
            m = self.markup_sentence(s.raw, modifiers=self.modifiers, targets=self.targets)
            rslts.append(m)
        for r in rslts:
            context.addMarkup(r)
        return context

    def most_common(self, lst):
        """
        This function selects the most occuring element in a list
        """
        return max(set(lst), key=lst.count)
    
    def radiag(self, reports):
        """
        Simple algorithm to interpret the context provided by pyContextNLP
        
        Currently the most frequent marker surrounding the target 
        determines wheter or not it is an RA-case. 
        
        -> maybe use numeric values within a spectrum:
            definite_negated_existence = 0
            definite_existence = 1
        -> if the following is the case:
            [prob_neg, def_neg, amb, prob_ex, prob_ex] ->
        """
        reports = reports.split('^')
        lst_prediction = []
        spec = []
        for report in reports:
            prediction = 'no ra found'
            context = self.readContext(report.lower())
            contextTree = xml.etree.ElementTree.fromstring(context.getXML())
            list_neg = ['probable_negated_existence', 'definite_negated_existence', 'ambivalent_existence']
            for sect in contextTree.findall('section'):
                for mk in sect.findall('ConTextMarkup'):
                    for nodes in mk.findall('nodes'):
                        if nodes.getchildren() != []:
                            for node in nodes.findall('node'):
                                if str(node[0].text) == ' target ':
                                    tags = len(node.getchildren()) - 2
                                    for tag in range(tags):
                                        if len(node[tag+2][1].text) > 0:
                                            if node[tag+2][1].text.split("'")[1] not in list_neg:
                                                prediction = 'y'
                                                lst_prediction.append('y')
                                            else :
                                                prediction = 'n'
                                                lst_prediction.append('n')
                                            spec.append(node[tag+2][1].text.split("'")[1])
                                    if prediction != 'n':
                                        lst_prediction.append('y')
            #if 'waarschijnlijk' in report.lower():
            #    display(HTML(html.mark_document_with_html(context,colors = clrs, default_color="black")))
        if lst_prediction != []:
            prediction = self.most_common(lst_prediction) # improve
        else : 
            prediction = 'n'
        return prediction, spec
    
    
    
        
        