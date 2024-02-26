'''
utility functions to process data
'''
LANGUAGE_MAPPING = {'A1':1,'A2':2,'B1':3,'B2':4,'C1':5,'C2':6}
DEGREE_MAPPING = {'none':0,'apprenticeship':1,'bachelor':1,'master':2,'doctorate':3}
SENIORITY_MAPPING = {'none':0,'junior':1,'midlevel':2,'senior':3}

def prepare_data(talent:dict, job:dict) -> tuple:
    '''
    creates tuple of features to be used for train and inference
    '''
    talent_langs =  extract_lang_skills(talent.get('languages'), False)
    job_langs = extract_lang_skills(job.get('languages'))
    lang_reqs_satisfied = meets_lang_requirements(talent_langs, job_langs)
    degree_diff = DEGREE_MAPPING[talent.get('degree')]-DEGREE_MAPPING[job.get('min_degree')]
    seniority_diff = get_seniority_diff(talent.get('seniority'), job.get('seniorities'))
    salary_diff = job.get('max_salary') - talent.get('salary_expectation')
    job_intersection_len = len(get_intersection(talent.get('job_roles'),job.get('job_roles')))
    return (lang_reqs_satisfied,degree_diff,seniority_diff,salary_diff,job_intersection_len)

def extract_lang_skills(x:dict, mh_param:bool=True) -> dict:
    'extracts language skills available if talent and required if job'
    if mh_param:
        return {i.get('title'):LANGUAGE_MAPPING[i.get('rating')] for i in x if i.get('must_have')}
    return {i.get('title'):LANGUAGE_MAPPING[i.get('rating')] for i in x}

def meets_lang_requirements(talent_lang:dict, job_lang:dict) -> int:
    'determines if language requirements for a job are met for a talent'
    for l in job_lang:
        if job_lang.get(l) > talent_lang.get(l,0):
            return 0
    return 1

def get_seniority_diff(talent_s:str, job_s:list) -> int:
    '''
    Calculates a metric to measure seniority of a talent 
    in relation to minimum seniority required for a job
    '''
    job_s = min(SENIORITY_MAPPING.get(i) for i in job_s)
    return SENIORITY_MAPPING.get(talent_s) - job_s

def get_intersection(lst1:list, lst2:list) -> list:
    'Returns unique item interection of 2 lists'
    return list(set(lst1).intersection(lst2))
