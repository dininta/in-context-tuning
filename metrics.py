import numpy as np
import string
import re
from collections import Counter
from rouge import Rouge
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef, f1_score


METRICS = {
    'acronym_identification': 'EM',
    'ade_corpus_v2-classification': 'Classification-F1',
    'ade_corpus_v2-dosage': 'EM',
    'ade_corpus_v2-effect': 'EM',
    'adversarialqa': 'QA-F1',
    'aeslc': 'Rouge-L',
    'ag_news': 'Classification-F1-fixed', # 'Classification-F1',
    'ai2_arc': 'QA-F1', # 'ACC',
    'amazon_polarity': 'Classification-F1',
    'anli': 'Classification-F1-fixed', # 'Classification-F1',
    'app_reviews': 'Pearson-Correlation', 
    'aqua_rat': 'QA-F1', # 'ACC',
    'art': 'ACC',
    'aslg_pc12': 'EM',
    'biomrc': 'QA-F1',
    'blimp-anaphor_gender_agreement': 'ACC',
    'blimp-anaphor_number_agreement': 'ACC',
    'blimp-determiner_noun_agreement_with_adj_irregular_1': 'ACC',
    'blimp-ellipsis_n_bar_1': 'ACC',
    'blimp-ellipsis_n_bar_2': 'ACC',
    'blimp-existential_there_quantifiers_1': 'ACC',
    'blimp-irregular_past_participle_adjectives': 'ACC',
    'blimp-sentential_negation_npi_licensor_present': 'Classification-F1-fixed', # 'ACC',
    'blimp-sentential_negation_npi_scope': 'Classification-F1-fixed', # 'ACC',
    'blimp-wh_questions_object_gap': 'ACC',
    'boolq': 'ACC',
    'break-QDMR': 'EM',
    'break-QDMR-high-level': 'EM',
    'circa': 'Classification-F1-fixed', # 'Classification-F1',
    'climate_fever': 'Classification-F1',
    'codah': 'QA-F1', # 'Classification-F1',
    'common_gen': 'Rouge-L',
    'commonsense_qa': 'QA-F1', # 'ACC',
    'cos_e': 'Rouge-L',
    'cosmos_qa': 'QA-F1', # 'ACC',
    'crawl_domain': 'EM',
    'crows_pairs': 'ACC',
    'dbpedia_14': 'Classification-F1-fixed', # 'Classification-F1',
    'definite_pronoun_resolution': 'ACC',
    'discovery': 'Classification-F1',
    'dream': 'QA-F1', # 'ACC',
    'duorc': 'QA-F1',
    'e2e_nlg_cleaned': 'Rouge-L',
    'eli5-askh': 'Rouge-L',
    'eli5-asks': 'Rouge-L',
    'eli5-eli5': 'Rouge-L',
    'emo': 'Classification-F1-fixed', # 'Classification-F1',
    'emotion': 'Classification-F1',
    'empathetic_dialogues': 'Rouge-L',
    'ethos-directed_vs_generalized': 'Classification-F1',
    'ethos-disability': 'Classification-F1-fixed', # 'Classification-F1',
    'ethos-gender': 'Classification-F1',
    'ethos-national_origin': 'Classification-F1',
    'ethos-race': 'Classification-F1-fixed', # 'Classification-F1',
    'ethos-religion': 'Classification-F1-fixed', # 'Classification-F1',
    'ethos-sexual_orientation': 'Classification-F1-fixed', # 'Classification-F1',
    'financial_phrasebank': 'Classification-F1-fixed', # 'Classification-F1',
    'freebase_qa': 'QA-F1', # 'EM',
    'gigaword': 'Rouge-L',
    'glue-cola': 'Classification-F1-fixed', # 'Matthew-Correlation',
    'glue-mnli': 'Classification-F1-fixed', # 'ACC',
    'glue-mrpc': 'Classification-F1-fixed', # 'ACC',
    'glue-qnli': 'Classification-F1-fixed', # 'ACC',
    'glue-qqp': 'Classification-F1-fixed', # 'ACC',
    'glue-rte': 'Classification-F1-fixed', # 'ACC',
    'glue-sst2': 'ACC',
    'glue-wnli': 'Classification-F1-fixed', # 'ACC',
    'google_wellformed_query': 'ACC',
    'hate_speech18': 'Classification-F1',
    'hate_speech_offensive': 'Classification-F1',
    'hatexplain': 'Classification-F1-fixed', # 'Classification-F1',
    'health_fact': 'Classification-F1',
    'hellaswag': 'QA-F1', # 'ACC',
    'hotpot_qa': 'QA-F1',
    'imdb': 'Classification-F1',
    'jeopardy': 'EM',
    'kilt_ay2': 'EM',
    'kilt_fever': 'ACC',
    'kilt_hotpotqa': 'EM',
    'kilt_nq': 'EM',
    'kilt_trex': 'EM',
    'kilt_wow': 'Rouge-L',
    'kilt_zsre': 'EM',
    'lama-conceptnet': 'EM',
    'lama-google_re': 'EM',
    'lama-squad': 'EM',
    'lama-trex': 'EM',
    'liar': 'Classification-F1',
    'limit': 'EM',
    'math_qa': 'QA-F1', # 'ACC',
    'mc_taco': 'ACC',
    'medical_questions_pairs': 'ACC',
    'mocha': 'Pearson-Correlation',
    'multi_news': 'Rouge-L',
    'numer_sense': 'EM',
    'onestop_english': 'Classification-F1',
    'openbookqa': 'QA-F1', # 'ACC',
    'paws': 'Classification-F1-fixed', # 'Classification-F1',
    'piqa': 'ACC',
    'poem_sentiment': 'Classification-F1',
    'proto_qa': 'EM',
    'qa_srl': 'EM',
    'qasc': 'QA-F1', # 'ACC',
    'quail': 'ACC',
    'quarel': 'QA-F1', # 'ACC',
    'quartz-no_knowledge': 'QA-F1', # 'ACC',
    'quartz-with_knowledge': 'QA-F1', # 'ACC',
    'quoref': 'QA-F1',
    'race-high': 'ACC',
    'race-middle': 'QA-F1', # 'ACC',
    'reddit_tifu-title': 'Rouge-L',
    'reddit_tifu-tldr': 'Rouge-L',
    'ropes': 'QA-F1',
    'rotten_tomatoes': 'Classification-F1',
    'samsum': 'Rouge-L',
    'scicite': 'Classification-F1',
    'sciq': 'QA-F1', # 'ACC',
    'scitail': 'Classification-F1-fixed', # 'Classification-F1',
    'search_qa': 'EM',
    'sick': 'Classification-F1-fixed', # 'Classification-F1',
    'sms_spam': 'Classification-F1',
    'social_i_qa': 'QA-F1', # 'ACC',
    'spider': 'EM',
    'squad-with_context': 'QA-F1',
    'squad-no_context': 'EM',
    'superglue-cb': 'Classification-F1-fixed', # 'ACC',
    'superglue-copa': 'QA-F1', # 'ACC',
    'superglue-multirc': 'EM',
    'superglue-record': 'QA-F1',
    'superglue-rte': 'Classification-F1-fixed', # 'ACC',
    'superglue-wic': 'ACC',
    'superglue-wsc': 'ACC',
    'swag': 'QA-F1', # 'ACC',
    'tab_fact': 'Classification-F1',
    'trec': 'Classification-F1',
    'trec-finegrained': 'Classification-F1',
    'tweet_eval-emoji': 'Classification-F1',
    'tweet_eval-emotion': 'Classification-F1',
    'tweet_eval-hate': 'Classification-F1',
    'tweet_eval-irony': 'Classification-F1-fixed', # 'Classification-F1',
    'tweet_eval-offensive': 'Classification-F1',
    'tweet_eval-sentiment': 'Classification-F1',
    'tweet_eval-stance_abortion': 'Classification-F1',
    'tweet_eval-stance_atheism': 'Classification-F1',
    'tweet_eval-stance_climate': 'Classification-F1',
    'tweet_eval-stance_feminist': 'Classification-F1',
    'tweet_eval-stance_hillary': 'Classification-F1',
    'tweet_qa': 'QA-F1',
    'web_questions': 'EM',
    'wiki_auto': 'Classification-F1',
    'wiki_bio': 'Rouge-L',
    'wiki_qa': 'Classification-F1-fixed', # 'Classification-F1',
    'wiki_split': 'Rouge-L',
    'wikisql': 'EM',
    'wino_grande': 'QA-F1', # 'ACC',
    'wiqa': 'QA-F1', # 'ACC',
    'xsum': 'Rouge-L',
    'yahoo_answers_topics': 'Classification-F1',
    'yelp_polarity': 'Classification-F1',
    'yelp_review_full': 'Pearson-Correlation'
}

LABELS = {
    "dbpedia_14": [
        "company", "educational institution", "artist", "athlete", "office holder", "mean of transportation",
        "building", "natural place", "village", "animal", "plant", "album", "film", "written work"],
    "emo": ["others", "happy", "sad", "angry"],
    "ethos-race": ["false", "true"],
    "ethos-religion": ["false", "true"],
    "financial_phrasebank" : ["negative", "neutral", "positive"],
    "wiki_qa": ["yes", "no"],
    "anli": ["entailment", "neutral", "contradiction"],
    "glue-mnli": ["entailment", "neutral", "contradiction"],
    "glue-qnli": ["yes", "no"],
    "glue-rte": ["yes", "no"],
    "glue-wnli": ["yes", "no"],
    "scitail": ["entailment", "neutral"],
    "sick": ["entailment", "neutral", "contradiction"],
    "superglue-cb": ["entailment", "neutral", "contradiction"],
    "superglue-rte": ["yes", "no"],
    "glue-mrpc": ["yes", "no"],
    "glue-qqp": ["yes", "no"],
    "medical_questions_pairs": ["similar", "dissimilar"],
    "paws": ["yes", "no"],
    "ag_news": ["world politics", "sports", "business", "science and technology"],
    "circa": ["yes", "no", "in the middle, neither yes nor no", "yes, subject to some conditions", "other"],
    "ethos-disability": ["false", "true"],
    "ethos-sexual_orientation": ["false", "true"],
    "glue-cola": ["yes", "no"],
    "hatexplain": ["hate speech", "normal", "offensive"],
    "tweet_eval-irony": ["yes", "no"],
    "blimp-sentential_negation_npi_licensor_present": ["sentence 1", "sentence 2"],
    "blimp-sentential_negation_npi_scope": ["sentence 1", "sentence 2"],
}

def evaluate(predictions, targets, metric, task_name=None):
    def cast_to_float(predictions):
        new_predictions = []
        for prediction in predictions:
            try:
                new_predictions.append(float(prediction.strip()))
            except:
                new_predictions.append(float('NaN'))
        assert len(new_predictions) == len(predictions)
        return new_predictions

    assert len(predictions) == len(targets)

    if metric == "EM":
        ems = []
        for (prediction, dp) in zip(predictions, targets):
            ems.append(get_exact_match_over_list(prediction, dp))
        return np.mean(ems)
    elif metric == "ACC":
        accs = []
        for (prediction, dp) in zip(predictions, targets):
            accs.append(get_accuracy_over_list(prediction, dp))
        return np.mean(accs)
    elif metric == "QA-F1":
        f1s = []
        for (prediction, dp) in zip(predictions, targets):
            f1s.append(get_f1_over_list(prediction, dp))
        return np.mean(f1s)
    elif metric == "Classification-F1":
        return f1_score([dp[0] for dp in targets], predictions, average="macro")
    elif metric == "Matthew-Correlation":
        return get_matthews_corr(targets, predictions)
    elif metric == "Pearson-Correlation":
        predictions = cast_to_float(predictions)
        return pearsonr([float(dp[0]) for dp in targets], predictions)[0]
    elif metric == "Rouge-L":
        rouges = []
        for (prediction, dp) in zip(predictions, targets):
            rouges.append(get_rouge_over_list(prediction, dp))
        return np.mean(rouges)
    elif metric == "Classification-F1-fixed":
        if type(targets[0])==list:
            targets = [x[0].lower() for x in targets]
        predictions = [x.lower() for x in predictions]
        return f1_score(targets, predictions, average='macro', labels=LABELS[task_name])

def get_matthews_corr(targets, predictions):
    # only cola is using this...?
    new_predictions = []
    for prediction in predictions:
        if prediction.strip() == "acceptable":
            new_predictions.append(1.0)
        else:
            new_predictions.append(0.0)
    new_gold = []
    for dp in targets:
        if dp[0] == "acceptable":
            new_gold.append(1.0)
        else:
            new_gold.append(0.0)
    return matthews_corrcoef(new_gold, new_predictions)

def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def accuracy(prediction, ground_truth):
    return prediction.lower() == ground_truth.lower()

def get_rouge_over_list(prediction, groundtruth):
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    if len(remove_punc(prediction)) == 0:
        return 0.0 # during early stages, it might generate nothin?
    # print(prediction)
    rouge = Rouge()
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([rouge.get_scores(prediction, gt, avg=True)["rouge-l"]["f"] for gt in groundtruth])
    return rouge.get_scores(prediction, groundtruth, avg=True)["rouge-l"]["f"]

def get_accuracy_over_list(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([accuracy(prediction, gt) for gt in groundtruth])
    return accuracy(prediction, groundtruth)

def get_f1_over_list(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([qa_f1_score(prediction, gt) for gt in groundtruth])
    return qa_f1_score(prediction, groundtruth)

def get_exact_match_over_list(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match_over_list(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
