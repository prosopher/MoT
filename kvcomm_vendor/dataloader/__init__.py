from importlib import import_module


def get_evaluator(test_task: str):
    if test_task == "countries":
        return import_module("dataloader.countries").CountriesEvaluator()
    elif test_task == "tipsheets":
        return import_module("dataloader.tipsheets").TipsheetsEvaluator()
    elif test_task == "boolq":
        return import_module("dataloader.boolq").BoolQEvaluator()
    elif test_task == "pubmedqa":
        return import_module("dataloader.pubmedqa").PubMedQAEvaluator()
    elif test_task == "squad":
        return import_module("dataloader.squad").SQuADEvaluator()
    elif test_task == "newsqa":
        return import_module("dataloader.newsqa").NewsQAEvaluator()
    elif test_task == "hotpotqa":
        return import_module("dataloader.hotpotqa").HotpotQAEvaluator()
    elif test_task == "hotpotqa_full":
        return import_module("dataloader.hotpotqa").HotpotQAEvaluator(n_samples=None)
    elif test_task == "qasper":
        return import_module("dataloader.qasper").QaSperEvaluator()
    elif test_task == "qasper_full":
        return import_module("dataloader.qasper").QaSperEvaluator(n_samples=None)
    elif test_task == "musique":
        return import_module("dataloader.musique").MuSiQueEvaluator()
    elif test_task == "musique_full":
        return import_module("dataloader.musique").MuSiQueEvaluator(n_samples=None)
    elif test_task == "multifieldqa_en":
        return import_module("dataloader.multifieldqa_en").MultiFieldQAEnEvaluator()
    elif test_task == "twowikimqa":
        return import_module("dataloader.twowikimqa").TwoWikiMQAEvaluator()
    elif test_task == "tmath":
        return import_module("dataloader.tmath").TMathEvaluator()
    elif test_task == "repobench":
        return import_module("dataloader.repobench").RepoBenchEvaluator()
    elif test_task == "samsum":
        return import_module("dataloader.samsum").SAMSumEvaluator()
    else:
        raise ValueError(f"Unsupported task name: {test_task}")


def get_multi_agent_evaluator(test_task: str):
    if test_task == "hotpotqa":
        return import_module("dataloader.hotpotqa").HotpotQAEvaluator(multi_agent=True)
    elif test_task == "musique":
        return import_module("dataloader.musique").MuSiQueEvaluator(multi_agent=True)
    elif test_task == "twowikimqa":
        return import_module("dataloader.twowikimqa").TwoWikiMQAEvaluator(multi_agent=True)
    else:
        raise ValueError(f"Unsupported task name: {test_task}")


def get_mix_evaluator(test_task: str, mix_method: str):
    if test_task == "countries_tipsheets":
        return import_module("dataloader.countries_tipsheets").CountriesTipsheetsEvaluator(mix_method=mix_method)
    elif test_task == "countries_multifieldqa":
        return import_module("dataloader.countries_multifieldqa_en").CountriesMultiFieldQAEvaluator(mix_method=mix_method)
    else:
        raise ValueError(f"Unsupported task name: {test_task}")
