from typing import List
import plotly.express as px
from pt_vid.entity.TestResult import TestResult

class Plot:

    @staticmethod
    def _extract_data(test_results: List[TestResult], metric: str, p_pos, p_ner):
        results = []
        
        for pos in p_pos:
            data = []
            
            for ner in p_ner:
                for test_result in test_results:
                    if test_result['p_pos'] == pos and test_result['p_ner'] == ner:
                        data.append(test_result[metric])
            
            results.append(data)
        
        return results

    def heatmap(test_results: List[TestResult]):
        figs = []

        results = {}

        for test_result in test_results:
            if results.get(str(test_result.training_result.training_datasets_names)) is None:
                results[str(test_result.training_result.training_datasets_names)] = []

            results[str(test_result.training_result.training_datasets_names)].append({
                'p_pos': test_result.training_result.p_pos,
                'p_ner': test_result.training_result.p_ner,
                'test_f1_score': test_result.f1_score,
                'test_accuracy': test_result.accuracy
            })

        for dataset_name in results.keys():
            for metric in ['test_f1_score', 'test_accuracy']:
                p_pos = list(set([result['p_pos'] for result in results[dataset_name]]))
                p_ner = list(set([result['p_ner'] for result in results[dataset_name]]))

                # Sort the values
                p_pos.sort()
                p_ner.sort()

                fig = px.imshow(
                    Plot._extract_data(results[dataset_name], metric, p_pos, p_ner),
                    labels=dict(x="p_pos", y="p_ner", color=metric),
                    x=p_pos,
                    y=p_ner,
                    text_auto=True,
                )

                fig.update_yaxes(autorange="reversed")

                figs.append(fig)

        return figs