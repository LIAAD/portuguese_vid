from typing import List
from collections import defaultdict
import plotly.express as px
from pt_vid.entity.TestResult import TestResult

class Plot:
    
    @staticmethod
    def _extract_data(test_results: List[dict], metric: str, p_pos, p_ner):
        # Create a matrix using nested list comprehensions.
        return [
            [
                next((r[metric] for r in test_results if r['p_pos'] == pos and r['p_ner'] == ner), None)
                for ner in p_ner
            ]
            for pos in p_pos
        ]
    
    @staticmethod
    def heatmap(test_results: List[TestResult]):
        figs = []
        grouped = defaultdict(list)
        
        # Group test results by training dataset names.
        for t in test_results:
            key = str(t.training_result.training_datasets_names)
            grouped[key].append({
                'p_pos': t.training_result.p_pos,
                'p_ner': t.training_result.p_ner,
                'test_f1_score': t.f1_score,
                'test_accuracy': t.accuracy
            })

        # Process each group.
        for dataset_name, data in grouped.items():
            p_pos = sorted({record['p_pos'] for record in data})
            p_ner = sorted({record['p_ner'] for record in data})
            
            for metric in ['test_f1_score', 'test_accuracy']:
                matrix = Plot._extract_data(data, metric, p_pos, p_ner)
                fig = px.imshow(
                    matrix,
                    labels=dict(x="p_pos", y="p_ner", color=metric),
                    x=p_pos,
                    y=p_ner,
                    text_auto=True,
                )
                fig.update_yaxes(autorange="reversed")
                figs.append(fig)

        return figs
